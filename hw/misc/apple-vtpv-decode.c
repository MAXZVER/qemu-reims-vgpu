/*
 * apple-vtpv-decode.c - hardware H.264 decoder for the Apple VTPV device
 * (ZCode, 2026-10-22).  Decodes the guest's CMSampleBuffer bitstream via
 * VK_KHR_video_decode_queue on the host GPU (NVDEC on the RTX 4060 verified
 * against ffmpeg ground truth on driver 617.42) and returns NV12 for the
 * write-back into the exported canvas pages.
 *
 * Self-contained: runtime-loads vulkan-1.dll, uses the bundled Vulkan
 * headers (vk/include).  Verified pipeline contract:
 *   - session sized from caps, stdProfileIdc from the stream SPS
 *   - codec RESET in its own submit with an EMPTY parameters object
 *   - Annex-B slices WITH 3-byte start codes at buffer offset 0;
 *     pSliceOffsets point at each start code; H.264 keeps offset 0
 *   - Std PPS MUST carry entropy_coding_mode_flag from the bitstream or
 *     CABAC streams decode to nothing
 *   - Std picture info from the real slice header; frames keep
 *     top/bottom field flags = 0
 *   - result-status query inside the coding scope is mandatory
 *   - copies run ONLY on the decode queue (graphics queue = DEVICE_LOST);
 *     617.42 silently drops the plane-1 copy - UV then falls back to
 *     neutral 0x80 and vtpv_dec_has_chroma() reports false
 */
#include "qemu/osdep.h"
#include "qemu/timer.h"
#include "hw/misc/apple-vtpv-decode.h"

#include <windows.h>
#include "vk/include/vulkan/vulkan.h"
#include "vk/include/vk_video/vulkan_video_codec_h264std.h"
#include "vk/include/vk_video/vulkan_video_codec_h264std_decode.h"
#include "apple-vtpv-uvspv.h"

/* ---------------- bit reader + NAL header parsing ---------------- */

typedef struct VDBitReader {
    const uint8_t *d;
    int pos;
} VDBitReader;

static int vd_bit(VDBitReader *b)
{
    int v = (b->d[b->pos >> 3] >> (7 - (b->pos & 7))) & 1;
    b->pos++;
    return v;
}

static uint32_t vd_bits(VDBitReader *b, int n)
{
    uint32_t v = 0;
    while (n--) {
        v = (v << 1) | vd_bit(b);
    }
    return v;
}

static uint32_t vd_ue(VDBitReader *b)
{
    int z = 0;
    while (!vd_bit(b)) {
        if (++z > 31) {
            return 0;
        }
    }
    return (1u << z) - 1 + (z ? vd_bits(b, z) : 0);
}

static int32_t vd_se(VDBitReader *b)
{
    uint32_t k = vd_ue(b);
    return (k & 1) ? (int32_t)((k + 1) / 2) : -(int32_t)(k / 2);
}

typedef struct VDSps {
    int profile_idc, log2_max_frame_num, poc_type, log2_max_poc_lsb;
    int frame_mbs_only;
    int w, h;
} VDSps;

typedef struct VDSlice {
    int frame_num, idr_pic_id, poc_lsb;
    int is_idr, is_intra, is_ref;
} VDSlice;

static bool vd_parse_sps(const uint8_t *n, StdVideoH264SequenceParameterSet *std,
                         VDSps *out)
{
    VDBitReader b = { n + 1, 0 };
    memset(std, 0, sizeof(*std));
    memset(out, 0, sizeof(*out));
    std->profile_idc = vd_bits(&b, 8);
    std->flags.constraint_set0_flag = vd_bit(&b);
    std->flags.constraint_set1_flag = vd_bit(&b);
    std->flags.constraint_set2_flag = vd_bit(&b);
    std->flags.constraint_set3_flag = vd_bit(&b);
    std->flags.constraint_set4_flag = vd_bit(&b);
    std->flags.constraint_set5_flag = vd_bit(&b);
    vd_bits(&b, 2);
    std->level_idc = vd_bits(&b, 8);
    std->seq_parameter_set_id = vd_ue(&b);
    switch ((int)std->profile_idc) {
    case 100: case 110: case 122: case 244: case 44: case 83: case 86:
    case 118: case 128: case 138: case 139: case 134: case 135:
        std->chroma_format_idc = vd_ue(&b);
        if (std->chroma_format_idc == 3) {
            std->flags.separate_colour_plane_flag = vd_bit(&b);
        }
        std->bit_depth_luma_minus8 = vd_ue(&b);
        std->bit_depth_chroma_minus8 = vd_ue(&b);
        std->flags.qpprime_y_zero_transform_bypass_flag = vd_bit(&b);
        std->flags.seq_scaling_matrix_present_flag = vd_bit(&b);
        if (std->flags.seq_scaling_matrix_present_flag) {
            return false;  /* scaling lists unsupported in this module */
        }
        break;
    default:
        std->chroma_format_idc = 1;
        break;
    }
    std->log2_max_frame_num_minus4 = vd_ue(&b);
    std->pic_order_cnt_type = vd_ue(&b);
    if (std->pic_order_cnt_type == 0) {
        std->log2_max_pic_order_cnt_lsb_minus4 = vd_ue(&b);
    } else if (std->pic_order_cnt_type == 1) {
        std->flags.delta_pic_order_always_zero_flag = vd_bit(&b);
        std->offset_for_non_ref_pic = vd_se(&b);
        std->offset_for_top_to_bottom_field = vd_se(&b);
        uint32_t cyc = vd_ue(&b);
        for (uint32_t i = 0; i < cyc; i++) {
            vd_se(&b);
        }
    } else if (std->pic_order_cnt_type > 2) {
        return false;
    }
    std->max_num_ref_frames = vd_ue(&b);
    std->flags.gaps_in_frame_num_value_allowed_flag = vd_bit(&b);
    std->pic_width_in_mbs_minus1 = vd_ue(&b);
    std->pic_height_in_map_units_minus1 = vd_ue(&b);
    std->flags.frame_mbs_only_flag = vd_bit(&b);
    if (!std->flags.frame_mbs_only_flag) {
        std->flags.mb_adaptive_frame_field_flag = vd_bit(&b);
    }
    std->flags.direct_8x8_inference_flag = vd_bit(&b);
    std->flags.frame_cropping_flag = vd_bit(&b);
    if (std->flags.frame_cropping_flag) {
        std->frame_crop_left_offset = vd_ue(&b);
        std->frame_crop_right_offset = vd_ue(&b);
        std->frame_crop_top_offset = vd_ue(&b);
        std->frame_crop_bottom_offset = vd_ue(&b);
    }
    std->flags.vui_parameters_present_flag = vd_bit(&b);  /* VUI skipped */

    out->profile_idc = std->profile_idc;
    out->log2_max_frame_num = std->log2_max_frame_num_minus4 + 4;
    out->poc_type = std->pic_order_cnt_type;
    out->log2_max_poc_lsb = std->log2_max_pic_order_cnt_lsb_minus4 + 4;
    out->frame_mbs_only = std->flags.frame_mbs_only_flag;
    out->w = (std->pic_width_in_mbs_minus1 + 1) * 16;
    out->h = (std->pic_height_in_map_units_minus1 + 1) * 16 *
             (std->flags.frame_mbs_only_flag ? 1 : 2);
    return true;
}

static void vd_parse_pps(const uint8_t *n, StdVideoH264PictureParameterSet *std,
                         const StdVideoH264SequenceParameterSet *sps)
{
    VDBitReader b = { n + 1, 0 };
    memset(std, 0, sizeof(*std));
    std->pic_parameter_set_id = vd_ue(&b);
    std->seq_parameter_set_id = vd_ue(&b);
    std->flags.entropy_coding_mode_flag = vd_bit(&b);  /* THE critical flag */
    std->flags.bottom_field_pic_order_in_frame_present_flag = vd_bit(&b);
    if (vd_ue(&b)) {
        return;  /* slice groups unsupported */
    }
    std->num_ref_idx_l0_default_active_minus1 = vd_ue(&b);
    std->num_ref_idx_l1_default_active_minus1 = vd_ue(&b);
    std->flags.weighted_pred_flag = vd_bit(&b);
    std->weighted_bipred_idc = vd_bits(&b, 2);
    std->pic_init_qp_minus26 = vd_se(&b);
    std->pic_init_qs_minus26 = vd_se(&b);
    std->chroma_qp_index_offset = vd_se(&b);
    std->flags.deblocking_filter_control_present_flag = vd_bit(&b);
    std->flags.constrained_intra_pred_flag = vd_bit(&b);
    std->flags.redundant_pic_cnt_present_flag = vd_bit(&b);
    switch ((int)sps->profile_idc) {
    case 100: case 110: case 122: case 244: case 44:
        std->flags.transform_8x8_mode_flag = vd_bit(&b);
        std->flags.pic_scaling_matrix_present_flag = vd_bit(&b);
        if (std->flags.pic_scaling_matrix_present_flag) {
            std->flags.pic_scaling_matrix_present_flag = 0;  /* not carried */
        }
        std->second_chroma_qp_index_offset = vd_se(&b);
        break;
    default:
        std->second_chroma_qp_index_offset = std->chroma_qp_index_offset;
        break;
    }
}

static void vd_parse_slice(const uint8_t *n, const VDSps *sps, VDSlice *out)
{
    int nal_ref_idc = (n[0] >> 5) & 3;
    int nal_type = n[0] & 0x1f;
    memset(out, 0, sizeof(*out));
    out->is_idr = (nal_type == 5);
    out->is_ref = nal_ref_idc != 0;
    VDBitReader b = { n + 1, 0 };
    vd_ue(&b);                 /* first_mb_in_slice */
    uint32_t st = vd_ue(&b);
    out->is_intra = (st % 5) == 2;
    vd_ue(&b);                 /* pps id */
    out->frame_num = vd_bits(&b, sps->log2_max_frame_num);
    if (!sps->frame_mbs_only) {
        if (vd_bit(&b)) {
            vd_bit(&b);
        }
    }
    if (out->is_idr) {
        out->idr_pic_id = vd_ue(&b);
    }
    if (sps->poc_type == 0) {
        out->poc_lsb = vd_bits(&b, sps->log2_max_poc_lsb);
    }
}

/* ---------------- loader + proc table ---------------- */

static struct {
    HMODULE lib;
    void *(VKAPI_PTR *GetInstanceProcAddr)(VkInstance, const char *);
    void *(VKAPI_PTR *GetDeviceProcAddr)(VkDevice, const char *);
    VkInstance inst;
    VkPhysicalDevice pd;
} L;

typedef struct VdProcs {
    PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices;
    PFN_vkGetPhysicalDeviceProperties2 GetPhysicalDeviceProperties2;
    PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties2 GetPhysicalDeviceQueueFamilyProperties2;
    PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR GetPhysicalDeviceVideoCapabilitiesKHR;
    PFN_vkCreateDevice CreateDevice;
    PFN_vkDestroyDevice DestroyDevice;
    PFN_vkGetDeviceQueue GetDeviceQueue;
    PFN_vkCreateVideoSessionKHR CreateVideoSessionKHR;
    PFN_vkDestroyVideoSessionKHR DestroyVideoSessionKHR;
    PFN_vkGetVideoSessionMemoryRequirementsKHR GetVideoSessionMemoryRequirementsKHR;
    PFN_vkBindVideoSessionMemoryKHR BindVideoSessionMemoryKHR;
    PFN_vkCreateVideoSessionParametersKHR CreateVideoSessionParametersKHR;
    PFN_vkDestroyVideoSessionParametersKHR DestroyVideoSessionParametersKHR;
    PFN_vkCreateCommandPool CreateCommandPool;
    PFN_vkDestroyCommandPool DestroyCommandPool;
    PFN_vkAllocateCommandBuffers AllocateCommandBuffers;
    PFN_vkBeginCommandBuffer BeginCommandBuffer;
    PFN_vkEndCommandBuffer EndCommandBuffer;
    PFN_vkQueueSubmit QueueSubmit;
    PFN_vkQueueWaitIdle QueueWaitIdle;
    PFN_vkCreateBuffer CreateBuffer;
    PFN_vkDestroyBuffer DestroyBuffer;
    PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements;
    PFN_vkAllocateMemory AllocateMemory;
    PFN_vkFreeMemory FreeMemory;
    PFN_vkMapMemory MapMemory;
    PFN_vkUnmapMemory UnmapMemory;
    PFN_vkFlushMappedMemoryRanges FlushMappedMemoryRanges;
    PFN_vkBindBufferMemory BindBufferMemory;
    PFN_vkCreateImage CreateImage;
    PFN_vkDestroyImage DestroyImage;
    PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements;
    PFN_vkBindImageMemory BindImageMemory;
    PFN_vkCreateImageView CreateImageView;
    PFN_vkDestroyImageView DestroyImageView;
    PFN_vkCmdPipelineBarrier2 CmdPipelineBarrier2;
    PFN_vkCmdBeginVideoCodingKHR CmdBeginVideoCodingKHR;
    PFN_vkCmdControlVideoCodingKHR CmdControlVideoCodingKHR;
    PFN_vkCmdEndVideoCodingKHR CmdEndVideoCodingKHR;
    PFN_vkCmdDecodeVideoKHR CmdDecodeVideoKHR;
    PFN_vkCmdBeginQuery CmdBeginQuery;
    PFN_vkCmdEndQuery CmdEndQuery;
    PFN_vkCmdResetQueryPool CmdResetQueryPool;
    PFN_vkCreateQueryPool CreateQueryPool;
    PFN_vkGetQueryPoolResults GetQueryPoolResults;
    PFN_vkCmdCopyImageToBuffer CmdCopyImageToBuffer;
    PFN_vkInvalidateMappedMemoryRanges InvalidateMappedMemoryRanges;
    PFN_vkCreateSampler CreateSampler;
    PFN_vkDestroySampler DestroySampler;
    PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout;
    PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayout;
    PFN_vkCreatePipelineLayout CreatePipelineLayout;
    PFN_vkDestroyPipelineLayout DestroyPipelineLayout;
    PFN_vkCreateShaderModule CreateShaderModule;
    PFN_vkDestroyShaderModule DestroyShaderModule;
    PFN_vkCreateComputePipelines CreateComputePipelines;
    PFN_vkDestroyPipeline DestroyPipeline;
    PFN_vkCreateDescriptorPool CreateDescriptorPool;
    PFN_vkDestroyDescriptorPool DestroyDescriptorPool;
    PFN_vkAllocateDescriptorSets AllocateDescriptorSets;
    PFN_vkUpdateDescriptorSets UpdateDescriptorSets;
    PFN_vkCmdBindPipeline CmdBindPipeline;
    PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets;
    PFN_vkCmdDispatch CmdDispatch;
} VdProcs;

static VdProcs P;

#define VDPI(name) P.name = (PFN_vk##name)L.GetInstanceProcAddr(L.inst, "vk" #name)
#define VDP(name) P.name = (PFN_vk##name)L.GetDeviceProcAddr(dev, "vk" #name)

static bool vd_load_procs_inst(void)
{
    VDPI(EnumeratePhysicalDevices);
    VDPI(GetPhysicalDeviceProperties2);
    VDPI(GetPhysicalDeviceMemoryProperties);
    VDPI(GetPhysicalDeviceQueueFamilyProperties2);
    VDPI(GetPhysicalDeviceVideoCapabilitiesKHR);
    VDPI(CreateDevice);
    return P.EnumeratePhysicalDevices && P.CreateDevice;
}

static bool vd_load_procs_dev(VkDevice dev)
{
    VDP(DestroyDevice);
    VDP(GetDeviceQueue);
    VDP(CreateVideoSessionKHR);
    VDP(DestroyVideoSessionKHR);
    VDP(GetVideoSessionMemoryRequirementsKHR);
    VDP(BindVideoSessionMemoryKHR);
    VDP(CreateVideoSessionParametersKHR);
    VDP(DestroyVideoSessionParametersKHR);
    VDP(CreateCommandPool);
    VDP(DestroyCommandPool);
    VDP(AllocateCommandBuffers);
    VDP(BeginCommandBuffer);
    VDP(EndCommandBuffer);
    VDP(QueueSubmit);
    VDP(QueueWaitIdle);
    VDP(CreateBuffer);
    VDP(DestroyBuffer);
    VDP(GetBufferMemoryRequirements);
    VDP(AllocateMemory);
    VDP(FreeMemory);
    VDP(MapMemory);
    VDP(UnmapMemory);
    VDP(FlushMappedMemoryRanges);
    VDP(BindBufferMemory);
    VDP(CreateImage);
    VDP(DestroyImage);
    VDP(GetImageMemoryRequirements);
    VDP(BindImageMemory);
    VDP(CreateImageView);
    VDP(DestroyImageView);
    VDP(CmdPipelineBarrier2);
    VDP(CmdBeginVideoCodingKHR);
    VDP(CmdControlVideoCodingKHR);
    VDP(CmdEndVideoCodingKHR);
    VDP(CmdDecodeVideoKHR);
    VDP(CmdBeginQuery);
    VDP(CmdEndQuery);
    VDP(CmdResetQueryPool);
    VDP(CreateQueryPool);
    VDP(GetQueryPoolResults);
    VDP(CmdCopyImageToBuffer);
    VDP(InvalidateMappedMemoryRanges);
    VDP(CreateSampler);
    VDP(DestroySampler);
    VDP(CreateDescriptorSetLayout);
    VDP(DestroyDescriptorSetLayout);
    VDP(CreatePipelineLayout);
    VDP(DestroyPipelineLayout);
    VDP(CreateShaderModule);
    VDP(DestroyShaderModule);
    VDP(CreateComputePipelines);
    VDP(DestroyPipeline);
    VDP(CreateDescriptorPool);
    VDP(DestroyDescriptorPool);
    VDP(AllocateDescriptorSets);
    VDP(UpdateDescriptorSets);
    VDP(CmdBindPipeline);
    VDP(CmdBindDescriptorSets);
    VDP(CmdDispatch);
    return P.CmdDecodeVideoKHR && P.CreateVideoSessionKHR &&
           P.CmdCopyImageToBuffer && P.AllocateCommandBuffers;
}

/* pick first allowed memory type; prefer one with `want` flags */
static bool vd_mem_type(uint32_t bits, uint32_t want, uint32_t *out)
{
    VkPhysicalDeviceMemoryProperties mp;
    P.GetPhysicalDeviceMemoryProperties(L.pd, &mp);
    *out = 0xffffffffu;
    for (uint32_t t = 0; t < mp.memoryTypeCount; t++) {
        if ((bits & (1u << t))) {
            if (*out == 0xffffffffu) {
                *out = t;
            }
            if (want == 0 || (mp.memoryTypes[t].propertyFlags & want) == want) {
                *out = t;
                return true;
            }
        }
    }
    return *out != 0xffffffffu;
}

/* ---------------- decoder context ---------------- */

typedef struct VtpvDecoder {
    uint64_t t0;
    VkInstance inst;
    VkPhysicalDevice pd;
    VkDevice dev;
    VkQueue q;
    uint32_t dfam;
    VkVideoSessionKHR session;
    VkVideoSessionParametersKHR params;
    VkCommandPool pool;
    VkQueryPool qpool;
    VkBuffer bsbuf;
    VkDeviceMemory bsmem;
    uint64_t bs_cap;
    uint64_t bs_size_align;
    /* picture ring: each decoded frame gets its own image (coincide mode:
     * the picture doubles as its DPB reference) so P-frames can reference
     * previous pictures across calls */
    VkImage img[4];
    VkImageView imgview[4];
    VkDeviceMemory imgmem[4];
    uint32_t ring_pos;
    bool ring_used[4];
    bool slot_valid[4];
    StdVideoDecodeH264ReferenceInfo slot_ref[4];
    VkVideoDecodeH264DpbSlotInfoKHR slot_dpb[4];
    VkVideoPictureResourceInfoKHR slot_res[4];
    VkBuffer rbbuf;
    VkDeviceMemory rbmem;
    void *rbmap;

    /* chroma via compute shader: 617.42 silently drops every plane-1 copy
     * path (buffer and image, both queue families), so UV is sampled from
     * the decode image on a compute-capable family instead */
    int uv_on;
    int uv_fam;
    VkQueue uvq;
    VkCommandPool uvpool;
    VkSampler uvsamp;
    VkImageView uvview[4];
    VkDescriptorSetLayout uvdsl;
    VkPipelineLayout uvpl;
    VkPipeline uvpipe;
    VkDescriptorPool uvdpool;
    VkDescriptorSet uvset;
    VkBuffer uvbuf;
    VkDeviceMemory uvmem;
    float *uvmap;
    size_t uvn;

    uint32_t w, h;
    uint8_t sps_hash[8], pps_hash[8];
    StdVideoH264SequenceParameterSet std_sps;
    StdVideoH264PictureParameterSet std_pps;
    VDSps sps;
    bool chroma_ok;
    int poc_prev_ref;   /* pic_order_cnt_type=2 derivation (H.264 8.2.1.3) */
} VtpvDecoder;

static void vd_hash(const uint8_t *d, uint32_t n, uint8_t out[8])
{
    uint64_t h = 1469598103934665603ull;
    for (uint32_t i = 0; i < n; i++) {
        h ^= d[i];
        h *= 1099511628211ull;
    }
    for (int i = 0; i < 8; i++) {
        out[i] = (uint8_t)(h >> (i * 8));
    }
}

VtpvDecoder *vtpv_dec_open(const char *gpu_pref,
                           const uint8_t *sps_nal, uint32_t sps_len,
                           const uint8_t *pps_nal, uint32_t pps_len)
{
    VtpvDecoder *d = g_new0(VtpvDecoder, 1);
    VkResult r;
    d->t0 = qemu_clock_get_ms(QEMU_CLOCK_REALTIME);

    L.lib = LoadLibraryA("vulkan-1.dll");
    if (!L.lib) {
        fprintf(stderr, "vtpv-dec: vulkan-1.dll unavailable\n");
        goto fail0;
    }
    L.GetInstanceProcAddr =
        (void *(VKAPI_PTR *)(VkInstance, const char *))
        GetProcAddress(L.lib, "vkGetInstanceProcAddr");
    L.GetDeviceProcAddr =
        (void *(VKAPI_PTR *)(VkDevice, const char *))
        GetProcAddress(L.lib, "vkGetDeviceProcAddr");
    if (!L.GetInstanceProcAddr || !L.GetDeviceProcAddr) {
        goto fail0;
    }

    PFN_vkCreateInstance createInst = (PFN_vkCreateInstance)
        L.GetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance");
    VkApplicationInfo ai = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "apple-vtpv",
        .apiVersion = VK_MAKE_VERSION(1, 3, 0),
    };
    VkInstanceCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &ai,
    };
    r = createInst(&ici, NULL, &d->inst);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "vtpv-dec: instance rc=%d\n", (int)r);
        goto fail0;
    }
    L.inst = d->inst;
    if (!vd_load_procs_inst()) {
        fprintf(stderr, "vtpv-dec: instance procs missing\n");
        goto fail1;
    }

    bool want_int = gpu_pref &&
                    (!strcmp(gpu_pref, "int") || !strcmp(gpu_pref, "intel"));
    uint32_t nd = 0;
    P.EnumeratePhysicalDevices(d->inst, &nd, NULL);
    if (!nd) {
        goto fail1;
    }
    VkPhysicalDevice *devs = g_new(VkPhysicalDevice, nd);
    P.EnumeratePhysicalDevices(d->inst, &nd, devs);
    for (uint32_t k = 0; k < nd; k++) {
        VkPhysicalDeviceProperties2 p2 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        P.GetPhysicalDeviceProperties2(devs[k], &p2);
        bool is_intel = p2.properties.vendorID == 0x8086;
        if ((want_int && is_intel) || (!want_int && !is_intel)) {
            d->pd = devs[k];
            break;
        }
    }
    g_free(devs);
    if (!d->pd) {
        fprintf(stderr, "vtpv-dec: no matching gpu (pref=%s)\n",
                gpu_pref ? gpu_pref : "nv");
        goto fail1;
    }
    L.pd = d->pd;

    if (!vd_parse_sps(sps_nal, &d->std_sps, &d->sps)) {
        fprintf(stderr, "vtpv-dec: unsupported SPS\n");
        goto fail1;
    }
    vd_parse_pps(pps_nal, &d->std_pps, &d->std_sps);
    d->w = d->sps.w;
    d->h = d->sps.h;
    vd_hash(sps_nal, sps_len, d->sps_hash);
    vd_hash(pps_nal, pps_len, d->pps_hash);
    if (d->std_sps.max_num_ref_frames > 2) {
        d->std_sps.max_num_ref_frames = 2;  /* 617.42 validation quirk */
    }

    StdVideoH264ProfileIdc prof = STD_VIDEO_H264_PROFILE_IDC_HIGH;
    switch ((int)d->sps.profile_idc) {
    case 66: case 88:
        prof = STD_VIDEO_H264_PROFILE_IDC_BASELINE;
        break;
    case 77:
        prof = STD_VIDEO_H264_PROFILE_IDC_MAIN;
        break;
    default:
        break;
    }
    VkVideoDecodeH264ProfileInfoKHR hprof = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PROFILE_INFO_KHR,
        .stdProfileIdc = prof,
        .pictureLayout = VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_PROGRESSIVE_KHR,
    };
    VkVideoProfileInfoKHR vprof = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR,
        .pNext = &hprof,
        .videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR,
        .chromaSubsampling = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR,
        .lumaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
        .chromaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
    };

    uint32_t nq = 0;
    P.GetPhysicalDeviceQueueFamilyProperties2(d->pd, &nq, NULL);
    VkQueueFamilyProperties2 *qs = g_new0(VkQueueFamilyProperties2, nq);
    VkQueueFamilyVideoPropertiesKHR *qv =
        g_new0(VkQueueFamilyVideoPropertiesKHR, nq);
    for (uint32_t k = 0; k < nq; k++) {
        qs[k].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
        qs[k].pNext = &qv[k];
        qv[k].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_VIDEO_PROPERTIES_KHR;
    }
    P.GetPhysicalDeviceQueueFamilyProperties2(d->pd, &nq, qs);
    int dfam = -1;
    for (uint32_t k = 0; k < nq; k++) {
        if ((qs[k].queueFamilyProperties.queueFlags &
             VK_QUEUE_VIDEO_DECODE_BIT_KHR) &&
            (qv[k].videoCodecOperations &
             VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR)) {
            dfam = k;
            break;
        }
    }
    g_free(qs);
    g_free(qv);
    if (dfam < 0) {
        fprintf(stderr, "vtpv-dec: no h264 decode queue\n");
        goto fail1;
    }
    d->dfam = dfam;

    /* compute family for the UV shader path (see struct comment) */
    d->uv_fam = -1;
    {
        VkPhysicalDeviceProperties2 pdp2 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        P.GetPhysicalDeviceProperties2(d->pd, &pdp2);
        if (pdp2.properties.vendorID == 0x10de &&
            !getenv("VTPV_DEC_NOSHADER")) {
            uint32_t nq2 = 0;
            P.GetPhysicalDeviceQueueFamilyProperties2(d->pd, &nq2, NULL);
            VkQueueFamilyProperties2 *qp = g_new0(VkQueueFamilyProperties2, nq2);
            for (uint32_t k = 0; k < nq2; k++) {
                qp[k].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
            }
            P.GetPhysicalDeviceQueueFamilyProperties2(d->pd, &nq2, qp);
            for (uint32_t k = 0; k < nq2; k++) {
                if ((int)k != dfam &&
                    (qp[k].queueFamilyProperties.queueFlags &
                     VK_QUEUE_COMPUTE_BIT)) {
                    d->uv_fam = k;
                    break;
                }
            }
            g_free(qp);
            d->uv_on = d->uv_fam >= 0;
        }
    }

    VkVideoDecodeCapabilitiesKHR dcap = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR };
    VkVideoDecodeH264CapabilitiesKHR hcap = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_CAPABILITIES_KHR,
        .pNext = &dcap };
    VkVideoCapabilitiesKHR cap = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR, .pNext = &hcap };
    r = P.GetPhysicalDeviceVideoCapabilitiesKHR(d->pd, &vprof, &cap);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "vtpv-dec: caps rc=%d\n", (int)r);
        goto fail1;
    }
    d->bs_size_align = cap.minBitstreamBufferSizeAlignment
                       ? cap.minBitstreamBufferSizeAlignment : 256;

    const char *devext[] = {
        VK_KHR_VIDEO_QUEUE_EXTENSION_NAME,
        VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME,
        VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME,
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
        VK_KHR_FORMAT_FEATURE_FLAGS_2_EXTENSION_NAME,
        VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
        VK_KHR_BIND_MEMORY_2_EXTENSION_NAME,
        VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
    };
    float one = 1.f;
    VkDeviceQueueCreateInfo dq[2];
    uint32_t ndq = 0;
    dq[ndq++] = (VkDeviceQueueCreateInfo){
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = d->dfam,
        .queueCount = 1,
        .pQueuePriorities = &one,
    };
    if (d->uv_on) {
        dq[ndq++] = (VkDeviceQueueCreateInfo){
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = (uint32_t)d->uv_fam,
            .queueCount = 1,
            .pQueuePriorities = &one,
        };
    }
    VkPhysicalDeviceSynchronization2Features sync2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
        .synchronization2 = true,
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &sync2,
        .queueCreateInfoCount = ndq,
        .pQueueCreateInfos = dq,
        .enabledExtensionCount = ARRAY_SIZE(devext),
        .ppEnabledExtensionNames = devext,
    };
    r = P.CreateDevice(d->pd, &dci, NULL, &d->dev);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "vtpv-dec: device rc=%d\n", (int)r);
        goto fail1;
    }
    if (!vd_load_procs_dev(d->dev)) {
        fprintf(stderr, "vtpv-dec: device procs missing\n");
        goto fail2;
    }
    P.GetDeviceQueue(d->dev, d->dfam, 0, &d->q);
    if (d->uv_on) {
        P.GetDeviceQueue(d->dev, d->uv_fam, 0, &d->uvq);
    }

    VkExtensionProperties stdHdr = cap.stdHeaderVersion;
    VkVideoSessionCreateInfoKHR vsci = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_CREATE_INFO_KHR,
        .queueFamilyIndex = d->dfam,
        .maxCodedExtent = { d->w, d->h },
        .pictureFormat = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
        .referencePictureFormat = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
        .maxDpbSlots = cap.maxDpbSlots ? cap.maxDpbSlots : 4,
        .maxActiveReferencePictures = cap.maxActiveReferencePictures
                                      ? cap.maxActiveReferencePictures : 4,
        .pVideoProfile = &vprof,
        .pStdHeaderVersion = &stdHdr,
    };
    r = P.CreateVideoSessionKHR(d->dev, &vsci, NULL, &d->session);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "vtpv-dec: session rc=%d\n", (int)r);
        goto fail2;
    }
    uint32_t nb = 0;
    P.GetVideoSessionMemoryRequirementsKHR(d->dev, d->session, &nb, NULL);
    if (nb > 8) {
        nb = 8;
    }
    if (nb) {
        VkVideoSessionMemoryRequirementsKHR mr[8];
        VkBindVideoSessionMemoryInfoKHR bi[8];
        VkDeviceMemory mems[8];
        memset(mr, 0, sizeof(mr));
        memset(bi, 0, sizeof(bi));
        memset(mems, 0, sizeof(mems));
        for (uint32_t k = 0; k < nb; k++) {
            mr[k].sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_MEMORY_REQUIREMENTS_KHR;
        }
        P.GetVideoSessionMemoryRequirementsKHR(d->dev, d->session, &nb, mr);
        for (uint32_t k = 0; k < nb; k++) {
            uint32_t mt;
            if (!vd_mem_type(mr[k].memoryRequirements.memoryTypeBits,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &mt)) {
                goto fail3;
            }
            VkMemoryAllocateInfo mai = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .allocationSize = mr[k].memoryRequirements.size,
                .memoryTypeIndex = mt,
            };
            r = P.AllocateMemory(d->dev, &mai, NULL, &mems[k]);
            if (r != VK_SUCCESS) {
                fprintf(stderr, "vtpv-dec: session mem rc=%d\n", (int)r);
                goto fail3;
            }
            bi[k].sType = VK_STRUCTURE_TYPE_BIND_VIDEO_SESSION_MEMORY_INFO_KHR;
            bi[k].memoryBindIndex = mr[k].memoryBindIndex;
            bi[k].memory = mems[k];
            bi[k].memoryOffset = 0;
            bi[k].memorySize = mr[k].memoryRequirements.size;
        }
        r = P.BindVideoSessionMemoryKHR(d->dev, d->session, nb, bi);
        if (r != VK_SUCCESS) {
            fprintf(stderr, "vtpv-dec: bind session rc=%d\n", (int)r);
            goto fail3;
        }
    }

    VkVideoDecodeH264SessionParametersAddInfoKHR add = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR,
        .stdSPSCount = 1,
        .pStdSPSs = &d->std_sps,
        .stdPPSCount = 1,
        .pStdPPSs = &d->std_pps,
    };
    VkVideoDecodeH264SessionParametersCreateInfoKHR h264ci = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR,
        .maxStdSPSCount = 1,
        .maxStdPPSCount = 1,
        .pParametersAddInfo = &add,
    };
    VkVideoSessionParametersCreateInfoKHR pci = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,
        .pNext = &h264ci,
        .videoSession = d->session,
    };
    r = P.CreateVideoSessionParametersKHR(d->dev, &pci, NULL, &d->params);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "vtpv-dec: params rc=%d\n", (int)r);
        goto fail3;
    }

    VkCommandPoolCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = d->dfam,
    };
    if (P.CreateCommandPool(d->dev, &cpci, NULL, &d->pool) != VK_SUCCESS) {
        goto fail4;
    }
    VkQueryPoolCreateInfo qpci = {
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .pNext = &vprof,
        .queryType = VK_QUERY_TYPE_RESULT_STATUS_ONLY_KHR,
        .queryCount = 8,
    };
    if (P.CreateQueryPool(d->dev, &qpci, NULL, &d->qpool) != VK_SUCCESS) {
        fprintf(stderr, "vtpv-dec: query pool failed\n");
        goto fail5;
    }

    /* codec RESET: its own submit with an EMPTY parameters object */
    {
        VkVideoDecodeH264SessionParametersCreateInfoKHR eh264 = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR,
        };
        VkVideoSessionParametersCreateInfoKHR epci = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,
            .pNext = &eh264,
            .videoSession = d->session,
        };
        VkVideoSessionParametersKHR ep = VK_NULL_HANDLE;
        if (P.CreateVideoSessionParametersKHR(d->dev, &epci, NULL, &ep)
            == VK_SUCCESS) {
            VkCommandBuffer cb;
            VkCommandBufferAllocateInfo cai = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .commandPool = d->pool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1,
            };
            if (P.AllocateCommandBuffers(d->dev, &cai, &cb) == VK_SUCCESS) {
                VkCommandBufferBeginInfo bi2 = {
                    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
                P.BeginCommandBuffer(cb, &bi2);
                VkVideoBeginCodingInfoKHR rbeg = {
                    .sType = VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR,
                    .videoSession = d->session,
                    .videoSessionParameters = ep,
                };
                P.CmdBeginVideoCodingKHR(cb, &rbeg);
                VkVideoCodingControlInfoKHR rctl = {
                    .sType = VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR,
                    .flags = VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR,
                };
                P.CmdControlVideoCodingKHR(cb, &rctl);
                VkVideoEndCodingInfoKHR rend = {
                    .sType = VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR };
                P.CmdEndVideoCodingKHR(cb, &rend);
                P.EndCommandBuffer(cb);
                VkCommandBuffer cbs[1] = { cb };
                VkSubmitInfo si = {
                    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    .commandBufferCount = 1,
                    .pCommandBuffers = cbs,
                };
                r = P.QueueSubmit(d->q, 1, &si, VK_NULL_HANDLE);
                if (r == VK_SUCCESS) {
                    P.QueueWaitIdle(d->q);
                }
                fprintf(stderr, "vtpv-dec: codec reset rc=%d\n", (int)r);
            }
            P.DestroyVideoSessionParametersKHR(d->dev, ep, NULL);
        }
    }

    d->bs_cap = 256 * 1024;
    {
        VkBufferCreateInfo bci = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = d->bs_cap,
            .usage = VK_BUFFER_USAGE_VIDEO_DECODE_SRC_BIT_KHR,
        };
        if (P.CreateBuffer(d->dev, &bci, NULL, &d->bsbuf) != VK_SUCCESS) {
            goto fail5;
        }
        VkMemoryRequirements bmr;
        P.GetBufferMemoryRequirements(d->dev, d->bsbuf, &bmr);
        uint32_t mt;
        if (!vd_mem_type(bmr.memoryTypeBits,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, &mt)) {
            goto fail6;
        }
        VkMemoryAllocateInfo bmai = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = bmr.size,
            .memoryTypeIndex = mt,
        };
        if (P.AllocateMemory(d->dev, &bmai, NULL, &d->bsmem) != VK_SUCCESS ||
            P.BindBufferMemory(d->dev, d->bsbuf, d->bsmem, 0) != VK_SUCCESS) {
            goto fail6;
        }
    }

    {
        uint32_t fams[1] = { d->dfam };
        VkVideoProfileListInfoKHR plist = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR,
            .profileCount = 1,
            .pProfiles = &vprof,
        };
        int dpb_array = getenv("VTPV_DEC_DPBARRAY") != NULL;
        VkImageCreateInfo iici = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = &plist,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
            .extent = { d->w, d->h, 1 },
            .mipLevels = 1,
            .arrayLayers = dpb_array ? 4 : 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR |
                     VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR |
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                     (d->uv_on ? VK_IMAGE_USAGE_SAMPLED_BIT : 0),
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = fams,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        if (dpb_array) {
            /* canonical NV shape: one DPB image, ring slots = array layers */
            if (P.CreateImage(d->dev, &iici, NULL, &d->img[0]) != VK_SUCCESS) {
                goto fail6;
            }
            VkMemoryRequirements imr;
            P.GetImageMemoryRequirements(d->dev, d->img[0], &imr);
            uint32_t mt;
            if (!vd_mem_type(imr.memoryTypeBits,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &mt)) {
                goto fail7;
            }
            VkMemoryAllocateInfo imai = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .allocationSize = imr.size,
                .memoryTypeIndex = mt,
            };
            if (P.AllocateMemory(d->dev, &imai, NULL, &d->imgmem[0])
                    != VK_SUCCESS ||
                P.BindImageMemory(d->dev, d->img[0], d->imgmem[0], 0)
                    != VK_SUCCESS) {
                goto fail7;
            }
            for (uint32_t ri = 0; ri < 4; ri++) {
                d->img[ri] = d->img[0];
                VkImageViewCreateInfo vci = {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    .pNext = &plist,
                    .image = d->img[0],
                    .viewType = VK_IMAGE_VIEW_TYPE_2D,
                    .format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
                    .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, ri, 1 },
                };
                if (P.CreateImageView(d->dev, &vci, NULL,
                                      &d->imgview[ri]) != VK_SUCCESS) {
                    goto fail8;
                }
            }
        } else
        for (uint32_t ri = 0; ri < 4; ri++) {
            if (P.CreateImage(d->dev, &iici, NULL, &d->img[ri]) != VK_SUCCESS) {
                goto fail6;
            }
            VkMemoryRequirements imr;
            P.GetImageMemoryRequirements(d->dev, d->img[ri], &imr);
            uint32_t mt;
            if (!vd_mem_type(imr.memoryTypeBits,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &mt)) {
                goto fail7;
            }
            VkMemoryDedicatedAllocateInfo iddi = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
                .image = d->img[ri],
            };
            VkMemoryAllocateInfo imai = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .pNext = &iddi,
                .allocationSize = imr.size,
                .memoryTypeIndex = mt,
            };
            if (P.AllocateMemory(d->dev, &imai, NULL,
                                 &d->imgmem[ri]) != VK_SUCCESS ||
                P.BindImageMemory(d->dev, d->img[ri], d->imgmem[ri], 0)
                    != VK_SUCCESS) {
                goto fail7;
            }
            VkImageViewCreateInfo vci = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .pNext = &plist,
                .image = d->img[ri],
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
                .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
            };
            if (P.CreateImageView(d->dev, &vci, NULL,
                                  &d->imgview[ri]) != VK_SUCCESS) {
                goto fail8;
            }
        }
    }

    /* UV shader context: plane-1 views, sampler, SSBO, compute pipeline */
    if (d->uv_on) {
        bool ok = true;
        for (uint32_t ri = 0; ri < 4 && ok; ri++) {
            VkImageViewCreateInfo uvi = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = d->img[ri],
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = VK_FORMAT_R8G8_UNORM,
                .subresourceRange = { VK_IMAGE_ASPECT_PLANE_1_BIT, 0, 1, 0, 1 },
            };
            ok = P.CreateImageView(d->dev, &uvi, NULL, &d->uvview[ri])
                 == VK_SUCCESS;
        }
        VkSamplerCreateInfo sci = {
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_NEAREST,
            .minFilter = VK_FILTER_NEAREST,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .maxLod = 0.25f,
        };
        if (ok && P.CreateSampler(d->dev, &sci, NULL, &d->uvsamp)
            != VK_SUCCESS) {
            ok = false;
        }
        d->uvn = (size_t)(d->w / 2) * (d->h / 2);
        VkBufferCreateInfo ubc = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = d->uvn * 8,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        };
        if (ok && P.CreateBuffer(d->dev, &ubc, NULL, &d->uvbuf)
            != VK_SUCCESS) {
            ok = false;
        }
        if (ok) {
            VkMemoryRequirements umr;
            uint32_t mt;
            P.GetBufferMemoryRequirements(d->dev, d->uvbuf, &umr);
            if (!vd_mem_type(umr.memoryTypeBits,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, &mt)) {
                ok = false;
            } else {
                VkMemoryAllocateInfo umai = {
                    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    .allocationSize = umr.size,
                    .memoryTypeIndex = mt,
                };
                if (P.AllocateMemory(d->dev, &umai, NULL, &d->uvmem)
                        != VK_SUCCESS ||
                    P.BindBufferMemory(d->dev, d->uvbuf, d->uvmem, 0)
                        != VK_SUCCESS ||
                    P.MapMemory(d->dev, d->uvmem, 0, VK_WHOLE_SIZE, 0,
                                (void **)&d->uvmap) != VK_SUCCESS) {
                    ok = false;
                }
            }
        }
        if (ok) {
            VkDescriptorSetLayoutBinding lb[2] = {
                { .binding = 0,
                  .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                  .descriptorCount = 1,
                  .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                  .pImmutableSamplers = &d->uvsamp },
                { .binding = 1,
                  .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                  .descriptorCount = 1,
                  .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
            };
            VkDescriptorSetLayoutCreateInfo dslc = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = 2,
                .pBindings = lb,
            };
            if (P.CreateDescriptorSetLayout(d->dev, &dslc, NULL, &d->uvdsl)
                != VK_SUCCESS) {
                ok = false;
            }
        }
        if (ok) {
            VkPipelineLayoutCreateInfo plc = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount = 1,
                .pSetLayouts = &d->uvdsl,
            };
            if (P.CreatePipelineLayout(d->dev, &plc, NULL, &d->uvpl)
                != VK_SUCCESS) {
                ok = false;
            }
        }
        if (ok) {
            VkShaderModuleCreateInfo smc = {
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .codeSize = sizeof(vtpv_uv_spv),
                .pCode = vtpv_uv_spv,
            };
            VkShaderModule sm;
            if (P.CreateShaderModule(d->dev, &smc, NULL, &sm) != VK_SUCCESS) {
                ok = false;
            } else {
                VkComputePipelineCreateInfo cpc = {
                    .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                    .stage = {
                        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                        .module = sm,
                        .pName = "main",
                    },
                    .layout = d->uvpl,
                };
                if (P.CreateComputePipelines(d->dev, VK_NULL_HANDLE, 1,
                                             &cpc, NULL, &d->uvpipe)
                    != VK_SUCCESS) {
                    ok = false;
                }
                P.DestroyShaderModule(d->dev, sm, NULL);
            }
        }
        if (ok) {
            VkDescriptorPoolSize dps[2] = {
                { .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                  .descriptorCount = 4 },
                { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                  .descriptorCount = 4 },
            };
            VkDescriptorPoolCreateInfo dpc = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets = 4,
                .poolSizeCount = 2,
                .pPoolSizes = dps,
            };
            if (P.CreateDescriptorPool(d->dev, &dpc, NULL, &d->uvdpool)
                != VK_SUCCESS) {
                ok = false;
            }
        }
        if (ok) {
            VkDescriptorSetAllocateInfo dsai = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = d->uvdpool,
                .descriptorSetCount = 1,
                .pSetLayouts = &d->uvdsl,
            };
            if (P.AllocateDescriptorSets(d->dev, &dsai, &d->uvset)
                != VK_SUCCESS) {
                ok = false;
            }
        }
        if (ok) {
            VkDescriptorImageInfo dii = {
                .sampler = d->uvsamp,
                .imageView = d->uvview[0],
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };
            VkDescriptorBufferInfo dbi = {
                .buffer = d->uvbuf,
                .offset = 0,
                .range = d->uvn * 8,
            };
            VkWriteDescriptorSet dw[2] = {
                { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                  .dstSet = d->uvset, .dstBinding = 0, .descriptorCount = 1,
                  .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                  .pImageInfo = &dii },
                { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                  .dstSet = d->uvset, .dstBinding = 1, .descriptorCount = 1,
                  .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                  .pBufferInfo = &dbi },
            };
            P.UpdateDescriptorSets(d->dev, 2, dw, 0, NULL);
            VkCommandPoolCreateInfo upc = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .queueFamilyIndex = (uint32_t)d->uv_fam,
            };
            if (P.CreateCommandPool(d->dev, &upc, NULL, &d->uvpool)
                != VK_SUCCESS) {
                ok = false;
            }
        }
        /* the descriptor references uvview[0]; rewrite it per frame for the
         * current ring slot before each dispatch */
        if (!ok) {
            fprintf(stderr, "vtpv-dec: UV shader init failed, "
                    "chroma falls back to 0x80\n");
            d->uv_on = 0;
        } else {
            fprintf(stderr, "vtpv-dec: UV shader on family %d\n", d->uv_fam);
        }
    }

    {
        VkBufferCreateInfo rbc = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = (VkDeviceSize)d->w * d->h * 3 / 2 + 256,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        };
        if (P.CreateBuffer(d->dev, &rbc, NULL, &d->rbbuf) != VK_SUCCESS) {
            goto fail8;
        }
        VkMemoryRequirements rmr;
        P.GetBufferMemoryRequirements(d->dev, d->rbbuf, &rmr);
        uint32_t mt;
        if (!vd_mem_type(rmr.memoryTypeBits,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, &mt)) {
            goto fail9;
        }
        VkMemoryAllocateInfo rmai = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = rmr.size,
            .memoryTypeIndex = mt,
        };
        if (P.AllocateMemory(d->dev, &rmai, NULL, &d->rbmem) != VK_SUCCESS ||
            P.BindBufferMemory(d->dev, d->rbbuf, d->rbmem, 0) != VK_SUCCESS ||
            P.MapMemory(d->dev, d->rbmem, 0, VK_WHOLE_SIZE, 0, &d->rbmap)
                != VK_SUCCESS) {
            goto fail9;
        }
    }

    fprintf(stderr, "vtpv-dec: open %ux%d prof=%d cabac=%d dpb=%u align=%llu\n",
            d->w, d->h, (int)prof,
            (int)d->std_pps.flags.entropy_coding_mode_flag,
            cap.maxDpbSlots, (unsigned long long)d->bs_size_align);
    return d;

fail9:
    if (d->rbbuf) {
        P.DestroyBuffer(d->dev, d->rbbuf, NULL);
    }
fail8:
    for (uint32_t ri = 0; ri < 4; ri++) {
        if (d->imgview[ri]) {
            P.DestroyImageView(d->dev, d->imgview[ri], NULL);
        }
        if (d->imgmem[ri]) {
            P.FreeMemory(d->dev, d->imgmem[ri], NULL);
        }
    }
fail7:
    for (uint32_t ri = 0; ri < 4; ri++) {
        if (d->img[ri]) {
            P.DestroyImage(d->dev, d->img[ri], NULL);
        }
    }
fail6:
    if (d->bsbuf) {
        P.DestroyBuffer(d->dev, d->bsbuf, NULL);
    }
    if (d->bsmem) {
        P.FreeMemory(d->dev, d->bsmem, NULL);
    }
fail5:
    P.DestroyCommandPool(d->dev, d->pool, NULL);
fail4:
    if (d->params) {
        P.DestroyVideoSessionParametersKHR(d->dev, d->params, NULL);
    }
fail3:
    P.DestroyVideoSessionKHR(d->dev, d->session, NULL);
fail2:
    P.DestroyDevice(d->dev, NULL);
fail1:
    {
        PFN_vkDestroyInstance di = (PFN_vkDestroyInstance)
            L.GetInstanceProcAddr(d->inst, "vkDestroyInstance");
        if (di) {
            di(d->inst, NULL);
        }
    }
fail0:
    fprintf(stderr, "vtpv-dec: open failed\n");
    g_free(d);
    return NULL;
}

int vtpv_dec_frame(VtpvDecoder *d, const uint8_t *frame, uint32_t frame_len,
                   const uint32_t *slice_off, uint32_t nsl,
                   uint8_t *nv12_out)
{
    if (!d || frame_len > d->bs_cap || nsl == 0) {
        return -1;
    }
    VkResult r;
    uint64_t t0 = qemu_clock_get_ms(QEMU_CLOCK_REALTIME);
    VDSlice sh;
    vd_parse_slice(frame + slice_off[0] + 3, &d->sps, &sh);
    fprintf(stderr, "vtpv-dec: slice idr=%d intra=%d ref=%d frame_num=%u "
            "poc=%u (log2fn=%u log2poc=%u poct=%u)\n",
            (int)sh.is_idr, (int)sh.is_intra, (int)sh.is_ref,
            sh.frame_num, sh.poc_lsb,
            (unsigned)(d->sps.log2_max_frame_num),
            (unsigned)(d->sps.log2_max_poc_lsb),
            (unsigned)(d->sps.poc_type));

    /* picture order count: the driver resolves reference lists by POC, so
     * every frame claiming POC 0 breaks list0 (concealment output).
     * type 2 has no poc_lsb in the slice header - derive it instead */
    int poc;
    if (sh.is_idr) {
        poc = 0;
    } else if (d->sps.poc_type == 2) {
        poc = d->poc_prev_ref + (sh.is_ref ? 2 : 1);
    } else {
        poc = sh.poc_lsb;
    }
    if (sh.is_ref) {
        d->poc_prev_ref = poc;
    }
    void *p = NULL;
    if (P.MapMemory(d->dev, d->bsmem, 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS) {
        return -1;
    }
    memset(p, 0, (size_t)d->bs_cap);
    memcpy(p, frame, frame_len);
    VkMappedMemoryRange fl = {
        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .memory = d->bsmem,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    };
    P.FlushMappedMemoryRanges(d->dev, 1, &fl);
    P.UnmapMemory(d->dev, d->bsmem);

    VkDeviceSize range = ((VkDeviceSize)frame_len + d->bs_size_align - 1) &
                         ~(d->bs_size_align - 1);
    if (range > d->bs_cap) {
        range = d->bs_cap;
    }

    VkCommandBuffer cb;
    VkCommandBufferAllocateInfo cai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = d->pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (P.AllocateCommandBuffers(d->dev, &cai, &cb) != VK_SUCCESS) {
        return -1;
    }
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    P.BeginCommandBuffer(cb, &bi);

    /* picture ring slot for this frame; IDR flushes all references */
    uint32_t cur = d->ring_pos & 3;
    if (sh.is_idr) {
        for (uint32_t k = 0; k < 4; k++) {
            d->slot_valid[k] = false;
        }
    }
    d->ring_pos++;

    VkVideoPictureResourceInfoKHR res = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR,
        .codedOffset = { 0, 0 },
        .codedExtent = { d->w, d->h },
        .baseArrayLayer = cur,
        .imageViewBinding = d->imgview[cur],
    };
    /* bind previously decoded pictures as ACTIVE references for P/B frames,
     * deactivate the ring slot being reused, and declare the current
     * resource as an additional inactive slot */
    VkVideoReferenceSlotInfoKHR refs[6];
    uint32_t nrefs = 0;
    for (uint32_t k = 0; k < 4 && nrefs < 4; k++) {
        if (k == cur || !d->slot_valid[k]) {
            continue;
        }
        refs[nrefs].sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
        refs[nrefs].pNext = &d->slot_dpb[k];  /* update form: rebind meta */
        refs[nrefs].slotIndex = (int32_t)k;
        refs[nrefs].pPictureResource = &d->slot_res[k];
        nrefs++;
    }
    if (d->slot_valid[cur]) {
        /* re-activating a live slot below: deactivate it first */
        refs[nrefs].sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
        refs[nrefs].pNext = NULL;
        refs[nrefs].slotIndex = (int32_t)cur;
        refs[nrefs].pPictureResource = NULL;
        nrefs++;
    }
    VkVideoReferenceSlotInfoKHR inactive = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,
        .slotIndex = -1,
        .pPictureResource = &res,
    };
    if (nrefs == 0 || !getenv("VTPV_DEC_NOINACT")) {
        refs[nrefs++] = inactive;
    }
    int refsetup = getenv("VTPV_DEC_REFSETUP") != NULL;
    /* dst picture into decode-write layout */
    VkImageMemoryBarrier2 pre = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
        .srcAccessMask = 0,
        .dstStageMask = VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR,
        .oldLayout = d->ring_used[cur] ? VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR
                                       : VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = d->img[cur],
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, cur, 1 },
    };
    VkDependencyInfo pred = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &pre,
    };
    P.CmdPipelineBarrier2(cb, &pred);

    if (refsetup) {
        /* official-sample shape: the setup slot also appears among the
         * BeginCoding reference slots */
        static StdVideoDecodeH264ReferenceInfo sref_tmp;
        static VkVideoDecodeH264DpbSlotInfoKHR sdpb_tmp;
        sref_tmp.FrameNum = sh.frame_num;
        sref_tmp.PicOrderCnt[0] = poc;
        sref_tmp.PicOrderCnt[1] = poc;
        sdpb_tmp.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR;
        sdpb_tmp.pNext = NULL;
        sdpb_tmp.pStdReferenceInfo = &sref_tmp;
        refs[nrefs].sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
        refs[nrefs].pNext = &sdpb_tmp;
        refs[nrefs].slotIndex = (int32_t)cur;
        refs[nrefs].pPictureResource = &res;
        nrefs++;
    }
    VkVideoBeginCodingInfoKHR beg = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR,
        .videoSession = d->session,
        .videoSessionParameters = d->params,
        .referenceSlotCount = nrefs,
        .pReferenceSlots = refs,
    };
    P.CmdBeginVideoCodingKHR(cb, &beg);
    uint32_t qidx = cur;
    P.CmdResetQueryPool(cb, d->qpool, qidx, 1);
    P.CmdBeginQuery(cb, d->qpool, qidx, 0);

    StdVideoDecodeH264PictureInfo stdpic = {
        .flags = {
            .is_intra = sh.is_intra,
            .IdrPicFlag = sh.is_idr,
            .is_reference = sh.is_ref,
        },
        .seq_parameter_set_id = 0,
        .pic_parameter_set_id = 0,
        .frame_num = sh.frame_num,
        .idr_pic_id = sh.idr_pic_id,
        .PicOrderCnt = { poc, poc },
    };
    /* NB: this std header rev has no pRefLists on the decode picture info -
     * the driver matches slice-header frame_nums against the ACTIVE slot
     * metadata (FrameNum/POC registered below after each decode) */

    VkVideoDecodeH264PictureInfoKHR h264info = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PICTURE_INFO_KHR,
        .pStdPictureInfo = &stdpic,
        .sliceCount = nsl,
        .pSliceOffsets = slice_off,
    };
    StdVideoDecodeH264ReferenceInfo stdref = {
        .FrameNum = sh.frame_num,
        .PicOrderCnt = { poc, poc },
    };
    VkVideoDecodeH264DpbSlotInfoKHR dpbslot = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR,
        .pStdReferenceInfo = &stdref,
    };
    VkVideoReferenceSlotInfoKHR setup = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,
        .pNext = &dpbslot,
        .slotIndex = (int32_t)cur,   /* each ring picture owns its slot */
        .pPictureResource = &res,
    };
    /* the decode command itself must list the active references with their
     * DPB slot info - BeginCoding only binds them; without this the driver
     * predicts from nothing (the "P = residual on black" of 617.42 was this,
     * proven with video-re/pprobe.c, 2026-10-22) */
    VkVideoReferenceSlotInfoKHR drefs[4];
    uint32_t ndrefs = 0;
    for (uint32_t k = 0; k < 4; k++) {
        if (k == cur || !d->slot_valid[k]) {
            continue;
        }
        drefs[ndrefs].sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
        drefs[ndrefs].pNext = &d->slot_dpb[k];
        drefs[ndrefs].slotIndex = (int32_t)k;
        drefs[ndrefs].pPictureResource = &d->slot_res[k];
        ndrefs++;
    }
    VkVideoDecodeInfoKHR dec = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR,
        .pNext = &h264info,
        .srcBuffer = d->bsbuf,
        .srcBufferOffset = 0,
        .srcBufferRange = range,
        .dstPictureResource = res,
        .pSetupReferenceSlot = &setup,
        .referenceSlotCount = ndrefs,
        .pReferenceSlots = ndrefs ? drefs : NULL,
    };
    P.CmdDecodeVideoKHR(cb, &dec);
    P.CmdEndQuery(cb, d->qpool, qidx);
    VkVideoEndCodingInfoKHR endi = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR };
    P.CmdEndVideoCodingKHR(cb, &endi);

    if (!getenv("VTPV_DEC_BARE")) {
    VkImageMemoryBarrier2 tr = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR,
        .srcAccessMask = VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR,
        .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
        .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = d->img[cur],
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, cur, 1 },
    };
    VkDependencyInfo trd = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &tr,
    };
    P.CmdPipelineBarrier2(cb, &trd);
    if (!getenv("VTPV_DEC_TRONLY")) {
    VkBufferImageCopy cp0 = {
        .bufferOffset = 0,
        .bufferRowLength = d->w,
        .bufferImageHeight = d->h,
        .imageSubresource = { VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 0, 1 },
        .imageExtent = { d->w, d->h, 1 },
    };
    P.CmdCopyImageToBuffer(cb, d->img[cur], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           d->rbbuf, 1, &cp0);
    if (!d->uv_on && !getenv("VTPV_DEC_NOCP1")) {
        VkBufferImageCopy cp1 = {
            .bufferOffset = (VkDeviceSize)d->w * d->h,
            .bufferRowLength = d->w,
            .bufferImageHeight = d->h / 2,
            .imageSubresource = { VK_IMAGE_ASPECT_PLANE_1_BIT, 0, 0, 1 },
            .imageExtent = { d->w, d->h / 2, 1 },
        };
        P.CmdCopyImageToBuffer(cb, d->img[cur],
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               d->rbbuf, 1, &cp1);
    }
    VkBufferMemoryBarrier2 rb = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
        .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = d->rbbuf,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    };
    VkDependencyInfo rbd = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &rb,
    };
    P.CmdPipelineBarrier2(cb, &rbd);
    if (d->uv_on) {
        /* hand the picture over to the compute family for the UV shader;
         * 617.42 drops every plane-1 copy, so plane 1 can only leave the
         * image through a sampled read */
        VkImageMemoryBarrier2 rel = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = d->dfam,
            .dstQueueFamilyIndex = (uint32_t)d->uv_fam,
            .image = d->img[cur],
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, cur, 1 },
        };
        VkDependencyInfo reld = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &rel,
        };
        P.CmdPipelineBarrier2(cb, &reld);
    } else if (!getenv("VTPV_DEC_NOBACK")) {
    /* picture back to the DPB layout: the session keeps it bound as a
     * reference slot and a TRANSFER_SRC reference trips submit validation */
    VkImageMemoryBarrier2 back = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR |
                         VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = d->img[cur],
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, cur, 1 },
    };
    VkDependencyInfo backd = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &back,
    };
    P.CmdPipelineBarrier2(cb, &backd);
    d->ring_used[cur] = true;
    }
    } /* !VTPV_DEC_TRONLY */
    } /* !VTPV_DEC_BARE */
    /* NB: uv_on leaves the picture on the compute family in SHADER_READ_ONLY
     * (per-frame reopen mode); VTPV_DEC_REUSE ring mode wants the else path */
    P.EndCommandBuffer(cb);

    VkCommandBuffer cbs[1] = { cb };
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = cbs,
    };
    r = P.QueueSubmit(d->q, 1, &si, VK_NULL_HANDLE);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "vtpv-dec: QueueSubmit rc=%d nsl=%u refs=%u\n",
                (int)r, nsl, nrefs);
        return -1;
    }
    r = P.QueueWaitIdle(d->q);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "vtpv-dec: WaitIdle rc=%d\n", (int)r);
        return -1;
    }

    int32_t status = 0x7fffffff;
    P.GetQueryPoolResults(d->dev, d->qpool, cur, 1,
                          sizeof(status), &status,
                          sizeof(status),
                          VK_QUERY_RESULT_WITH_STATUS_BIT_KHR |
                          VK_QUERY_RESULT_WAIT_BIT);
    if (status != 1) {
        fprintf(stderr, "vtpv-dec: decode status=%d\n", (int)status);
        return -1;
    }

    /* register the decoded picture as an active DPB slot for later
     * inter frames (the ring picture doubles as its reference) */
    d->slot_ref[cur] = stdref;
    d->slot_dpb[cur].sType =
        VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR;
    d->slot_dpb[cur].pNext = NULL;
    d->slot_dpb[cur].pStdReferenceInfo = &d->slot_ref[cur];
    d->slot_res[cur] = res;
    d->slot_valid[cur] = true;

    if (getenv("VTPV_DEC_BARE")) {
        fprintf(stderr, "vtpv-dec: bare frame %u ok\n", d->ring_pos - 1);
        return 0;
    }

    bool uv_done = false;
    if (d->uv_on) {
        VkCommandBufferBeginInfo ubi = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        };
        VkDescriptorImageInfo dii = {
            .sampler = d->uvsamp,
            .imageView = d->uvview[cur],
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        VkWriteDescriptorSet dw = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = d->uvset,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &dii,
        };
        P.UpdateDescriptorSets(d->dev, 1, &dw, 0, NULL);

        VkCommandBuffer ucb;
        VkCommandBufferAllocateInfo uai = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = d->uvpool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        if (P.AllocateCommandBuffers(d->dev, &uai, &ucb) == VK_SUCCESS &&
            P.BeginCommandBuffer(ucb, &ubi) == VK_SUCCESS) {
            VkImageMemoryBarrier2 acq = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT |
                                 VK_ACCESS_2_MEMORY_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .srcQueueFamilyIndex = d->dfam,
                .dstQueueFamilyIndex = (uint32_t)d->uv_fam,
                .image = d->img[cur],
                .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, cur, 1 },
            };
            VkDependencyInfo ad = {
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &acq,
            };
            P.CmdPipelineBarrier2(ucb, &ad);
            P.CmdBindPipeline(ucb, VK_PIPELINE_BIND_POINT_COMPUTE, d->uvpipe);
            P.CmdBindDescriptorSets(ucb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    d->uvpl, 0, 1, &d->uvset, 0, NULL);
            P.CmdDispatch(ucb, (d->w / 2 + 7) / 8, (d->h / 2 + 7) / 8, 1);
            VkBufferMemoryBarrier2 sbb = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
                .buffer = d->uvbuf,
                .offset = 0,
                .size = VK_WHOLE_SIZE,
            };
            VkDependencyInfo sd1 = {
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .bufferMemoryBarrierCount = 1,
                .pBufferMemoryBarriers = &sbb,
            };
            P.CmdPipelineBarrier2(ucb, &sd1);
            if (P.EndCommandBuffer(ucb) == VK_SUCCESS) {
                VkCommandBuffer ucbs[1] = { ucb };
                VkSubmitInfo usi = {
                    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    .commandBufferCount = 1,
                    .pCommandBuffers = ucbs,
                };
                if (P.QueueSubmit(d->uvq, 1, &usi, VK_NULL_HANDLE)
                        == VK_SUCCESS &&
                    P.QueueWaitIdle(d->uvq) == VK_SUCCESS) {
                    VkMappedMemoryRange ifl = {
                        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                        .memory = d->uvmem,
                        .size = VK_WHOLE_SIZE,
                    };
                    P.InvalidateMappedMemoryRanges(d->dev, 1, &ifl);
                    uv_done = true;
                } else {
                    fprintf(stderr, "vtpv-dec: UV dispatch failed\n");
                }
            }
        }
    }

    fprintf(stderr, "vtpv-dec: frame took %llu ms\n",
            (unsigned long long)(qemu_clock_get_ms(QEMU_CLOCK_REALTIME) - t0));
    const uint8_t *src = d->rbmap;
    memcpy(nv12_out, src, (size_t)d->w * d->h);
    uint32_t uvsz = d->w * d->h / 2;
    if (uv_done) {
        /* shader output: float pairs (u, v) per chroma pixel */
        uint8_t *dst = nv12_out + (size_t)d->w * d->h;
        for (size_t i = 0; i < d->uvn && i * 2 < uvsz; i++) {
            dst[2 * i] = (uint8_t)(d->uvmap[2 * i] + 0.5f);
            dst[2 * i + 1] = (uint8_t)(d->uvmap[2 * i + 1] + 0.5f);
        }
        d->chroma_ok = true;
        return 0;
    }
    const uint8_t *uvsrc = src + (size_t)d->w * d->h;
    bool uv_real = false;
    for (uint32_t i = 0; i < uvsz && i < 64; i++) {
        if (uvsrc[i] != 0xab && uvsrc[i] != 0x00) {
            uv_real = true;
            break;
        }
    }
    if (uv_real) {
        d->chroma_ok = true;
    }
    if (d->chroma_ok) {
        memcpy(nv12_out + (size_t)d->w * d->h, uvsrc, uvsz);
    } else {
        memset(nv12_out + (size_t)d->w * d->h, 0x80, uvsz);
    }
    return 0;
}

bool vtpv_dec_match(const VtpvDecoder *d,
                    const uint8_t *sps_nal, uint32_t sps_len,
                    const uint8_t *pps_nal, uint32_t pps_len)
{
    uint8_t hs[8], hp[8];
    if (!d) {
        return false;
    }
    vd_hash(sps_nal, sps_len, hs);
    vd_hash(pps_nal, pps_len, hp);
    return memcmp(hs, d->sps_hash, 8) == 0 && memcmp(hp, d->pps_hash, 8) == 0;
}

uint32_t vtpv_dec_width(const VtpvDecoder *d)  { return d->w; }
uint32_t vtpv_dec_height(const VtpvDecoder *d) { return d->h; }
bool vtpv_dec_has_chroma(const VtpvDecoder *d) { return d->chroma_ok; }

void vtpv_dec_close(VtpvDecoder *d)
{
    if (!d) {
        return;
    }
    if (d->rbmap && d->rbmem) {
        P.UnmapMemory(d->dev, d->rbmem);
    }
    if (d->uvmap && d->uvmem) {
        P.UnmapMemory(d->dev, d->uvmem);
    }
    if (d->uvbuf) {
        P.DestroyBuffer(d->dev, d->uvbuf, NULL);
    }
    if (d->uvmem) {
        P.FreeMemory(d->dev, d->uvmem, NULL);
    }
    if (d->uvpool) {
        P.DestroyCommandPool(d->dev, d->uvpool, NULL);
    }
    if (d->uvpipe) {
        P.DestroyPipeline(d->dev, d->uvpipe, NULL);
    }
    if (d->uvdpool) {
        P.DestroyDescriptorPool(d->dev, d->uvdpool, NULL);
    }
    if (d->uvpl) {
        P.DestroyPipelineLayout(d->dev, d->uvpl, NULL);
    }
    if (d->uvdsl) {
        P.DestroyDescriptorSetLayout(d->dev, d->uvdsl, NULL);
    }
    if (d->uvsamp) {
        P.DestroySampler(d->dev, d->uvsamp, NULL);
    }
    for (uint32_t ri = 0; ri < 4; ri++) {
        if (d->uvview[ri]) {
            P.DestroyImageView(d->dev, d->uvview[ri], NULL);
        }
    }
    if (d->rbbuf) {
        P.DestroyBuffer(d->dev, d->rbbuf, NULL);
    }
    if (d->rbmem) {
        P.FreeMemory(d->dev, d->rbmem, NULL);
    }
    int img_arr = d->img[0] && d->img[1] == d->img[0];
    for (uint32_t ri = 0; ri < 4; ri++) {
        if (d->imgview[ri]) {
            P.DestroyImageView(d->dev, d->imgview[ri], NULL);
        }
        if (img_arr && ri > 0) {
            continue;
        }
        if (d->imgmem[ri]) {
            P.FreeMemory(d->dev, d->imgmem[ri], NULL);
        }
        if (d->img[ri]) {
            P.DestroyImage(d->dev, d->img[ri], NULL);
        }
    }
    if (d->bsbuf) {
        P.DestroyBuffer(d->dev, d->bsbuf, NULL);
    }
    if (d->bsmem) {
        P.FreeMemory(d->dev, d->bsmem, NULL);
    }
    if (d->pool) {
        P.DestroyCommandPool(d->dev, d->pool, NULL);
    }
    if (d->params) {
        P.DestroyVideoSessionParametersKHR(d->dev, d->params, NULL);
    }
    if (d->session) {
        P.DestroyVideoSessionKHR(d->dev, d->session, NULL);
    }
    if (d->dev) {
        P.DestroyDevice(d->dev, NULL);
    }
    if (d->inst) {
        PFN_vkDestroyInstance di = (PFN_vkDestroyInstance)
            L.GetInstanceProcAddr(d->inst, "vkDestroyInstance");
        if (di) {
            di(d->inst, NULL);
        }
    }
    g_free(d);
}
