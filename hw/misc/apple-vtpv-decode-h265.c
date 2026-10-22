/*
 * apple-vtpv-decode-h265.c - hardware HEVC decoder for the Apple VTPV device
 * (ZCode, 2026-10-22).  Same verified pipeline contract as the H264 module
 * (apple-vtpv-decode.c): per-frame session, codec RESET with an empty
 * parameters object in its own submit, Annex-B slice segments with 3-byte
 * start codes at offset 0, status query inside the coding scope, copies on
 * the decode queue only.  Supports Main / Main-10 8-bit streams without
 * scaling lists, tiles or PCM (the VideoToolbox tile-flow shape).
 */
#include "qemu/osdep.h"
#include "hw/misc/apple-vtpv-decode.h"
#include "qemu/timer.h"

#include <windows.h>
#include "vk/include/vulkan/vulkan.h"
#include "vk/include/vk_video/vulkan_video_codec_h265std.h"
#include "vk/include/vk_video/vulkan_video_codec_h265std_decode.h"
#include "apple-vtpv-uvspv.h"

/* reuse the H264 module's loader/proc table (single process, one loader) */
#define VD265_SHARED_LOADER 1
extern void *vd265_get_proc(const char *name);

typedef struct VD265Procs {
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
} VD265Procs;

static VD265Procs Q;

typedef struct {
    HMODULE lib;
    void *(VKAPI_PTR *GetInstanceProcAddr)(VkInstance, const char *);
    void *(VKAPI_PTR *GetDeviceProcAddr)(VkDevice, const char *);
    VkInstance inst;
    VkPhysicalDevice pd;
} VD265Loader;

static VD265Loader L265;

#define QI(name) Q.name = (PFN_vk##name)L265.GetInstanceProcAddr(L265.inst, "vk" #name)
#define QD(name) Q.name = (PFN_vk##name)L265.GetDeviceProcAddr(dev, "vk" #name)

/* ---------------- bit reader + HEVC header parsing ---------------- */

typedef struct BR265 {
    const uint8_t *d;
    int pos;
} BR265;

static int b2_bit(BR265 *b)
{
    int v = (b->d[b->pos >> 3] >> (7 - (b->pos & 7))) & 1;
    b->pos++;
    return v;
}

static uint32_t b2_bits(BR265 *b, int n)
{
    uint32_t v = 0;
    while (n--) {
        v = (v << 1) | b2_bit(b);
    }
    return v;
}

static uint32_t b2_ue(BR265 *b)
{
    int z = 0;
    while (!b2_bit(b)) {
        if (++z > 31) {
            return 0;
        }
    }
    return (1u << z) - 1 + (z ? b2_bits(b, z) : 0);
}

static int32_t b2_se(BR265 *b)
{
    uint32_t k = b2_ue(b);
    return (k & 1) ? (int32_t)((k + 1) / 2) : -(int32_t)(k / 2);
}

/* st_ref_pic_set(i) - H.265 7.3.7: values not needed for IDR decode, but
 * the bit position must be consumed to reach the fields after the list */
static void h265_parse_st_rps(BR265 *b, int idx, int num_delta_pocs[65])
{
    int inter_pred = 0;
    if (idx != 0) {
        inter_pred = b2_bit(b);
    }
    int ndp = 0;
    if (inter_pred) {
        int delta_idx = 0;
        if (idx == 0) {
            delta_idx = b2_ue(b);
        }
        int delta_rps_sign = b2_bit(b);
        uint32_t abs_delta_rps_minus1 = b2_ue(b);
        (void)delta_rps_sign;
        (void)abs_delta_rps_minus1;
        int ref = num_delta_pocs[idx - 1 - delta_idx];
        for (int j = 0; j <= ref; j++) {
            int used = b2_bit(b);
            if (!used) {
                b2_bit(b);
            }
        }
        ndp = ref + 1;
    } else {
        uint32_t npos = b2_ue(b);
        uint32_t nneg = b2_ue(b);
        for (uint32_t j = 0; j < npos; j++) {
            b2_ue(b);
            b2_bit(b);
        }
        for (uint32_t j = 0; j < nneg; j++) {
            b2_ue(b);
            b2_bit(b);
        }
        ndp = (int)(npos + nneg);
    }
    num_delta_pocs[idx] = ndp;
}


typedef struct H265SpsInfo {
    int w, h;
    int profile_idc, level_idc;
    int vps_id, sps_id;
    int sub_layers;
} H265SpsInfo;

typedef struct VD265 {
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
    uint64_t bs_cap, bs_size_align;
    VkImage img;
    VkImageView imgview;
    VkDeviceMemory imgmem;
    VkBuffer rbbuf;
    VkDeviceMemory rbmem;
    void *rbmap;

    /* chroma via compute shader (see apple-vtpv-decode.c: 617.42 drops
     * every plane-1 copy path) */
    int uv_on;
    int uv_fam;
    VkQueue uvq;
    VkCommandPool uvpool;
    VkSampler uvsamp;
    VkImageView uvview;
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
    uint8_t vps_hash[8], sps_hash[8], pps_hash[8];
    /* std parameter sets kept alive for the session parameters object */
    StdVideoH265VideoParameterSet std_vps;
    StdVideoH265SequenceParameterSet std_sps;
    StdVideoH265PictureParameterSet std_pps;
    StdVideoH265ProfileTierLevel std_ptl;
    StdVideoH265DecPicBufMgr std_dpbm;
    H265SpsInfo sps;
    bool chroma_ok;
} VD265;

static uint32_t h265_strip_epb(uint8_t *dst, const uint8_t *src, uint32_t n)
{
    uint32_t o = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (i + 2 < n && src[i] == 0 && src[i + 1] == 0 && src[i + 2] == 3) {
            dst[o++] = 0;
            dst[o++] = 0;
            i += 2;   /* skip the 03; next byte copied as-is */
        } else {
            dst[o++] = src[i];
        }
    }
    return o;
}

/* HEVC NAL: 2-byte header, type = (b0 >> 1) & 0x3f */
static int h265_nal_type(const uint8_t *n)
{
    return (n[0] >> 1) & 0x3f;
}

/* profile_tier_level(1, maxSubLayersMinus1=0) */
static bool h265_parse_ptl(BR265 *b, StdVideoH265ProfileTierLevel *ptl,
                           int *profile_idc, int *level_idc)
{
    int general_profile_space = b2_bits(b, 2);
    int general_tier_flag = b2_bit(b);
    *profile_idc = b2_bits(b, 5);
    uint32_t compat = b2_bits(b, 32);
    ptl->flags.general_progressive_source_flag = b2_bit(b);
    ptl->flags.general_interlaced_source_flag = b2_bit(b);
    ptl->flags.general_non_packed_constraint_flag = b2_bit(b);
    ptl->flags.general_frame_only_constraint_flag = b2_bit(b);
    /* remaining 44 bits of the 48-bit constraint flags */
    b2_bits(b, 44);
    *level_idc = b2_bits(b, 8);
    (void)general_profile_space;
    (void)general_tier_flag;
    (void)compat;
    return true;
}

static uint32_t g_h265_nal_len;
static bool h265_parse_sps(const uint8_t *n, VD265 *d)
{
    static uint8_t clean[512];
    uint32_t cn = h265_strip_epb(clean, n + 2, g_h265_nal_len > 2 ? g_h265_nal_len - 2 : 0);
    if (cn > sizeof(clean)) cn = sizeof(clean);
    BR265 b = { clean, 0 };
    StdVideoH265SequenceParameterSet *s = &d->std_sps;
    memset(s, 0, sizeof(*s));
    memset(&d->std_ptl, 0, sizeof(d->std_ptl));
    memset(&d->std_dpbm, 0, sizeof(d->std_dpbm));

    s->sps_video_parameter_set_id = b2_bits(&b, 4);
    s->sps_max_sub_layers_minus1 = b2_bits(&b, 3);
    int temporal_id_nesting = b2_bit(&b);
    (void)temporal_id_nesting;
    int prof = 0, lvl = 0;
    if (!h265_parse_ptl(&b, &d->std_ptl, &prof, &lvl)) {
        return false;
    }
    if (s->sps_max_sub_layers_minus1 != 0) {
        return false;  /* single layer only in this module */
    }
    s->sps_seq_parameter_set_id = b2_ue(&b);
    s->chroma_format_idc = b2_ue(&b);
    if (s->chroma_format_idc == 3) {
        b2_bit(&b);
    }
    s->pic_width_in_luma_samples = b2_ue(&b);
    s->pic_height_in_luma_samples = b2_ue(&b);
    int conf = b2_bit(&b);
    if (conf) {
        s->conf_win_left_offset = b2_ue(&b);
        s->conf_win_right_offset = b2_ue(&b);
        s->conf_win_top_offset = b2_ue(&b);
        s->conf_win_bottom_offset = b2_ue(&b);
    }
    s->bit_depth_luma_minus8 = b2_ue(&b);
    s->bit_depth_chroma_minus8 = b2_ue(&b);
    s->log2_max_pic_order_cnt_lsb_minus4 = b2_ue(&b);
    int sub_ordering = b2_bit(&b);
    (void)sub_ordering;
    /* sps_sub_layer_ordering_info: with 0 sub layers exactly one triple */
    d->std_dpbm.max_dec_pic_buffering_minus1[0] = b2_ue(&b);
    d->std_dpbm.max_num_reorder_pics[0] = b2_ue(&b);
    d->std_dpbm.max_latency_increase_plus1[0] = b2_ue(&b);
    s->log2_min_luma_coding_block_size_minus3 = b2_ue(&b);
    s->log2_diff_max_min_luma_coding_block_size = b2_ue(&b);
    s->log2_min_luma_transform_block_size_minus2 = b2_ue(&b);
    s->log2_diff_max_min_luma_transform_block_size = b2_ue(&b);
    s->max_transform_hierarchy_depth_inter = b2_ue(&b);
    s->max_transform_hierarchy_depth_intra = b2_ue(&b);
    if (b2_bit(&b)) {
        return false;  /* scaling_list_enabled - unsupported */
    }
    int amp_enabled = b2_bit(&b);
    s->flags.amp_enabled_flag = amp_enabled;
    int sao = b2_bit(&b);
    s->flags.sample_adaptive_offset_enabled_flag = sao;
    if (b2_bit(&b)) {
        return false;  /* pcm_enabled - unsupported */
    }
    int num_st = b2_ue(&b);
    s->num_short_term_ref_pic_sets = num_st;
    {
        int ndp[65] = { 0 };
        for (int k = 0; k < num_st && k < 64; k++) {
            h265_parse_st_rps(&b, k, ndp);
        }
    }
    if (b2_bit(&b)) {
        return false;  /* long_term_ref_pics_present */
    }
    s->flags.sps_temporal_mvp_enabled_flag = b2_bit(&b);
    s->flags.strong_intra_smoothing_enabled_flag = b2_bit(&b);
    if (b2_bit(&b)) {
        /* vui_parameters_present - skipped per the std contract */
    }
    s->pProfileTierLevel = &d->std_ptl;
    s->pDecPicBufMgr = &d->std_dpbm;

    d->sps.w = s->pic_width_in_luma_samples;
    d->sps.h = s->pic_height_in_luma_samples;
    d->sps.profile_idc = prof;
    d->sps.level_idc = lvl;
    d->sps.vps_id = s->sps_video_parameter_set_id;
    d->sps.sps_id = s->sps_seq_parameter_set_id;
    d->sps.sub_layers = s->sps_max_sub_layers_minus1;
    return d->sps.w > 0 && d->sps.h > 0 && (d->sps.w & 7) == 0 && (d->sps.h & 7) == 0;
}

static void h265_parse_pps(const uint8_t *n, uint32_t nlen, VD265 *d)
{
    static uint8_t clean[512];
    h265_strip_epb(clean, n + 2, nlen > 2 ? nlen - 2 : 0);
    BR265 b = { clean, 0 };
    StdVideoH265PictureParameterSet *p = &d->std_pps;
    memset(p, 0, sizeof(*p));
    p->pps_pic_parameter_set_id = b2_ue(&b);
    p->pps_seq_parameter_set_id = b2_ue(&b);
    p->flags.dependent_slice_segments_enabled_flag = b2_bit(&b);
    p->flags.output_flag_present_flag = b2_bit(&b);
    p->num_extra_slice_header_bits = b2_bits(&b, 3);
    p->flags.sign_data_hiding_enabled_flag = b2_bit(&b);
    p->flags.cabac_init_present_flag = b2_bit(&b);
    p->num_ref_idx_l0_default_active_minus1 = b2_ue(&b);
    p->num_ref_idx_l1_default_active_minus1 = b2_ue(&b);
    p->init_qp_minus26 = b2_se(&b);
    p->flags.constrained_intra_pred_flag = b2_bit(&b);
    p->flags.transform_skip_enabled_flag = b2_bit(&b);
    int cu_qp_delta = b2_bit(&b);
    p->flags.cu_qp_delta_enabled_flag = cu_qp_delta;
    if (cu_qp_delta) {
        p->diff_cu_qp_delta_depth = b2_ue(&b);
    }
    p->pps_cb_qp_offset = b2_se(&b);
    p->pps_cr_qp_offset = b2_se(&b);
    p->flags.pps_slice_chroma_qp_offsets_present_flag = b2_bit(&b);
    p->flags.weighted_pred_flag = b2_bit(&b);
    p->flags.weighted_bipred_flag = b2_bit(&b);
    p->flags.transquant_bypass_enabled_flag = b2_bit(&b);
    int tiles = b2_bit(&b);
    p->flags.tiles_enabled_flag = tiles;
    if (tiles) {
        return;  /* tiles unsupported: leave the set partially filled */
    }
    p->flags.entropy_coding_sync_enabled_flag = b2_bit(&b);
    p->flags.pps_loop_filter_across_slices_enabled_flag = b2_bit(&b);
    int deblock_present = b2_bit(&b);
    p->flags.deblocking_filter_control_present_flag = deblock_present;
    if (deblock_present) {
        p->flags.deblocking_filter_override_enabled_flag = b2_bit(&b);
        int disabled = b2_bit(&b);
        p->flags.pps_deblocking_filter_disabled_flag = disabled;
        if (disabled) {
            p->pps_beta_offset_div2 = b2_se(&b);
            p->pps_tc_offset_div2 = b2_se(&b);
        }
    }
    int scl = b2_bit(&b);
    p->flags.pps_scaling_list_data_present_flag = scl;
    if (scl) {
        return;  /* scaling list data - unsupported */
    }
    p->flags.lists_modification_present_flag = b2_bit(&b);
    p->log2_parallel_merge_level_minus2 = b2_ue(&b);
    p->flags.slice_segment_header_extension_present_flag = b2_bit(&b);
}

/* slice segment header (enough for the std picture info of an IRAP frame) */
typedef struct H265Slice {
    int is_irap, slice_type;   /* slice_type: 2 = I */
    int poc_lsb;
    int no_output_prior;
} H265Slice;

static void h265_parse_slice(const uint8_t *n, VD265 *d, H265Slice *out)
{
    int nt = h265_nal_type(n);
    memset(out, 0, sizeof(*out));
    out->is_irap = (nt >= 16 && nt <= 23);   /* BLA..RSV_IRAP_VCL23 */
    static uint8_t clean[256];
    h265_strip_epb(clean, n + 2, 64);
    BR265 b = { clean, 0 };
    int first = b2_bit(&b);
    (void)first;
    if (out->is_irap) {
        out->no_output_prior = b2_bit(&b);
    }
    b2_ue(&b);   /* slice_pic_parameter_set_id */
    /* (first slice => no dependent/segment address) */
    out->slice_type = b2_ue(&b);
    if (!out->is_irap) {
        /* slice_pic_order_cnt_lsb only when not IDR (needs sps log2) */
        out->poc_lsb = b2_bits(&b, d->std_sps.log2_max_pic_order_cnt_lsb_minus4 + 4);
    }
}

static void h265_hash(const uint8_t *dd, uint32_t n, uint8_t out[8])
{
    uint64_t h = 1469598103934665603ull;
    for (uint32_t i = 0; i < n; i++) {
        h ^= dd[i];
        h *= 1099511628211ull;
    }
    for (int i = 0; i < 8; i++) {
        out[i] = (uint8_t)(h >> (i * 8));
    }
}

static bool q_mem_type(uint32_t bits, uint32_t want, uint32_t *out)
{
    VkPhysicalDeviceMemoryProperties mp;
    Q.GetPhysicalDeviceMemoryProperties(L265.pd, &mp);
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

VtpvDecoder *vtpv_dec265_open(const char *gpu_pref,
                              const uint8_t *vps_nal, uint32_t vps_len,
                              const uint8_t *sps_nal, uint32_t sps_len,
                              const uint8_t *pps_nal, uint32_t pps_len)
{
    VD265 *d = g_new0(VD265, 1);
    VkResult r;
    d->t0 = qemu_clock_get_ms(QEMU_CLOCK_REALTIME);

    L265.lib = LoadLibraryA("vulkan-1.dll");
    if (!L265.lib) {
        goto fail0;
    }
    L265.GetInstanceProcAddr =
        (void *(VKAPI_PTR *)(VkInstance, const char *))
        GetProcAddress(L265.lib, "vkGetInstanceProcAddr");
    L265.GetDeviceProcAddr =
        (void *(VKAPI_PTR *)(VkDevice, const char *))
        GetProcAddress(L265.lib, "vkGetDeviceProcAddr");
    if (!L265.GetInstanceProcAddr || !L265.GetDeviceProcAddr) {
        goto fail0;
    }
    PFN_vkCreateInstance createInst = (PFN_vkCreateInstance)
        L265.GetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance");
    VkApplicationInfo ai = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "apple-vtpv-h265",
        .apiVersion = VK_MAKE_VERSION(1, 3, 0),
    };
    VkInstanceCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &ai,
    };
    r = createInst(&ici, NULL, &d->inst);
    if (r != VK_SUCCESS) {
        goto fail0;
    }
    L265.inst = d->inst;
    QI(EnumeratePhysicalDevices);
    QI(GetPhysicalDeviceProperties2);
    QI(GetPhysicalDeviceMemoryProperties);
    QI(GetPhysicalDeviceQueueFamilyProperties2);
    QI(GetPhysicalDeviceVideoCapabilitiesKHR);
    QI(CreateDevice);
    if (!Q.EnumeratePhysicalDevices || !Q.CreateDevice) {
        goto fail1;
    }

    bool want_int = gpu_pref &&
                    (!strcmp(gpu_pref, "int") || !strcmp(gpu_pref, "intel"));
    uint32_t nd = 0;
    Q.EnumeratePhysicalDevices(d->inst, &nd, NULL);
    if (!nd) {
        goto fail1;
    }
    VkPhysicalDevice devs[8];
    Q.EnumeratePhysicalDevices(d->inst, &nd, devs);
    for (uint32_t k = 0; k < nd && k < 8; k++) {
        VkPhysicalDeviceProperties2 p2 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        Q.GetPhysicalDeviceProperties2(devs[k], &p2);
        bool is_intel = p2.properties.vendorID == 0x8086;
        if ((want_int && is_intel) || (!want_int && !is_intel)) {
            d->pd = devs[k];
            break;
        }
    }
    if (!d->pd) {
        goto fail1;
    }
    L265.pd = d->pd;

    /* parse the parameter sets */
    memset(&d->std_vps, 0, sizeof(d->std_vps));
    if (vps_nal && vps_len >= 3) {
        static uint8_t clean[512];
        h265_strip_epb(clean, vps_nal + 2, vps_len - 2);
        BR265 b = { clean, 0 };
        d->std_vps.vps_video_parameter_set_id = b2_bits(&b, 4);
        int base_internal = b2_bit(&b);   /* vps_base_layer flags (2 bits) */
        int base_available = b2_bit(&b);
        (void)base_internal;
        (void)base_available;
        b2_bits(&b, 6);   /* vps_max_layers_minus1 */
        int max_sub = b2_bits(&b, 3);
        int tl_nesting = b2_bit(&b);
        if (max_sub != 0) {
            goto fail1;
        }
        d->std_vps.vps_max_sub_layers_minus1 = max_sub;
        d->std_vps.flags.vps_temporal_id_nesting_flag = tl_nesting;
        StdVideoH265ProfileTierLevel ptl;
        int pf = 0, lv = 0;
        h265_parse_ptl(&b, &ptl, &pf, &lv);
    }
    g_h265_nal_len = sps_len;
    if (!h265_parse_sps(sps_nal, d)) {
        fprintf(stderr, "vtpv-dec265: unsupported SPS\n");
        goto fail1;
    }
    h265_parse_pps(pps_nal, pps_len, d);
    d->w = d->sps.w;
    d->h = d->sps.h;
    h265_hash(vps_nal, vps_len, d->vps_hash);
    h265_hash(sps_nal, sps_len, d->sps_hash);
    h265_hash(pps_nal, pps_len, d->pps_hash);

    StdVideoH265ProfileIdc prof =
        d->sps.profile_idc == 2 ? STD_VIDEO_H265_PROFILE_IDC_MAIN_10 :
        d->sps.profile_idc == 3 ? STD_VIDEO_H265_PROFILE_IDC_MAIN_STILL_PICTURE :
        d->sps.profile_idc == 4 ? STD_VIDEO_H265_PROFILE_IDC_FORMAT_RANGE_EXTENSIONS :
        d->sps.profile_idc == 9 ? STD_VIDEO_H265_PROFILE_IDC_SCC_EXTENSIONS :
        STD_VIDEO_H265_PROFILE_IDC_MAIN;
    VkVideoDecodeH265ProfileInfoKHR hprof = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PROFILE_INFO_KHR,
        .stdProfileIdc = prof,
    };
    VkVideoProfileInfoKHR vprof = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR,
        .pNext = &hprof,
        .videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR,
        .chromaSubsampling = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR,
        .lumaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
        .chromaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
    };

    uint32_t nq = 0;
    Q.GetPhysicalDeviceQueueFamilyProperties2(d->pd, &nq, NULL);
    VkQueueFamilyProperties2 qs[16];
    VkQueueFamilyVideoPropertiesKHR qv[16];
    memset(qs, 0, sizeof(qs));
    memset(qv, 0, sizeof(qv));
    for (uint32_t k = 0; k < nq && k < 16; k++) {
        qs[k].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
        qs[k].pNext = &qv[k];
        qv[k].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_VIDEO_PROPERTIES_KHR;
    }
    Q.GetPhysicalDeviceQueueFamilyProperties2(d->pd, &nq, qs);
    int dfam = -1;
    for (uint32_t k = 0; k < nq && k < 16; k++) {
        if ((qs[k].queueFamilyProperties.queueFlags &
             VK_QUEUE_VIDEO_DECODE_BIT_KHR) &&
            (qv[k].videoCodecOperations &
             VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR)) {
            dfam = k;
            break;
        }
    }
    if (dfam < 0) {
        fprintf(stderr, "vtpv-dec265: no h265 decode queue\n");
        goto fail1;
    }
    d->dfam = dfam;

    /* compute family for the UV shader path */
    d->uv_fam = -1;
    {
        VkPhysicalDeviceProperties2 pdp2 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        Q.GetPhysicalDeviceProperties2(d->pd, &pdp2);
        if (pdp2.properties.vendorID == 0x10de &&
            !getenv("VTPV_DEC_NOSHADER")) {
            uint32_t nq2 = 0;
            Q.GetPhysicalDeviceQueueFamilyProperties2(d->pd, &nq2, NULL);
            VkQueueFamilyProperties2 *qp = g_new0(VkQueueFamilyProperties2, nq2);
            for (uint32_t k = 0; k < nq2; k++) {
                qp[k].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
            }
            Q.GetPhysicalDeviceQueueFamilyProperties2(d->pd, &nq2, qp);
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
    VkVideoDecodeH265CapabilitiesKHR hcap = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_CAPABILITIES_KHR,
        .pNext = &dcap };
    VkVideoCapabilitiesKHR cap = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR, .pNext = &hcap };
    r = Q.GetPhysicalDeviceVideoCapabilitiesKHR(d->pd, &vprof, &cap);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "vtpv-dec265: caps rc=%d\n", (int)r);
        goto fail1;
    }
    d->bs_size_align = cap.minBitstreamBufferSizeAlignment
                       ? cap.minBitstreamBufferSizeAlignment : 256;

    const char *devext[] = {
        VK_KHR_VIDEO_QUEUE_EXTENSION_NAME,
        VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME,
        VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME,
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
    r = Q.CreateDevice(d->pd, &dci, NULL, &d->dev);
    if (r != VK_SUCCESS) {
        goto fail1;
    }
    {
        VkDevice dev = d->dev;
        QD(DestroyDevice);
        QD(GetDeviceQueue);
        QD(CreateVideoSessionKHR);
        QD(DestroyVideoSessionKHR);
        QD(GetVideoSessionMemoryRequirementsKHR);
        QD(BindVideoSessionMemoryKHR);
        QD(CreateVideoSessionParametersKHR);
        QD(DestroyVideoSessionParametersKHR);
        QD(CreateCommandPool);
        QD(DestroyCommandPool);
        QD(AllocateCommandBuffers);
        QD(BeginCommandBuffer);
        QD(EndCommandBuffer);
        QD(QueueSubmit);
        QD(QueueWaitIdle);
        QD(CreateBuffer);
        QD(DestroyBuffer);
        QD(GetBufferMemoryRequirements);
        QD(AllocateMemory);
        QD(FreeMemory);
        QD(MapMemory);
        QD(UnmapMemory);
        QD(FlushMappedMemoryRanges);
        QD(BindBufferMemory);
        QD(CreateImage);
        QD(DestroyImage);
        QD(GetImageMemoryRequirements);
        QD(BindImageMemory);
        QD(CreateImageView);
        QD(DestroyImageView);
        QD(CmdPipelineBarrier2);
        QD(CmdBeginVideoCodingKHR);
        QD(CmdControlVideoCodingKHR);
        QD(CmdEndVideoCodingKHR);
        QD(CmdDecodeVideoKHR);
        QD(CmdBeginQuery);
        QD(CmdEndQuery);
        QD(CmdResetQueryPool);
        QD(CreateQueryPool);
        QD(GetQueryPoolResults);
        QD(CmdCopyImageToBuffer);
        QD(InvalidateMappedMemoryRanges);
        QD(CreateSampler);
        QD(DestroySampler);
        QD(CreateDescriptorSetLayout);
        QD(DestroyDescriptorSetLayout);
        QD(CreatePipelineLayout);
        QD(DestroyPipelineLayout);
        QD(CreateShaderModule);
        QD(DestroyShaderModule);
        QD(CreateComputePipelines);
        QD(DestroyPipeline);
        QD(CreateDescriptorPool);
        QD(DestroyDescriptorPool);
        QD(AllocateDescriptorSets);
        QD(UpdateDescriptorSets);
        QD(CmdBindPipeline);
        QD(CmdBindDescriptorSets);
        QD(CmdDispatch);
    }
    if (!Q.CmdDecodeVideoKHR || !Q.CreateVideoSessionKHR) {
        goto fail2;
    }
    Q.GetDeviceQueue(d->dev, d->dfam, 0, &d->q);
    if (d->uv_on) {
        Q.GetDeviceQueue(d->dev, d->uv_fam, 0, &d->uvq);
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
    r = Q.CreateVideoSessionKHR(d->dev, &vsci, NULL, &d->session);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "vtpv-dec265: session rc=%d\n", (int)r);
        goto fail2;
    }
    uint32_t nb = 0;
    Q.GetVideoSessionMemoryRequirementsKHR(d->dev, d->session, &nb, NULL);
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
        Q.GetVideoSessionMemoryRequirementsKHR(d->dev, d->session, &nb, mr);
        for (uint32_t k = 0; k < nb; k++) {
            uint32_t mt;
            if (!q_mem_type(mr[k].memoryRequirements.memoryTypeBits,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &mt)) {
                goto fail3;
            }
            VkMemoryAllocateInfo mai = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .allocationSize = mr[k].memoryRequirements.size,
                .memoryTypeIndex = mt,
            };
            r = Q.AllocateMemory(d->dev, &mai, NULL, &mems[k]);
            if (r != VK_SUCCESS) {
                goto fail3;
            }
            bi[k].sType = VK_STRUCTURE_TYPE_BIND_VIDEO_SESSION_MEMORY_INFO_KHR;
            bi[k].memoryBindIndex = mr[k].memoryBindIndex;
            bi[k].memory = mems[k];
            bi[k].memoryOffset = 0;
            bi[k].memorySize = mr[k].memoryRequirements.size;
        }
        r = Q.BindVideoSessionMemoryKHR(d->dev, d->session, nb, bi);
        if (r != VK_SUCCESS) {
            fprintf(stderr, "vtpv-dec265: bind rc=%d\n", (int)r);
            goto fail3;
        }
    }

    VkVideoDecodeH265SessionParametersAddInfoKHR add = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_PARAMETERS_ADD_INFO_KHR,
        .stdVPSCount = 1,
        .pStdVPSs = &d->std_vps,
        .stdSPSCount = 1,
        .pStdSPSs = &d->std_sps,
        .stdPPSCount = 1,
        .pStdPPSs = &d->std_pps,
    };
    VkVideoDecodeH265SessionParametersCreateInfoKHR hci = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_PARAMETERS_CREATE_INFO_KHR,
        .maxStdVPSCount = 1,
        .maxStdSPSCount = 1,
        .maxStdPPSCount = 1,
        .pParametersAddInfo = &add,
    };
    VkVideoSessionParametersCreateInfoKHR pci = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,
        .pNext = &hci,
        .videoSession = d->session,
    };
    r = Q.CreateVideoSessionParametersKHR(d->dev, &pci, NULL, &d->params);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "vtpv-dec265: params rc=%d\n", (int)r);
        goto fail3;
    }

    VkCommandPoolCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = d->dfam,
    };
    if (Q.CreateCommandPool(d->dev, &cpci, NULL, &d->pool) != VK_SUCCESS) {
        goto fail4;
    }
    VkQueryPoolCreateInfo qpci = {
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .pNext = &vprof,
        .queryType = VK_QUERY_TYPE_RESULT_STATUS_ONLY_KHR,
        .queryCount = 1,
    };
    if (Q.CreateQueryPool(d->dev, &qpci, NULL, &d->qpool) != VK_SUCCESS) {
        goto fail5;
    }

    /* codec RESET in its own submit (empty parameters) */
    {
        VkVideoDecodeH265SessionParametersCreateInfoKHR eh = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_PARAMETERS_CREATE_INFO_KHR };
        VkVideoSessionParametersCreateInfoKHR ep = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,
            .pNext = &eh,
            .videoSession = d->session };
        VkVideoSessionParametersKHR epms;
        if (Q.CreateVideoSessionParametersKHR(d->dev, &ep, NULL, &epms)
            == VK_SUCCESS) {
            VkCommandBuffer cb;
            VkCommandBufferAllocateInfo cai = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .commandPool = d->pool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1 };
            if (Q.AllocateCommandBuffers(d->dev, &cai, &cb) == VK_SUCCESS) {
                VkCommandBufferBeginInfo bi2 = {
                    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
                Q.BeginCommandBuffer(cb, &bi2);
                VkVideoBeginCodingInfoKHR rbeg = {
                    .sType = VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR,
                    .videoSession = d->session,
                    .videoSessionParameters = epms };
                Q.CmdBeginVideoCodingKHR(cb, &rbeg);
                VkVideoCodingControlInfoKHR rctl = {
                    .sType = VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR,
                    .flags = VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR };
                Q.CmdControlVideoCodingKHR(cb, &rctl);
                VkVideoEndCodingInfoKHR rend = {
                    .sType = VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR };
                Q.CmdEndVideoCodingKHR(cb, &rend);
                Q.EndCommandBuffer(cb);
                VkSubmitInfo si = {
                    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    .commandBufferCount = 1,
                    .pCommandBuffers = &cb };
                r = Q.QueueSubmit(d->q, 1, &si, VK_NULL_HANDLE);
                if (r == VK_SUCCESS) {
                    Q.QueueWaitIdle(d->q);
                }
            }
            Q.DestroyVideoSessionParametersKHR(d->dev, epms, NULL);
        }
    }

    d->bs_cap = 512 * 1024;
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = d->bs_cap,
        .usage = VK_BUFFER_USAGE_VIDEO_DECODE_SRC_BIT_KHR,
    };
    if (Q.CreateBuffer(d->dev, &bci, NULL, &d->bsbuf) != VK_SUCCESS) {
        goto fail5;
    }
    VkMemoryRequirements bmr;
    Q.GetBufferMemoryRequirements(d->dev, d->bsbuf, &bmr);
    uint32_t bmt;
    if (!q_mem_type(bmr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, &bmt)) {
        goto fail6;
    }
    VkMemoryAllocateInfo bmai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = bmr.size,
        .memoryTypeIndex = bmt,
    };
    if (Q.AllocateMemory(d->dev, &bmai, NULL, &d->bsmem) != VK_SUCCESS ||
        Q.BindBufferMemory(d->dev, d->bsbuf, d->bsmem, 0) != VK_SUCCESS) {
        goto fail6;
    }

    {
        uint32_t fams[1] = { d->dfam };
        VkVideoProfileListInfoKHR plist = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR,
            .profileCount = 1,
            .pProfiles = &vprof,
        };
        VkImageCreateInfo iici = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = &plist,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
            .extent = { d->w, d->h, 1 },
            .mipLevels = 1,
            .arrayLayers = 1,
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
        if (Q.CreateImage(d->dev, &iici, NULL, &d->img) != VK_SUCCESS) {
            goto fail6;
        }
        VkMemoryRequirements imr;
        Q.GetImageMemoryRequirements(d->dev, d->img, &imr);
        uint32_t imt;
        if (!q_mem_type(imr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &imt)) {
            goto fail7;
        }
        VkMemoryDedicatedAllocateInfo iddi = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
            .image = d->img,
        };
        VkMemoryAllocateInfo imai = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = &iddi,
            .allocationSize = imr.size,
            .memoryTypeIndex = imt,
        };
        if (Q.AllocateMemory(d->dev, &imai, NULL, &d->imgmem) != VK_SUCCESS ||
            Q.BindImageMemory(d->dev, d->img, d->imgmem, 0) != VK_SUCCESS) {
            goto fail7;
        }
        VkImageViewCreateInfo vci = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = &plist,
            .image = d->img,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
        };
        if (Q.CreateImageView(d->dev, &vci, NULL, &d->imgview) != VK_SUCCESS) {
            goto fail8;
        }
    }

    /* UV shader context (single image; see H264 module for the story) */
    if (d->uv_on) {
        bool ok = true;
        VkImageViewCreateInfo uvi = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = d->img,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_R8G8_UNORM,
            .subresourceRange = { VK_IMAGE_ASPECT_PLANE_1_BIT, 0, 1, 0, 1 },
        };
        ok = Q.CreateImageView(d->dev, &uvi, NULL, &d->uvview) == VK_SUCCESS;
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
        if (ok && Q.CreateSampler(d->dev, &sci, NULL, &d->uvsamp)
            != VK_SUCCESS) {
            ok = false;
        }
        d->uvn = (size_t)(d->w / 2) * (d->h / 2);
        VkBufferCreateInfo ubc = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = d->uvn * 8,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        };
        if (ok && Q.CreateBuffer(d->dev, &ubc, NULL, &d->uvbuf)
            != VK_SUCCESS) {
            ok = false;
        }
        if (ok) {
            VkMemoryRequirements umr;
            Q.GetBufferMemoryRequirements(d->dev, d->uvbuf, &umr);
            uint32_t mt;
            if (!q_mem_type(umr.memoryTypeBits,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, &mt)) {
                ok = false;
            } else {
                VkMemoryAllocateInfo umai = {
                    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    .allocationSize = umr.size,
                    .memoryTypeIndex = mt,
                };
                if (Q.AllocateMemory(d->dev, &umai, NULL, &d->uvmem)
                        != VK_SUCCESS ||
                    Q.BindBufferMemory(d->dev, d->uvbuf, d->uvmem, 0)
                        != VK_SUCCESS ||
                    Q.MapMemory(d->dev, d->uvmem, 0, VK_WHOLE_SIZE, 0,
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
            if (Q.CreateDescriptorSetLayout(d->dev, &dslc, NULL, &d->uvdsl)
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
            if (Q.CreatePipelineLayout(d->dev, &plc, NULL, &d->uvpl)
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
            if (Q.CreateShaderModule(d->dev, &smc, NULL, &sm) != VK_SUCCESS) {
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
                if (Q.CreateComputePipelines(d->dev, VK_NULL_HANDLE, 1,
                                             &cpc, NULL, &d->uvpipe)
                    != VK_SUCCESS) {
                    ok = false;
                }
                Q.DestroyShaderModule(d->dev, sm, NULL);
            }
        }
        if (ok) {
            VkDescriptorPoolSize dps[2] = {
                { .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                  .descriptorCount = 1 },
                { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                  .descriptorCount = 1 },
            };
            VkDescriptorPoolCreateInfo dpc = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets = 1,
                .poolSizeCount = 2,
                .pPoolSizes = dps,
            };
            if (Q.CreateDescriptorPool(d->dev, &dpc, NULL, &d->uvdpool)
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
            if (Q.AllocateDescriptorSets(d->dev, &dsai, &d->uvset)
                != VK_SUCCESS) {
                ok = false;
            }
        }
        if (ok) {
            VkDescriptorImageInfo dii = {
                .sampler = d->uvsamp,
                .imageView = d->uvview,
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
            Q.UpdateDescriptorSets(d->dev, 2, dw, 0, NULL);
            VkCommandPoolCreateInfo upc = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .queueFamilyIndex = (uint32_t)d->uv_fam,
            };
            if (Q.CreateCommandPool(d->dev, &upc, NULL, &d->uvpool)
                != VK_SUCCESS) {
                ok = false;
            }
        }
        if (!ok) {
            fprintf(stderr, "vtpv-dec265: UV shader init failed, "
                    "chroma falls back to 0x80\n");
            d->uv_on = 0;
        } else {
            fprintf(stderr, "vtpv-dec265: UV shader on family %d\n",
                    d->uv_fam);
        }
    }

    {
        VkBufferCreateInfo rbc = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = (VkDeviceSize)d->w * d->h * 3 / 2 + 256,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        };
        if (Q.CreateBuffer(d->dev, &rbc, NULL, &d->rbbuf) != VK_SUCCESS) {
            goto fail8;
        }
        VkMemoryRequirements rmr;
        Q.GetBufferMemoryRequirements(d->dev, d->rbbuf, &rmr);
        uint32_t rmt;
        if (!q_mem_type(rmr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, &rmt)) {
            goto fail9;
        }
        VkMemoryAllocateInfo rmai = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = rmr.size,
            .memoryTypeIndex = rmt,
        };
        if (Q.AllocateMemory(d->dev, &rmai, NULL, &d->rbmem) != VK_SUCCESS ||
            Q.BindBufferMemory(d->dev, d->rbbuf, d->rbmem, 0) != VK_SUCCESS ||
            Q.MapMemory(d->dev, d->rbmem, 0, VK_WHOLE_SIZE, 0, &d->rbmap)
                != VK_SUCCESS) {
            goto fail9;
        }
    }

    fprintf(stderr, "vtpv-dec265: open %ux%d prof=%d took %llu ms\n",
            d->w, d->h, d->sps.profile_idc,
            (unsigned long long)(qemu_clock_get_ms(QEMU_CLOCK_REALTIME) - d->t0));
    return (VtpvDecoder *)d;

fail9:
    if (d->rbbuf) {
        Q.DestroyBuffer(d->dev, d->rbbuf, NULL);
    }
fail8:
    if (d->imgmem) {
        Q.FreeMemory(d->dev, d->imgmem, NULL);
    }
fail7:
    if (d->img) {
        Q.DestroyImage(d->dev, d->img, NULL);
    }
fail6:
    if (d->bsbuf) {
        Q.DestroyBuffer(d->dev, d->bsbuf, NULL);
    }
    if (d->bsmem) {
        Q.FreeMemory(d->dev, d->bsmem, NULL);
    }
fail5:
    Q.DestroyCommandPool(d->dev, d->pool, NULL);
fail4:
    if (d->params) {
        Q.DestroyVideoSessionParametersKHR(d->dev, d->params, NULL);
    }
fail3:
    Q.DestroyVideoSessionKHR(d->dev, d->session, NULL);
fail2:
    Q.DestroyDevice(d->dev, NULL);
fail1:
    {
        PFN_vkDestroyInstance di = (PFN_vkDestroyInstance)
            L265.GetInstanceProcAddr(d->inst, "vkDestroyInstance");
        if (di) {
            di(d->inst, NULL);
        }
    }
fail0:
    fprintf(stderr, "vtpv-dec265: open failed\n");
    g_free(d);
    return NULL;
}

int vtpv_dec265_frame(VtpvDecoder *dec, const uint8_t *frame, uint32_t frame_len,
                      const uint32_t *slice_off, uint32_t nsl,
                      uint8_t *nv12_out)
{
    VD265 *d = (VD265 *)dec;
    if (!d || frame_len > d->bs_cap || nsl == 0) {
        return -1;
    }
    uint64_t t0 = qemu_clock_get_ms(QEMU_CLOCK_REALTIME);
    H265Slice sh;
    h265_parse_slice(frame + slice_off[0] + 3, d, &sh);

    void *p = NULL;
    if (Q.MapMemory(d->dev, d->bsmem, 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS) {
        return -1;
    }
    memset(p, 0, (size_t)d->bs_cap);
    memcpy(p, frame, frame_len);
    VkMappedMemoryRange fl = {
        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .memory = d->bsmem,
        .size = VK_WHOLE_SIZE,
    };
    Q.FlushMappedMemoryRanges(d->dev, 1, &fl);
    Q.UnmapMemory(d->dev, d->bsmem);

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
    if (Q.AllocateCommandBuffers(d->dev, &cai, &cb) != VK_SUCCESS) {
        return -1;
    }
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    Q.BeginCommandBuffer(cb, &bi);

    VkVideoPictureResourceInfoKHR res = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR,
        .codedOffset = { 0, 0 },
        .codedExtent = { d->w, d->h },
        .baseArrayLayer = 0,
        .imageViewBinding = d->imgview,
    };
    VkVideoReferenceSlotInfoKHR inactive = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,
        .slotIndex = -1,
        .pPictureResource = &res,
    };
    VkVideoBeginCodingInfoKHR beg = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR,
        .videoSession = d->session,
        .videoSessionParameters = d->params,
        .referenceSlotCount = 1,
        .pReferenceSlots = &inactive,
    };
    Q.CmdBeginVideoCodingKHR(cb, &beg);
    Q.CmdResetQueryPool(cb, d->qpool, 0, 1);
    Q.CmdBeginQuery(cb, d->qpool, 0, 0);

    int nal0_type = h265_nal_type(frame + slice_off[0] + 3);
    StdVideoDecodeH265PictureInfo stdpic = {
        .flags = {
            .IrapPicFlag = sh.is_irap,
            .IdrPicFlag = (nal0_type == 19 || nal0_type == 20),
            .IsReference = 1,
            .short_term_ref_pic_set_sps_flag = 0,
        },
        .sps_video_parameter_set_id = d->std_sps.sps_video_parameter_set_id,
        .pps_seq_parameter_set_id = d->std_pps.pps_seq_parameter_set_id,
        .pps_pic_parameter_set_id = d->std_pps.pps_pic_parameter_set_id,
        .PicOrderCntVal = sh.is_irap ? 0 : sh.poc_lsb,
        .NumBitsForSTRefPicSetInSlice = 0,
    };
    VkVideoDecodeH265PictureInfoKHR h265i = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PICTURE_INFO_KHR,
        .pStdPictureInfo = &stdpic,
        .sliceSegmentCount = nsl,
        .pSliceSegmentOffsets = slice_off,
    };
    StdVideoDecodeH265ReferenceInfo stdref = {
        .flags.used_for_long_term_reference = 0,
        .PicOrderCntVal = stdpic.PicOrderCntVal,
    };
    VkVideoDecodeH265DpbSlotInfoKHR dslot = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_DPB_SLOT_INFO_KHR,
        .pStdReferenceInfo = &stdref,
    };
    VkVideoReferenceSlotInfoKHR setup = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,
        .pNext = &dslot,
        .slotIndex = 0,
        .pPictureResource = &res,
    };
    VkVideoDecodeInfoKHR dec2 = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR,
        .pNext = &h265i,
        .srcBuffer = d->bsbuf,
        .srcBufferOffset = 0,
        .srcBufferRange = range,
        .dstPictureResource = res,
        .pSetupReferenceSlot = &setup,
    };
    Q.CmdDecodeVideoKHR(cb, &dec2);
    Q.CmdEndQuery(cb, d->qpool, 0);
    VkVideoEndCodingInfoKHR endi = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR };
    Q.CmdEndVideoCodingKHR(cb, &endi);

    VkImageMemoryBarrier2 tr = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR,
        .srcAccessMask = VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR,
        .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
        .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .image = d->img,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
    };
    VkDependencyInfo trd = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &tr,
    };
    Q.CmdPipelineBarrier2(cb, &trd);
    VkBufferImageCopy cp0 = {
        .bufferOffset = 0,
        .bufferRowLength = d->w,
        .bufferImageHeight = d->h,
        .imageSubresource = { VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 0, 1 },
        .imageExtent = { d->w, d->h, 1 },
    };
    Q.CmdCopyImageToBuffer(cb, d->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           d->rbbuf, 1, &cp0);
    if (!d->uv_on) {
        VkBufferImageCopy cp1 = {
            .bufferOffset = (VkDeviceSize)d->w * d->h,
            .bufferRowLength = d->w,
            .bufferImageHeight = d->h / 2,
            .imageSubresource = { VK_IMAGE_ASPECT_PLANE_1_BIT, 0, 0, 1 },
            .imageExtent = { d->w, d->h / 2, 1 },
        };
        Q.CmdCopyImageToBuffer(cb, d->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               d->rbbuf, 1, &cp1);
    }
    VkBufferMemoryBarrier2 rb = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
        .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
        .buffer = d->rbbuf,
        .size = VK_WHOLE_SIZE,
    };
    VkDependencyInfo rbd = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &rb,
    };
    Q.CmdPipelineBarrier2(cb, &rbd);
    if (d->uv_on) {
        /* hand the picture to the compute family for the UV shader */
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
            .image = d->img,
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
        };
        VkDependencyInfo reld = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &rel,
        };
        Q.CmdPipelineBarrier2(cb, &reld);
    }
    Q.EndCommandBuffer(cb);

    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
    };
    VkResult r2 = Q.QueueSubmit(d->q, 1, &si, VK_NULL_HANDLE);
    if (r2 != VK_SUCCESS) {
        fprintf(stderr, "vtpv-dec265: submit rc=%d\n", (int)r2);
        return -1;
    }
    r2 = Q.QueueWaitIdle(d->q);
    if (r2 != VK_SUCCESS) {
        fprintf(stderr, "vtpv-dec265: wait rc=%d\n", (int)r2);
        return -1;
    }

    int32_t status = 0x7fffffff;
    Q.GetQueryPoolResults(d->dev, d->qpool, 0, 1, sizeof(status), &status,
                          sizeof(status),
                          VK_QUERY_RESULT_WITH_STATUS_BIT_KHR |
                          VK_QUERY_RESULT_WAIT_BIT);
    if (status != 1) {
        fprintf(stderr, "vtpv-dec265: status=%d\n", (int)status);
        return -1;
    }

    bool uv_done = false;
    if (d->uv_on) {
        VkCommandBufferBeginInfo ubi = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        };
        VkCommandBuffer ucb;
        VkCommandBufferAllocateInfo uai = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = d->uvpool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        if (Q.AllocateCommandBuffers(d->dev, &uai, &ucb) == VK_SUCCESS &&
            Q.BeginCommandBuffer(ucb, &ubi) == VK_SUCCESS) {
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
                .image = d->img,
                .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
            };
            VkDependencyInfo ad = {
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &acq,
            };
            Q.CmdPipelineBarrier2(ucb, &ad);
            Q.CmdBindPipeline(ucb, VK_PIPELINE_BIND_POINT_COMPUTE, d->uvpipe);
            Q.CmdBindDescriptorSets(ucb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    d->uvpl, 0, 1, &d->uvset, 0, NULL);
            Q.CmdDispatch(ucb, (d->w / 2 + 7) / 8, (d->h / 2 + 7) / 8, 1);
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
            Q.CmdPipelineBarrier2(ucb, &sd1);
            if (Q.EndCommandBuffer(ucb) == VK_SUCCESS) {
                VkCommandBuffer ucbs[1] = { ucb };
                VkSubmitInfo usi = {
                    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    .commandBufferCount = 1,
                    .pCommandBuffers = ucbs,
                };
                if (Q.QueueSubmit(d->uvq, 1, &usi, VK_NULL_HANDLE)
                        == VK_SUCCESS &&
                    Q.QueueWaitIdle(d->uvq) == VK_SUCCESS) {
                    VkMappedMemoryRange ifl = {
                        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                        .memory = d->uvmem,
                        .size = VK_WHOLE_SIZE,
                    };
                    Q.InvalidateMappedMemoryRanges(d->dev, 1, &ifl);
                    uv_done = true;
                } else {
                    fprintf(stderr, "vtpv-dec265: UV dispatch failed\n");
                }
            }
        }
    }

    fprintf(stderr, "vtpv-dec265: frame took %llu ms\n",
            (unsigned long long)(qemu_clock_get_ms(QEMU_CLOCK_REALTIME) - t0));

    const uint8_t *src = d->rbmap;
    memcpy(nv12_out, src, (size_t)d->w * d->h);
    uint32_t uvsz = d->w * d->h / 2;
    if (uv_done) {
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

bool vtpv_dec265_match(const VtpvDecoder *dec,
                       const uint8_t *vps, uint32_t vps_len,
                       const uint8_t *sps, uint32_t sps_len,
                       const uint8_t *pps, uint32_t pps_len)
{
    VD265 *d = (VD265 *)dec;
    if (!d) {
        return false;
    }
    uint8_t hv[8], hs[8], hp[8];
    h265_hash(vps, vps_len, hv);
    h265_hash(sps, sps_len, hs);
    h265_hash(pps, pps_len, hp);
    return memcmp(hv, d->vps_hash, 8) == 0 &&
           memcmp(hs, d->sps_hash, 8) == 0 &&
           memcmp(hp, d->pps_hash, 8) == 0;
}

uint32_t vtpv_dec265_width(const VtpvDecoder *dec)
{
    return ((VD265 *)dec)->w;
}

uint32_t vtpv_dec265_height(const VtpvDecoder *dec)
{
    return ((VD265 *)dec)->h;
}

void vtpv_dec265_close(VtpvDecoder *dec)
{
    VD265 *d = (VD265 *)dec;
    if (!d) {
        return;
    }
    if (d->rbmap && d->rbmem) {
        Q.UnmapMemory(d->dev, d->rbmem);
    }
    if (d->uvmap && d->uvmem) {
        Q.UnmapMemory(d->dev, d->uvmem);
    }
    if (d->uvbuf) {
        Q.DestroyBuffer(d->dev, d->uvbuf, NULL);
    }
    if (d->uvmem) {
        Q.FreeMemory(d->dev, d->uvmem, NULL);
    }
    if (d->uvpool) {
        Q.DestroyCommandPool(d->dev, d->uvpool, NULL);
    }
    if (d->uvpipe) {
        Q.DestroyPipeline(d->dev, d->uvpipe, NULL);
    }
    if (d->uvdpool) {
        Q.DestroyDescriptorPool(d->dev, d->uvdpool, NULL);
    }
    if (d->uvpl) {
        Q.DestroyPipelineLayout(d->dev, d->uvpl, NULL);
    }
    if (d->uvdsl) {
        Q.DestroyDescriptorSetLayout(d->dev, d->uvdsl, NULL);
    }
    if (d->uvsamp) {
        Q.DestroySampler(d->dev, d->uvsamp, NULL);
    }
    if (d->uvview) {
        Q.DestroyImageView(d->dev, d->uvview, NULL);
    }
    if (d->rbbuf) {
        Q.DestroyBuffer(d->dev, d->rbbuf, NULL);
    }
    if (d->rbmem) {
        Q.FreeMemory(d->dev, d->rbmem, NULL);
    }
    if (d->imgview) {
        Q.DestroyImageView(d->dev, d->imgview, NULL);
    }
    if (d->imgmem) {
        Q.FreeMemory(d->dev, d->imgmem, NULL);
    }
    if (d->img) {
        Q.DestroyImage(d->dev, d->img, NULL);
    }
    if (d->bsbuf) {
        Q.DestroyBuffer(d->dev, d->bsbuf, NULL);
    }
    if (d->bsmem) {
        Q.FreeMemory(d->dev, d->bsmem, NULL);
    }
    if (d->pool) {
        Q.DestroyCommandPool(d->dev, d->pool, NULL);
    }
    if (d->params) {
        Q.DestroyVideoSessionParametersKHR(d->dev, d->params, NULL);
    }
    if (d->session) {
        Q.DestroyVideoSessionKHR(d->dev, d->session, NULL);
    }
    if (d->dev) {
        Q.DestroyDevice(d->dev, NULL);
    }
    if (d->inst) {
        PFN_vkDestroyInstance di = (PFN_vkDestroyInstance)
            L265.GetInstanceProcAddr(d->inst, "vkDestroyInstance");
        if (di) {
            di(d->inst, NULL);
        }
    }
    g_free(d);
}
