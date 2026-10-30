/*
 * TEMP (lab): apple-vtpv-dx-enc.c - D3D12 Video Encode backend for the VTPV
 * device's ENCODE side (Zcode, task77, 30-10).  H.264 first, on the NVIDIA
 * adapter by default ("nv", the RTX 4060 of the Intel stand) or the Intel
 * iGPU ("int"), NV12 in host memory, Annex-B bytes out (SPS/PPS+IDR per
 * frame while the GOP is 1 - every sample self-contained, which is exactly
 * what the first end-to-end frame needs).
 *
 * All signatures checked against C:/msys64/ucrt64/include/d3d12video.h
 * (the task65 header lesson): CreateVideoEncoder/CreateVideoEncoderHeap on
 * ID3D12VideoDevice3, EncodeFrame + ResolveEncoderOutputMetadata on
 * ID3D12VideoEncodeCommandList2 (queue/list type VIDEO_ENCODE = 6); video
 * command lists have NO CopyTextureRegion - the NV12 upload rides its own
 * COPY queue and hands the texture over through COMMON state (the decode
 * path's pattern).
 *
 * Failure discipline: any D3D12 step fails -> open returns NULL / encode
 * returns nonzero, resources are released; the caller (apple-vtpv.c) keeps
 * answering encoder requests with the typed refusal, never a hang.
 * Validate standalone with lab/tools/session-2026-10-02/hwsurvey/dxeprobe.c
 * (the iprobe.c precedent) BEFORE wiring the device to it.
 */
#define COBJMACROS
#include <initguid.h>
#include <windows.h>
#include <d3d12.h>
#include <d3d12video.h>
#include <dxgi1_6.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include "apple-vtpv-dx.h"

struct VtpvDxE {
    UINT w, h;
    ID3D12Device *dev;
    ID3D12VideoDevice3 *vd;
    ID3D12CommandQueue *q;          /* VIDEO_ENCODE */
    ID3D12CommandAllocator *alloc;
    ID3D12VideoEncodeCommandList2 *cl;
    ID3D12CommandQueue *cq;         /* COPY: NV12 upload */
    ID3D12CommandAllocator *calloc_;
    ID3D12GraphicsCommandList *ccl;
    ID3D12Fence *fence;
    HANDLE ev;
    UINT64 fence_val;
    ID3D12VideoEncoder *enc;
    ID3D12VideoEncoderHeap *heap;
    ID3D12Resource *upload;          /* NV12 staging, per-plane footprints */
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp[2];
    UINT nrows[2];
    UINT64 rowsz[2];                 /* GetCopyableFootprints writes UINT64 row sizes */
    ID3D12Resource *tex;             /* encode input */
    ID3D12Resource *recon;           /* reconstructed picture */
    ID3D12Resource *bits;            /* compressed bitstream (readback) */
    ID3D12Resource *hwmeta;          /* HW layout metadata (default heap) */
    ID3D12Resource *rmeta;           /* resolved metadata (readback) */
    D3D12_VIDEO_ENCODER_PROFILE_DESC prof;
    D3D12_VIDEO_ENCODER_LEVEL_SETTING level;
    /* task80: the encode-time configuration lives here so the SUPPORT query
     * in open() and the recording in encode() use the SAME instances */
    D3D12_VIDEO_ENCODER_SEQUENCE_GOP_STRUCTURE_H264 gop;
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264 cc;
    D3D12_VIDEO_ENCODER_RATE_CONTROL_CBR cbr;
    D3D12_VIDEO_ENCODER_RATE_CONTROL rc;
    UINT64 bits_cap, meta_cap;
    UINT frame;
};

static UINT dxe_align(UINT v, UINT a)
{
    return (v + a - 1) & ~(a - 1);
}

VtpvDxE *vtpv_dxe_open(const char *gpu, uint32_t w, uint32_t h,
                       uint32_t fps, uint32_t bitrate)
{
    struct VtpvDxE *e = calloc(1, sizeof(*e));
    (void)fps;
    (void)bitrate;                    /* CBR fixed inside encode() for v1 */
    if (!e) {
        return NULL;
    }
    e->w = w;
    e->h = h;

    /* adapter: same pick logic as the decoder path - nv = NVIDIA, else Intel */
    UINT vendor = gpu && strcmp(gpu, "nv") == 0 ? 0x10deu : 0x8086u;
    IDXGIFactory4 *fac = NULL;
    IDXGIAdapter1 *ad = NULL;
    if (SUCCEEDED(CreateDXGIFactory1(&IID_IDXGIFactory4, (void **)&fac))) {
        for (UINT i = 0; !ad && IDXGIFactory4_EnumAdapters1(fac, i, &ad) != DXGI_ERROR_NOT_FOUND; i++) {
            DXGI_ADAPTER_DESC1 d1;
            IDXGIAdapter1_GetDesc1(ad, &d1);
            if (d1.VendorId != vendor) {
                IDXGIAdapter1_Release(ad);
                ad = NULL;
            }
        }
        IDXGIFactory4_Release(fac);
    }
    HRESULT hr = D3D12CreateDevice((IUnknown *)ad, D3D_FEATURE_LEVEL_12_0,
                                   &IID_ID3D12Device, (void **)&e->dev);
    if (ad) {
        IDXGIAdapter1_Release(ad);
    }
    if (FAILED(hr) || !e->dev) {
        fprintf(stderr, "vtpv-dx-enc: no D3D12 device (gpu=%s hr=%lx)\n",
                gpu ? gpu : "int", (unsigned long)hr);
        vtpv_dxe_close(e);
        return NULL;
    }
    if (FAILED(ID3D12Device_QueryInterface(e->dev, &IID_ID3D12VideoDevice3,
                                           (void **)&e->vd))) {
        fprintf(stderr, "vtpv-dx-enc: ID3D12VideoDevice3 missing (OS < 20H1?)\n");
        vtpv_dxe_close(e);
        return NULL;
    }

    /* resource requirements: bitstream/metadata sizes and alignments */
    /* Opus 10-30 (dxeprobe crash in d3d12core): the requirements query
     * dereferences Profile - set it before the call, not only for the
     * encoder create further down. */
    static D3D12_VIDEO_ENCODER_PROFILE_H264 rr_prof = D3D12_VIDEO_ENCODER_PROFILE_H264_HIGH;
    D3D12_FEATURE_DATA_VIDEO_ENCODER_RESOURCE_REQUIREMENTS rr = {
        .Codec = D3D12_VIDEO_ENCODER_CODEC_H264,
        .Profile = { .DataSize = sizeof(rr_prof), .pH264Profile = &rr_prof },
        .InputFormat = DXGI_FORMAT_NV12,
        .PictureTargetResolution = { w, h },
        .IsSupported = FALSE,
    };
    ID3D12VideoDevice_CheckFeatureSupport((ID3D12VideoDevice *)e->vd,
                                          D3D12_FEATURE_VIDEO_ENCODER_RESOURCE_REQUIREMENTS,
                                          &rr, sizeof(rr));
    UINT bit_align = rr.CompressedBitstreamBufferAccessAlignment
                     ? rr.CompressedBitstreamBufferAccessAlignment : 256u;
    UINT meta_align = rr.EncoderMetadataBufferAccessAlignment
                      ? rr.EncoderMetadataBufferAccessAlignment : 256u;
    e->meta_cap = rr.MaxEncoderOutputMetadataBufferSize
                  ? dxe_align(rr.MaxEncoderOutputMetadataBufferSize, meta_align)
                  : dxe_align(1u << 20, meta_align);

    /* task80: the SUPPORT query - the debug-layer substitute on this host
     * (no d3d12SDKLayers.dll).  ValidationFlags name the exact broken field;
     * the query mirrors what encode() submits by construction (shared
     * session structs). */
    e->gop = (D3D12_VIDEO_ENCODER_SEQUENCE_GOP_STRUCTURE_H264){
        .GOPLength = 1, .PPicturePeriod = 0,
        .pic_order_cnt_type = 0,
        /* task81: minus4 = 4 (256 frame_num/POC-lsb values) - with 0 the
         * raw FrameDecodingOrderNumber outgrew the 16-value field at
         * frame 16 and the recording turned invalid */
        .log2_max_frame_num_minus4 = 4,
        .log2_max_pic_order_cnt_lsb_minus4 = 4,
    };
    e->cc = (D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264){
        .ConfigurationFlags = D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_FLAG_ENABLE_CABAC_ENCODING,
        .DirectModeConfig = D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_DIRECT_MODES_DISABLED,
        .DisableDeblockingFilterConfig =
            D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODE_FLAG_0_ALL_LUMA_CHROMA_SLICE_BLOCK_EDGES_ALWAYS_FILTERED,
    };
    e->cbr = (D3D12_VIDEO_ENCODER_RATE_CONTROL_CBR){
        .InitialQP = 26, .MinQP = 1, .MaxQP = 51,
        .MaxFrameBitSize = 400000,      /* ~3x the CBR per-frame budget */
        .TargetBitRate = 4000000,
        .VBVCapacity = 1000000,
        .InitialVBVFullness = 750000,
    };
    e->rc = (D3D12_VIDEO_ENCODER_RATE_CONTROL){
        .Mode = D3D12_VIDEO_ENCODER_RATE_CONTROL_MODE_CBR,
        .Flags = D3D12_VIDEO_ENCODER_RATE_CONTROL_FLAG_NONE,
        .ConfigParams = { .DataSize = sizeof(e->cbr), .pConfiguration_CBR = &e->cbr },
        .TargetFrameRate = { 30, 1 },
    };
    {
        D3D12_VIDEO_ENCODER_PICTURE_RESOLUTION_DESC sup_res = { w, h };
        D3D12_VIDEO_ENCODER_PROFILE_H264 sup_prof = D3D12_VIDEO_ENCODER_PROFILE_H264_HIGH;
        D3D12_VIDEO_ENCODER_LEVELS_H264 sup_lev = D3D12_VIDEO_ENCODER_LEVELS_H264_52;
        D3D12_FEATURE_DATA_VIDEO_ENCODER_RESOLUTION_SUPPORT_LIMITS rsl = { 0 };
        D3D12_FEATURE_DATA_VIDEO_ENCODER_SUPPORT sup = {
            .Codec = D3D12_VIDEO_ENCODER_CODEC_H264,
            .InputFormat = DXGI_FORMAT_NV12,
            .CodecConfiguration = { .DataSize = sizeof(e->cc), .pH264Config = &e->cc },
            .CodecGopSequence = { .DataSize = sizeof(e->gop), .pH264GroupOfPictures = &e->gop },
            .RateControl = e->rc,
            .IntraRefresh = D3D12_VIDEO_ENCODER_INTRA_REFRESH_MODE_NONE,  /* mingw: an enum here */
            .SubregionFrameEncoding = D3D12_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_MODE_FULL_FRAME,
            .ResolutionsListCount = 1,
            .pResolutionList = &sup_res,
            .MaxReferenceFramesInDPB = 1,
            .SuggestedProfile = { .DataSize = sizeof(sup_prof), .pH264Profile = &sup_prof },
            .SuggestedLevel = { .DataSize = sizeof(sup_lev), .pH264LevelSetting = &sup_lev },
            .pResolutionDependentSupport = &rsl,
        };
        HRESULT sh = ID3D12VideoDevice_CheckFeatureSupport(
            (ID3D12VideoDevice *)e->vd, D3D12_FEATURE_VIDEO_ENCODER_SUPPORT,
            &sup, sizeof(sup));
        fprintf(stderr,
                "vtpv-dx-enc: SUPPORT hr=%lx supported=%u supportFlags=%x "
                "validation=%x suggProf=%d suggLevel=%d\n",
                (unsigned long)sh, (unsigned)(sup.SupportFlags & 1),
                (unsigned)sup.SupportFlags, (unsigned)sup.ValidationFlags,
                (int)sup_prof, (int)sup_lev);
        if (sup.ValidationFlags & 0x1)    fprintf(stderr, "  validation: CODEC_NOT_SUPPORTED\n");
        if (sup.ValidationFlags & 0x8)    fprintf(stderr, "  validation: INPUT_FORMAT_NOT_SUPPORTED\n");
        if (sup.ValidationFlags & 0x10)   fprintf(stderr, "  validation: CODEC_CONFIGURATION_NOT_SUPPORTED\n");
        if (sup.ValidationFlags & 0x20)   fprintf(stderr, "  validation: RATE_CONTROL_MODE_NOT_SUPPORTED\n");
        if (sup.ValidationFlags & 0x40)   fprintf(stderr, "  validation: RATE_CONTROL_CONFIGURATION_NOT_SUPPORTED\n");
        if (sup.ValidationFlags & 0x80)   fprintf(stderr, "  validation: INTRA_REFRESH_NOT_SUPPORTED\n");
        if (sup.ValidationFlags & 0x100)  fprintf(stderr, "  validation: SUBREGION_LAYOUT_NOT_SUPPORTED\n");
        if (sup.ValidationFlags & 0x200)  fprintf(stderr, "  validation: RESOLUTION_NOT_IN_LIST\n");
        if (sup.ValidationFlags & 0x800) {
            fprintf(stderr, "  validation: GOP_STRUCTURE - GOPLength 1 -> 32 (IDR stays per-frame)\n");
            e->gop.GOPLength = 32;
        }
        if (FAILED(sh) || (!(sup.SupportFlags & 1) && sup.ValidationFlags & 0x1)) {
            vtpv_dxe_close(e);
            return NULL;
        }
    }

    /* queues + lists: VIDEO_ENCODE for the encoder, COPY for the NV12 upload */
    D3D12_COMMAND_QUEUE_DESC qd = { .Type = D3D12_COMMAND_LIST_TYPE_VIDEO_ENCODE };
    D3D12_COMMAND_QUEUE_DESC cd = { .Type = D3D12_COMMAND_LIST_TYPE_COPY };
    if (FAILED(ID3D12Device_CreateCommandQueue(e->dev, &qd, &IID_ID3D12CommandQueue,
                                               (void **)&e->q)) ||
        FAILED(ID3D12Device_CreateCommandAllocator(
                   e->dev, D3D12_COMMAND_LIST_TYPE_VIDEO_ENCODE, &IID_ID3D12CommandAllocator,
                   (void **)&e->alloc)) ||
        FAILED(ID3D12Device_CreateCommandList(
                   e->dev, 0, D3D12_COMMAND_LIST_TYPE_VIDEO_ENCODE, e->alloc, NULL,
                   &IID_ID3D12VideoEncodeCommandList2, (void **)&e->cl)) ||
        FAILED(ID3D12Device_CreateCommandQueue(e->dev, &cd, &IID_ID3D12CommandQueue,
                                               (void **)&e->cq)) ||
        FAILED(ID3D12Device_CreateCommandAllocator(
                   e->dev, D3D12_COMMAND_LIST_TYPE_COPY, &IID_ID3D12CommandAllocator,
                   (void **)&e->calloc_)) ||
        FAILED(ID3D12Device_CreateCommandList(
                   e->dev, 0, D3D12_COMMAND_LIST_TYPE_COPY, e->calloc_, NULL,
                   &IID_ID3D12GraphicsCommandList, (void **)&e->ccl)) ||
        FAILED(ID3D12VideoEncodeCommandList2_Close(e->cl)) ||
        FAILED(ID3D12GraphicsCommandList_Close(e->ccl)) ||
        FAILED(ID3D12Device_CreateFence(e->dev, 0, D3D12_FENCE_FLAG_NONE,
                                        &IID_ID3D12Fence, (void **)&e->fence))) {
        fprintf(stderr, "vtpv-dx-enc: queue/list/fence setup failed\n");
        vtpv_dxe_close(e);
        return NULL;
    }
    e->ev = CreateEventA(NULL, FALSE, FALSE, NULL);

    /* encoder + heap: H.264 HIGH, level 5.2 */
    static D3D12_VIDEO_ENCODER_PROFILE_H264 prof = D3D12_VIDEO_ENCODER_PROFILE_H264_HIGH;
    static D3D12_VIDEO_ENCODER_LEVELS_H264 lev = D3D12_VIDEO_ENCODER_LEVELS_H264_52;
    e->prof.DataSize = sizeof(prof);
    e->prof.pH264Profile = &prof;
    e->level.DataSize = sizeof(lev);
    e->level.pH264LevelSetting = &lev;

    D3D12_VIDEO_ENCODER_DESC ed = {
        .EncodeCodec = D3D12_VIDEO_ENCODER_CODEC_H264,
        .EncodeProfile = e->prof,
        .InputFormat = DXGI_FORMAT_NV12,
        .CodecConfiguration = { .DataSize = sizeof(e->cc), .pH264Config = &e->cc },
        .MaxMotionEstimationPrecision = D3D12_VIDEO_ENCODER_MOTION_ESTIMATION_PRECISION_MODE_MAXIMUM,
    };
    D3D12_VIDEO_ENCODER_PICTURE_RESOLUTION_DESC res = { w, h };
    D3D12_VIDEO_ENCODER_HEAP_DESC hd = {
        .EncodeCodec = D3D12_VIDEO_ENCODER_CODEC_H264,
        .EncodeProfile = e->prof,
        .EncodeLevel = e->level,
        .ResolutionsListCount = 1,
        .pResolutionList = &res,
    };
    if (FAILED(ID3D12VideoDevice3_CreateVideoEncoder(
                   e->vd, &ed, &IID_ID3D12VideoEncoder, (void **)&e->enc)) ||
        FAILED(ID3D12VideoDevice3_CreateVideoEncoderHeap(
                   e->vd, &hd, &IID_ID3D12VideoEncoderHeap, (void **)&e->heap))) {
        fprintf(stderr, "vtpv-dx-enc: encoder/heap create failed (no H.264 encode on this adapter?)\n");
        vtpv_dxe_close(e);
        return NULL;
    }

    UINT64 up_bytes = 0;
    /* resources */
    D3D12_HEAP_PROPERTIES up = { .Type = D3D12_HEAP_TYPE_UPLOAD };
    /* Opus 10-30 (dxeprobe rc -5 even with EncodeFrame/Resolve skipped): a
     * READBACK-heap resource is pinned to COPY_DEST - it can neither be
     * transitioned nor written by the encoder, so recording barriers on it made
     * the list invalid.  A CUSTOM write-back L0 heap is CPU-readable and
     * takes VIDEO_ENCODE_WRITE. */
    D3D12_HEAP_PROPERTIES rb = { .Type = D3D12_HEAP_TYPE_CUSTOM,
                                 .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                                 .MemoryPoolPreference = D3D12_MEMORY_POOL_L0 };
    D3D12_HEAP_PROPERTIES df = { .Type = D3D12_HEAP_TYPE_DEFAULT };
    D3D12_RESOURCE_DESC texd = {
        .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        .Width = w, .Height = h,
        .DepthOrArraySize = 1, .MipLevels = 1,
        .Format = DXGI_FORMAT_NV12,
        .SampleDesc = { 1, 0 },
        /* task79: the input texture crosses queues via the COMMON-state
         * handoff (COPY -> ENCODE); textures need this flag for that */
        .Flags = getenv("DXE_NO_SIMUL") ? D3D12_RESOURCE_FLAG_NONE : D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS,  /* TEMP bisect */
    };
    /* task78: NV12 is planar - the upload layout IS the copyable footprints
     * of the two subresources, and the copy records one
     * CopyTextureRegion per plane (fp[0] = Y R8_UNORM w x h,
     * fp[1] = UV R8G8_UNORM w x h/2) */
    ID3D12Device_GetCopyableFootprints(e->dev, &texd, 0, 2, 0, e->fp,
                                       e->nrows, e->rowsz, &up_bytes);
    e->bits_cap = dxe_align((UINT64)w * h, bit_align);   /* H.264 worst-case room */
    D3D12_RESOURCE_DESC bufd = {
        .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
        .Width = up_bytes,
        .Height = 1, .DepthOrArraySize = 1, .MipLevels = 1,
        .SampleDesc = { 1, 0 }, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
    };
    D3D12_RESOURCE_DESC bitd = bufd;
    bitd.Width = e->bits_cap;
    D3D12_RESOURCE_DESC metad = bufd;
    metad.Width = e->meta_cap;
    /* task82: the recon surface - RECONSTRUCTED_FRAMES_REQUIRE_TEXTURE_
     * ARRAYS (supportFlags 0x20) wants an ARRAY; a reference surface is
     * VIDEO_ENCODE_REFERENCE_ONLY paired with DENY_SHADER_RESOURCE */
    D3D12_RESOURCE_DESC recond = texd;
    recond.DepthOrArraySize = 2;
    recond.Flags = D3D12_RESOURCE_FLAG_VIDEO_ENCODE_REFERENCE_ONLY |
                   D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
    if (FAILED(ID3D12Device_CreateCommittedResource(
                   e->dev, &up, D3D12_HEAP_FLAG_NONE, &bufd,
                   D3D12_RESOURCE_STATE_GENERIC_READ, NULL,
                   &IID_ID3D12Resource, (void **)&e->upload)) ||
        FAILED(ID3D12Device_CreateCommittedResource(
                   e->dev, &df, D3D12_HEAP_FLAG_NONE, &texd,
                   D3D12_RESOURCE_STATE_COMMON, NULL,
                   &IID_ID3D12Resource, (void **)&e->tex)) ||
        FAILED(ID3D12Device_CreateCommittedResource(
                   e->dev, &df, D3D12_HEAP_FLAG_NONE, &recond,
                   D3D12_RESOURCE_STATE_COMMON, NULL,
                   &IID_ID3D12Resource, (void **)&e->recon)) ||
        FAILED(ID3D12Device_CreateCommittedResource(
                   e->dev, &rb, D3D12_HEAP_FLAG_NONE, &bitd,
                   D3D12_RESOURCE_STATE_COMMON, NULL,
                   &IID_ID3D12Resource, (void **)&e->bits)) ||
        FAILED(ID3D12Device_CreateCommittedResource(
                   e->dev, &df, D3D12_HEAP_FLAG_NONE, &metad,
                   D3D12_RESOURCE_STATE_COMMON, NULL,
                   &IID_ID3D12Resource, (void **)&e->hwmeta)) ||
        FAILED(ID3D12Device_CreateCommittedResource(
                   e->dev, &rb, D3D12_HEAP_FLAG_NONE, &metad,
                   D3D12_RESOURCE_STATE_COMMON, NULL,
                   &IID_ID3D12Resource, (void **)&e->rmeta))) {
        fprintf(stderr, "vtpv-dx-enc: resource alloc failed\n");
        vtpv_dxe_close(e);
        return NULL;
    }
    e->frame = 0;
    fprintf(stderr, "vtpv-dx-enc: open %ux%u bits_cap=%llu meta_cap=%llu\n",
            w, h, (unsigned long long)e->bits_cap, (unsigned long long)e->meta_cap);
    return e;
}

/* task81: the H.264 parameter sets D3D12VE never writes - the app emits
 * them (ffmpeg's d3d12va_encode / Mesa precedent).  Fields mirror the
 * session config exactly: profile High + CABAC, level 52, 4:2:0,
 * log2_max_frame_num/pic_order_cnt_lsb minus4 = 4, max_num_ref_frames 1,
 * MB dims + luma crop for non-mod16 heights, deblocking-control present.
 * Annex-B out: [start code][SPS][start code][PPS]; the guest delivery
 * (3b glue) converts to AVCC lengths + avcC. */
struct dxe_bw {
    uint8_t *buf;
    uint32_t cap, len;
    uint32_t bits;
    int nbits;
};

static void dxe_bw_bit(struct dxe_bw *b, uint32_t v)
{
    b->bits = (b->bits << 1) | (v & 1u);
    if (++b->nbits == 8) {
        if (b->len < b->cap) {
            b->buf[b->len++] = (uint8_t)b->bits;
        }
        b->bits = 0;
        b->nbits = 0;
    }
}

static void dxe_bw_bits(struct dxe_bw *b, uint32_t v, int n)
{
    for (int i = n - 1; i >= 0; i--) {
        dxe_bw_bit(b, (v >> i) & 1u);
    }
}

static void dxe_bw_ue(struct dxe_bw *b, uint32_t v)
{
    uint32_t vp = v + 1;
    int n = 0;
    while (n < 31 && (vp >> n) > 1) {
        n++;
    }
    dxe_bw_bits(b, 0, n);
    dxe_bw_bits(b, vp, n + 1);
}

static void dxe_bw_se(struct dxe_bw *b, int32_t v)
{
    dxe_bw_ue(b, (uint32_t)(v <= 0 ? -2 * (int64_t)v : 2 * (int64_t)v - 1));
}

static void dxe_bw_rbsp_trailing(struct dxe_bw *b)
{
    dxe_bw_bit(b, 1);
    while (b->nbits) {
        dxe_bw_bit(b, 0);
    }
}

/* task82: emulation prevention - insert 0x03 after any 00 00 before a
 * byte <= 3 so start-code-like sequences never appear inside a NAL */
static uint32_t dxe_epb(const uint8_t *src, uint32_t len, uint8_t *dst, uint32_t cap)
{
    uint32_t n = 0;
    int zeros = 0;
    for (uint32_t i = 0; i < len; i++) {
        if (zeros >= 2 && src[i] <= 3 && n + 1 < cap) {
            dst[n++] = 3;
            zeros = 0;
        }
        if (n < cap) {
            dst[n++] = src[i];
            zeros = src[i] == 0 ? zeros + 1 : 0;
        }
    }
    return n;
}

static uint32_t dxe_write_sps_pps(struct VtpvDxE *e, uint8_t *out, uint32_t cap)
{
    uint8_t rbsp[64];
    uint32_t total = 0;
    struct dxe_bw b = { rbsp, sizeof(rbsp), 0, 0 };
    dxe_bw_bits(&b, 0x67, 8);         /* NAL: ref_idc 3, type 7 (SPS) */
    /* task82: H.264 7.3.2.1 - profile/constraints/level come BEFORE
     * seq_parameter_set_id; writing sps_id first shifted every following
     * field (the '67 b2 00 1a' NAL = that one ue bit + profile bits) */
    dxe_bw_bits(&b, 100, 8);          /* profile_idc: High */
    dxe_bw_bits(&b, 0, 8);            /* constraint flags + reserved */
    dxe_bw_bits(&b, 52, 8);           /* level_idc (heap level 5.2) */
    dxe_bw_ue(&b, 0);                 /* seq_parameter_set_id */
    dxe_bw_ue(&b, 1);                 /* chroma_format_idc: 4:2:0 */
    dxe_bw_ue(&b, 0);                 /* bit_depth_luma_minus8: ue(v), not u(8) (Opus 10-30) */
    dxe_bw_ue(&b, 0);                 /* bit_depth_chroma_minus8: ue(v) */
    dxe_bw_bit(&b, 0);                /* qpprime_y_zero_transform_bypass */
    dxe_bw_bit(&b, 0);                /* seq_scaling_matrix_present */
    dxe_bw_ue(&b, e->gop.log2_max_frame_num_minus4);
    dxe_bw_ue(&b, 0);                 /* pic_order_cnt_type */
    dxe_bw_ue(&b, e->gop.log2_max_pic_order_cnt_lsb_minus4);
    dxe_bw_ue(&b, 1);                 /* max_num_ref_frames */
    dxe_bw_bit(&b, 0);                /* gaps_in_frame_num_value_allowed */
    dxe_bw_ue(&b, (e->w + 15) / 16 - 1);
    dxe_bw_ue(&b, (e->h + 15) / 16 - 1);
    dxe_bw_bit(&b, 1);                /* frame_mbs_only_flag */
    dxe_bw_bit(&b, 1);                /* direct_8x8_inference_flag */
    if ((e->w & 15) || (e->h & 15)) { /* frame_cropping in luma samples */
        dxe_bw_bit(&b, 1);
        dxe_bw_ue(&b, 0);
        dxe_bw_ue(&b, ((e->w + 15) / 16) * 16 - e->w);   /* right */
        dxe_bw_ue(&b, 0);
        dxe_bw_ue(&b, ((e->h + 15) / 16) * 16 - e->h);   /* bottom */
    } else {
        dxe_bw_bit(&b, 0);
    }
    dxe_bw_bit(&b, 0);                /* vui_parameters_present */
    dxe_bw_rbsp_trailing(&b);

    memcpy(out, "\x00\x00\x00\x01", 4);
    total = 4 + dxe_epb(rbsp, b.len, out + 4, cap - 4);

    b = (struct dxe_bw){ rbsp, sizeof(rbsp), 0, 0 };
    dxe_bw_bits(&b, 0x68, 8);         /* NAL: ref_idc 3, type 8 (PPS) */
    dxe_bw_ue(&b, 0);                 /* pic_parameter_set_id */
    dxe_bw_ue(&b, 0);                 /* seq_parameter_set_id */
    dxe_bw_bit(&b, 1);                /* entropy_coding_mode: CABAC */
    dxe_bw_bit(&b, 0);                /* bottom_field_pic_order_in_frame_present */
    dxe_bw_ue(&b, 0);                 /* num_slice_groups_minus1 */
    dxe_bw_ue(&b, 0);                 /* num_ref_idx_l0_default_active_minus1 */
    dxe_bw_ue(&b, 0);                 /* num_ref_idx_l1_default_active_minus1 */
    dxe_bw_bit(&b, 0);                /* weighted_pred_flag */
    dxe_bw_bits(&b, 0, 2);            /* weighted_bipred_idc */
    dxe_bw_se(&b, 0);                 /* pic_init_qp_minus26 */
    dxe_bw_se(&b, 0);                 /* pic_init_qs_minus26 */
    dxe_bw_se(&b, 0);                 /* chroma_qp_index_offset */
    dxe_bw_bit(&b, 1);                /* deblocking_filter_control_present */
    dxe_bw_bit(&b, 0);                /* constrained_intra_pred */
    dxe_bw_bit(&b, 0);                /* redundant_pic_cnt_present */
    dxe_bw_rbsp_trailing(&b);

    memcpy(out + total, "\x00\x00\x00\x01", 4);
    total += 4 + dxe_epb(rbsp, b.len, out + total + 4, cap - total - 4);
    return total;
}

int vtpv_dxe_encode(VtpvDxE *eh, const uint8_t *y, uint32_t y_stride,
                    const uint8_t *uv, uint32_t uv_stride, int force_key,
                    uint8_t *out, uint32_t out_cap, uint32_t *out_len)
{
    struct VtpvDxE *e = (struct VtpvDxE *)eh;
    if (!e || !e->enc || !y || !uv) {
        return -1;
    }
    *out_len = 0;
    (void)force_key;                 /* GOP 1: every frame is an IDR */

    /* 1. stage NV12 into the upload buffer */
    uint8_t *dst = NULL;
    D3D12_RANGE rng = { 0, 0 };
    if (FAILED(ID3D12Resource_Map(e->upload, 0, &rng, (void **)&dst))) {
        return -2;
    }
    for (UINT r = 0; r < e->nrows[0]; r++) {
        memcpy(dst + e->fp[0].Offset + (size_t)r * e->fp[0].Footprint.RowPitch,
               y + (size_t)r * y_stride, e->rowsz[0]);
    }
    for (UINT r = 0; r < e->nrows[1]; r++) {
        memcpy(dst + e->fp[1].Offset + (size_t)r * e->fp[1].Footprint.RowPitch,
               uv + (size_t)r * uv_stride, e->rowsz[1]);
    }
    ID3D12Resource_Unmap(e->upload, 0, NULL);

    /* 2. upload -> input texture on the COPY queue, hand off via COMMON;
     * one CopyTextureRegion per plane, footprints/offsets from open() */
    D3D12_TEXTURE_COPY_LOCATION srcp[2] = {
        { .pResource = e->upload,
          .Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
          .PlacedFootprint = e->fp[0] },
        { .pResource = e->upload,
          .Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
          .PlacedFootprint = e->fp[1] },
    };
    D3D12_TEXTURE_COPY_LOCATION dstloc = { .pResource = e->tex,
                                           .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                                           .SubresourceIndex = 0 };
    D3D12_RESOURCE_BARRIER cbar[2] = {
        { .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
          .Transition = { .pResource = e->tex,
                          .StateBefore = D3D12_RESOURCE_STATE_COMMON,
                          .StateAfter = D3D12_RESOURCE_STATE_COPY_DEST } },
    };
    ID3D12GraphicsCommandList_Reset(e->ccl, e->calloc_, NULL);
    ID3D12GraphicsCommandList_ResourceBarrier(e->ccl, 1, cbar);
    ID3D12GraphicsCommandList_CopyTextureRegion(e->ccl, &dstloc, 0, 0, 0, &srcp[0], NULL);
    dstloc.SubresourceIndex = 1;
    ID3D12GraphicsCommandList_CopyTextureRegion(e->ccl, &dstloc, 0, 0, 0, &srcp[1], NULL);
    cbar[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    cbar[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    ID3D12GraphicsCommandList_ResourceBarrier(e->ccl, 1, cbar);
    if (FAILED(ID3D12GraphicsCommandList_Close(e->ccl))) {
        return -3;
    }
    ID3D12CommandList *cll = (ID3D12CommandList *)e->ccl;
    ID3D12CommandQueue_ExecuteCommandLists(e->cq, 1, &cll);
    e->fence_val++;
    if (FAILED(ID3D12CommandQueue_Signal(e->cq, e->fence, e->fence_val)) ||
        FAILED(ID3D12Fence_SetEventOnCompletion(e->fence, e->fence_val, e->ev)) ||
        WaitForSingleObject(e->ev, 2000) != WAIT_OBJECT_0) {
        return -4;
    }
    ID3D12CommandAllocator_Reset(e->calloc_);

    /* 3. per-frame control: the session's config (the SUPPORT query ran
     * against these very instances); GOP-1/32 per the auto-fix, always IDR
     * via pic.FrameType so every sample stays self-contained */
    /* task82: the full reference set - USED_AS_REFERENCE_PICTURE with one
     * DPB slot (recon array slice 0); if this still fails Close, the
     * DXE_NO_RECON NULL-recon path is the 3b ship (intra-only works) */
    D3D12_VIDEO_ENCODER_REFERENCE_PICTURE_DESCRIPTOR_H264 refd = {
        .ReconstructedPictureResourceIndex = 0,
        .IsLongTermReference = FALSE,
        .PictureOrderCountNumber = (e->frame * 2u) & 0xffu,
        .FrameDecodingOrderNumber = e->frame & 0xffu,
        .TemporalLayerIndex = 0,
    };
    D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_H264 pic = {
        .FrameType = D3D12_VIDEO_ENCODER_FRAME_TYPE_H264_IDR_FRAME,
        /* wrap inside log2_max_frame_num/pic_order_cnt_lsb (8 bits with
         * minus4 = 4) - the raw counters died at frame 16 */
        .PictureOrderCountNumber = (e->frame * 2u) & 0xffu,
        .FrameDecodingOrderNumber = e->frame & 0xffu,
        .List0ReferenceFramesCount = 0,
        .List1ReferenceFramesCount = 0,
        .ReferenceFramesReconPictureDescriptorsCount = getenv("DXE_NO_RECON") ? 0 : 1,
        .pReferenceFramesReconPictureDescriptors = getenv("DXE_NO_RECON") ? NULL : &refd,
    };
    ID3D12Resource *recon_arr[1] = { e->recon };
    D3D12_VIDEO_ENCODER_PICTURE_CONTROL_DESC pd = {
        .Flags = getenv("DXE_NO_RECON") ? D3D12_VIDEO_ENCODER_PICTURE_CONTROL_FLAG_NONE : D3D12_VIDEO_ENCODER_PICTURE_CONTROL_FLAG_USED_AS_REFERENCE_PICTURE,
        .PictureControlCodecData = { .DataSize = sizeof(pic), .pH264PicData = &pic },
        .ReferenceFrames = { getenv("DXE_NO_RECON") ? 0 : 1, getenv("DXE_NO_RECON") ? NULL : recon_arr, NULL },
    };
    D3D12_VIDEO_ENCODER_SEQUENCE_CONTROL_DESC sq = {
        .RateControl = e->rc,
        .PictureTargetResolution = { e->w, e->h },
        .SelectedLayoutMode = D3D12_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_MODE_FULL_FRAME,
        .CodecGopSequence = { .DataSize = sizeof(e->gop), .pH264GroupOfPictures = &e->gop },
    };
    D3D12_VIDEO_ENCODER_ENCODEFRAME_INPUT_ARGUMENTS ia = {
        .SequenceControlDesc = sq,
        .PictureControlDesc = pd,
        .pInputFrame = e->tex,
        .InputFrameSubresource = 0,
    };
    D3D12_VIDEO_ENCODER_ENCODEFRAME_OUTPUT_ARGUMENTS oa = {
        .Bitstream = { .pBuffer = e->bits, .FrameStartOffset = 0 },
        .ReconstructedPicture = { .pReconstructedPicture = getenv("DXE_NO_RECON") ? NULL : e->recon },  /* TEMP bisect */
        .EncoderOutputMetadata = { .pBuffer = e->hwmeta, .Offset = 0 },
    };

    /* 4. encode + resolve on the VIDEO_ENCODE queue */
    D3D12_RESOURCE_BARRIER bar[4] = { 0 };   /* Opus 10-30: + the bitstream buffer */
    int nb = 0;
    bar[nb++] = (D3D12_RESOURCE_BARRIER){
        .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        .Transition = { .pResource = e->tex,
                        .StateBefore = D3D12_RESOURCE_STATE_COMMON,
                        .StateAfter = D3D12_RESOURCE_STATE_VIDEO_ENCODE_READ },
    };
    bar[nb++] = (D3D12_RESOURCE_BARRIER){
        .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        .Transition = { .pResource = e->recon,
                        .StateBefore = D3D12_RESOURCE_STATE_COMMON,
                        .StateAfter = D3D12_RESOURCE_STATE_VIDEO_ENCODE_WRITE },
    };
    bar[nb++] = (D3D12_RESOURCE_BARRIER){
        .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        .Transition = { .pResource = e->hwmeta,
                        .StateBefore = D3D12_RESOURCE_STATE_COMMON,
                        .StateAfter = D3D12_RESOURCE_STATE_VIDEO_ENCODE_WRITE },
    };
    bar[nb++] = (D3D12_RESOURCE_BARRIER){
        .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        .Transition = { .pResource = e->bits,
                        .StateBefore = D3D12_RESOURCE_STATE_COMMON,
                        .StateAfter = D3D12_RESOURCE_STATE_VIDEO_ENCODE_WRITE },
    };
    ID3D12VideoEncodeCommandList2_Reset(e->cl, e->alloc);
    ID3D12VideoEncodeCommandList2_ResourceBarrier(e->cl, nb, bar);
    if (!getenv("DXE_SKIP_ENCODE")) {   /* TEMP bisect (Opus 10-30) */
        ID3D12VideoEncodeCommandList2_EncodeFrame(e->cl, e->enc, e->heap, &ia, &oa);
    }
    /* Opus 10-30 (dxeprobe rc -5 = Close failed): ResolveEncoderOutputMetadata
     * reads the hardware metadata in VIDEO_ENCODE_READ and writes the resolved
     * buffer in VIDEO_ENCODE_WRITE; everything else goes back to COMMON. */
#define DXE_T(res, a_, b_) { .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,         .Transition = { .pResource = (res), .StateBefore = (a_), .StateAfter = (b_) } }
    D3D12_RESOURCE_BARRIER rb_[5] = {
        DXE_T(e->hwmeta, D3D12_RESOURCE_STATE_VIDEO_ENCODE_WRITE, D3D12_RESOURCE_STATE_VIDEO_ENCODE_READ),
        DXE_T(e->rmeta, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_VIDEO_ENCODE_WRITE),
        DXE_T(e->recon, D3D12_RESOURCE_STATE_VIDEO_ENCODE_WRITE, D3D12_RESOURCE_STATE_COMMON),
        DXE_T(e->tex, D3D12_RESOURCE_STATE_VIDEO_ENCODE_READ, D3D12_RESOURCE_STATE_COMMON),
        DXE_T(e->bits, D3D12_RESOURCE_STATE_VIDEO_ENCODE_WRITE, D3D12_RESOURCE_STATE_COMMON),
    };
    ID3D12VideoEncodeCommandList2_ResourceBarrier(e->cl, 5, rb_);

    D3D12_VIDEO_ENCODER_RESOLVE_METADATA_INPUT_ARGUMENTS ri = {
        .EncoderCodec = D3D12_VIDEO_ENCODER_CODEC_H264,
        .EncoderProfile = e->prof,
        .EncoderInputFormat = DXGI_FORMAT_NV12,
        .EncodedPictureEffectiveResolution = { e->w, e->h },
        .HWLayoutMetadata = { .pBuffer = e->hwmeta, .Offset = 0 },
    };
    D3D12_VIDEO_ENCODER_RESOLVE_METADATA_OUTPUT_ARGUMENTS ro = {
        .ResolvedLayoutMetadata = { .pBuffer = e->rmeta, .Offset = 0 },
    };
    if (!getenv("DXE_SKIP_RESOLVE")) {   /* TEMP bisect (Opus 10-30) */
        ID3D12VideoEncodeCommandList2_ResolveEncoderOutputMetadata(e->cl, &ri, &ro);
    }
    D3D12_RESOURCE_BARRIER rb2_[2] = {
        DXE_T(e->hwmeta, D3D12_RESOURCE_STATE_VIDEO_ENCODE_READ, D3D12_RESOURCE_STATE_COMMON),
        DXE_T(e->rmeta, D3D12_RESOURCE_STATE_VIDEO_ENCODE_WRITE, D3D12_RESOURCE_STATE_COMMON),
    };
    ID3D12VideoEncodeCommandList2_ResourceBarrier(e->cl, 2, rb2_);
#undef DXE_T

    if (FAILED(ID3D12VideoEncodeCommandList2_Close(e->cl))) {
        return -5;
    }
    cll = (ID3D12CommandList *)e->cl;
    ID3D12CommandQueue_ExecuteCommandLists(e->q, 1, &cll);
    e->fence_val++;
    if (FAILED(ID3D12CommandQueue_Signal(e->q, e->fence, e->fence_val)) ||
        FAILED(ID3D12Fence_SetEventOnCompletion(e->fence, e->fence_val, e->ev)) ||
        WaitForSingleObject(e->ev, 5000) != WAIT_OBJECT_0) {
        return -6;
    }
    ID3D12CommandAllocator_Reset(e->alloc);

    /* 5. read the resolved metadata + the bitstream */
    D3D12_VIDEO_ENCODER_OUTPUT_METADATA *md = NULL;
    if (FAILED(ID3D12Resource_Map(e->rmeta, 0, &rng, (void **)&md))) {
        return -7;
    }
    UINT64 written = md->EncodedBitstreamWrittenBytesCount;
    UINT64 errf = md->EncodeErrorFlags;
    ID3D12Resource_Unmap(e->rmeta, 0, NULL);
    if (errf != 0) {
        fprintf(stderr, "vtpv-dx-enc: frame %u encode error flags %llx\n",
                e->frame, (unsigned long long)errf);
        return -8;
    }
    if (written > out_cap) {
        written = out_cap;
    }
    uint8_t *srcb = NULL;
    if (FAILED(ID3D12Resource_Map(e->bits, 0, &rng, (void **)&srcb))) {
        return -9;
    }
    /* task81: SPS/PPS ahead of every sample (GOP 1 = all IDR) */
    uint8_t hdr[128];
    uint32_t hlen = dxe_write_sps_pps(e, hdr, sizeof(hdr));
    if ((uint64_t)hlen + written > out_cap) {
        written = out_cap > hlen ? out_cap - hlen : 0;
    }
    memcpy(out, hdr, hlen);
    memcpy(out + hlen, srcb, (size_t)written);
    ID3D12Resource_Unmap(e->bits, 0, NULL);
    *out_len = hlen + (uint32_t)written;
    e->frame++;
    return 0;
}

void vtpv_dxe_close(VtpvDxE *eh)
{
    struct VtpvDxE *e = (struct VtpvDxE *)eh;
    if (!e) {
        return;
    }
    if (e->frame) {
        fprintf(stderr, "vtpv-dx-enc: close after %u frames\n", e->frame);
    }
    if (e->upload) ID3D12Resource_Release(e->upload);
    if (e->tex) ID3D12Resource_Release(e->tex);
    if (e->recon) ID3D12Resource_Release(e->recon);
    if (e->bits) ID3D12Resource_Release(e->bits);
    if (e->hwmeta) ID3D12Resource_Release(e->hwmeta);
    if (e->rmeta) ID3D12Resource_Release(e->rmeta);
    if (e->heap) ID3D12VideoEncoderHeap_Release(e->heap);
    if (e->enc) ID3D12VideoEncoder_Release(e->enc);
    if (e->cl) ID3D12VideoEncodeCommandList2_Release(e->cl);
    if (e->ccl) ID3D12GraphicsCommandList_Release(e->ccl);
    if (e->alloc) ID3D12CommandAllocator_Release(e->alloc);
    if (e->calloc_) ID3D12CommandAllocator_Release(e->calloc_);
    if (e->ev) CloseHandle(e->ev);
    if (e->fence) ID3D12Fence_Release(e->fence);
    if (e->q) ID3D12CommandQueue_Release(e->q);
    if (e->cq) ID3D12CommandQueue_Release(e->cq);
    if (e->vd) ID3D12VideoDevice3_Release(e->vd);
    if (e->dev) ID3D12Device_Release(e->dev);
    free(e);
}
