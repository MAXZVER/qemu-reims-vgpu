/*
 * TEMP (lab): apple-vtpv-dx.c - see apple-vtpv-dx.h.  Host-tested copy of
 * lab/tools/session-2026-10-02/video-re/vtpvdx/vtpv-dx.c (keep them in sync).
 * Recipe (all verified on Intel UHD 770, driver 32.0.101.7088/7092, lab
 * journal 2026-10-22-opus-video-intel.md):
 *
 *  - libavcodec's native decoders + the d3d12va hwaccel on the adapter picked
 *    by PCI vendor (DXGI index), thread_count 1.
 *  - OWN hw frame pool: d3d12va sizes surfaces to the display size, which a
 *    1080p H.264 (coded 1088) overruns -> garbage.  Align like dxva2: 16 for
 *    H.264, 128 for HEVC/VP9/AV1.
 *  - Decode order: H.264/HEVC output is reordered (and delayed) by libavcodec
 *    even with AV_CODEC_FLAG_LOW_DELAY, so get_buffer2 is wrapped: it is called
 *    once per picture in decode order, and when avcodec_send_packet() returns
 *    that picture has been submitted (its fence value is set).  VP9/AV1 output
 *    is not delayed but hidden frames also allocate buffers, so those codecs use
 *    avcodec_receive_frame().
 *  - Readback: a COPY queue on the same device waits on the picture's decode
 *    fence (GPU-side) and CopyTextureRegion()s both planes into a buffer placed
 *    in plain VirtualAlloc memory (ID3D12Device3::OpenExistingHeapFromAddress).
 *    No staging texture, no CPU copy.  4K: ~7 ms of iGPU copy per picture, so
 *    the pipelined mode keeps two such buffers and overlaps the copy of N with
 *    the parse/decode of N+1.
 */
#include "qemu/osdep.h"
#include "hw/misc/apple-vtpv-dx.h"

/* COM C macros and the aggregate-return wrappers (GetDesc) for d3d12.h; the
 * D3D12/DXGI IIDs are instantiated here (selectany) via initguid.h */
#define COBJMACROS
#define WIDL_C_INLINE_WRAPPERS
#include <initguid.h>
#include <dxgi.h>
#include <d3d12.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wundef"
#pragma GCC diagnostic ignored "-Wstrict-prototypes"
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_d3d12va.h>
#include <libavutil/pixdesc.h>
#pragma GCC diagnostic pop

#define MAXPEND 8
/* TEMP lab (async readback, Opus 13:1x): 4 slots so the decode thread keeps
 * 2-3 pictures in flight - a submit costs the CPU-side parse only, the
 * readback wait lands on vtpv_dx_flush (collect), overlapped with the next
 * submits.  Depth 2 (the old value) pipelined exactly one picture. */
#define NSLOT 4

typedef struct {
    ID3D12CommandAllocator *alloc;
    ID3D12GraphicsCommandList *list;
    uint8_t *mem;               /* plain process memory (VirtualAlloc) */
    ID3D12Heap *heap;
    ID3D12Resource *buf;        /* placed buffer covering mem */
    UINT64 fv;                  /* copy fence value of the picture in it */
    int busy;                   /* holds a picture whose callback is pending */
    ID3D12CommandAllocator *valloc;  /* 28-10 (240fps): this slot's P010->NV12 blit */
    ID3D12VideoProcessCommandList *vlist;
    VtpvDxFrame out;
} Slot;

struct VtpvDx {
    /* keep-alive: after ~200 ms without work the iGPU drops into a deep idle
     * state and the next picture pays ~450 ms (host repro 2026-10-22, gaps of
     * 200 ms vs 250 ms); a tiny copy every ka_ms keeps it awake while the
     * session lives.  VTPVDX_KEEPALIVE_MS=0 disables it. */
    HANDLE ka_thread;
    volatile LONG ka_stop;
    int ka_ms;
    ID3D12CommandAllocator *ka_alloc;
    ID3D12GraphicsCommandList *ka_list;
    ID3D12Resource *ka_src, *ka_dst;
    ID3D12Fence *ka_fence;
    HANDLE ka_ev;
    UINT64 ka_fv;
    AVCodecContext *c;
    AVBufferRef *hw;
    enum AVPixelFormat hwfmt;
    int d3d12;
    int cuda;
    int soft;                   /* "sw": libavcodec on the CPU, no device */
    int hook;                   /* decode-order output via get_buffer2 */
    int pipelined;
    VtpvDxOut out;
    void *opaque;
    AVPacket *pkt;
    AVFrame *fr, *sw;
    AVFrame *pend[MAXPEND];     /* pictures allocated during the current packet */
    int npend;
    /* D3D12 readback into process memory */
    ID3D12Device *dev;
    ID3D12CommandQueue *q;
    ID3D12Fence *fence;
    HANDLE ev;
    UINT64 fv;
    Slot slot[NSLOT];
    int next;                   /* slot the next copy goes to */
    SIZE_T memsz;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp[2];
    UINT64 tex_w;
    UINT tex_h;
    DXGI_FORMAT tex_fmt;
    int nplanes;                /* 2: NV12/P010; 1: the packed HEVC range-extension outputs */
    /* 28-10 (240fps): GPU P010->NV12 - a VIDEO_PROCESS queue blits the decode
     * target into an NV12 texture before the readback: the copy moves half
     * the bytes and fire() does no CPU downshift.  vp_fail = CPU fallback */
    ID3D12VideoDevice *vd;
    ID3D12VideoProcessor *vp;
    ID3D12CommandQueue *vpq;
    ID3D12Fence *vpf;
    UINT64 vpfv;
    ID3D12Resource *nv12;       /* conversion target, decoder-sized */
    UINT64 nv12_w;
    UINT nv12_h;
    int vp_fail;
    uint8_t *conv;              /* packed -> luma + interleaved CbCr, for the callback */
    size_t conv_cap;
    /* stats */
    LARGE_INTEGER qf;
    double t_parse, t_gpu;
    int npics;
    /* pool key (vtpv_dx_close) and the time it went there */
    int nv;
    VtpvDxCodec codec;
    uint8_t *cfg;
    uint32_t cfg_len;
    volatile double pooled_at;
};

/*
 * Reuse.  Every HEIF tile comes in its own decoder session, and each open paid
 * the D3D12 device, libavcodec's context, the copy queue and on the first
 * picture the decoder heap: 12-60 ms a tile against 1.6 ms for a picture on an
 * open decoder.  The guest's reply clerk times a slow 'dtil' out (TV, 10-23:
 * "timeout waiting for pending reply ... 'dtil'", -19093); CMPhoto then drops
 * the image and frees state a later tile's callback still uses (TV, Weather
 * crashed in it).  So one device per api and GPU for the process, and closed
 * decoders wait POOL_TTL_MS in a pool for a session with the same stream
 * configuration (the tiles of one image share it).
 */
#define POOL_MAX 8
#define POOL_TTL_MS 5000.0
static SRWLOCK pool_lock = SRWLOCK_INIT;
static VtpvDx *pool[POOL_MAX];
static AVBufferRef *dev_cache[2][2];        /* [cuda][nv] */

static void dx_destroy(VtpvDx *d);

static double now_ms(const VtpvDx *d)
{
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1000.0 / (double)d->qf.QuadPart;
}

static int adapter_for(UINT vendor)
{
    IDXGIFactory1 *fac;
    if (FAILED(CreateDXGIFactory1(&IID_IDXGIFactory1, (void **)&fac))) return -1;
    IDXGIAdapter1 *ad;
    int found = -1;
    for (UINT i = 0; IDXGIFactory1_EnumAdapters1(fac, i, &ad) != DXGI_ERROR_NOT_FOUND; i++) {
        DXGI_ADAPTER_DESC1 desc;
        IDXGIAdapter1_GetDesc1(ad, &desc);
        if (desc.VendorId == vendor && found < 0) found = (int)i;
        IDXGIAdapter1_Release(ad);
    }
    IDXGIFactory1_Release(fac);
    return found;
}

static enum AVPixelFormat get_fmt(AVCodecContext *c, const enum AVPixelFormat *f)
{
    VtpvDx *d = c->opaque;
    for (; *f != AV_PIX_FMT_NONE; f++) {
        if (*f != d->hwfmt) continue;
        AVBufferRef *fc = NULL;
        if (avcodec_get_hw_frames_parameters(c, c->hw_device_ctx, *f, &fc) < 0) return AV_PIX_FMT_NONE;
        AVHWFramesContext *hf = (AVHWFramesContext *)fc->data;
        int a = c->codec_id == AV_CODEC_ID_H264 ? 16 : 128;
        hf->width = (c->coded_width + a - 1) & ~(a - 1);
        hf->height = (c->coded_height + a - 1) & ~(a - 1);
        if (av_hwframe_ctx_init(fc) < 0) { av_buffer_unref(&fc); return AV_PIX_FMT_NONE; }
        av_buffer_unref(&c->hw_frames_ctx);
        c->hw_frames_ctx = fc;
        return *f;
    }
    fprintf(stderr, "vtpv-dx: hardware format not offered (unsupported stream?)\n");
    return AV_PIX_FMT_NONE;
}

static int get_buf(AVCodecContext *c, AVFrame *f, int flags)
{
    VtpvDx *d = c->opaque;
    int r = avcodec_default_get_buffer2(c, f, flags);
    if (r >= 0 && d->npend < MAXPEND) {
        AVFrame *k = av_frame_alloc();
        if (k && av_frame_ref(k, f) >= 0) d->pend[d->npend++] = k;
        else av_frame_free(&k);
    }
    return r;
}

static int queue_init(VtpvDx *d)
{
    AVD3D12VADeviceContext *dctx = ((AVHWDeviceContext *)d->hw->data)->hwctx;
    d->dev = dctx->device;
    D3D12_COMMAND_QUEUE_DESC qd = { .Type = D3D12_COMMAND_LIST_TYPE_COPY };
    if (FAILED(ID3D12Device_CreateCommandQueue(d->dev, &qd, &IID_ID3D12CommandQueue, (void **)&d->q)) ||
        FAILED(ID3D12Device_CreateFence(d->dev, 0, D3D12_FENCE_FLAG_NONE, &IID_ID3D12Fence, (void **)&d->fence)))
        return -1;
    for (int i = 0; i < NSLOT; i++) {
        Slot *s = &d->slot[i];
        if (FAILED(ID3D12Device_CreateCommandAllocator(d->dev, D3D12_COMMAND_LIST_TYPE_COPY,
                                                       &IID_ID3D12CommandAllocator, (void **)&s->alloc)) ||
            FAILED(ID3D12Device_CreateCommandList(d->dev, 0, D3D12_COMMAND_LIST_TYPE_COPY, s->alloc, NULL,
                                                  &IID_ID3D12GraphicsCommandList, (void **)&s->list)))
            return -1;
        ID3D12GraphicsCommandList_Close(s->list);
        if (FAILED(ID3D12Device_CreateCommandAllocator(d->dev, D3D12_COMMAND_LIST_TYPE_VIDEO_PROCESS,
                                                       &IID_ID3D12CommandAllocator, (void **)&s->valloc)) ||
            FAILED(ID3D12Device_CreateCommandList(d->dev, 0, D3D12_COMMAND_LIST_TYPE_VIDEO_PROCESS, s->valloc,
                                                  NULL, &IID_ID3D12VideoProcessCommandList, (void **)&s->vlist)))
            return -1;
        ID3D12VideoProcessCommandList_Close(s->vlist);
    }
    d->ev = CreateEventA(NULL, FALSE, FALSE, NULL);
    return 0;
}

static DWORD WINAPI ka_main(LPVOID p)
{
    VtpvDx *d = p;
    while (!d->ka_stop) {
        Sleep((DWORD)d->ka_ms);
        if (d->ka_stop) break;
        if (d->pooled_at > 0 && now_ms(d) - d->pooled_at > POOL_TTL_MS) continue;   /* expired in the pool */
        ID3D12CommandAllocator_Reset(d->ka_alloc);
        ID3D12GraphicsCommandList_Reset(d->ka_list, d->ka_alloc, NULL);
        ID3D12GraphicsCommandList_CopyBufferRegion(d->ka_list, d->ka_dst, 0, d->ka_src, 0, 256);
        ID3D12GraphicsCommandList_Close(d->ka_list);
        ID3D12CommandList *l = (ID3D12CommandList *)d->ka_list;
        ID3D12CommandQueue_ExecuteCommandLists(d->q, 1, &l);
        ID3D12CommandQueue_Signal(d->q, d->ka_fence, ++d->ka_fv);
        if (ID3D12Fence_GetCompletedValue(d->ka_fence) < d->ka_fv) {
            ID3D12Fence_SetEventOnCompletion(d->ka_fence, d->ka_fv, d->ka_ev);
            WaitForSingleObject(d->ka_ev, 2000);
        }
    }
    return 0;
}

static void ka_start(VtpvDx *d)
{
    const char *e = getenv("VTPVDX_KEEPALIVE_MS");
    d->ka_ms = e ? atoi(e) : 100;
    if (d->ka_ms <= 0) return;
    D3D12_HEAP_PROPERTIES hp = { .Type = D3D12_HEAP_TYPE_DEFAULT };
    D3D12_RESOURCE_DESC bd = { .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Width = 256, .Height = 1,
        .DepthOrArraySize = 1, .MipLevels = 1, .Format = DXGI_FORMAT_UNKNOWN, .SampleDesc = { 1, 0 },
        .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR };
    if (FAILED(ID3D12Device_CreateCommittedResource(d->dev, &hp, D3D12_HEAP_FLAG_NONE, &bd,
               D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource, (void **)&d->ka_src)) ||
        FAILED(ID3D12Device_CreateCommittedResource(d->dev, &hp, D3D12_HEAP_FLAG_NONE, &bd,
               D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource, (void **)&d->ka_dst)) ||
        FAILED(ID3D12Device_CreateCommandAllocator(d->dev, D3D12_COMMAND_LIST_TYPE_COPY,
               &IID_ID3D12CommandAllocator, (void **)&d->ka_alloc)) ||
        FAILED(ID3D12Device_CreateCommandList(d->dev, 0, D3D12_COMMAND_LIST_TYPE_COPY, d->ka_alloc, NULL,
               &IID_ID3D12GraphicsCommandList, (void **)&d->ka_list)) ||
        FAILED(ID3D12Device_CreateFence(d->dev, 0, D3D12_FENCE_FLAG_NONE, &IID_ID3D12Fence,
               (void **)&d->ka_fence))) {
        fprintf(stderr, "vtpv-dx: keep-alive setup failed\n");
        return;
    }
    ID3D12GraphicsCommandList_Close(d->ka_list);
    d->ka_ev = CreateEventA(NULL, FALSE, FALSE, NULL);
    d->ka_thread = CreateThread(NULL, 0, ka_main, d, 0, NULL);
}

static void ka_stop(VtpvDx *d)
{
    if (d->ka_thread) {
        InterlockedExchange(&d->ka_stop, 1);
        WaitForSingleObject(d->ka_thread, INFINITE);
        CloseHandle(d->ka_thread);
        d->ka_thread = NULL;
    }
    if (d->ka_list) ID3D12GraphicsCommandList_Release(d->ka_list);
    if (d->ka_alloc) ID3D12CommandAllocator_Release(d->ka_alloc);
    if (d->ka_src) ID3D12Resource_Release(d->ka_src);
    if (d->ka_dst) ID3D12Resource_Release(d->ka_dst);
    if (d->ka_fence) ID3D12Fence_Release(d->ka_fence);
    if (d->ka_ev) CloseHandle(d->ka_ev);
    d->ka_list = NULL; d->ka_alloc = NULL; d->ka_src = d->ka_dst = NULL; d->ka_fence = NULL; d->ka_ev = NULL;
}

static void wait_fv(VtpvDx *d, UINT64 fv)
{
    if (ID3D12Fence_GetCompletedValue(d->fence) < fv) {
        ID3D12Fence_SetEventOnCompletion(d->fence, fv, d->ev);
        WaitForSingleObject(d->ev, INFINITE);
    }
}

static void slots_free(VtpvDx *d)
{
    for (int i = 0; i < NSLOT; i++) {
        Slot *s = &d->slot[i];
        if (s->buf) { ID3D12Resource_Release(s->buf); s->buf = NULL; }
        if (s->heap) { ID3D12Heap_Release(s->heap); s->heap = NULL; }
        if (s->mem) { VirtualFree(s->mem, 0, MEM_RELEASE); s->mem = NULL; }
    }
}

static int slot_mem(VtpvDx *d, Slot *s)
{
    s->mem = VirtualAlloc(NULL, d->memsz, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!s->mem) return -1;
    ID3D12Device3 *d3;
    if (FAILED(ID3D12Device_QueryInterface(d->dev, &IID_ID3D12Device3, (void **)&d3))) return -1;
    HRESULT h = ID3D12Device3_OpenExistingHeapFromAddress(d3, s->mem, &IID_ID3D12Heap, (void **)&s->heap);
    ID3D12Device3_Release(d3);
    if (FAILED(h)) { fprintf(stderr, "vtpv-dx: OpenExistingHeapFromAddress -> %#lx\n", (unsigned long)h); return -1; }
    D3D12_RESOURCE_DESC bd = { .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Width = d->memsz, .Height = 1,
        .DepthOrArraySize = 1, .MipLevels = 1, .Format = DXGI_FORMAT_UNKNOWN, .SampleDesc = { 1, 0 },
        .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR, .Flags = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER };
    if (FAILED(ID3D12Device_CreatePlacedResource(d->dev, s->heap, 0, &bd, D3D12_RESOURCE_STATE_COPY_DEST, NULL,
                                                 &IID_ID3D12Resource, (void **)&s->buf)))
        return -1;
    return 0;
}

static void fire(VtpvDx *d, Slot *s)
{
    if (!s->busy) return;
    double a = now_ms(d);
    wait_fv(d, s->fv);
    d->t_gpu += now_ms(d) - a;
    s->busy = 0;
    if (d->nplanes == 1) {
        /* YUY2 / Y210 (4:2:2) and AYUV / Y410 (4:4:4) as luma + CbCr pairs, 8 bits */
        const uint8_t *src = s->mem + d->fp[0].Offset;
        uint32_t pitch = d->fp[0].Footprint.RowPitch, w = s->out.width, h = s->out.height;
        int c422 = d->tex_fmt == DXGI_FORMAT_YUY2 || d->tex_fmt == DXGI_FORMAT_Y210;
        uint32_t cw = c422 ? (w + 1) / 2 : w;
        size_t need = (size_t)w * h + (size_t)cw * 2 * h;
        if (d->conv_cap < need) {
            free(d->conv);
            d->conv = malloc(need);
            d->conv_cap = d->conv ? need : 0;
            if (!d->conv) return;
        }
        uint8_t *yp = d->conv, *cp = d->conv + (size_t)w * h;
        for (uint32_t y = 0; y < h; y++) {
            const uint8_t *r = src + (size_t)y * pitch;
            uint8_t *yo = yp + (size_t)y * w, *co = cp + (size_t)y * cw * 2;
            switch (d->tex_fmt) {
            case DXGI_FORMAT_YUY2:          /* Y0 U Y1 V */
                for (uint32_t x = 0; x < cw; x++) {
                    yo[2 * x] = r[4 * x];
                    if (2 * x + 1 < w) yo[2 * x + 1] = r[4 * x + 2];
                    co[2 * x] = r[4 * x + 1];
                    co[2 * x + 1] = r[4 * x + 3];
                }
                break;
            case DXGI_FORMAT_Y210: {        /* the same, 16-bit samples, data in the high bits */
                const uint16_t *q = (const uint16_t *)r;
                for (uint32_t x = 0; x < cw; x++) {
                    yo[2 * x] = q[4 * x] >> 8;
                    if (2 * x + 1 < w) yo[2 * x + 1] = q[4 * x + 2] >> 8;
                    co[2 * x] = q[4 * x + 1] >> 8;
                    co[2 * x + 1] = q[4 * x + 3] >> 8;
                }
                break;
            }
            case DXGI_FORMAT_AYUV:          /* V U Y A */
                for (uint32_t x = 0; x < w; x++) {
                    yo[x] = r[4 * x + 2];
                    co[2 * x] = r[4 * x + 1];
                    co[2 * x + 1] = r[4 * x];
                }
                break;
            default: {                      /* Y410: U 0-9, Y 10-19, V 20-29 */
                const uint32_t *q = (const uint32_t *)r;
                for (uint32_t x = 0; x < w; x++) {
                    yo[x] = (q[x] >> 12) & 0xff;
                    co[2 * x] = (q[x] >> 2) & 0xff;
                    co[2 * x + 1] = (q[x] >> 22) & 0xff;
                }
                break;
            }
            }
        }
        s->out.y = yp;
        s->out.y_pitch = w;
        s->out.uv = cp;
        s->out.uv_pitch = cw * 2;
        s->out.planar = 0;
        s->out.sx = c422 ? 1 : 0;
        s->out.sy = 0;
    }
    /* 28-10 (240fps): with the GPU conversion the slot already holds 8-bit
     * NV12 - the CPU downshift below is only the vp_fail fallback */
    if (!d->vp && d->nplanes == 2 && d->tex_fmt == DXGI_FORMAT_P010) {
        /* TEMP lab main10 (28.10): P010 = 10 bits in the HIGH bits of 16-bit
         * samples, and every consumer (the stage copy, vtpv_fill_surface, the
         * guest's '420v' pool - VideoToolbox's own default output for a Main10
         * stream is 8-bit 420v) reads 8-bit rows: without this the stage took
         * the row's first w bytes of a 2-per-sample plane - a striped half
         * picture.  Downshift both planes like the packed 4:2:2/4:4:4 branch */
        uint32_t w = s->out.width, h = s->out.height;
        uint32_t cw = (w + 1) / 2, ch = (h + 1) / 2;
        size_t need = (size_t)w * h + (size_t)cw * 2 * ch;
        if (d->conv_cap < need) {
            free(d->conv);
            d->conv = malloc(need);
            d->conv_cap = d->conv ? need : 0;
            if (!d->conv) return;
        }
        uint8_t *yp = d->conv, *cp = d->conv + (size_t)w * h;
        const uint8_t *sy = s->mem + d->fp[0].Offset, *sc = s->mem + d->fp[1].Offset;
        for (uint32_t y = 0; y < h; y++) {
            const uint16_t *r = (const uint16_t *)(sy + (size_t)y * d->fp[0].Footprint.RowPitch);
            uint8_t *o = yp + (size_t)y * w;
            for (uint32_t x = 0; x < w; x++) {
                o[x] = (uint8_t)(r[x] >> 8);
            }
        }
        for (uint32_t y = 0; y < ch; y++) {
            const uint16_t *r = (const uint16_t *)(sc + (size_t)y * d->fp[1].Footprint.RowPitch);
            uint8_t *o = cp + (size_t)y * cw * 2;
            for (uint32_t x = 0; x < cw * 2; x++) {
                o[x] = (uint8_t)(r[x] >> 8);
            }
        }
        s->out.y = yp;
        s->out.y_pitch = w;
        s->out.uv = cp;
        s->out.uv_pitch = cw * 2;
    }
    d->out(d->opaque, &s->out);
}

/* 28-10 (240fps): the P010->NV12 conversion target and the VIDEO_PROCESS
 * queue, once per decoder texture size; any failure sets vp_fail and the
 * CPU downshift in fire() stays */
static int vp_setup(VtpvDx *d, const D3D12_RESOURCE_DESC *td)
{
    if (d->vp_fail) {
        return -1;
    }
    if (d->vp && d->nv12 && d->nv12_w == td->Width && d->nv12_h == td->Height) {
        return 0;
    }
    if (d->nv12) { ID3D12Resource_Release(d->nv12); d->nv12 = NULL; }
    if (!d->vd &&
        FAILED(ID3D12Device_QueryInterface(d->dev, &IID_ID3D12VideoDevice, (void **)&d->vd))) {
        goto fail;
    }
    if (d->vp) {                    /* its size ranges are this texture's */
        ID3D12VideoProcessor_Release(d->vp);
        d->vp = NULL;
    }
    {
        UINT tw = (UINT)td->Width, th = td->Height;
        D3D12_VIDEO_PROCESS_INPUT_STREAM_DESC in = {
            .Format = DXGI_FORMAT_P010,
            .ColorSpace = DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P709,
            .SourceAspectRatio = { 1, 1 }, .DestinationAspectRatio = { 1, 1 },
            .FrameRate = { 60, 1 },
            .SourceSizeRange = { .MaxWidth = tw, .MaxHeight = th, .MinWidth = tw, .MinHeight = th },
            .DestinationSizeRange = { .MaxWidth = tw, .MaxHeight = th, .MinWidth = tw, .MinHeight = th },
        };
        D3D12_VIDEO_PROCESS_OUTPUT_STREAM_DESC out = {
            .Format = DXGI_FORMAT_NV12,
            .ColorSpace = DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P709,
            .FrameRate = { 60, 1 },
        };
        if (FAILED(ID3D12VideoDevice_CreateVideoProcessor(d->vd, 0, &out, 1, &in,
                                                          &IID_ID3D12VideoProcessor,
                                                          (void **)&d->vp))) {
            goto fail;
        }
    }
    if (!d->vpq) {
        D3D12_COMMAND_QUEUE_DESC qd = { .Type = D3D12_COMMAND_LIST_TYPE_VIDEO_PROCESS };
        if (FAILED(ID3D12Device_CreateCommandQueue(d->dev, &qd, &IID_ID3D12CommandQueue,
                                                   (void **)&d->vpq)) ||
            FAILED(ID3D12Device_CreateFence(d->dev, 0, D3D12_FENCE_FLAG_NONE, &IID_ID3D12Fence,
                                            (void **)&d->vpf))) {
            goto fail;
        }
    }
    D3D12_HEAP_PROPERTIES hp = { .Type = D3D12_HEAP_TYPE_DEFAULT };
    D3D12_RESOURCE_DESC nd = *td;
    nd.Format = DXGI_FORMAT_NV12;
    nd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    nd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    if (FAILED(ID3D12Device_CreateCommittedResource(d->dev, &hp, D3D12_HEAP_FLAG_NONE, &nd,
               D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource, (void **)&d->nv12))) {
        goto fail;
    }
    d->nv12_w = td->Width;
    d->nv12_h = td->Height;
    fprintf(stderr, "vtpv-dx: GPU P010->NV12 %llux%llu\n",
            (unsigned long long)td->Width, (unsigned long long)td->Height);
    return 0;
fail:
    d->vp_fail = 1;
    fprintf(stderr, "vtpv-dx: video-processor setup failed - CPU downshift stays\n");
    return -1;
}

/* (re)size the process-memory buffers for this decoder texture */
static int slots_setup(VtpvDx *d, ID3D12Resource *tex, int sub)
{
    D3D12_RESOURCE_DESC td = ID3D12Resource_GetDesc(tex);
    if (d->slot[0].buf && td.Width == d->tex_w && td.Height == d->tex_h && td.Format == d->tex_fmt) return 0;
    for (int i = 0; i < NSLOT; i++) fire(d, &d->slot[i]);   /* resolution change: drain */
    slots_free(d);
    d->tex_fmt = td.Format;
    d->nplanes = td.Format == DXGI_FORMAT_NV12 || td.Format == DXGI_FORMAT_P010 ? 2 : 1;
    /* 28-10 (240fps): through the video processor the readback copies the
     * CONVERTED NV12 texture - footprints and half the bytes from it */
    if (d->nplanes == 2 && td.Format == DXGI_FORMAT_P010) {
        vp_setup(d, &td);
    }
    D3D12_RESOURCE_DESC nv = td, *fd = &td;
    if (d->vp && d->nv12 && td.Format == DXGI_FORMAT_P010) {
        nv.Format = DXGI_FORMAT_NV12;
        fd = &nv;
        sub = 0;
    }
    UINT64 off = 0;
    for (int p = 0; p < d->nplanes; p++) {
        UINT s = (UINT)sub + (UINT)p * fd->DepthOrArraySize * fd->MipLevels;
        UINT64 t;
        ID3D12Device_GetCopyableFootprints(d->dev, fd, s, 1, off, &d->fp[p], NULL, NULL, &t);
        off = (d->fp[p].Offset + t + 511) & ~511ull;
    }
    d->memsz = (SIZE_T)((off + 65535) & ~65535ull);
    for (int i = 0; i < NSLOT; i++)
        if (slot_mem(d, &d->slot[i]) < 0) return -1;
    d->tex_w = td.Width;
    d->tex_h = td.Height;
    return 0;
}

/* one decoded picture: GPU copy into a slot; synchronous mode calls back now,
 * pipelined mode calls back the PREVIOUS picture once this copy is queued */
static int emit(VtpvDx *d, AVFrame *f, int64_t tag)
{
    VtpvDxFrame o = { .width = (uint32_t)d->c->width, .height = (uint32_t)d->c->height, .tag = tag,
                      .sx = 1, .sy = 1 };
    d->npics++;
    if (!d->d3d12) {
        /* d3d11/cuda: libavcodec's readback; sw: the frame is in memory already */
        AVFrame *m = f;
        if (!d->soft) {
            double a = now_ms(d);
            av_frame_unref(d->sw);
            if (av_hwframe_transfer_data(d->sw, f, 0) < 0) return -1;
            d->t_gpu += now_ms(d) - a;
            m = d->sw;
        }
        o.y = m->data[0];
        o.y_pitch = (uint32_t)m->linesize[0];
        o.uv = m->data[1];
        o.uv_pitch = (uint32_t)m->linesize[1];
        switch (m->format) {
        case AV_PIX_FMT_NV12:
            break;
        case AV_PIX_FMT_GRAY8:
            o.gray = 1;
            break;
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_YUVJ420P:
        case AV_PIX_FMT_YUV422P:
        case AV_PIX_FMT_YUVJ422P:
        case AV_PIX_FMT_YUV444P:
        case AV_PIX_FMT_YUVJ444P: {
            const AVPixFmtDescriptor *pd = av_pix_fmt_desc_get(m->format);
            o.planar = 1;
            o.v = m->data[2];
            o.v_pitch = (uint32_t)m->linesize[2];
            o.sx = pd->log2_chroma_w;
            o.sy = pd->log2_chroma_h;
            break;
        }
        default:
            fprintf(stderr, "vtpv-dx: picture format %s not handled\n", av_get_pix_fmt_name(m->format));
            return -1;
        }
        d->out(d->opaque, &o);
        return 0;
    }
    AVD3D12VAFrame *df = (AVD3D12VAFrame *)f->data[0];
    if (slots_setup(d, df->texture, df->subresource_index) < 0) return -1;
    Slot *s;
    if (d->pipelined) {
        /* async readback: prefer a free slot so the submit returns without a
         * fence wait; when the pool is full the oldest picture fires first
         * (backpressure at the pool depth) */
        s = NULL;
        for (int i = 0; i < NSLOT && !s; i++) {
            Slot *c = &d->slot[(d->next + i) % NSLOT];
            if (!c->busy) {
                s = c;
            }
        }
        if (!s) {
            s = &d->slot[d->next];
            fire(d, s);
        }
    } else {
        s = &d->slot[d->next];
        fire(d, s);                                 /* free it (pipelined: picture N-2) */
    }
    D3D12_RESOURCE_DESC td = ID3D12Resource_GetDesc(df->texture);
    ID3D12Resource *srcTex = df->texture;
    UINT srcSub = (UINT)df->subresource_index;
    bool vpBlit = d->vp && d->nv12 && d->tex_fmt == DXGI_FORMAT_P010;
    if (vpBlit) {
        /* 28-10 (240fps): the P010->NV12 blit rides this SLOT's video list
         * (its allocator resets only after the slot's previous picture fired
         * - the same discipline as the copy list); every wait is GPU-side */
        ID3D12CommandAllocator_Reset(s->valloc);
        ID3D12VideoProcessCommandList_Reset(s->vlist, s->valloc);
        D3D12_RESOURCE_BARRIER tovp[2] = {
            { .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
              .Transition = { .pResource = df->texture, .Subresource = df->subresource_index,
                              .StateBefore = D3D12_RESOURCE_STATE_COMMON,
                              .StateAfter = D3D12_RESOURCE_STATE_VIDEO_PROCESS_READ } },
            { .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
              .Transition = { .pResource = d->nv12, .Subresource = 0,
                              .StateBefore = D3D12_RESOURCE_STATE_COMMON,
                              .StateAfter = D3D12_RESOURCE_STATE_VIDEO_PROCESS_WRITE } },
        };
        ID3D12VideoProcessCommandList_ResourceBarrier(s->vlist, 2, tovp);
        D3D12_VIDEO_PROCESS_INPUT_STREAM_ARGUMENTS vin = { 0 };
        vin.InputStream[0].pTexture2D = df->texture;
        vin.InputStream[0].Subresource = df->subresource_index;
        D3D12_RECT whole = { 0, 0, (LONG)td.Width, (LONG)td.Height };
        vin.Transform.SourceRectangle = whole;
        vin.Transform.DestinationRectangle = whole;
        vin.RateInfo.OutputIndex = 0;
        vin.RateInfo.InputFrameOrField = 0;
        D3D12_VIDEO_PROCESS_OUTPUT_STREAM_ARGUMENTS vout = { 0 };
        vout.OutputStream[0].pTexture2D = d->nv12;
        vout.OutputStream[0].Subresource = 0;
        vout.TargetRectangle = whole;
        ID3D12VideoProcessCommandList_ProcessFrames(s->vlist, d->vp, &vout, 1, &vin);
        D3D12_RESOURCE_BARRIER back[2] = { tovp[0], tovp[1] };
        back[0].Transition.StateBefore = D3D12_RESOURCE_STATE_VIDEO_PROCESS_READ;
        back[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
        back[1].Transition.StateBefore = D3D12_RESOURCE_STATE_VIDEO_PROCESS_WRITE;
        back[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
        ID3D12VideoProcessCommandList_ResourceBarrier(s->vlist, 2, back);
        ID3D12VideoProcessCommandList_Close(s->vlist);
        ID3D12CommandQueue_Wait(d->vpq, df->sync_ctx.fence, df->sync_ctx.fence_value);
        ID3D12CommandList *vl = (ID3D12CommandList *)s->vlist;
        ID3D12CommandQueue_ExecuteCommandLists(d->vpq, 1, &vl);
        ID3D12CommandQueue_Signal(d->vpq, d->vpf, ++d->vpfv);
        srcTex = d->nv12;
        srcSub = 0;
    }
    ID3D12CommandAllocator_Reset(s->alloc);
    ID3D12GraphicsCommandList_Reset(s->list, s->alloc, NULL);
    for (int p = 0; p < d->nplanes; p++) {
        D3D12_TEXTURE_COPY_LOCATION src = { .pResource = srcTex,
            .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            .SubresourceIndex = srcSub + (UINT)p * td.DepthOrArraySize * td.MipLevels };
        D3D12_TEXTURE_COPY_LOCATION dst = { .pResource = s->buf,
            .Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, .PlacedFootprint = d->fp[p] };
        ID3D12GraphicsCommandList_CopyTextureRegion(s->list, &dst, 0, 0, 0, &src, NULL);
    }
    ID3D12GraphicsCommandList_Close(s->list);
    if (vpBlit) {
        ID3D12CommandQueue_Wait(d->q, d->vpf, d->vpfv);
    } else {
        ID3D12CommandQueue_Wait(d->q, df->sync_ctx.fence, df->sync_ctx.fence_value);
    }
    ID3D12CommandList *l = (ID3D12CommandList *)s->list;
    ID3D12CommandQueue_ExecuteCommandLists(d->q, 1, &l);
    ID3D12CommandQueue_Signal(d->q, d->fence, ++d->fv);
    s->fv = d->fv;
    o.y = s->mem + d->fp[0].Offset;
    o.y_pitch = d->fp[0].Footprint.RowPitch;
    if (d->nplanes == 2) {      /* packed outputs get their planes in fire() */
        o.uv = s->mem + d->fp[1].Offset;
        o.uv_pitch = d->fp[1].Footprint.RowPitch;
    }
    s->out = o;
    s->busy = 1;
    d->next = (int)((s - d->slot) + 1) % NSLOT;
    if (!d->pipelined) {
        fire(d, s);
    }
    /* pipelined: nothing fires here - the pool fills to NSLOT-1 pictures and
     * vtpv_dx_flush() (the collect step) fires them oldest-first */
    return 0;
}

VtpvDx *vtpv_dx_open(const char *gpu, const char *api, VtpvDxCodec codec,
                     const uint8_t *cfg, uint32_t cfg_len, VtpvDxOut out, void *opaque)
{
    static const enum AVCodecID ids[] = { AV_CODEC_ID_H264, AV_CODEC_ID_HEVC, AV_CODEC_ID_VP9, AV_CODEC_ID_AV1 };
    int cuda = api && strcmp(api, "cuda") == 0;
    int soft = api && strcmp(api, "sw") == 0;
    int d3d12 = !cuda && !soft && !(api && strcmp(api, "d3d11") == 0);
    int nv = cuda || (gpu && strcmp(gpu, "nv") == 0);
    if (cuda || d3d12) {
        /* a pooled decoder of this stream configuration; drop the expired ones */
        VtpvDx *hit = NULL, *old[POOL_MAX];
        int nold = 0;
        AcquireSRWLockExclusive(&pool_lock);
        for (int i = 0; i < POOL_MAX; i++) {
            VtpvDx *p = pool[i];
            if (!p) continue;
            if (now_ms(p) - p->pooled_at > POOL_TTL_MS) {
                old[nold++] = p;
                pool[i] = NULL;
            } else if (!hit && p->cuda == cuda && p->d3d12 == d3d12 && p->nv == nv && p->codec == codec &&
                       p->cfg_len == cfg_len && (!cfg_len || memcmp(p->cfg, cfg, cfg_len) == 0)) {
                hit = p;
                pool[i] = NULL;
            }
        }
        ReleaseSRWLockExclusive(&pool_lock);
        for (int i = 0; i < nold; i++) dx_destroy(old[i]);
        if (hit) {
            hit->out = out;
            hit->opaque = opaque;
            hit->pipelined = 0;
            hit->t_parse = hit->t_gpu = 0;
            hit->npics = 0;
            hit->pooled_at = 0;
            return hit;
        }
    }
    VtpvDx *d = calloc(1, sizeof(*d));
    if (!d) return NULL;
    QueryPerformanceFrequency(&d->qf);
    d->out = out;
    d->opaque = opaque;
    d->cuda = cuda;
    d->soft = soft;
    d->d3d12 = d3d12;
    d->nv = nv;
    d->codec = codec;
    if (cfg_len) {
        d->cfg = malloc(cfg_len);
        if (!d->cfg) goto fail;
        memcpy(d->cfg, cfg, cfg_len);
        d->cfg_len = cfg_len;
    }
    d->hook = d->d3d12 && (codec == VTPV_DX_H264 || codec == VTPV_DX_HEVC);
    d->hwfmt = d->cuda ? AV_PIX_FMT_CUDA : d->d3d12 ? AV_PIX_FMT_D3D12 : AV_PIX_FMT_D3D11;
    if (d->soft) {
        /* no device: libavcodec's own decoder on this thread */
    } else if (d->cuda || d->d3d12) {
        /* one device per api and GPU for the process (see POOL_MAX) */
        AcquireSRWLockExclusive(&pool_lock);
        AVBufferRef **dc = &dev_cache[d->cuda][d->nv];
        if (!*dc) {
            if (d->cuda) {
                /* CUDA numbers its own devices: the first NVIDIA GPU */
                if (av_hwdevice_ctx_create(dc, AV_HWDEVICE_TYPE_CUDA, "0", NULL, 0) < 0) {
                    fprintf(stderr, "vtpv-dx: cuda device failed (libavcodec without nvdec, or no NVIDIA GPU)\n");
                }
            } else {
                int ai = adapter_for(d->nv ? 0x10de : 0x8086);
                char devstr[16];
                snprintf(devstr, sizeof devstr, "%d", ai);
                if (ai < 0) {
                    fprintf(stderr, "vtpv-dx: no %s adapter\n", gpu ? gpu : "int");
                } else if (av_hwdevice_ctx_create(dc, AV_HWDEVICE_TYPE_D3D12VA, devstr, NULL, 0) < 0) {
                    fprintf(stderr, "vtpv-dx: d3d12 device on adapter %d failed\n", ai);
                }
            }
        }
        d->hw = *dc ? av_buffer_ref(*dc) : NULL;
        ReleaseSRWLockExclusive(&pool_lock);
        if (!d->hw) goto fail;
    } else {
        int ai = adapter_for(gpu && strcmp(gpu, "nv") == 0 ? 0x10de : 0x8086);
        if (ai < 0) { fprintf(stderr, "vtpv-dx: no %s adapter\n", gpu ? gpu : "int"); goto fail; }
        char devstr[16];
        snprintf(devstr, sizeof devstr, "%d", ai);
        if (av_hwdevice_ctx_create(&d->hw, AV_HWDEVICE_TYPE_D3D11VA, devstr, NULL, 0) < 0) {
            fprintf(stderr, "vtpv-dx: d3d11 device on adapter %d failed\n", ai);
            goto fail;
        }
    }
    const AVCodec *dec = avcodec_find_decoder_by_name(avcodec_get_name(ids[codec]));
    if (!dec) goto fail;
    d->c = avcodec_alloc_context3(dec);
    d->c->opaque = d;
    if (!d->soft) {
        d->c->hw_device_ctx = av_buffer_ref(d->hw);
        d->c->get_format = get_fmt;
    }
    if (d->hook) d->c->get_buffer2 = get_buf;
    d->c->thread_count = 1;
    if (cfg && cfg_len) {
        d->c->extradata = av_mallocz(cfg_len + AV_INPUT_BUFFER_PADDING_SIZE);
        memcpy(d->c->extradata, cfg, cfg_len);
        d->c->extradata_size = (int)cfg_len;
    }
    if (avcodec_open2(d->c, dec, NULL) < 0) { fprintf(stderr, "vtpv-dx: codec open failed\n"); goto fail; }
    if (d->d3d12 && queue_init(d) < 0) { fprintf(stderr, "vtpv-dx: copy queue setup failed\n"); goto fail; }
    if (d->d3d12) ka_start(d);
    d->pkt = av_packet_alloc();
    d->fr = av_frame_alloc();
    d->sw = av_frame_alloc();
    return d;
fail:
    dx_destroy(d);
    return NULL;
}

void vtpv_dx_set_pipelined(VtpvDx *d, int on)
{
    if (!on) vtpv_dx_flush(d);
    d->pipelined = on && d->d3d12;
}

void vtpv_dx_flush(VtpvDx *d)
{
    /* oldest first: the slot after the last one written */
    for (int i = 0; i < NSLOT; i++) fire(d, &d->slot[(d->next + i) % NSLOT]);
}

int vtpv_dx_decode(VtpvDx *d, const uint8_t *sample, uint32_t len, int64_t tag)
{
    int rc = 0;
    double a = now_ms(d);
    if (av_new_packet(d->pkt, (int)len) < 0) return -1;
    memcpy(d->pkt->data, sample, len);
    d->pkt->pts = tag;
    d->npend = 0;
    int r = avcodec_send_packet(d->c, d->pkt);
    av_packet_unref(d->pkt);
    d->t_parse += now_ms(d) - a;
    if (r < 0 && r != AVERROR(EAGAIN)) rc = -1;
    if (d->hook) {
        /* decode order: the pictures this packet allocated are submitted now */
        for (int i = 0; i < d->npend; i++) {
            if (emit(d, d->pend[i], tag) < 0) rc = -1;
            av_frame_free(&d->pend[i]);
        }
        d->npend = 0;
    }
    for (;;) {
        a = now_ms(d);
        r = avcodec_receive_frame(d->c, d->fr);
        d->t_parse += now_ms(d) - a;
        if (r < 0) break;
        if (!d->hook && emit(d, d->fr, d->fr->pts) < 0) rc = -1;
        av_frame_unref(d->fr);
    }
    return rc;
}

void vtpv_dx_stats(const VtpvDx *d, double *ms_parse, double *ms_gpu, int *pictures)
{
    int n = d->npics ? d->npics : 1;
    if (ms_parse) *ms_parse = d->t_parse / n;
    if (ms_gpu) *ms_gpu = d->t_gpu / n;
    if (pictures) *pictures = d->npics;
}

void vtpv_dx_close(VtpvDx *d)
{
    if (!d) return;
    if (d->cuda || d->d3d12) {
        /* to the pool: pictures in flight are dropped (their copies finished
         * first, nothing writes the slots later), the decoder forgets its
         * references */
        for (int i = 0; i < d->npend; i++) av_frame_free(&d->pend[i]);
        d->npend = 0;
        if (d->fence && d->q) {
            ID3D12CommandQueue_Signal(d->q, d->fence, ++d->fv);
            wait_fv(d, d->fv);
        }
        for (int i = 0; i < NSLOT; i++) d->slot[i].busy = 0;
        avcodec_flush_buffers(d->c);
        d->out = NULL;
        d->opaque = NULL;
        d->pooled_at = now_ms(d);
        VtpvDx *evict = d;
        AcquireSRWLockExclusive(&pool_lock);
        for (int i = 0; i < POOL_MAX && evict; i++) {
            if (!pool[i]) {
                pool[i] = d;
                evict = NULL;
            }
        }
        if (evict) {
            /* full: the oldest goes */
            int o = 0;
            for (int i = 1; i < POOL_MAX; i++) {
                if (pool[i]->pooled_at < pool[o]->pooled_at) o = i;
            }
            evict = pool[o];
            pool[o] = d;
        }
        ReleaseSRWLockExclusive(&pool_lock);
        if (evict) dx_destroy(evict);
        return;
    }
    dx_destroy(d);
}

void vtpv_dx_discard(VtpvDx *d)
{
    if (d) dx_destroy(d);
}

static void dx_destroy(VtpvDx *d)
{
    if (!d) return;
    ka_stop(d);
    for (int i = 0; i < d->npend; i++) av_frame_free(&d->pend[i]);
    if (d->fence && d->q) {
        /* nothing may still write into the memory we free; pending pictures are dropped */
        ID3D12CommandQueue_Signal(d->q, d->fence, ++d->fv);
        wait_fv(d, d->fv);
    }
    slots_free(d);
    for (int i = 0; i < NSLOT; i++) {
        if (d->slot[i].list) ID3D12GraphicsCommandList_Release(d->slot[i].list);
        if (d->slot[i].alloc) ID3D12CommandAllocator_Release(d->slot[i].alloc);
        if (d->slot[i].vlist) ID3D12VideoProcessCommandList_Release(d->slot[i].vlist);
        if (d->slot[i].valloc) ID3D12CommandAllocator_Release(d->slot[i].valloc);
    }
    if (d->vpq) ID3D12CommandQueue_Release(d->vpq);
    if (d->vpf) ID3D12Fence_Release(d->vpf);
    if (d->vp) ID3D12VideoProcessor_Release(d->vp);
    if (d->vd) ID3D12VideoDevice_Release(d->vd);
    if (d->nv12) ID3D12Resource_Release(d->nv12);
    if (d->q) ID3D12CommandQueue_Release(d->q);
    if (d->fence) ID3D12Fence_Release(d->fence);
    if (d->ev) CloseHandle(d->ev);
    free(d->conv);
    av_frame_free(&d->fr);
    av_frame_free(&d->sw);
    av_packet_free(&d->pkt);
    avcodec_free_context(&d->c);
    av_buffer_unref(&d->hw);
    free(d->cfg);
    free(d);
}
