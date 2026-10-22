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
#pragma GCC diagnostic pop

#define MAXPEND 8
#define NSLOT 2

typedef struct {
    ID3D12CommandAllocator *alloc;
    ID3D12GraphicsCommandList *list;
    uint8_t *mem;               /* plain process memory (VirtualAlloc) */
    ID3D12Heap *heap;
    ID3D12Resource *buf;        /* placed buffer covering mem */
    UINT64 fv;                  /* copy fence value of the picture in it */
    int busy;                   /* holds a picture whose callback is pending */
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
    /* stats */
    LARGE_INTEGER qf;
    double t_parse, t_gpu;
    int npics;
};

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
    d->out(d->opaque, &s->out);
}

/* (re)size the process-memory buffers for this decoder texture */
static int slots_setup(VtpvDx *d, ID3D12Resource *tex, int sub)
{
    D3D12_RESOURCE_DESC td = ID3D12Resource_GetDesc(tex);
    if (d->slot[0].buf && td.Width == d->tex_w && td.Height == d->tex_h) return 0;
    for (int i = 0; i < NSLOT; i++) fire(d, &d->slot[i]);   /* resolution change: drain */
    slots_free(d);
    UINT64 off = 0;
    for (int p = 0; p < 2; p++) {
        UINT s = (UINT)sub + (UINT)p * td.DepthOrArraySize * td.MipLevels;
        UINT64 t;
        ID3D12Device_GetCopyableFootprints(d->dev, &td, s, 1, off, &d->fp[p], NULL, NULL, &t);
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
    VtpvDxFrame o = { .width = (uint32_t)d->c->width, .height = (uint32_t)d->c->height, .tag = tag };
    d->npics++;
    if (!d->d3d12) {
        double a = now_ms(d);
        av_frame_unref(d->sw);
        if (av_hwframe_transfer_data(d->sw, f, 0) < 0) return -1;
        d->t_gpu += now_ms(d) - a;
        o.y = d->sw->data[0];
        o.y_pitch = (uint32_t)d->sw->linesize[0];
        o.uv = d->sw->data[1];
        o.uv_pitch = (uint32_t)d->sw->linesize[1];
        d->out(d->opaque, &o);
        return 0;
    }
    AVD3D12VAFrame *df = (AVD3D12VAFrame *)f->data[0];
    if (slots_setup(d, df->texture, df->subresource_index) < 0) return -1;
    Slot *s = &d->slot[d->next];
    fire(d, s);                                 /* free it (pipelined: picture N-2) */
    D3D12_RESOURCE_DESC td = ID3D12Resource_GetDesc(df->texture);
    ID3D12CommandAllocator_Reset(s->alloc);
    ID3D12GraphicsCommandList_Reset(s->list, s->alloc, NULL);
    for (int p = 0; p < 2; p++) {
        D3D12_TEXTURE_COPY_LOCATION src = { .pResource = df->texture,
            .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            .SubresourceIndex = (UINT)df->subresource_index + (UINT)p * td.DepthOrArraySize * td.MipLevels };
        D3D12_TEXTURE_COPY_LOCATION dst = { .pResource = s->buf,
            .Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, .PlacedFootprint = d->fp[p] };
        ID3D12GraphicsCommandList_CopyTextureRegion(s->list, &dst, 0, 0, 0, &src, NULL);
    }
    ID3D12GraphicsCommandList_Close(s->list);
    ID3D12CommandQueue_Wait(d->q, df->sync_ctx.fence, df->sync_ctx.fence_value);
    ID3D12CommandList *l = (ID3D12CommandList *)s->list;
    ID3D12CommandQueue_ExecuteCommandLists(d->q, 1, &l);
    ID3D12CommandQueue_Signal(d->q, d->fence, ++d->fv);
    s->fv = d->fv;
    o.y = s->mem + d->fp[0].Offset;
    o.y_pitch = d->fp[0].Footprint.RowPitch;
    o.uv = s->mem + d->fp[1].Offset;
    o.uv_pitch = d->fp[1].Footprint.RowPitch;
    s->out = o;
    s->busy = 1;
    d->next = (d->next + 1) % NSLOT;
    if (d->pipelined) fire(d, &d->slot[d->next]);   /* picture N-1, copied while N decoded */
    else fire(d, s);
    return 0;
}

VtpvDx *vtpv_dx_open(const char *gpu, const char *api, VtpvDxCodec codec,
                     const uint8_t *cfg, uint32_t cfg_len, VtpvDxOut out, void *opaque)
{
    static const enum AVCodecID ids[] = { AV_CODEC_ID_H264, AV_CODEC_ID_HEVC, AV_CODEC_ID_VP9, AV_CODEC_ID_AV1 };
    VtpvDx *d = calloc(1, sizeof(*d));
    if (!d) return NULL;
    QueryPerformanceFrequency(&d->qf);
    d->out = out;
    d->opaque = opaque;
    d->d3d12 = !(api && strcmp(api, "d3d11") == 0);
    d->hook = codec == VTPV_DX_H264 || codec == VTPV_DX_HEVC;
    d->hwfmt = d->d3d12 ? AV_PIX_FMT_D3D12 : AV_PIX_FMT_D3D11;
    int ai = adapter_for(gpu && strcmp(gpu, "nv") == 0 ? 0x10de : 0x8086);
    if (ai < 0) { fprintf(stderr, "vtpv-dx: no %s adapter\n", gpu ? gpu : "int"); goto fail; }
    char devstr[16];
    snprintf(devstr, sizeof devstr, "%d", ai);
    if (av_hwdevice_ctx_create(&d->hw, d->d3d12 ? AV_HWDEVICE_TYPE_D3D12VA : AV_HWDEVICE_TYPE_D3D11VA,
                               devstr, NULL, 0) < 0) {
        fprintf(stderr, "vtpv-dx: %s device on adapter %d failed\n", d->d3d12 ? "d3d12" : "d3d11", ai);
        goto fail;
    }
    const AVCodec *dec = avcodec_find_decoder_by_name(avcodec_get_name(ids[codec]));
    if (!dec) goto fail;
    d->c = avcodec_alloc_context3(dec);
    d->c->opaque = d;
    d->c->hw_device_ctx = av_buffer_ref(d->hw);
    d->c->get_format = get_fmt;
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
    vtpv_dx_close(d);
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
    }
    if (d->q) ID3D12CommandQueue_Release(d->q);
    if (d->fence) ID3D12Fence_Release(d->fence);
    if (d->ev) CloseHandle(d->ev);
    av_frame_free(&d->fr);
    av_frame_free(&d->sw);
    av_packet_free(&d->pkt);
    avcodec_free_context(&d->c);
    av_buffer_unref(&d->hw);
    free(d);
}
