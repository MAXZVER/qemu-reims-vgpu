/*
 * TEMP (lab): apple-vtpv-dx.h - Intel (or any DXGI adapter) video decode backend for the Apple
 * VTPV device: libavcodec parses, D3D12 Video (DXVA) decodes on the chosen
 * adapter, the adapter's own copy engine writes NV12 into plain process memory.
 *
 * Codecs: H.264 (incl. multi-slice), HEVC, VP9, AV1 - all bit-exact against
 * software on Intel UHD 770 (lab journal 2026-10-22-opus-video-intel.md).
 *
 * Output order is DECODE order, one callback per decoded picture, called
 * synchronously from vtpv_dx_decode() - the VideoToolbox contract (VT does not
 * reorder unless the client asks for temporal processing).
 */
#ifndef HW_MISC_APPLE_VTPV_DX_H
#define HW_MISC_APPLE_VTPV_DX_H

typedef struct VtpvDx VtpvDx;

typedef enum {
    VTPV_DX_H264,   /* sample: 4-byte length-prefixed NALs; cfg: avcC record */
    VTPV_DX_HEVC,   /* sample: length-prefixed NALs; cfg: hvcC record */
    VTPV_DX_VP9,    /* sample: one VP9 frame (or superframe); cfg: vpcC or NULL */
    VTPV_DX_AV1,    /* sample: AV1 OBUs (low-overhead format); cfg: av1C or NULL */
} VtpvDxCodec;

typedef struct {
    const uint8_t *y;     /* luma plane, valid only during the callback */
    const uint8_t *uv;    /* interleaved CbCr plane; the Cb plane when planar */
    uint32_t y_pitch, uv_pitch;
    uint32_t width, height;   /* display size (cropped) */
    int64_t tag;          /* the tag passed with the sample that produced it */
    int planar;           /* uv = Cb plane and v = Cr plane (else uv interleaves CbCr) */
    const uint8_t *v;
    uint32_t v_pitch;
    int sx, sy;           /* chroma subsampling shifts: 1,1 = 4:2:0; 1,0 = 4:2:2; 0,0 = 4:4:4 */
    int gray;             /* luma only (monochrome HEVC: a HEIF alpha plane); uv, v unused */
    int wide;             /* 16-bit samples, data in the high bits (P010 pass-through, 'x420') */
} VtpvDxFrame;

typedef void (*VtpvDxOut)(void *opaque, const VtpvDxFrame *f);

/*
 * gpu: "int" (Intel, default) or "nv"; api: "d3d12" (default), "d3d11"
 * (d3d11 = libavcodec staging readback with a CPU copy) or "cuda" (NVDEC on the
 * NVIDIA GPU, libavcodec readback; the only one of the three libavcodec offers
 * HEVC 4:4:4 on - HEIF images with full chroma) or "sw" (libavcodec on the
 * host CPU: the last resort for streams no GPU here decodes - HEVC 4:2:2, the
 * Tahoe desktop picture; RTX 40 NVDEC has no 4:2:2 and libavcodec offers no
 * D3D path for HEVC range extensions).
 * Returns NULL and prints the reason on failure.
 */
VtpvDx *vtpv_dx_open(const char *gpu, const char *api, VtpvDxCodec codec,
                     const uint8_t *cfg, uint32_t cfg_len,
                     VtpvDxOut out, void *opaque);

/* Decode one sample; calls out() for each picture it decoded (normally one,
 * none for a VP9/AV1 sample that only carries a hidden frame).  0 = ok. */
int vtpv_dx_decode(VtpvDx *d, const uint8_t *sample, uint32_t len, int64_t tag);

/* Pipelined mode (VT asynchronous decompression): the copy of picture N runs on
 * the GPU while picture N+1 is parsed and decoded; N's callback fires from the
 * next vtpv_dx_decode() or from vtpv_dx_flush().  Off by default (synchronous:
 * every callback before vtpv_dx_decode() returns). */
void vtpv_dx_set_pipelined(VtpvDx *d, int on);

/* 0003 (Task 18): keep P010 as 16-bit samples instead of the GPU NV12
 * conversion - for 10-bit video sessions whose pool buffer is 'x420'. */
void vtpv_dx_set_wide(VtpvDx *d, int on);

/* Emit every picture still in flight (VTDecompressionSessionWaitForAsynchronousFrames
 * / FinishDelayedFrames). */
void vtpv_dx_flush(VtpvDx *d);

/* Average milliseconds per picture: CPU time in libavcodec, GPU wait+copy */
void vtpv_dx_stats(const VtpvDx *d, double *ms_parse, double *ms_gpu, int *pictures);

/* The session is done with the decoder.  A GPU decoder goes to a small pool
 * and the next vtpv_dx_open() of the same api, GPU, codec and configuration
 * record takes it back flushed (every HEIF tile comes in its own decoder
 * session); pictures still in flight are dropped, the callback never runs
 * again for this caller. */
void vtpv_dx_close(VtpvDx *d);

/* Close for good: a decoder that refused its stream. */
void vtpv_dx_discard(VtpvDx *d);

/* TEMP lab (Zcode 29-10 task61): the handle census of this backend - every
 * kernel object it creates (CreateEventA / CreateThread /
 * OpenExistingHeapFromAddress), so a growing QEMU HandleCount is attributed
 * to it or it is acquitted from the log alone.  The audit that motivated
 * it: dx_destroy/ka_stop close everything they made and the decoder pool is
 * bounded; these counters say it live. */
typedef struct VtpvDxHandles {
    int open;       /* live decoders (created - destroyed) */
    int pooled;     /* of those, sitting in the reuse pool */
    int events;     /* live CreateEventA events */
    int heaps;      /* live OpenExistingHeapFromAddress heaps */
    int threads;    /* live keep-alive threads */
} VtpvDxHandles;

void vtpv_dx_handles(VtpvDxHandles *h);

/* TEMP (lab, Zcode 10-30 task77): D3D12 Video Encode backend
 * (apple-vtpv-dx-enc.c).  H.264 first, NV12 in host memory in, Annex-B out;
 * GOP 1 so every sample is a self-contained SPS/PPS+IDR.  gpu: "nv" (default,
 * the RTX 4060) or "int".  Any failure -> NULL / nonzero, the caller keeps
 * refusing encoder requests - never a hang.  Validate standalone with
 * lab/tools/session-2026-10-02/hwsurvey/dxeprobe.c before wiring it in. */
typedef struct VtpvDxE VtpvDxE;

VtpvDxE *vtpv_dxe_open(const char *gpu, uint32_t w, uint32_t h,
                       uint32_t fps, uint32_t bitrate);

/* encode one NV12 frame; out receives Annex-B bytes (SPS/PPS on IDR frames);
 * 0 = ok. */
int vtpv_dxe_encode(VtpvDxE *e, const uint8_t *y, uint32_t y_stride,
                    const uint8_t *uv, uint32_t uv_stride, int force_key,
                    uint8_t *out, uint32_t out_cap, uint32_t *out_len);

void vtpv_dxe_close(VtpvDxE *e);

#endif /* HW_MISC_APPLE_VTPV_DX_H */
