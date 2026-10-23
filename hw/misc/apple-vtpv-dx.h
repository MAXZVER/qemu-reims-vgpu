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
    const uint8_t *uv;    /* interleaved CbCr plane */
    uint32_t y_pitch, uv_pitch;
    uint32_t width, height;   /* display size (cropped) */
    int64_t tag;          /* the tag passed with the sample that produced it */
} VtpvDxFrame;

typedef void (*VtpvDxOut)(void *opaque, const VtpvDxFrame *f);

/*
 * gpu: "int" (Intel, default) or "nv"; api: "d3d12" (default) or "d3d11"
 * (d3d11 = libavcodec staging readback with a CPU copy).
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

/* Emit every picture still in flight (VTDecompressionSessionWaitForAsynchronousFrames
 * / FinishDelayedFrames). */
void vtpv_dx_flush(VtpvDx *d);

/* Average milliseconds per picture: CPU time in libavcodec, GPU wait+copy */
void vtpv_dx_stats(const VtpvDx *d, double *ms_parse, double *ms_gpu, int *pictures);

void vtpv_dx_close(VtpvDx *d);

#endif /* HW_MISC_APPLE_VTPV_DX_H */
