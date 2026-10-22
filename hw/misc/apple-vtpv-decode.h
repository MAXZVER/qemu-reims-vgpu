/*
 * apple-vtpv-decode.h - hardware H.264 decode for the Apple VTPV device.
 * Self-contained module (runtime-loads vulkan-1.dll); see .c for the
 * verified pipeline recipe.
 */
#ifndef APPLE_VTPV_DECODE_H
#define APPLE_VTPV_DECODE_H

#include <stdint.h>
#include <stdbool.h>

typedef struct VtpvDecoder VtpvDecoder;

/*
 * Open a decoder for one stream shape.
 *   gpu_pref: "nv" (discrete, first non-Intel) or "int" (first Intel);
 *             NULL = "nv".
 *   sps_nal/pps_nal: NAL units WITH their 1-byte header, WITHOUT start
 *   codes (as split from the Annex-B sample buffer).
 * Returns NULL and logs the reason on failure.
 */
VtpvDecoder *vtpv_dec_open(const char *gpu_pref,
                           const uint8_t *sps_nal, uint32_t sps_len,
                           const uint8_t *pps_nal, uint32_t pps_len);

/*
 * Decode one IDR frame.
 *   frame: the Annex-B slice section (start codes + NALs, exactly the
 *   bytes the guest put in the sample buffer after SPS/PPS).
 *   slice_off: byte offsets of each slice's start code within frame.
 * Returns 0 and fills nv12_out (w*h Y + w*h/2 interleaved UV) on success.
 */
int vtpv_dec_frame(VtpvDecoder *d, const uint8_t *frame, uint32_t frame_len,
                   const uint32_t *slice_off, uint32_t nsl,
                   uint8_t *nv12_out);

bool vtpv_dec_match(const VtpvDecoder *d,
                    const uint8_t *sps_nal, uint32_t sps_len,
                    const uint8_t *pps_nal, uint32_t pps_len);
uint32_t vtpv_dec_width(const VtpvDecoder *d);
uint32_t vtpv_dec_height(const VtpvDecoder *d);
bool vtpv_dec_has_chroma(const VtpvDecoder *d);  /* plane-1 copy worked? */

void vtpv_dec_close(VtpvDecoder *d);

/* HEVC path (apple-vtpv-decode-h265.c) - same contract, adds VPS */
VtpvDecoder *vtpv_dec265_open(const char *gpu_pref,
                              const uint8_t *vps_nal, uint32_t vps_len,
                              const uint8_t *sps_nal, uint32_t sps_len,
                              const uint8_t *pps_nal, uint32_t pps_len);
int vtpv_dec265_frame(VtpvDecoder *d, const uint8_t *frame, uint32_t frame_len,
                      const uint32_t *slice_off, uint32_t nsl,
                      uint8_t *nv12_out);
bool vtpv_dec265_match(const VtpvDecoder *d,
                       const uint8_t *vps, uint32_t vps_len,
                       const uint8_t *sps, uint32_t sps_len,
                       const uint8_t *pps, uint32_t pps_len);
uint32_t vtpv_dec265_width(const VtpvDecoder *d);
uint32_t vtpv_dec265_height(const VtpvDecoder *d);
void vtpv_dec265_close(VtpvDecoder *d);

#endif /* APPLE_VTPV_DECODE_H */
