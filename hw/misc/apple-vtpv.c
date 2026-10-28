/*
 * TEMP (lab): Apple paravirtual VideoToolbox device v2 (PCI 106b:1a03).
 *
 * The x86 Tahoe guest ships the VideoToolboxParavirtualizationSupport user
 * framework but no kernel binary for AppleVideoToolboxParavirtualization.kext,
 * so nothing ever binds 106b:1a03.  This device is the host side of our own
 * replacement kext (com.reims.vtpv): a plain register-block transport over
 * BAR0.  The kext publishes IOService class
 * "AppleVideoToolboxParavirtualizationDriver" with the user-client contract the
 * framework expects (selector 0 buffers, 1/2 UUID, 3 send, 4 re-arm) and talks
 * to these registers.
 *
 * Transport: the kext allocates the guest->host message buffer and the
 * host->guest IODataQueue memory (both physically contiguous, DMA-mapped) and
 * writes their GPAs here.  A message send is MSG_SIZE/MSG_NSURF/SURF0..4 +
 * MSG_KICK; this device DMAs the message out of the g2h buffer, parses the
 * 'para' atom header, logs everything (msglog= file) and hands the bytes to
 * the handler.  Host->guest replies are appended into the slot's IODataQueue
 * memory (xnu layout, single producer) followed by an INTx kick; the kext
 * mach-notifies the client port and the framework dequeues.
 *
 * Multiple processes open separate user clients; each gets a slot (CLIENT_ID)
 * so its h2g queue is not cross-dequeued.  Only present with
 * -device apple-vtpv-pci.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/lab-elog.h"
#include "hw/misc/apple-vtpv-decode.h"
#include "hw/misc/apple-vtpv-dx.h"
#include "hw/misc/apple-vtpv-replies.h"
#include "qemu/module.h"
#include "qapi/error.h"
#include "hw/pci/pci_device.h"
#include "hw/pci/msi.h"
#include "hw/pci/pci_ids.h"
#include "hw/core/qdev-properties.h"
#include "qemu/timer.h"
#include "qemu/thread.h"
#include "system/whpx-wp.h"
#include "hw/display/reims-vgpu-dirty.h"
#include <zlib.h>

#define TYPE_APPLE_VTPV "apple-vtpv-pci"
OBJECT_DECLARE_SIMPLE_TYPE(AppleVtpv, APPLE_VTPV)

/* what the dx decoder's callbacks wrote during one message */
typedef struct {
    uint64_t written;
    uint32_t w, h;
    int pictures;
    bool to_stage;          /* outbuf: keep the picture (tight NV12) in stage */
    uint8_t csx, csy;       /* the staged picture's chroma shifts (1,1 = NV12) */
    bool cgray;             /* the staged picture is luma only */
    uint8_t *stage;
    size_t stage_cap;
} VtpvDxSink;

/* one IOSurface attached to the message being handled, as the kext
 * described it (VtpvSurfDesc, vtpv.cpp): pages cover the surface's range in
 * its memory descriptor, org = in-page offset of its first byte, plane
 * offsets count from that byte.  Valid only during that kick - the kext
 * unpins the surface right after it. */
typedef struct {
    uint32_t id, pixfmt, width, height, planes, alloc;
    uint32_t off[2], bpr[2], pw[2], ph[2];
    uint32_t npages, org;
    uint64_t *pages;
} VtpvKickSurf;

/* frames waiting for their 'Dbuf' reply.  A player runs well ahead of its
 * pool: while the guest waits for displayed buffers to come back it keeps
 * sending 'dfrm' (ox13: the wallpaper's Dbuf replies stalled for a minute at
 * login).  Samples are small (4K HEVC P-frames ~1.2 KB), so the table is
 * large; a drop completes the frame with an error the player sees. */
#define VTPV_FRAMES 1024
/* uuids alive at once: decoder sessions (wallpaper, Safari tabs...) and the
 * per-process client uuids that only ask capabilities (ox17: 16 overflowed) */
#define VTPV_SESS 64
#define VTPV_DBUF_INFLIGHT 4 /* 'Dbuf' requests outstanding per session */

/* one guest decoder session, keyed by the uuid every message carries at
 * 0x10: its own decoder, stream shape and frames.  ox9: one shared table let
 * the 4K HEVC wallpaper session starve Safari's ("frame table full" from
 * YouTube's frame 3), and asking a 'Dbuf' per 'dfrm' at once overran the
 * 16 KiB host->guest queue - the player queues a GOP ahead of display and
 * the guest answers 'Dbuf's one by one.  Frames now wait here and only
 * VTPV_DBUF_INFLIGHT requests are in the queue at any time. */
typedef struct VtpvSess {
    bool used;
    uint8_t uuid[16];
    uint64_t last_use;
    struct AppleVtpv *owner;
    struct {
        uint64_t frame_id;
        uint64_t seq;               /* arrival order = decode order */
        uint8_t dbuf_id[8];         /* raw reply-id bytes of our 'Dbuf' */
        bool in_use;
        bool dbuf_sent;
        bool queued;                /* handed to the decode thread (dec_lock) */
        bool decoded;               /* the decode thread is done: nv12 or nothing */
        bool shed;                  /* backlog: decoded for its references only, completed dropped */
        bool dno;                   /* TEMP lab dno: the player asked kVTDecodeFrame_DoNotOutputFrame */
        bool pace_drop;             /* TEMP lab dec-pace: decode for references, complete dropped */
        uint8_t *nv12;              /* outbuf: decoded picture waiting for its buffer */
        uint32_t w, h;
        uint8_t *sample;            /* outbuf: or the compressed sample, decoded at the reply */
        uint32_t sample_len;
    } frame[VTPV_FRAMES];
    uint64_t next_seq;
    uint32_t dbuf_out;              /* 'Dbuf's sent and not answered yet */
    uint32_t shed_n;                /* frames shed and not completed yet */
    uint32_t pace_warm, pace_cold; /* diag: recycled vs fresh answer surfaces */
    uint64_t pace_seen[256];
    uint32_t pace_seen_n, pace_seen_i;
    /* dbuf-gate (TEMP lab): the hidden-layer signature - the layer stops
     * returning displayed buffers, the pool answers every 'Dbuf' with a
     * NEVER-SEEN surface (cold), forever.  Gate: hold new 'Dbuf's (one probe
     * out at a time, DBUF_GATE_PROBE_US apart) until a probe's answer carries
     * a RECYCLED surface (the layer consumes again = visible) or the session
     * goes quiet.  The held-back 'Dbuf's are the player's own back-pressure:
     * its buffer acquisition blocks, the pipeline pauses - what a real Mac's
     * display link does to a hidden layer. */
    int slot;                       /* the slot its 'Dbuf's go out on */
    bool dpba_sent;
    uint32_t dpba_w, dpba_h;        /* size the last 'Dpba' announced */
    VtpvDx *dx;
    int dx_codec;                   /* VtpvDxCodec of the open decoder, -1 none */
    bool tile_live;                 /* its 'dsts' succeeded: one HW tile session at a time (24c) */
    uint8_t dx_cfg[4096];           /* avcC/hvcC/vpcC/av1C record it was opened with */
    uint32_t dx_cfg_len;
    uint8_t dx_api;                 /* VTPV_API_*: the decoder this session moved to */
    VtpvDxSink dx_sink;
} VtpvSess;

/* the decoders a session tries, in order: the GPU's D3D12 Video (Intel by
 * default: H.264/HEVC/VP9/AV1, HEVC 4:2:2 and 4:4:4 through the range-extension
 * profiles), NVDEC on the NVIDIA GPU (what D3D12 there refuses), and libavcodec
 * on the host CPU only for a stream no GPU here decodes (12-bit HEVC) - never
 * an error the guest would show as a black or green picture */
enum { VTPV_API_D3D12, VTPV_API_CUDA, VTPV_API_SW, VTPV_API_NONE };
static const char *const vtpv_api_name[] = { "d3d12", "cuda", "sw" };

#define PCI_DEVICE_ID_APPLE_VTPV 0x1a03

/* register offsets (BAR0, u32 slots; 64-bit values take two slots) */
#define VTPV_MAGIC      0x00    /* RO  'VTP1' */
#define VTPV_VERSION    0x04    /* RO  2 */
#define VTPV_CLIENT_ID  0x08    /* RW  slot selector for the regs below */
#define VTPV_G2H_GPA    0x10    /* RW  u64 */
#define VTPV_G2H_LEN    0x18    /* RW  u64 */
#define VTPV_H2G_GPA    0x20    /* RW  u64 */
#define VTPV_H2G_LEN    0x28    /* RW  u64 */
#define VTPV_UUID_GPA   0x30    /* RW  u64 */
#define VTPV_UUID_OP    0x38    /* W1  1=register 2=deregister */
#define VTPV_MSG_SIZE   0x40    /* W1  message bytes in g2h buffer */
#define VTPV_MSG_NSURF  0x44    /* W1  number of surface ids */
#define VTPV_SURF0      0x48    /* W1  x5 */
#define VTPV_MSG_KICK   0x5c    /* W1  process the message */
#define VTPV_IRQ_PEND   0x60    /* RO  bitmask of slots with pending kick */
#define VTPV_IRQ_ACK    0x64    /* W1  clear the interrupt */
#define VTPV_CAPS       0x68    /* RO  host codec capabilities */
#define VTPV_MSGS_IN    0x6c    /* RO  messages received */
#define VTPV_MSGS_OUT   0x70    /* RO  replies enqueued */
#define VTPV_SELFTEST   0x78    /* W1  value=size: canned h2g message + kick */
#define VTPV_DEBUGVAL   0x7c    /* RW  scratch */
#define VTPV_SURFMAP_COUNT 0x80 /* W1  pages in the exported surface */
#define VTPV_SURFMAP_KICK  0x84 /* W1  latch marker (bumped per update) */
#define VTPV_SURFMAP_GPA0  0x90 /* W1  u64 x16: guest-physical pages */
#define VTPV_SURFMAP_META  0x110/* W1  u32 x4: w<<16|h, bpr0, plane1_off, bpr1 */
#define VTPV_SURFMAP_LIST  0x120/* W1  u64: GPA of a little-endian u64 page list
                                 * (SURFMAP_COUNT entries, read at SURFMAP_KICK);
                                 * 0 = the 16 direct GPA0 registers.  Older
                                 * devices ignore the write, so a kext can
                                 * always program both. */
#define VTPV_SURFDESC   0x128   /* W1  u64: GPA of the VtpvSurfDesc describing
                                 * the surfaces of the next kick (0 = none) */
#define VTPV_SURFDESC_MAGIC 0x31445356u /* 'VSD1' */
#define VTPV_KICKDESC   0x130   /* W1  u64 (v3): GPA of the VtpvKickDesc - one
                                 * MMIO write for a send's whole register
                                 * sequence (slot, surf[], total, surfdesc) */
#define VTPV_KICKDESC_MAGIC 0x31494b56u /* 'VKI1' */
#define VTPV_KICK_HOLD  0x138   /* W1  u64: GPA of the sender's u32 pair {token,
                                 * done}: the kext keeps this kick's surfaces
                                 * pinned when the device writes a nonzero token,
                                 * and unpins them once done (written later, in
                                 * token order per slot) reaches it.  0 / never
                                 * written = the surfaces live for the kick only */
#define VTPV_REG_SIZE   0x1000

#define VTPV_MAGIC_VAL  0x31505456  /* "VTP1" */
#define VTPV_MAX_SLOT   16
/* per-VM cap on one message; the guest's h2g dequeue buffer is 0x4000 */
#define VTPV_MSG_MAX    (4 * 1024 * 1024)
#define VTPV_H2G_ENTRY_MAX 0x3f00

typedef struct AppleVtpvSlot {
    uint64_t g2h_gpa, g2h_len;
    uint64_t h2g_gpa, h2g_len;
    uint64_t uuid_gpa;
    uint32_t surf[5];
    uint32_t last_kick;
    /* host->guest messages the guest's queue had no room for, in order.
     * Apple's host sits on virtio and just keeps them; so do we - the guest
     * stops draining while its pool waits for displayed buffers (ox13) and
     * a lost 'dfrm' reply or 'Demt' would strand its session. */
    GQueue outbox;
    /* the guest's pump drains on a notification and stops at a message it
     * cannot dispatch; what is left waits for the next one - re-notify a
     * queue whose head stood still (vtpv_outbox_tick) */
    uint32_t seen_head;
    int64_t seen_ms;
    int stall_tries;
    int64_t stuck_since_ms;        /* the head has not moved since (a wedge) */
} AppleVtpvSlot;

struct AppleVtpv {
    PCIDevice parent_obj;
    MemoryRegion mmio;

    AppleVtpvSlot slot[VTPV_MAX_SLOT];
    uint32_t client_id;

    uint64_t msgs_in, msgs_out;
    uint32_t irq_pending;           /* bitmask */
    char *msglog;
    char *reply_dcap;               /* stage-2: file whose bytes answer 'decs' */
    char *reply_plusdec;            /* stage-2: file whose bytes answer '+dec' */
    char *reply_generic;            /* stage-3: ',ver'+',err' template for any other sync
                                     * query (dsta/dinv/...); the request's message type
                                     * is patched into the reply header */
    char *reply_dsup;               /* stage-3: 'dsup' also wants ',sup' (CFDictionary) */
    char *dpba_file;                /* stage-4: plist with pixel buffer attributes sent
                                   * as 'Dpba' after StartSession so the guest session
                                   * builds its output pool (nA31: CreatePixelBuffer
                                   *WithOptions err 3840 without it) */
    uint32_t verbose;
    uint32_t debugval;              /* RW scratch (kext MSI-state telemetry) */

    /* scratch for the current message */
    uint8_t *msg;
    size_t msg_cap;

    /* stage-4: non-tile decode completion.  The guest's first-fragment 'dfrm'
     * (',frm'+',dfl'+',fop'+',sbf') is ASYNC - no sync reply exists.  The real
     * host pulls an output pixel buffer from the guest ('Dbuf' -> reply with
     * ',err'+',uui') and then completes the frame ('Demt' ',frm'+',err'+',inf'+
     * ',uui' -> VTDecoderSessionEmitDecodedFrame).  Without 'Demt' the guest's
     * VTDecompressionSessionWaitForAsynchronousFrames waits forever. */
    VtpvSess sess[VTPV_SESS];
    VtpvSess *cur;                  /* session of the message being handled */
    QEMUTimer *outbox_timer;
    uint64_t sess_clock;
    /* decode thread: a frame is decoded as soon as its 'Dbuf' goes out, so
     * the guest's answer only waits for the copy into its buffer.  Decoding
     * inside the kick held the vCPU ~10 ms per 4K picture and, worse, the
     * guest's message thread: its 'dfrm' replies queued behind our work and
     * every one missed the reply clerk's timeout (ox11: 13867 timeouts). */
    QemuThread dec_thread;
    QemuMutex dec_lock;
    QemuCond dec_cond;
    struct {
        VtpvSess *se;
        int idx;
    } dec_job[VTPV_SESS * VTPV_DBUF_INFLIGHT];
    uint32_t dec_head, dec_n;
    VtpvSess *dec_busy;             /* session the thread is decoding for */
    const uint8_t *ob_sample;       /* this dfrm's sample (inside msg) for vtpv_send_dbuf */
    uint32_t ob_sample_len;
    bool ob_first;                  /* this dfrm's picture was decoded in place (stream start) */
    uint64_t host_msg_id;
    uint64_t h2g_renotify;          /* stalled host->guest queues notified again */
    int64_t kick_open_us;           /* CLIENT_ID write that opened the kext's register sequence */
    /* 'TDem's waiting tdem_delay_us after their 'dtil' reply (vtpv_tdem_*) */
    QEMUTimer *tdem_timer;
    GQueue tdem_q;
    uint32_t tdem_delay_us;
    /* ack-gated 'TDem' window: the guest runs replies and our messages on one
     * serial dispatch queue and a tile handler holds it for 10-70 ms, so a
     * burst's 'TDem's stack the next sessions' 'dtil' replies behind them -
     * past the reply clerk's 1.1 s, whose timeout makes the guest self-emit
     * the tile with the error, drop the image and free the state its sibling
     * sessions' still-queued handlers then crash on (Maps/TV/Weather/Podcasts).
     * At most WINDOW 'TDem's per slot are unanswered; a 'TDem' ack (flags 8)
     * comes after EmitDecodedTile - the handler included - returned, so a
     * reply never waits behind more than WINDOW handlers.  The real host gets
     * this spread for free: its emits follow the actual GPU decode. */
    uint32_t tdem_unacked[VTPV_MAX_SLOT];
    uint32_t tdem_window;
    bool tdem_fast;               /* tdem-fast=on: pace by h2g ring free space, no fixed holds */
    bool tile_uui;
    bool canvas_probe;
    bool dbuf_gate;                /* TEMP lab: gate 'Dbuf's on the hidden-layer cold-surface signature */
    bool dbuf_gate_live;           /* TEMP lab: dbuf-gate=live - the flag file is the switch */
    /* dbuf-gate state lives on the DEVICE: the hidden layer's sessions churn
     * (dsts/dinv cycles restart a fresh decoder every few seconds) and the
     * leak is continuous across them - a per-session window can never fill */
    bool gate_on;
    int64_t gate_probe_us;
    uint32_t gate_cold_win;
    int64_t gate_win_us;
    uint32_t gate_cold_prev;
    uint64_t gate_seen[64];
    uint32_t gate_seen_n, gate_seen_i;
    /* dec-pace (TEMP lab): the pool is the clock - a 'Dbuf' ask costs one
     * credit, a warm (recycled-surface) answer earns one back; cold answers
     * (the pool growing = decode-ahead the display never consumes) earn
     * nothing, so a session runs ahead of its consumer by VTPV_PACE_CREDIT
     * frames at most - what a real Mac's full VT output queue does. */
    bool dec_pace;
    bool dec_pace_live;
    int64_t pace_live_check_us;
    QEMUTimer *pace_timer;
    uint8_t pace_uuid[16];
    uint32_t pace_holds;
    int64_t pace_ivl_us;           /* dec-pace ask interval (REIMS_VGPU_LAB_DEC_PACE_US) */
    bool pace_drop_next;           /* dec-pace v3: the next allocated frame is decode-only */
    uint64_t stat_dfrm, stat_ask, stat_drop, stat_dec;
    uint64_t stat_send, stat_evals, stat_done, stat_tidskip, stat_shed;
    /* dec-tid-max (TEMP lab, Opus 10-28): the Aerial clips are dyadic
     * HEVC temporal layers - tid>=3 frames are droppable without corrupting
     * the tid<=2 sublayer (an honest 60 fps).  -1 = decode everything. */
    int32_t dec_tid_max;
    bool dec_tid_live;
    int64_t tid_live_check_us;
    struct {                /* dec-pace v6: completions beyond the budget wait here */
        bool used;
        int slot;
        uint64_t frame_id;
        int32_t err;
        uint32_t inf;
        uint8_t uuid16[16];
        uint8_t uuid[16];
    } pace_pend[8];
    struct {                /* dec-pace v7: a dfrm REPLY held until a token - the
                             * reply clerk is what DecodeFrame waits on, so this
                             * is the real "decoder busy"; completions stay free */
        bool used;
        int slot;
        uint32_t len;
        uint8_t data[160];
    } pace_rep[4];
    int64_t pace_toks_us, pace_last_us;   /* dec-pace v4: DEVICE-level bucket - the
                                           * wallpaper decodes through >=2 alternating
                                           * sessions (ae17: adjacent diag lines carry
                                           * different warm counters), so a per-session
                                           * bucket multiplies the budget and never binds */
    int64_t gate_live_check_us;
    int64_t gate_diag_us;                /* tile-uui=off: 'TDem' carries no ',uui' - the guest emits its own
                                   * canvas (rec[+0x20], retained at DecodeTile) instead of resolving
                                   * our echoed UUID in its registry; immune to late-emit staleness */
    int64_t tdem_last_us;
    /* RE 25x (Ghidra, guest VideoToolbox + VideoToolboxParavirtualizationSupport)
     * + rfq1/rfq3 matrix: the guest takes TWO reply paths.  The plain sync sends
     * (decs/+dec/dsts/dsup, the Dbuf request) complete their wait with our old
     * flags=4 replies (Opus: CreateInstance ok every run) - but the 'dtil'
     * reply must be delivered by the pump through DeliverReply, which the
     * dispatchers reach only on flags bit 3 (0x8; the guest's own 'TDem' acks
     * carry 0x8 on the wire).  With 4 the 'dtil' reply was routed to the event
     * handler, dropped as an unknown 4CC and the clerk timed out (-19093 from
     * DeliverReply's lookup) -> CMPhoto fell back to CPU decode.  flags=8
     * EVERYWHERE is worse: the 'decs' handshake breaks, apps never open the HW
     * path at all and the "clean" logs are the same CPU fallback one level
     * earlier (rfq3 f8b: zero dtil in the msglog).  So: 8 on the 'dtil' reply
     * only, 4 everywhere else (Opus's split, COORD 23:1x).  Property for A/B. */
    uint32_t dtil_reply_flags;
    uint32_t dtil_reply_delay_us;
    int64_t reply_due_us[VTPV_MAX_SLOT];
    /* tdem-kick-gated: a slot's 'TDem' waits for that client's NEXT kick -
     * proof its previous DecodeTile caller returned and armed CMPhoto's
     * per-tile state (the taint happens-before gap; 8/40 ms lags did not
     * close it).  Cap so a stream's last tile still leaves. */
    bool tdem_kick_gated;
    uint64_t kick_count[VTPV_MAX_SLOT];
    uint64_t tdem_gate_cap_us;
    uint32_t tile_dump_left;       /* tile-dump=N: dump the first N fills */
    uint32_t tile_dump_total;
    bool tile_dump_read;           /* tile-dump-read=on: canvas readback too (raced
                                    * the memory layer twice - default off) */
    /* 24c: one live HW tile session at a time - the parallel tile sessions one
     * image runs (Maps/TV/Weather/Podcasts) crashed CMPhoto's tile handler on
     * every build through 24b; while they are parallel, a second 'dsts' fails
     * and ImageIO decodes that image's remaining tiles itself */
    uint32_t tile_sessions;
    bool tiles_one_session;         /* diagnostic: one live HW tile session */
    /* outbuf=on: the app's own output buffers - each decoded picture is held
     * until the guest answers our 'Dbuf' with a pool buffer (its IOSurface
     * attached), then written into that surface and completed by 'Demt' */
    bool outbuf;
    bool decthread;                 /* decode ahead on the decode thread (A/B: off = at the reply) */
    uint32_t backlog;               /* frames waiting for a 'Dbuf' past which the oldest are shed (0: never) */
    uint64_t shed_total;
    /* TEMP lab (prop dno=on): the 'dfrm' ',dfl' decode flags.  Bit 1 is
     * kVTDecodeFrame_DoNotOutputFrame - a player that needs a frame only as a
     * reference (catching up, seeking) asks for it decoded and NOT output; a
     * real decoder then calls back with no image and kVTDecodeInfo_FrameDropped.
     * This device asked the guest's pool for a buffer ('Dbuf') and emitted a
     * picture for every such frame, so a player behind its clock spent its pool
     * and its time on pictures it had said it would throw away - the live
     * wallpaper's spiral (140+ 'dfrm'/s, 115 % CPU) is the suspect.  With dno
     * on, such a frame is decoded for its references in decode order and
     * completed dropped without a buffer, through the backlog's shed path. */
    bool dno;
    uint32_t cur_dfl;               /* ',dfl' of the 'dfrm' being handled */
    uint32_t dfl_seen[8];           /* distinct ',dfl' values already printed */
    uint32_t dfl_seen_n;
    uint64_t dfl_dno_n;             /* frames that asked DoNotOutputFrame */
    bool tiles;                     /* off: refuse tile sessions ('dsts'), the guest decodes HEIF itself */
    uint64_t surfdesc_gpa;          /* VTPV_SURFDESC of the next kick */
    uint64_t kickdesc_gpa;          /* VTPV_KICKDESC latch (two 32-bit halves) */
    bool kickdesc;                  /* report version 3 + take one-write sends */
    VtpvKickSurf ksurf[5];
    uint32_t nksurf;
    uint64_t last_frame_id;
    bool mock_dbuf;
    bool mock_demt;
    uint8_t mock_uuid[16];
    uint64_t frames_completed;

    /* write-back: pages + layout of the canvas the kext exported (SURFMAP regs) */
    bool writeback;
    bool hwdecode;                 /* ',sbf' -> host Vulkan H.264 decode */
    bool tile_fake_error;          /* DIAG: dtil replies + error TDem without the
                                    * in-kick decode - isolates the kick's hold of
                                    * the guest kext spinlock (clerk timeouts) */
    /* 25p (Opus): a 'dtil' decodes on the tile thread, not in the kick.  The
     * guest's DecodeTile wants its reply a few ms after the send (25oc msglog vs
     * the guest log: Weather's clerk timeouts land ~8 ms after the send, right
     * as our kick - reply enqueued at +1 ms, then a 4-5 ms decode - returned,
     * and every kext access waited out the decode on the BQL): 12 of 12 Weather
     * tile sessions failed (clerk timeout or CMPhoto -17104) and the images
     * were decoded again on the guest CPU.  The kick now returns right after
     * the reply; the kext holds the canvas pins until the device's done word
     * passes the kick's token (VTPV_KICK_HOLD) */
    bool tile_async;
    bool tdem_ver;                 /* A/B: ',ver' in the 'TDem' too */
    bool tile_sync;                /* A/B: decode, then an ',inf' 0 reply, no 'TDem' */
    uint64_t kick_hold_gpa;        /* VTPV_KICK_HOLD of the kick being handled */
    uint32_t tile_token[VTPV_MAX_SLOT];
    QemuThread tile_thread;
    GQueue tile_q;                 /* VtpvTileJob, waiting (under dec_lock) */
    GQueue tile_done;              /* decoded, for vtpv_tile_complete (under dec_lock) */
    struct VtpvTileJob *tile_busy;
    QEMUBH *tile_bh;
    char *dec_gpu_str;             /* "nv" (default) / "int" */
    VtpvDecoder *dec;
    uint8_t avcC_sps[64];          /* SPS/PPS cached from dsts ',fds' avcC */
    uint32_t avcC_sps_len;
    uint8_t avcC_pps[64];
    uint32_t avcC_pps_len;
    int dec_codec;                 /* 0 = h264, 1 = hevc (from fdsc tag) */
    int dec_codec_last;            /* codec of the currently open decoder */
    uint8_t h265_vps[128];
    uint32_t h265_vps_len;
    uint8_t h265_sps[256];
    uint32_t h265_sps_len;
    uint8_t h265_pps[128];
    uint32_t h265_pps_len;
    /* hwbackend=dx (default): libavcodec + D3D12 Video on the Intel iGPU
     * (apple-vtpv-dx.c) - codec and config record straight from the format
     * description, samples unchanged; hwbackend=vk keeps the Vulkan modules */
    char *hwbackend_str;
    uint64_t surfmap_gpa[16];
    uint32_t surfmap_count;
    uint32_t surfmap_kick;
    uint32_t surfmap_meta[4];
    uint64_t surfmap_list_gpa;     /* VTPV_SURFMAP_LIST */
    uint64_t *surfmap_list;        /* pages read from it at the last kick */
    uint32_t surfmap_list_n;
};

/* ------------------------------------------------------------------ */
/* xnu IODataQueue layout (IODataQueueShared.h): the framework's userspace dequeues entries
 * we append into the shared memory.  Single producer (us), single consumer per slot.
 * THE HEADER ORDER IS {queueSize@0, head@4, tail@8} - NOT {head,tail,size} (the 2026-10-21
 * n983 bug: with the wrong order queueSize read as 0 and IODataQueueDequeue failed with
 * kIOReturnError on every wake). */

struct io_dq_header {
    uint32_t queue_size;
    uint32_t head;
    uint32_t tail;
};

static bool vtpv_dq_enqueue(AppleVtpv *s, int slot, const uint8_t *data, uint32_t len)
{
    AppleVtpvSlot *sl = &s->slot[slot];
    struct io_dq_header h;
    /* the consumer advances head by entrySize+4 UNPADDED (IODataQueueLib source);
     * the producer must advance tail identically or a phantom zero-size entry
     * appears between them (n986: head=382 vs tail=380 -> -19093 dispatch) */
    uint32_t entry = 4 + len;
    uint32_t freesz, tail_off;

    if (!sl->h2g_gpa || sl->h2g_len < sizeof(h) + 16 + entry) {
        lab_elog( "apple-vtpv: enq fail: no h2g buffer\n");
        return false;
    }
    if (entry > VTPV_H2G_ENTRY_MAX) {
        lab_elog( "apple-vtpv: enq fail: entry %#x too big\n", entry);
        return false;
    }
    if (dma_memory_read(pci_get_address_space(PCI_DEVICE(s)), sl->h2g_gpa, &h, sizeof(h), MEMTXATTRS_UNSPECIFIED)) {
        lab_elog( "apple-vtpv: enq fail: header read\n");
        return false;
    }
    h.queue_size = le32_to_cpu(h.queue_size);
    h.head = le32_to_cpu(h.head);
    h.tail = le32_to_cpu(h.tail);
    if (h.queue_size == 0 || h.queue_size + sizeof(h) > sl->h2g_len) {
        h.queue_size = sl->h2g_len - sizeof(h);
    }

    /* xnu IODataQueue::enqueue, step for step.  The producer never touches
     * head (the consumer may be reading it).  An entry that does not fit
     * before the end goes to offset 0 - only if it stays below head - and the
     * old tail gets a header whose size is the new entry's: the consumer's
     * IODataQueueDequeue wraps exactly when head + 4 + size passes the end.
     * ox12: a zero-size marker and a head reset made the consumer read a
     * phantom entry and then replay stale ones (the same 'Dbuf' answered
     * twice, 'Demt's for consumed buffers -> -19093). */
    bool wrap = false;
    if (h.tail >= h.head && h.tail <= h.queue_size && entry <= h.queue_size - h.tail) {
        tail_off = h.tail;
    } else if (h.tail >= h.head && h.head > entry) {
        tail_off = 0;
        wrap = true;
    } else if (h.tail < h.head && h.head - h.tail > entry) {
        tail_off = h.tail;
    } else {
        freesz = h.tail >= h.head ? h.head : h.head - h.tail;
        if (s->verbose > 1) {       /* expected: the message waits in the outbox */
            lab_elog( "apple-vtpv: queue full: entry %#x >= free %#x "
                    "(head=%u tail=%u qsize=%u)\n",
                    entry, freesz, h.head, h.tail, h.queue_size);
        }
        return false;
    }
    if (wrap && h.tail <= h.queue_size && h.queue_size - h.tail >= 4) {
        uint32_t le_mark = cpu_to_le32(len);
        if (dma_memory_write(pci_get_address_space(PCI_DEVICE(s)),
                             sl->h2g_gpa + sizeof(h) + h.tail,
                             &le_mark, 4, MEMTXATTRS_UNSPECIFIED)) {
            return false;
        }
    }

    {
        uint32_t le_len = cpu_to_le32(len);
        hwaddr at = sl->h2g_gpa + sizeof(h) + tail_off;
        if (dma_memory_write(pci_get_address_space(PCI_DEVICE(s)), at, &le_len, 4,
                             MEMTXATTRS_UNSPECIFIED) ||
            dma_memory_write(pci_get_address_space(PCI_DEVICE(s)), at + 4, data, len,
                             MEMTXATTRS_UNSPECIFIED)) {
            return false;
        }
    }
    h.tail = tail_off + entry;     /* may equal queue_size: the consumer wraps */
    uint32_t le_tail = cpu_to_le32(h.tail);
    dma_memory_write(pci_get_address_space(PCI_DEVICE(s)), sl->h2g_gpa + 8, &le_tail, 4,
                     MEMTXATTRS_UNSPECIFIED);

    s->msgs_out++;
    return true;
}

/* conservative free-bytes test of the slot's h2g ring (the enqueue fit rules,
 * read-only) - tdem-fast paces its sends by this instead of fixed holds */
static bool vtpv_dq_room(AppleVtpv *s, int slot, uint32_t entry)
{
    AppleVtpvSlot *sl = &s->slot[slot];
    struct io_dq_header h;
    if (!sl->h2g_gpa || sl->h2g_len < sizeof(h) + 16 + entry) {
        return false;
    }
    if (dma_memory_read(pci_get_address_space(PCI_DEVICE(s)), sl->h2g_gpa, &h, sizeof(h),
                        MEMTXATTRS_UNSPECIFIED)) {
        return false;
    }
    uint32_t qsize = le32_to_cpu(h.queue_size);
    uint32_t head = le32_to_cpu(h.head);
    uint32_t tail = le32_to_cpu(h.tail);
    if (qsize == 0 || qsize + sizeof(h) > sl->h2g_len) {
        qsize = sl->h2g_len - sizeof(h);
    }
    uint32_t fr = tail >= head ? qsize - (tail - head) : head - tail;
    return fr >= entry + 64;
}

static void vtpv_raise(AppleVtpv *s, int slot)
{
    bool was = s->irq_pending != 0;
    s->irq_pending |= 1u << slot;
    if (was) {
        return;
    }
    /* Either the guest enabled MSI (preferred), or assert INTx.  The 2026-10-21
     * n953 run had irq_pending=1 with no interrupt anywhere because this was
     * MSI-only while the guest never set the MSI enable bit.  The message
     * address/data are logged to stderr: n958 had MSI enabled+addressed by the
     * guest yet the kext handler still never fired, so the actual target of the
     * message is the next thing to see. */
    if (msi_enabled(PCI_DEVICE(s))) {
        if (s->verbose > 1) {
            MSIMessage msg = msi_get_message(PCI_DEVICE(s), 0);
            lab_elog( "apple-vtpv: msi_notify addr=%#llx data=%#x\n",
                    (unsigned long long)msg.address, msg.data);
        }
        msi_notify(PCI_DEVICE(s), 0);
    } else {
        if (s->verbose > 1) {
            lab_elog( "apple-vtpv: intx assert (msi off)\n");
        }
        pci_irq_assert(PCI_DEVICE(s));
    }
}

typedef struct {
    uint32_t len;
    uint8_t data[];
} VtpvOut;



/* move what the guest's queue now has room for, oldest first */
static bool vtpv_outbox_flush(AppleVtpv *s, int slot)
{
    AppleVtpvSlot *sl = &s->slot[slot];
    bool moved = false;
    VtpvOut *o;
    while ((o = g_queue_peek_head(&sl->outbox)) && vtpv_dq_enqueue(s, slot, o->data, o->len)) {
        g_free(g_queue_pop_head(&sl->outbox));
        moved = true;
    }
    return moved;
}

#define VTPV_STALL_MS 3
#define VTPV_STALL_TRIES 20
/* a head that stood still this long past the re-notifies is a wedge: nothing
 * behind it can be delivered (the reply clerks give up at 1.1 s), so the
 * queue is reset instead of left stuck (Weather's dtil replies ran 245 s
 * late behind one, tdem2 hold 23:52) */
#define VTPV_WEDGE_MS 1500

static void vtpv_tdem_purge_slot(AppleVtpv *s, int slot);

/* entries the guest has not taken yet: re-notify when its head stood still
 * VTPV_STALL_MS (a pump that stopped at an undispatchable message), at most
 * VTPV_STALL_TRIES times without progress (a client that died); true while
 * the queue holds entries */
static void vtpv_log4cc(char out[5], uint32_t v);

static bool vtpv_h2g_watch(AppleVtpv *s, int slot, int64_t now)
{
    AppleVtpvSlot *sl = &s->slot[slot];
    struct io_dq_header h;
    if (!sl->h2g_gpa ||
        dma_memory_read(pci_get_address_space(PCI_DEVICE(s)), sl->h2g_gpa, &h, sizeof(h),
                        MEMTXATTRS_UNSPECIFIED)) {
        return false;
    }
    uint32_t head = le32_to_cpu(h.head), tail = le32_to_cpu(h.tail), qs = le32_to_cpu(h.queue_size);
    if (head == tail || qs + sizeof(h) > sl->h2g_len || head > qs || tail > qs) {
        /* empty, or no queue there any more (a client that went: its pages hold anything).
         * What still waits in the outbox belongs to that client - dropping it here is
         * what keeps a dead client's replies out of the queue of the slot's next user. */
        if (!g_queue_is_empty(&sl->outbox) && (qs + sizeof(h) > sl->h2g_len || head > qs || tail > qs)) {
            VtpvOut *o;
            int dropped = 0;
            while ((o = g_queue_pop_head(&sl->outbox))) {
                g_free(o);
                dropped++;
            }
            vtpv_tdem_purge_slot(s, slot);
            lab_elog( "apple-vtpv: slot%d queue gone (qsize %u head %u tail %u) - "
                    "dropped %d outbox entries\n", slot, qs, head, tail, dropped);
        }
        sl->stall_tries = 0;
        return false;
    }
    if (head != sl->seen_head) {
        sl->seen_head = head;
        sl->seen_ms = now;
        sl->stall_tries = 0;
    } else if (now - sl->seen_ms >= VTPV_STALL_MS && sl->stall_tries < VTPV_STALL_TRIES) {
        if (sl->stall_tries == 0) {
            sl->stuck_since_ms = sl->seen_ms;
        }
        sl->seen_ms = now;
        sl->stall_tries++;
        s->h2g_renotify++;
        uint8_t e[4 + 0x30] = { 0 };
        dma_memory_read(pci_get_address_space(PCI_DEVICE(s)), sl->h2g_gpa + sizeof(h) + head, e,
                        MIN(sizeof(e), (size_t)(qs - head)), MEMTXATTRS_UNSPECIFIED);
        char t4[5];
        vtpv_log4cc(t4, ldl_be_p(e + 4 + 8));
        lab_elog( "apple-vtpv: slot%d h2g head %u tail %u qsize %u untouched %d ms - notify again (%d); "
                "head entry size %u type %s flags %#x uuid %02x%02x%02x%02x id %llu\n",
                slot, head, tail, qs, VTPV_STALL_MS, sl->stall_tries, ldl_le_p(e), t4, ldl_be_p(e + 4 + 12),
                e[4 + 16], e[4 + 17], e[4 + 18], e[4 + 19], (unsigned long long)ldq_be_p(e + 4 + 32));
        vtpv_raise(s, slot);
    } else if (sl->stall_tries >= VTPV_STALL_TRIES &&
               now - sl->stuck_since_ms >= VTPV_WEDGE_MS) {
        /* the pump will not take this entry and nothing behind it can pass:
         * reset the queue.  Every entry dropped here is over the reply
         * clerk's 1.1 s already - undeliverable in any case - while a wedged
         * queue holds every later reply of this slot's client hostage. */
        uint32_t z[3] = { h.queue_size, 0, 0 };
        dma_memory_write(pci_get_address_space(PCI_DEVICE(s)), sl->h2g_gpa, z, sizeof(z),
                         MEMTXATTRS_UNSPECIFIED);
        VtpvOut *o;
        int dropped = 0;
        while ((o = g_queue_pop_head(&sl->outbox))) {
            g_free(o);
            dropped++;
        }
        vtpv_tdem_purge_slot(s, slot);
        lab_elog( "apple-vtpv: slot%d h2g WEDGED %lld ms (head %u tail %u) - queue reset, "
                "outbox dropped %d\n", slot, (long long)(now - sl->stuck_since_ms), head, tail,
                dropped);
        sl->seen_head = 0;
        sl->seen_ms = now;
        sl->stall_tries = 0;
        return false;
    }
    if (sl->stall_tries >= VTPV_STALL_TRIES &&
        now - sl->stuck_since_ms < VTPV_WEDGE_MS) {
        return true;    /* keep watching: the wedge verdict is not due yet */
    }
    return sl->stall_tries < VTPV_STALL_TRIES;
}

/* the outbox drains as the guest consumes: retried every millisecond while
 * anything waits, and before every new message; queued entries are watched
 * the same way */
static void vtpv_shed_reap(AppleVtpv *s, VtpvSess *se);

static void vtpv_outbox_tick(void *opaque)
{
    AppleVtpv *s = opaque;
    bool left = false;
    int64_t now = qemu_clock_get_ms(QEMU_CLOCK_REALTIME);
    for (int i = 0; i < VTPV_SESS; i++) {
        if (s->sess[i].used && s->sess[i].shed_n) {
            vtpv_shed_reap(s, &s->sess[i]);
            left |= s->sess[i].shed_n != 0;
        }
    }
    for (int i = 0; i < VTPV_MAX_SLOT; i++) {
        if (vtpv_outbox_flush(s, i)) {
            vtpv_raise(s, i);
        }
        left |= !g_queue_is_empty(&s->slot[i].outbox);
        left |= vtpv_h2g_watch(s, i, now);
    }
    if (left) {
        timer_mod(s->outbox_timer, now + 1);
    }
}

/* a host->guest message that is never lost; true if it went straight in */
static bool vtpv_dq_push(AppleVtpv *s, int slot, const uint8_t *data, uint32_t len)
{
    AppleVtpvSlot *sl = &s->slot[slot];
    vtpv_outbox_flush(s, slot);
    if (g_queue_is_empty(&sl->outbox) && vtpv_dq_enqueue(s, slot, data, len)) {
        if (!timer_pending(s->outbox_timer)) {      /* watch it being taken */
            timer_mod(s->outbox_timer, qemu_clock_get_ms(QEMU_CLOCK_REALTIME) + 1);
        }
        return true;
    }
    VtpvOut *o = g_malloc(sizeof(*o) + len);
    o->len = len;
    memcpy(o->data, data, len);
    g_queue_push_tail(&sl->outbox, o);
    if (!timer_pending(s->outbox_timer)) {
        timer_mod(s->outbox_timer, qemu_clock_get_ms(QEMU_CLOCK_REALTIME) + 1);
    }
    return false;
}

static void vtpv_lower(AppleVtpv *s)
{
    if (!s->irq_pending && !msi_enabled(PCI_DEVICE(s))) {
        pci_irq_deassert(PCI_DEVICE(s));
    }
}

/* ------------------------------------------------------------------ */

static void vtpv_msg_realloc(AppleVtpv *s, size_t need)
{
    if (need <= s->msg_cap) {
        return;
    }
    size_t cap = s->msg_cap ? s->msg_cap : 65536;
    while (cap < need) {
        cap *= 2;
    }
    s->msg = g_realloc(s->msg, cap);
    s->msg_cap = cap;
}

static void vtpv_log4cc(char out[5], uint32_t v)
{
    out[0] = v >> 24; out[1] = v >> 16; out[2] = v >> 8; out[3] = v;
    out[4] = 0;
}

static void vtpv_hex(FILE *f, const uint8_t *p, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        fprintf(f, "%02x", p[i]);
        if ((i & 31) == 31) {
            fprintf(f, "\n");
        }
    }
    if (n && (n & 31)) {
        fprintf(f, "\n");
    }
}

/* ------------------------------------------------------------------ */
/* 'para' atom access: the guest's messages are [0x30-byte 'vtpv' box]
 * followed by sibling [size BE]['para'][paramID BE][tag BE][value] atoms. */

static const uint8_t *vtpv_memfind(const uint8_t *hay, uint32_t hn,
                                   const char *needle, uint32_t nn)
{
    if (hn < nn) {
        return NULL;
    }
    for (uint32_t i = 0; i + nn <= hn; i++) {
        if (memcmp(hay + i, needle, nn) == 0) {
            return hay + i;
        }
    }
    return NULL;
}

static const uint8_t *vtpv_para_find(const uint8_t *msg, uint64_t total,
                                     uint32_t param, uint32_t *vlen)
{
    uint64_t off = 0x30;
    while (off + 8 <= total) {
        uint32_t be_sz;
        memcpy(&be_sz, msg + off, 4);
        uint32_t sz = be32_to_cpu(be_sz);
        if (sz < 16 || off + sz > total) {
            return NULL;
        }
        if (memcmp(msg + off + 4, "para", 4) == 0) {
            uint32_t pid;
            memcpy(&pid, msg + off + 8, 4);
            if (be32_to_cpu(pid) == param) {
                if (vlen) {
                    *vlen = sz - 16;
                }
                return msg + off + 16;
            }
        }
        off += sz;
    }
    return NULL;
}

static int vtpv_para_put(uint8_t *p, uint32_t param, const char *tag,
                         const void *val, uint32_t vlen)
{
    stl_be_p(p, 16 + vlen);
    memcpy(p + 4, "para", 4);
    stl_be_p(p + 8, param);
    memcpy(p + 12, tag, 4);
    if (vlen) {
        memcpy(p + 16, val, vlen);
    }
    return 16 + vlen;
}

/* binary plist encoding of {} - the ',pop' options dict of 'Dbuf' (the guest
 * feeds it straight into VTDecoderSessionCreatePixelBufferWithOptions).  42
 * bytes: header, the dict object, a one-entry offset table, 32-byte trailer.
 * (It was 41 bytes - one trailer byte short - until 2026-10-22: the guest's
 * CopyCFDictionary failed with 3840, NSPropertyListReadCorruptError, and the
 * Dbuf handler returned without a reply - the "dead Dbuf" of nA37-43.) */
static const uint8_t vtpv_empty_plist[] = {
    0x62, 0x70, 0x6c, 0x69, 0x73, 0x74, 0x30, 0x30,
    0xd0, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x09
};
QEMU_BUILD_BUG_ON(sizeof(vtpv_empty_plist) != 42);

/* binary plist of the pixel buffer attributes 'Dpba' hands the guest session
 * (VTDecoderSessionSetPixelBufferAttributes -> its output pool):
 * { Height, IOSurfaceProperties = {}, PixelFormatType = '420v', Width }.
 * Objects: 0 = dict, 1-4 keys, 5-8 values; 1-byte offsets and refs. */
static uint32_t vtpv_pba_plist(uint8_t *out, uint32_t w, uint32_t h)
{
    static const char *keys[4] = { "Height", "IOSurfaceProperties", "PixelFormatType", "Width" };
    uint32_t vals[3] = { h, 0x34323076u /* '420v' */, w };
    uint8_t off[9];
    uint32_t n = 0;
    memcpy(out, "bplist00", 8);
    n = 8;
    off[0] = n;
    out[n++] = 0xd4;                       /* dict, 4 pairs */
    for (int i = 0; i < 4; i++) {
        out[n++] = 1 + i;                  /* key refs */
    }
    for (int i = 0; i < 4; i++) {
        out[n++] = 5 + i;                  /* value refs */
    }
    for (int i = 0; i < 4; i++) {
        size_t kl = strlen(keys[i]);
        off[1 + i] = n;
        if (kl < 15) {
            out[n++] = 0x50 | kl;          /* ASCII string */
        } else {
            out[n++] = 0x5f;
            out[n++] = 0x10;               /* length as a 1-byte int */
            out[n++] = kl;
        }
        memcpy(out + n, keys[i], kl);
        n += kl;
    }
    for (int i = 0, v = 0; i < 4; i++) {
        off[5 + i] = n;
        if (i == 1) {
            out[n++] = 0xd0;               /* empty dict */
            continue;
        }
        out[n++] = 0x12;                   /* 4-byte int */
        stl_be_p(out + n, vals[v++]);
        n += 4;
    }
    uint32_t table = n;
    memcpy(out + n, off, 9);
    n += 9;
    memset(out + n, 0, 32);                /* trailer */
    out[n + 6] = 1;                        /* offset int size */
    out[n + 7] = 1;                        /* object ref size */
    stq_be_p(out + n + 8, 9);              /* objects */
    stq_be_p(out + n + 16, 0);             /* top object */
    stq_be_p(out + n + 24, table);         /* offset table offset */
    return n + 32;
}

static bool vtpv_send_h2g2(AppleVtpv *s, int slot, uint32_t type, uint32_t flags,
                           const uint8_t *uuid16, const uint8_t *id8,
                           uint8_t *paras, uint32_t paras_len)
{
    uint8_t *buf = g_malloc0(0x30 + paras_len);
    stl_be_p(buf, 0x30);                     /* 'vtpv' atom size, not total */
    memcpy(buf + 4, "vtpv", 4);
    stl_be_p(buf + 8, type);
    stl_be_p(buf + 12, flags);               /* 1 = request (guest drops flags=0,
                                          * nA27), 4 = reply */
    memcpy(buf + 0x10, uuid16, 16);
    memcpy(buf + 0x20, id8, 8);
    if (paras_len) {
        memcpy(buf + 0x30, paras, paras_len);
    }
    {
        /* mirror every outgoing host->guest message into the msglog so the
         * header/atom bytes can be diffed against the guest's own requests */
        FILE *f = s->msglog ? fopen(s->msglog, "a") : NULL;
        if (f) {
            char t4[5];
            vtpv_log4cc(t4, type);
            fprintf(f, "=== OUT slot=%d time=%lld size=%llu type=%s\n", slot,
                    (long long)qemu_clock_get_ns(QEMU_CLOCK_REALTIME),
                    (unsigned long long)(0x30 + paras_len), t4);
            vtpv_hex(f, buf, 0x30 + paras_len);
            fprintf(f, "\n");
            fclose(f);
        }
    }
    bool ok = vtpv_dq_push(s, slot, buf, 0x30 + paras_len);
    vtpv_raise(s, slot);
    g_free(buf);
    return ok;
}

/* 'Dbuf': ask the guest to allocate this frame's output pixel buffer */
static bool vtpv_send_h2g(AppleVtpv *s, int slot, uint32_t type,
                          const uint8_t *uuid16, const uint8_t *id8,
                          uint8_t *paras, uint32_t paras_len)
{
    return vtpv_send_h2g2(s, slot, type, 1, uuid16, id8, paras, paras_len);
}

static void vtpv_send_demt2(AppleVtpv *s, int slot, uint64_t frame_id,
                            int32_t err, uint32_t inf, const uint8_t *uuid16,
                            const uint8_t *uuid);
static void vtpv_pump_dbuf(AppleVtpv *s, int slot, const uint8_t *uuid16);
static bool vtpv_dec_queue(AppleVtpv *s, VtpvSess *se, int idx);
static void vtpv_shed(AppleVtpv *s, VtpvSess *se);

static void vtpv_send_dbuf(AppleVtpv *s, int slot, uint64_t frame_id,
                           const uint8_t *uuid16)
{
    int i, free_i = -1;
    for (i = 0; i < VTPV_FRAMES; i++) {
        if (!s->cur->frame[i].in_use) {
            free_i = i;
            break;
        }
    }
    if (free_i < 0) {
        lab_elog( "apple-vtpv: frame table full, dropping frame %llu\n",
                (unsigned long long)frame_id);
        if (s->outbuf) {
            /* never leave the session waiting on a frame: complete it dropped */
            uint8_t zero_uuid[16] = { 0 };
            vtpv_send_demt2(s, slot, frame_id, 0, 1u << 1, uuid16, zero_uuid);
        }
        return;
    }
    s->cur->frame[free_i].frame_id = frame_id;
    s->cur->frame[free_i].seq = s->cur->next_seq++;
    s->cur->frame[free_i].in_use = true;
    s->cur->frame[free_i].dbuf_sent = false;
    s->cur->frame[free_i].queued = false;
    s->cur->frame[free_i].decoded = false;
    s->cur->frame[free_i].shed = false;
        s->cur->frame[free_i].dno = s->dno && s->decthread && (s->cur_dfl & (1u << 1));
        s->cur->frame[free_i].pace_drop = s->pace_drop_next;
        s->pace_drop_next = false;
        if (s->cur->frame[free_i].pace_drop) {
            s->stat_drop++;
        } else {
            s->stat_ask++;
        }
    g_free(s->cur->frame[free_i].nv12);
    s->cur->frame[free_i].nv12 = NULL;
    g_free(s->cur->frame[free_i].sample);
    s->cur->frame[free_i].sample = NULL;
    if (s->outbuf && s->ob_sample) {
        s->cur->frame[free_i].sample = g_memdup2(s->ob_sample, s->ob_sample_len);
        s->cur->frame[free_i].sample_len = s->ob_sample_len;
    } else if (s->outbuf && s->ob_first) {
        VtpvDxSink *k = &s->cur->dx_sink;
        size_t sz = (size_t)k->w * k->h + (size_t)((k->w + 1) & ~1u) * ((k->h + 1) / 2);
        s->cur->frame[free_i].nv12 = g_memdup2(k->stage, sz);
        s->cur->frame[free_i].w = k->w;
        s->cur->frame[free_i].h = k->h;
    }
    vtpv_pump_dbuf(s, slot, uuid16);
    vtpv_shed(s, s->cur);
}

#define DBUF_GATE_COLD_WIN 6       /* cold answers per 5 s window to trip; the measured hidden
                                   * leak is 16-32/10 s across boots.  Two consecutive windows >= 8
                                   * confirm, so a resolution-switch burst (one hot window) cannot
                                   * trip */
#define DBUF_GATE_PROBE_US 125000  /* gated: one 'Dbuf' out per 125 ms (8/s) */
#define VTPV_PACE_IVL_US 40000     /* dec-pace v2: min ask spacing (25/s) - a decoder this slow cannot feed a discard loop */
#define VTPV_PACE_BURST 4          /* dec-pace v2: burst depth (startup jitter, the VT queue size ballpark) */

/* a 'Dbuf' ANSWER arrived carrying these surfaces: cold = never seen for the
 * session (the pool is NOT recycling), warm = the layer returned it once */
static void vtpv_gate_note_answer(AppleVtpv *s, VtpvSess *se, uint32_t surf0)
{
    /* dbuf-gate=live: the flag file is the A/B switch (Opus 16:1x - one boot,
     * no boot-to-boot variance); the answer stream is the poll clock (~60/s
     * ungated, ~8/s gated - both far past the 250 ms throttle) */
    if (s->dbuf_gate_live) {
        int64_t nowl = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
        if (nowl >= s->gate_live_check_us) {
            s->gate_live_check_us = nowl + 250000;
            gchar *txt = NULL;
            if (g_file_get_contents("C:/reims-lab/live/vtpv-dbufgate", &txt, NULL, NULL) && txt) {
                s->dbuf_gate = g_str_has_prefix(g_strchomp(txt), "on");
                g_free(txt);
            }
        }
    }
    {
        static int g_note;
        if (g_note < 3) {
            g_note++;
            lab_elog( "apple-vtpv: note-answer #%d live=%d gate=%d\n",
                    g_note, s->dbuf_gate_live, s->dbuf_gate);
        }
    }
    if (!s->dbuf_gate) {
        return;
    }
    bool warm = false;
    for (uint32_t i = 0; i < s->gate_seen_n; i++) {
        if (s->gate_seen[i] == surf0) {
            warm = true;
            break;
        }
    }
    if (s->gate_seen_n < 64) {
        s->gate_seen[s->gate_seen_n++] = surf0;
    } else {
        s->gate_seen[s->gate_seen_i] = surf0;
        s->gate_seen_i = (s->gate_seen_i + 1) & 63;
    }
    int64_t now_ans = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
    /* the window runs on EVERY answer (v3's bug: the warm early-return made
     * the clean-window lift unreachable and deadlocked the gate ON) */
    if (s->gate_win_us == 0 || now_ans - s->gate_win_us >= 5000000) {
        s->gate_cold_prev = s->gate_cold_win;
        s->gate_win_us = now_ans;
        s->gate_cold_win = 0;
        if (s->gate_on && s->gate_cold_prev < 3) {
            s->gate_on = false;
            lab_elog("apple-vtpv: dbuf-gate LIFTED (clean window: %u cold)\n",
                    s->gate_cold_prev);
        }
    }
    if (warm) {
        return;
    }
    s->gate_cold_win++;
    if (s->dbuf_gate && s->gate_cold_win % 8 == 0) {
        lab_elog("apple-vtpv: gate-diag cold=%u in window (seen=%u)\n",
                s->gate_cold_win, s->gate_seen_n);
    }
    if (s->gate_cold_win >= DBUF_GATE_COLD_WIN && s->gate_cold_prev >= 4 && !s->gate_on) {
        s->gate_on = true;
        lab_elog("apple-vtpv: dbuf-gate ON (session: %u cold answer surfaces in 5 s)\n",
                s->gate_cold_win);
    }
}

/* dec-pace: 'Dbuf' answers are the pool clock.  A warm (recycled) surface
 * means the consumer returned a buffer: the session may ask one more. */
static void vtpv_pace_note_answer(AppleVtpv *s, VtpvSess *se, uint32_t surf0)
{
    if (!s->dec_pace && !s->dec_pace_live) {
        return;
    }
    bool warm = false;
    for (uint32_t i = 0; i < se->pace_seen_n; i++) {
        if (se->pace_seen[i] == surf0) {
            warm = true;
            break;
        }
    }
    if (se->pace_seen_n < 256) {
        se->pace_seen[se->pace_seen_n++] = surf0;
    } else {
        se->pace_seen[se->pace_seen_i] = surf0;
        se->pace_seen_i = (se->pace_seen_i + 1) & 255u;
    }
    if (warm) {
        se->pace_warm++;
    } else {
        se->pace_cold++;
    }
    if ((se->pace_warm + se->pace_cold) % 256 == 0) {
        lab_elog("apple-vtpv: pace-diag warm=%u cold=%u holds=%u "
                 "dec-stat dfrm=%llu ask=%llu drop=%llu dec=%llu done=%llu tidskip=%llu shed=%llu "
                 "dbg pace=%d dx=%d ivl=%lld toks=%lld send=%llu evals=%llu\n",
                se->pace_warm, se->pace_cold, s->pace_holds,
                (unsigned long long)s->stat_dfrm, (unsigned long long)s->stat_ask,
                (unsigned long long)s->stat_drop, (unsigned long long)s->stat_dec,
                (unsigned long long)s->stat_done, (unsigned long long)s->stat_tidskip,
                (unsigned long long)s->stat_shed,
                s->dec_pace ? 1 : 0, se->dx ? 1 : 0,
                (long long)s->pace_ivl_us, (long long)s->pace_toks_us,
                (unsigned long long)s->stat_send, (unsigned long long)s->stat_evals);
    }
}

/* dec-pace v3: token check; consume=true also spends a token (the pump's
 * real send); consume=false is the dfrm site's peek (its denied frames never
 * ask, so they never spend). */
static bool vtpv_pace_allow(AppleVtpv *s, VtpvSess *se, bool consume, int64_t floor_us)
{
    if (!s->dec_pace || !s->outbuf || !se->dx) {
        return true;
    }
    int64_t now = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
    int64_t cap = (int64_t)VTPV_PACE_BURST * s->pace_ivl_us;
    if (s->pace_last_us == 0) {
        s->pace_last_us = now;
    }
    s->pace_toks_us = MIN(cap, s->pace_toks_us + (now - s->pace_last_us));
    s->pace_last_us = now;
    s->stat_evals++;
    if (s->pace_toks_us < s->pace_ivl_us + floor_us) {
        return false;
    }
    if (consume) {
        s->pace_toks_us -= s->pace_ivl_us;
    }
    return true;
}

/* dec-pace v6: a frame completion either goes out now (a token was free) or
 * waits in the ring - the player's synchronous DecodeFrame loop then runs at
 * the budget instead of at decode speed.  Calls spend tokens with PRIORITY
 * over asks (the pump reserves one interval for them). */
static void vtpv_send_demt2(AppleVtpv *s, int slot, uint64_t frame_id,
                            int32_t err, uint32_t inf, const uint8_t *uuid16,
                            const uint8_t *uuid);
static void vtpv_pace_complete(AppleVtpv *s, VtpvSess *se, int slot,
                               uint64_t frame_id, int32_t err, uint32_t inf,
                               const uint8_t *uuid16, const uint8_t *uuid)
{
    /* v7: completions flow FREE - the dfrm reply hold (v7) is the only consumer;
     * v5/v6 showed the player tolerates no completion delay */
    vtpv_send_demt2(s, slot, frame_id, err, inf, uuid16, uuid);
    s->stat_done++;
}

/* ask the guest for buffers for the oldest waiting frames, at most
 * VTPV_DBUF_INFLIGHT at a time (the next goes out when one is answered) */
static void vtpv_pump_dbuf(AppleVtpv *s, int slot, const uint8_t *uuid16)
{
    VtpvSess *se = s->cur;
    se->slot = slot;
    vtpv_shed_reap(s, se);
    for (;;) {
        int next = -1;
        for (int i = 0; i < VTPV_FRAMES; i++) {
            if (se->frame[i].in_use && !se->frame[i].dbuf_sent && !se->frame[i].shed &&
                (next < 0 || se->frame[i].seq < se->frame[next].seq)) {
                next = i;
            }
        }
        if (next < 0) {
            return;
        }
        /* TEMP lab dno: the oldest waiting frame was asked not to be output -
         * decode it now, in decode order (every older frame is already queued
         * or shed), and complete it dropped without a buffer */
        if (se->frame[next].dno || se->frame[next].pace_drop) {
            if (se->frame[next].sample && !se->frame[next].nv12 &&
                !vtpv_dec_queue(s, se, next)) {
                /* v6: the decode queue is full - complete it dropped NOW (v5
                 * returned here leaving the frame unsent+unshed: completions
                 * froze and the pipeline never recovered) */
                uint8_t zero_uuid[16] = { 0 };
                uint64_t fid = se->frame[next].frame_id;
                qemu_mutex_lock(&s->dec_lock);
                g_free(se->frame[next].sample);
                se->frame[next].sample = NULL;
                g_free(se->frame[next].nv12);
                se->frame[next].nv12 = NULL;
                se->frame[next].in_use = false;
                se->frame[next].queued = false;
                se->frame[next].decoded = false;
                qemu_mutex_unlock(&s->dec_lock);
                vtpv_pace_complete(s, se, slot, fid, 0, 1u << 1, uuid16, zero_uuid);
                continue;
            }
            se->frame[next].shed = true;
            se->shed_n++;
            s->shed_total++;
            if (!timer_pending(s->outbox_timer)) {
                timer_mod(s->outbox_timer, qemu_clock_get_ms(QEMU_CLOCK_REALTIME) + 1);
            }
            continue;
        }
        if (se->dbuf_out >= VTPV_DBUF_INFLIGHT) {
            return;
        }
        if (s->dec_pace_live) {
            int64_t nowp = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
            if (nowp >= s->pace_live_check_us) {
                s->pace_live_check_us = nowp + 250000;
                gchar *txt = NULL;
                if (g_file_get_contents("C:/reims-lab/live/vtpv-decpace", &txt, NULL, NULL) && txt) {
                    bool pon = g_str_has_prefix(g_strchomp(txt), "on");
                    g_free(txt);
                    if (pon != s->dec_pace) {
                        s->dec_pace = pon;
                        lab_elog("apple-vtpv: dec-pace LIVE %s\n", pon ? "ON" : "OFF");
                    }
                }
            }
        }
        if (!vtpv_pace_allow(s, se, true, s->pace_ivl_us)) {
            s->pace_holds++;
            if (s->pace_holds == 1 || s->pace_holds % 100 == 0) {
                lab_elog("apple-vtpv: pace HOLD #%u toks=%lld dx=%d\n",
                        s->pace_holds, (long long)s->pace_toks_us, se->dx ? 1 : 0);
            }
            memcpy(s->pace_uuid, uuid16, 16);
            if (!timer_pending(s->pace_timer)) {
                timer_mod(s->pace_timer,
                          qemu_clock_get_us(QEMU_CLOCK_REALTIME) + s->pace_ivl_us / 4);
            }
            return;
        }
        if (s->dbuf_gate && s->gate_on) {
            /* hidden-like: throttle to ~8 'Dbuf'/s (the player degrades to a
             * hidden-tab cadence instead of freezing - v3's hard hold worked
             * but stalled the switch-back 824 ms; v4's probe-lift flapped on
             * the pool's recycled stock).  Lift = the clean window (v5). */
            int64_t now = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
            if (now < s->gate_probe_us) {
                return;
            }
            s->gate_probe_us = now + DBUF_GATE_PROBE_US;
        }
        uint8_t paras[128];
        uint32_t n = 0;
        uint64_t be_frm = cpu_to_be64(se->frame[next].frame_id);
        n += vtpv_para_put(paras + n, 0x2c66726du /* ',frm' */, "=s64",
                           &be_frm, 8);
        n += vtpv_para_put(paras + n, 0x2c706f70u /* ',pop' */, "=pls",
                           vtpv_empty_plist, sizeof(vtpv_empty_plist));
        s->host_msg_id++;
        stq_be_p(se->frame[next].dbuf_id, s->host_msg_id);
        uint8_t id8[8];
        stq_be_p(id8, s->host_msg_id);
        vtpv_send_h2g(s, slot, 0x44627566u /* 'Dbuf' */, uuid16, id8, paras, n);
        s->stat_send++;
        se->frame[next].dbuf_sent = true;  /* in the queue or its outbox: never lost */
        se->dbuf_out++;
        if (s->decthread && se->frame[next].sample && !se->frame[next].nv12) {
            vtpv_dec_queue(s, se, next);    /* decode while the guest finds a buffer */
        }
        if (s->verbose) {
            lab_elog( "apple-vtpv: frame %llu: Dbuf sent (id=%llu, %u out)\n",
                    (unsigned long long)se->frame[next].frame_id,
                    (unsigned long long)s->host_msg_id, se->dbuf_out);
        }
    }
}

/* dec-pace re-pump: a hold left frames waiting and no answer may ever come
 * (the guest is itself waiting on our ask) - retry every 10 ms. */
static void vtpv_pace_tick(void *opaque)
{
    AppleVtpv *s = opaque;
    /* v7 first: the held dfrm replies (they pace the submit loop itself) */
    for (uint32_t i = 0; i < ARRAY_SIZE(s->pace_rep); i++) {
        if (s->pace_rep[i].used) {
            if (!s->cur || vtpv_pace_allow(s, s->cur, true, 0)) {
                vtpv_dq_push(s, s->pace_rep[i].slot, s->pace_rep[i].data,
                             s->pace_rep[i].len);
                vtpv_raise(s, s->pace_rep[i].slot);
                s->pace_rep[i].used = false;
            } else {
                timer_mod(s->pace_timer,
                          qemu_clock_get_us(QEMU_CLOCK_REALTIME) + s->pace_ivl_us / 4);
                return;
            }
        }
    }
    /* v6: give the waiting completions their tokens, then pump */
    for (uint32_t i = 0; i < ARRAY_SIZE(s->pace_pend); i++) {
        if (s->pace_pend[i].used) {
            if (!s->cur) {
                return;
            }
            if (vtpv_pace_allow(s, s->cur, true, 0)) {
                vtpv_send_demt2(s, s->pace_pend[i].slot, s->pace_pend[i].frame_id,
                                s->pace_pend[i].err, s->pace_pend[i].inf,
                                s->pace_pend[i].uuid16, s->pace_pend[i].uuid);
                s->stat_done++;
                s->pace_pend[i].used = false;
            } else {
                timer_mod(s->pace_timer,
                          qemu_clock_get_us(QEMU_CLOCK_REALTIME) + s->pace_ivl_us / 4);
                return;
            }
        }
    }
    if (s->cur && s->cur->used) {
        vtpv_pump_dbuf(s, s->cur->slot, s->pace_uuid);
    }
}

/* backlog=N: a player decodes ahead of display, and while the guest's pool
 * has no buffer for our 'Dbuf's its frames pile up here.  AMD work VM 10-26:
 * the 4K HEVC live wallpaper sat at 1024 waiting (~7 s behind) - every frame
 * it got back was late, so it discarded it and fed more (235 'dfrm'/s, 92/s
 * dropped at the full table, and each drop broke the references of the
 * frames after it: 70k "Could not find ref with POC").  Past N waiting the
 * oldest frames are decoded anyway (the frames after them reference them),
 * in decode order on the decode thread, and completed as dropped without a
 * buffer - the backlog, and so the latency, stays at N and the player catches
 * up with the clock. */
static void vtpv_shed(AppleVtpv *s, VtpvSess *se)
{
    if (!s->backlog || !s->decthread) {
        return;
    }
    uint32_t waiting = 0;
    for (int i = 0; i < VTPV_FRAMES; i++) {
        waiting += se->frame[i].in_use && !se->frame[i].dbuf_sent && !se->frame[i].shed;
    }
    while (waiting > s->backlog) {
        int old = -1;
        for (int i = 0; i < VTPV_FRAMES; i++) {
            if (se->frame[i].in_use && !se->frame[i].dbuf_sent && !se->frame[i].shed &&
                (old < 0 || se->frame[i].seq < se->frame[old].seq)) {
                old = i;
            }
        }
        if (old < 0) {
            break;
        }
        if (se->frame[old].sample && !se->frame[old].nv12 && !vtpv_dec_queue(s, se, old)) {
            break;                  /* the decode queue is full: shed later */
        }
        se->frame[old].shed = true;
        se->shed_n++;
        s->shed_total++;
        waiting--;
    }
    if (se->shed_n && !timer_pending(s->outbox_timer)) {
        timer_mod(s->outbox_timer, qemu_clock_get_ms(QEMU_CLOCK_REALTIME) + 1);
    }
}

/* complete the shed frames the decode thread is done with, as dropped */
static void vtpv_shed_reap(AppleVtpv *s, VtpvSess *se)
{
    if (!se->shed_n) {
        return;
    }
    /* v7: the reap completes freely (v5's cap starved the player); the dfrm
     * reply hold is the loop's only throttle */
    uint32_t allowed = UINT32_MAX;
    uint64_t done[VTPV_FRAMES];
    uint32_t n = 0;
    qemu_mutex_lock(&s->dec_lock);
    for (int i = 0; i < VTPV_FRAMES; i++) {
        if (n >= allowed) {
            break;
        }
        if (se->frame[i].in_use && se->frame[i].shed &&
            (se->frame[i].decoded || !se->frame[i].queued)) {
            done[n++] = se->frame[i].frame_id;
            g_free(se->frame[i].nv12);
            se->frame[i].nv12 = NULL;
            g_free(se->frame[i].sample);
            se->frame[i].sample = NULL;
            se->frame[i].in_use = false;
            se->frame[i].queued = false;
            se->frame[i].decoded = false;
            se->frame[i].shed = false;
            se->shed_n--;
        }
    }
    qemu_mutex_unlock(&s->dec_lock);
    uint8_t zero_uuid[16] = { 0 };
    for (uint32_t i = 0; i < n; i++) {
        vtpv_send_demt2(s, se->slot, done[i], 0, 1u << 1, se->uuid, zero_uuid);
        s->stat_done++;
        s->stat_shed++;
    }
    if (n && s->verbose) {
        lab_elog( "apple-vtpv: shed %u frames (%llu in all, %u left)\n", n,
                (unsigned long long)s->shed_total, se->shed_n);
    }
}

/* 'Demt': complete the frame - the guest runs VTDecoderSessionEmitDecodedFrame
 * (the app's decode callback) with the buffer identified by the uuid */
static void vtpv_send_demt2(AppleVtpv *s, int slot, uint64_t frame_id,
                            int32_t err, uint32_t inf, const uint8_t *uuid16,
                            const uint8_t *uuid)
{
    uint8_t paras[128];
    uint32_t n = 0;
    uint64_t be_frm = cpu_to_be64(frame_id);
    int32_t be_err = cpu_to_be32(err);
    uint32_t be_inf = cpu_to_be32(inf);
    n += vtpv_para_put(paras + n, 0x2c66726du /* ',frm' */, "=s64",
                       &be_frm, 8);
    n += vtpv_para_put(paras + n, 0x2c657272u /* ',err' */, "=s32",
                       &be_err, 4);
    n += vtpv_para_put(paras + n, 0x2c696e66u /* ',inf' */, "=u32",
                       &be_inf, 4);
    n += vtpv_para_put(paras + n, 0x2c757569u /* ',uui' */, "=uui",
                       uuid, 16);
    s->host_msg_id++;
    uint8_t id8[8];
    stq_be_p(id8, s->host_msg_id);
    s->frames_completed++;
    if (s->verbose) {
        lab_elog( "apple-vtpv: frame %llu: Demt sent err=%d "
                "(frames_completed=%llu)\n",
                (unsigned long long)frame_id, err,
                (unsigned long long)s->frames_completed);
    }
    vtpv_send_h2g(s, slot, 0x44656d74u /* 'Demt' */, uuid16, id8, paras, n);
}

static void vtpv_send_demt(AppleVtpv *s, int slot, uint64_t frame_id,
                           int32_t err, const uint8_t *uuid16,
                           const uint8_t *uuid)
{
    vtpv_send_demt2(s, slot, frame_id, err, 0, uuid16, uuid);
}

/* 'Dpba': push pixel buffer attributes so the guest session builds its output
 * pool - without it VTDecoderSessionCreatePixelBufferWithOptions fails (err 3840,
 * nA31) and 'Dbuf' can never return a buffer. */
static void vtpv_send_dpba(AppleVtpv *s, int slot, const uint8_t *uuid16,
                           uint32_t w, uint32_t h)
{
    if (w && h) {
        /* the stream's real picture size (outbuf: from the first decoded
         * picture) - the pool then hands out buffers the picture fits */
        uint8_t plist[160], paras[224];
        uint32_t pl = vtpv_pba_plist(plist, w, h);
        uint32_t n = vtpv_para_put(paras, 0x2c706261u /* ',pba' */, "=pls", plist, pl);
        s->host_msg_id++;
        uint8_t id8[8];
        stq_be_p(id8, s->host_msg_id);
        s->cur->dpba_w = w;
        s->cur->dpba_h = h;
        if (s->verbose) {
            lab_elog( "apple-vtpv: Dpba sent %ux%u 420v (%u plist bytes)\n", w, h, pl);
        }
        vtpv_send_h2g(s, slot, 0x44706261u /* 'Dpba' */, uuid16, id8, paras, n);
        return;
    }
    long rlen = sizeof(vtpv_tpl_dpba);
    uint8_t *plist;
    if (!s->dpba_file) {
        plist = g_memdup2(vtpv_tpl_dpba, sizeof(vtpv_tpl_dpba));
    } else {
        FILE *rf = fopen(s->dpba_file, "rb");
        if (!rf) {
            lab_elog( "apple-vtpv: dpba template unreadable (%s)\n", s->dpba_file);
            return;
        }
        fseek(rf, 0, SEEK_END);
        rlen = ftell(rf);
        fseek(rf, 0, SEEK_SET);
        if (rlen <= 0 || rlen > 2048) {
            fclose(rf);
            return;
        }
        plist = g_malloc(rlen);
        if (fread(plist, 1, rlen, rf) != (size_t)rlen) {
            fclose(rf);
            g_free(plist);
            return;
        }
        fclose(rf);
    }
    uint8_t paras[2112];
    uint32_t n = vtpv_para_put(paras, 0x2c706261u /* ',pba' */, "=pls",
                               plist, (uint32_t)rlen);
    g_free(plist);
    s->host_msg_id++;
    uint8_t id8[8];
    stq_be_p(id8, s->host_msg_id);
    if (s->verbose) {
        lab_elog( "apple-vtpv: Dpba sent (%ld plist bytes)\n", rlen);
    }
    vtpv_send_h2g(s, slot, 0x44706261u /* 'Dpba' */, uuid16, id8, paras, n);
}

/* ------------------------------------------------------------------ */
/* hwbackend=dx: the serialized format description and sample go to
 * apple-vtpv-dx.c unchanged.  Format description layout (dump of a live
 * ',sbf', 2026-10-22): atoms [BE32 len][4cc]; 'codc' holds the codec 4CC
 * byte-reversed ('1cva' = avc1, '1cvh' = hvc1, '90pv' = vp09, '10va' = av01);
 * the SampleDescriptionExtensionAtoms value embeds the ISO sample entry, whose
 * 'avcC'/'hvcC'/'vpcC'/'av1C' box is the decoder configuration record. */

static bool vtpv_use_dx(AppleVtpv *s)
{
    return !s->hwbackend_str || strcmp(s->hwbackend_str, "vk") != 0;
}

/* first atom [BE32 len][tag] in buf whose length fits; returns its payload */
static const uint8_t *vtpv_atom_find(const uint8_t *buf, uint32_t n,
                                     const char *tag, uint32_t *plen)
{
    for (uint32_t k = 0; k + 8 <= n; k++) {
        if (memcmp(buf + k + 4, tag, 4) == 0) {
            uint32_t l = ldl_be_p(buf + k);
            if (l >= 8 && l <= n - k) {
                *plen = l - 8;
                return buf + k + 8;
            }
        }
    }
    return NULL;
}

/* A format description made with CMVideoFormatDescriptionCreate carries its
 * SampleDescriptionExtensionAtoms as a dictionary instead of an ISO sample
 * entry: the key is a bplist ASCII string object (0x54 + 4 bytes) and the
 * value follows in a 'datv' atom holding a bplist data object (0x4n, or 0x4f
 * + an int object for n >= 15).  Returns the record bytes or NULL. */
static const uint8_t *vtpv_fmt_dict_record(const uint8_t *fd, uint32_t n,
                                           const char *box, uint32_t *len)
{
    const char key[5] = { 0x54, box[0], box[1], box[2], box[3] };
    const uint8_t *k = vtpv_memfind(fd, n, key, sizeof(key));
    if (!k) {
        return NULL;
    }
    uint32_t rest = n - (uint32_t)(k - fd), dl = 0;
    const uint8_t *datv = vtpv_atom_find(k, rest, "datv", &dl);
    if (!datv || dl < 10 || memcmp(datv, "bplist00", 8) != 0) {
        return NULL;
    }
    const uint8_t *p = datv + 8, *end = datv + dl;
    uint8_t m = *p++;
    if ((m & 0xf0) != 0x40) {
        return NULL;
    }
    uint64_t sz = m & 0x0f;
    if (sz == 0x0f) {
        if (p >= end || (*p & 0xf0) != 0x10) {
            return NULL;
        }
        uint32_t ib = 1u << (*p++ & 0x0f);
        if (ib > 8 || p + ib > end) {
            return NULL;
        }
        sz = 0;
        for (uint32_t i = 0; i < ib; i++) {
            sz = (sz << 8) | *p++;
        }
    }
    if (sz == 0 || sz > (uint64_t)(end - p)) {
        return NULL;
    }
    *len = (uint32_t)sz;
    return p;
}

/* codec + configuration record of a format description; false if unknown */
static bool vtpv_dx_parse_fmt(const uint8_t *fd, uint32_t n, int *codec,
                              const uint8_t **cfg, uint32_t *cfg_len)
{
    static const struct {
        char codc[5];
        char entry[5];
        char box[5];
        int codec;
    } map[] = {
        { "1cva", "avc1", "avcC", VTPV_DX_H264 }, { "3cva", "avc3", "avcC", VTPV_DX_H264 },
        { "1cvh", "hvc1", "hvcC", VTPV_DX_HEVC }, { "1veh", "hev1", "hvcC", VTPV_DX_HEVC },
        { "90pv", "vp09", "vpcC", VTPV_DX_VP9 },  { "10va", "av01", "av1C", VTPV_DX_AV1 },
    };
    uint32_t cl = 0;
    const uint8_t *codc = vtpv_atom_find(fd, n, "codc", &cl);
    for (size_t i = 0; i < ARRAY_SIZE(map); i++) {
        /* a sample-buffer 'fdsc' names the codec in 'codc' (byte-reversed);
         * the session's ',fds' (dsta/dsts) is a bare ISO sample entry whose
         * type is the codec and whose child box is the record (ox3) */
        const uint8_t *body = fd;
        uint32_t bl = n;
        if (codc && cl >= 4) {
            if (memcmp(codc, map[i].codc, 4) != 0) {
                continue;
            }
        } else if (!(body = vtpv_atom_find(fd, n, map[i].entry, &bl))) {
            continue;
        }
        *codec = map[i].codec;
        *cfg = vtpv_atom_find(body, bl, map[i].box, cfg_len);
        if (!*cfg) {
            *cfg = vtpv_fmt_dict_record(fd, n, map[i].box, cfg_len);
        }
        if (map[i].codec == VTPV_DX_VP9) {
            *cfg = NULL;      /* libavcodec's VP9 reads no extradata */
            *cfg_len = 0;
        }
        return *cfg || map[i].codec == VTPV_DX_VP9 || map[i].codec == VTPV_DX_AV1;
    }
    return false;
}

/* vk paths: decide the codec before the slice filter runs - 'fdsc' may follow
 * 'sdat' in the sample buffer (NA56: the first HEVC dfrm parsed slices=0) */
static void vtpv_sbf_codec_first(AppleVtpv *s, const uint8_t *sbf, uint32_t n)
{
    uint32_t fl = 0, cl = 0;
    const uint8_t *fd = vtpv_atom_find(sbf, n, "fdsc", &fl);
    const uint8_t *cfg;
    int codec;
    if (fd && vtpv_dx_parse_fmt(fd, fl, &codec, &cfg, &cl)) {
        if (codec == VTPV_DX_H264) {
            s->dec_codec = 0;
        } else if (codec == VTPV_DX_HEVC) {
            s->dec_codec = 1;
        }
    }
}

/* wait until the decode thread holds nothing of this session: queued frames
 * are finished first (they belong to the stream as it was) */
static void vtpv_tile_drain(AppleVtpv *s, VtpvSess *se, bool cancel);

static void vtpv_dec_drain(AppleVtpv *s, VtpvSess *se)
{
    qemu_mutex_lock(&s->dec_lock);
    for (;;) {
        bool pending = s->dec_busy == se;
        for (uint32_t i = 0; i < s->dec_n && !pending; i++) {
            pending = s->dec_job[(s->dec_head + i) % ARRAY_SIZE(s->dec_job)].se == se;
        }
        if (!pending) {
            break;
        }
        qemu_cond_wait(&s->dec_cond, &s->dec_lock);
    }
    qemu_mutex_unlock(&s->dec_lock);
}

static void vtpv_dx_reset(AppleVtpv *s)
{
    vtpv_dec_drain(s, s->cur);
    vtpv_tile_drain(s, s->cur, false);
    vtpv_dx_close(s->cur->dx);
    s->cur->dx = NULL;
    s->cur->dx_api = VTPV_API_D3D12;
    s->cur->dx_codec = -1;
    s->cur->dx_cfg_len = 0;
    s->cur->dpba_w = s->cur->dpba_h = 0;      /* outbuf: the next stream announces its size again */
}

/* a session ends (its uuid deregistered, or evicted): decoder, waiting
 * frames and stage go with it */
static void vtpv_sess_free(AppleVtpv *s, VtpvSess *se)
{
    vtpv_dec_drain(s, se);
    vtpv_tile_drain(s, se, true);
    vtpv_dx_close(se->dx);
    for (int i = 0; i < VTPV_FRAMES; i++) {
        g_free(se->frame[i].nv12);
        g_free(se->frame[i].sample);
    }
    g_free(se->dx_sink.stage);
    memset(se, 0, sizeof(*se));
    se->dx_codec = -1;
    se->owner = s;
}

/* the session a message belongs to; a new uuid takes a free entry or the
 * least recently used one */
static VtpvSess *vtpv_sess_get(AppleVtpv *s, const uint8_t *uuid16)
{
    VtpvSess *lru = NULL, *free_se = NULL;
    for (int i = 0; i < VTPV_SESS; i++) {
        VtpvSess *se = &s->sess[i];
        if (se->used && memcmp(se->uuid, uuid16, 16) == 0) {
            se->last_use = ++s->sess_clock;
            return se;
        }
        if (!se->used) {
            free_se = free_se ? free_se : se;
        } else if (!lru || (!se->dx && lru->dx) ||
                   (!se->dx == !lru->dx && se->last_use < lru->last_use)) {
            lru = se;           /* oldest, sessions without a decoder first */
        }
    }
    VtpvSess *se = free_se ? free_se : lru;
    if (se->used) {
        lab_elog( "apple-vtpv: session table full, evicting the oldest\n");
        if (se->tile_live && s->tile_sessions) {
            s->tile_sessions--;
        }
    }
    vtpv_sess_free(s, se);
    se->used = true;
    se->pace_warm = 0;
    se->pace_cold = 0;
    se->pace_seen_n = 0;
    se->pace_seen_i = 0;
    memcpy(se->uuid, uuid16, 16);
    se->last_use = ++s->sess_clock;
    return se;
}

static uint32_t vtpv_sess_waiting(const VtpvSess *se)
{
    uint32_t n = 0;
    for (int i = 0; i < VTPV_FRAMES; i++) {
        n += se->frame[i].in_use;
    }
    return n;
}

static void vtpv_tdem_flush(AppleVtpv *s, const uint8_t *uuid16, bool drop);

static void vtpv_sess_drop(AppleVtpv *s, const uint8_t *uuid16)
{
    for (int i = 0; i < VTPV_SESS; i++) {
        VtpvSess *se = &s->sess[i];
        if (se->used && memcmp(se->uuid, uuid16, 16) == 0) {
            if (se->tile_live && s->tile_sessions) {
                s->tile_sessions--;
            }
            se->tile_live = false;
            if (s->cur == se) {
                s->cur = NULL;
            }
            vtpv_sess_free(s, se);
        }
    }
    vtpv_tdem_flush(s, uuid16, true);
    /* what still waits for room and names the UUID would reach a guest that
     * no longer dispatches it, and its pump stops at such a message */
    for (int i = 0; i < VTPV_MAX_SLOT; i++) {
        GQueue *q = &s->slot[i].outbox;
        for (GList *l = q->head; l;) {
            GList *next = l->next;
            VtpvOut *o = l->data;
            if (o->len >= 0x20 && memcmp(o->data + 0x10, uuid16, 16) == 0) {
                g_free(o);
                g_queue_delete_link(q, l);
            }
            l = next;
        }
    }
}

/* (re)arm the decoder for a format description; a new stream shape closes
 * the old decoder (its references belong to the old stream) */
static void vtpv_dx_set_fmt(AppleVtpv *s, const uint8_t *fd, uint32_t n)
{
    int codec;
    const uint8_t *cfg = NULL;
    uint32_t cfg_len = 0;
    if (!vtpv_dx_parse_fmt(fd, n, &codec, &cfg, &cfg_len) ||
        cfg_len > sizeof(s->cur->dx_cfg)) {
        return;
    }
    if (codec == s->cur->dx_codec && cfg_len == s->cur->dx_cfg_len &&
        (!cfg_len || memcmp(cfg, s->cur->dx_cfg, cfg_len) == 0)) {
        return;
    }
    vtpv_dx_reset(s);
    s->cur->dx_codec = codec;
    s->cur->dx_cfg_len = cfg_len;
    if (cfg_len) {
        memcpy(s->cur->dx_cfg, cfg, cfg_len);
    }
    lab_elog( "apple-vtpv: dx stream codec=%d cfg=%u bytes\n", codec, cfg_len);
}

/* exported surface pages: the indirect list when the kext gave one, else the
 * 16 direct registers (SURFMAP_COUNT is 16 bits wide - never index past them) */
static bool vtpv_page_ok(AppleVtpv *s, uint32_t pg)
{
    return s->surfmap_list ? pg < s->surfmap_list_n
                           : pg < MIN(s->surfmap_count, (uint32_t)ARRAY_SIZE(s->surfmap_gpa));
}

static uint64_t vtpv_page_gpa(AppleVtpv *s, uint32_t pg)
{
    return s->surfmap_list ? s->surfmap_list[pg] : s->surfmap_gpa[pg];
}

/* the kext exports the canvas from its first byte, so page 0 carries the
 * in-page offset of that byte (gpa0 = ...020 for a CVPixelBuffer plane at
 * +32, ox3) and every list entry names one virtual page: surface offset
 * `off` lives at org+off within the page run, each entry's page base is
 * the entry with its low bits dropped */
static uint64_t vtpv_surf_org(AppleVtpv *s)
{
    return vtpv_page_ok(s, 0) ? vtpv_page_gpa(s, 0) & 0xfff : 0;
}

static uint64_t vtpv_surf_page(AppleVtpv *s, uint32_t pg)
{
    return vtpv_page_gpa(s, pg) & ~(uint64_t)0xfff;
}

/*
 * Before the device stores into a guest page: the paravirtual GPU's dirty
 * tracking trusts write-protected pages until a guest store faults on them, and
 * a store of ours never faults - a decoded picture or HEIF tile the guest then
 * hands to the GPU (WallpaperAgent: the desktop picture) was read from the GPU's
 * stale copy (all zero: a green desktop). The fault hook is the "a store is about
 * to happen" the tracking already understands.
 */
static void vtpv_host_store(uint64_t gpa)
{
    if (whpx_wp_fault_hook) {
        whpx_wp_fault_hook(gpa & ~(uint64_t)0xfff);
    }
}

/* after the device filled an attached surface: its pages are written, for the
 * paravirtual GPU's tracking (vtpv_host_store's fault covers write-protected
 * pages only; a surface the GPU holds a copy of is re-read from these stamps) */
static void vtpv_note_surface(const VtpvKickSurf *ks)
{
    g_autofree uint64_t *pg = g_new(uint64_t, ks->npages ? ks->npages : 1);
    for (uint32_t i = 0; i < ks->npages; i++) {
        pg[i] = ks->pages[i] & ~(uint64_t)0xfff;
    }
    reims_vgpu_dirty_note_host_store(pg, ks->npages);
}

/* one row segment into the exported surface pages (4 KiB each) */
static uint64_t vtpv_wb_bytes(AppleVtpv *s, uint64_t off, const uint8_t *src,
                              uint32_t len)
{
    uint64_t written = 0;
    while (len) {
        uint64_t sa = off + vtpv_surf_org(s);
        uint32_t pg = sa >> 12;
        uint32_t in_pg = 4096 - (sa & 0xfff);
        uint32_t chunk = MIN(len, in_pg);
        if (vtpv_page_ok(s, pg)) {
            vtpv_host_store(vtpv_surf_page(s, pg));
            dma_memory_write(pci_get_address_space(PCI_DEVICE(s)),
                             vtpv_surf_page(s, pg) + (sa & 0xfff), src, chunk,
                             MEMTXATTRS_UNSPECIFIED);
            written += chunk;
        }
        off += chunk;
        src += chunk;
        len -= chunk;
    }
    return written;
}

/* one row segment into an attached surface (same page rule as the canvas:
 * offset `off` from the surface's first byte lives at org + off) */
static void vtpv_ksurf_bytes(AppleVtpv *s, const VtpvKickSurf *ks, uint64_t off,
                             const uint8_t *src, uint32_t len)
{
    if (off + len > ks->alloc) {
        return;
    }
    while (len) {
        uint64_t sa = off + ks->org;
        uint32_t pg = sa >> 12;
        uint32_t chunk = MIN(len, 4096 - (uint32_t)(sa & 0xfff));
        if (pg < ks->npages) {
            vtpv_host_store(ks->pages[pg]);
            dma_memory_write(pci_get_address_space(PCI_DEVICE(s)),
                             (ks->pages[pg] & ~(uint64_t)0xfff) + (sa & 0xfff), src, chunk,
                             MEMTXATTRS_UNSPECIFIED);
        }
        off += chunk;
        src += chunk;
        len -= chunk;
    }
}

static bool vtpv_ksurf_read(AppleVtpv *s, const VtpvKickSurf *ks, uint64_t off,
                           uint8_t *dst, uint32_t len)
{
    if (off + len > ks->alloc) {
        return false;
    }
    while (len) {
        uint64_t sa = off + ks->org;
        uint32_t pg = sa >> 12;
        uint32_t chunk = MIN(len, 4096 - (uint32_t)(sa & 0xfff));
        if (pg < ks->npages) {
            /* no vtpv_host_store here: a read must not arm the write-fault
             * tracker - it poisoned the GPU's page coherence and killed the
             * D3D12 device at dump time (kt1/kt1v2, 0xc0000374) */
            dma_memory_read(pci_get_address_space(PCI_DEVICE(s)),
                            (ks->pages[pg] & ~(uint64_t)0xfff) + (sa & 0xfff), dst, chunk,
                            MEMTXATTRS_UNSPECIFIED);
        } else {
            return false;
        }
        off += chunk;
        dst += chunk;
        len -= chunk;
    }
    return true;
}

/* tile-dump=N: the first N fills also drop {geometry, intended stage bytes,
 * canvas read-back} to E:/tmp-c/tiledump-<n>.bin - Opus's ask: CMPhoto
 * rejects the tile result even with no TDem (tile-sync: 84 canvases filled,
 * -17104 x48), so the CONTENT is under test (420v/f range, plane offsets,
 * bpr, org, tile position).  Compare against a host CPU decode of the same
 * tile's ',sbf' from the msglog. */
static void vtpv_tile_dump(AppleVtpv *s, const VtpvKickSurf *ks, const uint8_t *stage,
                           bool gray, uint32_t pw, uint32_t ph, uint32_t sx, uint32_t sy,
                           uint32_t w, uint32_t h, uint32_t ox, uint32_t oy, uint32_t dw,
                           uint32_t dh)
{
    if (!s->tile_dump_left) {
        return;
    }
    s->tile_dump_left--;
    uint32_t n = s->tile_dump_total++ ;
    char path[96];
    snprintf(path, sizeof(path), "E:/tmp-c/tiledump-%u.bin", n);
    FILE *f = fopen(path, "wb");
    if (!f) {
        return;
    }
    struct {
        uint32_t magic, pixfmt, planes, gray;
        uint32_t pw, ph, sx, sy, w, h, ox, oy, dw, dh;
        uint32_t cpw[2], cph[2], coff[2], cbpr[2], alloc, npages, org;
    } QEMU_PACKED hdr = {
        .magic = 0x44544C31u, .pixfmt = ks->pixfmt, .planes = ks->planes,
        .gray = gray, .pw = pw, .ph = ph, .sx = sx, .sy = sy, .w = w, .h = h,
        .ox = ox, .oy = oy, .dw = dw, .dh = dh, .alloc = ks->alloc,
        .npages = ks->npages, .org = ks->org,
    };
    for (int p = 0; p < 2; p++) {
        hdr.cpw[p] = ks->pw[p]; hdr.cph[p] = ks->ph[p];
        hdr.coff[p] = ks->off[p]; hdr.cbpr[p] = ks->bpr[p];
    }
    fwrite(&hdr, sizeof(hdr), 1, f);
    size_t stage_n = ks->planes > 1
        ? (size_t)pw * ph + (size_t)((pw + 1) / 2) * ((ph + 1) / 2) * 2
        : (size_t)pw * ph;
    fwrite(stage, stage_n, 1, f);
    /* canvas readback only behind tile-dump-read=on: it raced the memory
     * layer at dump time twice (0xc0000374), and the canvas is verified
     * byte-exact (kt1v2 dump: Y 0/262144, C 0/65536) so the stage suffices */
    if (s->tile_dump_read) {
        g_autofree uint8_t *rb = g_malloc(ks->bpr[0] * (uint64_t)ks->ph[0] + 16);
        if (vtpv_ksurf_read(s, ks, ks->off[0], rb, ks->bpr[0] * (uint64_t)ks->ph[0])) {
            fwrite(rb, ks->bpr[0] * (uint64_t)ks->ph[0], 1, f);
        }
        g_free(rb);
        g_autofree uint8_t *rbc = g_malloc((uint64_t)ks->bpr[1] * ks->ph[1] + 16);
        if (ks->planes > 1 && vtpv_ksurf_read(s, ks, ks->off[1], rbc,
                                              (uint64_t)ks->bpr[1] * ks->ph[1])) {
            fwrite(rbc, (uint64_t)ks->bpr[1] * ks->ph[1], 1, f);
        }
        g_free(rbc);
    }
    fclose(f);
    lab_elog( "apple-vtpv: tile dump %u -> %s (%.4s %ux%u at %u,%u dw %u dh %u)\n",
            n, path, (const char *)&ks->pixfmt, ks->pw[0], ks->ph[0], ox, oy, dw, dh);
}

/* a staged tight NV12 picture -> the guest pool buffer the app will receive */
static bool vtpv_fill_surface(AppleVtpv *s, const VtpvKickSurf *ks, const uint8_t *nv12,
                              uint32_t w, uint32_t h)
{
    if ((ks->pixfmt != 0x34323076u /* '420v' */ && ks->pixfmt != 0x34323066u /* '420f' */) ||
        ks->planes != 2) {
        lab_elog( "apple-vtpv: outbuf: surface %u is %.4s with %u planes - not NV12\n",
                ks->id, (const char *)&ks->pixfmt, ks->planes);
        return false;
    }
    uint32_t cw = (w + 1) & ~1u, ch = (h + 1) / 2;
    uint32_t dw = MIN(w, ks->pw[0]), dh = MIN(h, ks->ph[0]);
    uint32_t dcw = MIN(cw, ks->pw[1] * 2), dch = MIN(ch, ks->ph[1]);
    for (uint32_t y = 0; y < dh; y++) {
        vtpv_ksurf_bytes(s, ks, ks->off[0] + (uint64_t)y * ks->bpr[0], nv12 + (size_t)y * w, dw);
    }
    const uint8_t *uv = nv12 + (size_t)w * h;
    for (uint32_t y = 0; y < dch; y++) {
        vtpv_ksurf_bytes(s, ks, ks->off[1] + (uint64_t)y * ks->bpr[1], uv + (size_t)y * cw, dcw);
    }
    return true;
}

/* decoded picture -> guest surface: rows repacked to the surface pitches */
static void vtpv_dx_put(void *opaque, const VtpvDxFrame *f)
{
    VtpvSess *se = opaque;
    AppleVtpv *s = se->owner;
    VtpvDxSink *k = &se->dx_sink;
    uint32_t w = s->surfmap_meta[0] >> 16;
    uint32_t h = s->surfmap_meta[0] & 0xffffu;
    uint32_t bpr0 = s->surfmap_meta[1] ? s->surfmap_meta[1] : w;
    uint32_t p1 = s->surfmap_meta[2];
    uint32_t bpr1 = s->surfmap_meta[3] ? s->surfmap_meta[3] : w;
    uint32_t dw = MIN(f->width, w), dh = MIN(f->height, h);

    bool nv12 = !f->gray && !f->planar && f->sx == 1 && f->sy == 1;
    if (k->to_stage && f->gray) {
        size_t need = (size_t)f->width * f->height;
        if (k->stage_cap < need) {
            k->stage = g_realloc(k->stage, need);
            k->stage_cap = need;
        }
        for (uint32_t y = 0; y < f->height; y++) {
            memcpy(k->stage + (size_t)y * f->width, f->y + (size_t)y * f->y_pitch, f->width);
        }
        k->cgray = true;
        k->w = f->width;
        k->h = f->height;
        k->pictures++;
        return;
    }
    if (k->to_stage) {
        /* the picture waits in the stage (outbuf: for the guest's own buffer,
         * 'Dbuf'; a tile: for its canvas): tight luma w x h, then CbCr pairs,
         * (w >> sx) of them a row, (h >> sy) rows, both rounded up - NV12 for
         * 4:2:0, NV16 for 4:2:2, NV24 for 4:4:4 */
        uint32_t cw = (f->width + (1u << f->sx) - 1) >> f->sx;
        uint32_t ch = (f->height + (1u << f->sy) - 1) >> f->sy;
        size_t need = (size_t)f->width * f->height + (size_t)cw * 2 * ch;
        if (k->stage_cap < need) {
            k->stage = g_realloc(k->stage, need);
            k->stage_cap = need;
        }
        for (uint32_t y = 0; y < f->height; y++) {
            memcpy(k->stage + (size_t)y * f->width, f->y + (size_t)y * f->y_pitch, f->width);
        }
        uint8_t *uv = k->stage + (size_t)f->width * f->height;
        for (uint32_t y = 0; y < ch; y++) {
            uint8_t *d = uv + (size_t)y * cw * 2;
            if (!f->planar) {
                memcpy(d, f->uv + (size_t)y * f->uv_pitch, (size_t)cw * 2);
                continue;
            }
            const uint8_t *cb = f->uv + (size_t)y * f->uv_pitch, *cr = f->v + (size_t)y * f->v_pitch;
            for (uint32_t x = 0; x < cw; x++) {
                d[2 * x] = cb[x];
                d[2 * x + 1] = cr[x];
            }
        }
        k->csx = f->sx;
        k->csy = f->sy;
        k->cgray = false;
        k->w = f->width;
        k->h = f->height;
        k->pictures++;
        return;
    }
    if (!nv12) {
        lab_elog( "apple-vtpv: a non-NV12 picture outside a staged decode - dropped\n");
        return;
    }

    for (uint32_t y = 0; y < dh; y++) {
        k->written += vtpv_wb_bytes(s, (uint64_t)y * bpr0, f->y + (size_t)y * f->y_pitch, dw);
    }
    for (uint32_t y = 0; y < dh / 2; y++) {
        k->written += vtpv_wb_bytes(s, p1 + (uint64_t)y * bpr1,
                                    f->uv + (size_t)y * f->uv_pitch, dw);
    }
    if (s->verbose) {
        /* read back what landed in guest RAM (ox3: every pixel differed) */
        static unsigned dbg;
        if (dbg++ < 16) {
            uint8_t rb[2][8] = { { 0 } };
            uint64_t ro[2] = { 0, (uint64_t)(dh / 2) * bpr0 };
            for (int i = 0; i < 2; i++) {
                uint64_t sa = ro[i] + vtpv_surf_org(s);
                if (vtpv_page_ok(s, sa >> 12)) {
                    dma_memory_read(pci_get_address_space(PCI_DEVICE(s)),
                                    vtpv_surf_page(s, sa >> 12) + (sa & 0xfff), rb[i], 8,
                                    MEMTXATTRS_UNSPECIFIED);
                }
            }
            const uint8_t *s0 = f->y, *s1 = f->y + (size_t)(dh / 2) * f->y_pitch;
            lab_elog( "apple-vtpv: dx put %ux%u org=%llu y0 src %02x%02x%02x%02x "
                    "ram %02x%02x%02x%02x | y%u src %02x%02x%02x%02x ram %02x%02x%02x%02x\n",
                    dw, dh, (unsigned long long)vtpv_surf_org(s), s0[0], s0[1], s0[2], s0[3],
                    rb[0][0], rb[0][1], rb[0][2], rb[0][3], dh / 2, s1[0], s1[1], s1[2], s1[3],
                    rb[1][0], rb[1][1], rb[1][2], rb[1][3]);
        }
    }
    k->w = f->width;
    k->h = f->height;
    k->pictures++;
}

/* the 'sdat' sample of a 'dtil'/'dfrm' ',sbf' (its 'fdsc', if any, re-arms
 * the decoder first); NULL if there is nothing decodable */
static const uint8_t *vtpv_dx_sbf_sample(AppleVtpv *s, uint64_t total, uint64_t tag,
                                         uint32_t *plen)
{
    uint32_t sbf_len = 0, fl = 0, sl = 0;
    const uint8_t *sbf = vtpv_para_find(s->msg, total, 0x2c736266u /* ',sbf' */,
                                        &sbf_len);
    if (!sbf || sbf_len < 16) {
        lab_elog( "apple-vtpv: dx: no ',sbf' in frame %llu\n",
                (unsigned long long)tag);
        return false;
    }
    const uint8_t *fd = vtpv_atom_find(sbf, sbf_len, "fdsc", &fl);
    const uint8_t *sample = vtpv_atom_find(sbf, sbf_len, "sdat", &sl);
    if (fd) {
        vtpv_dx_set_fmt(s, fd, fl);
    }
    if (!sample || s->cur->dx_codec < 0) {
        static int dumps;
        lab_elog( "apple-vtpv: dx: frame %llu sample=%u codec=%d - not decodable\n",
                (unsigned long long)tag, sample ? sl : 0, s->cur->dx_codec);
        if (s->msglog && dumps < 3) {
            /* keep the first unparsed sample buffers next to the wire log */
            g_autofree char *path = g_strdup_printf("%s.dxfail%d.bin", s->msglog, dumps++);
            FILE *df = fopen(path, "wb");
            if (df) {
                fwrite(sbf, 1, sbf_len, df);
                fclose(df);
            }
        }
        return NULL;
    }
    *plen = sl;
    return sample;
}

/* the frame's HEVC temporal_id: the first VCL NAL of the (AVCC-form,
 * [BE32 len][NAL]...) sample; -1 when unparseable */
static int vtpv_sample_tid(const uint8_t *sample, uint32_t len)
{
    uint32_t off = 0;
    while (off + 6 <= len) {
        uint32_t nl = ((uint32_t)sample[off] << 24) | ((uint32_t)sample[off + 1] << 16) |
                      ((uint32_t)sample[off + 2] << 8) | sample[off + 3];
        if (nl < 2 || off + 4 + nl > len) {
            return -1;
        }
        const uint8_t *nal = sample + off + 4;
        uint32_t ntype = (nal[0] >> 1) & 0x3f;
        if (ntype <= 31) {          /* a VCL slice NAL carries the frame's tid */
            return (nal[1] & 7) - 1;
        }
        off += 4 + nl;
    }
    return -1;
}

/* the stream class a decoder rung answers for: codec, and for HEVC the hvcC's
 * chroma format and luma depth (a HEIF alpha plane is monochrome, the Tahoe
 * picture 4:2:2) */
static unsigned vtpv_stream_class(const VtpvSess *se)
{
    unsigned chroma = 1, depth = 0;
    if (se->dx_codec == VTPV_DX_HEVC && se->dx_cfg_len > 18) {
        chroma = se->dx_cfg[16] & 3;
        depth = MIN(se->dx_cfg[17] & 7, 3);
    }
    return ((unsigned)se->dx_codec & 3) << 4 | chroma << 2 | depth;
}

/* the rung that last decoded each stream class (0 = not learned, else api + 1):
 * a session starts there instead of opening and failing the GPU decoders first -
 * 80 ms a HEIF alpha tile, long enough for ImageIO to drop the image before the
 * tile's 'TDem' came (TV crashed in CMPhoto's handler) */
static uint8_t vtpv_learned_api[64];

/* tiles never ride the CUDA rung: a video stream learns cuda for its class
 * and the session's TILE decodes then inherit it, but an odd HEIF tile NALU
 * ("Duplicate POC / Skipping invalid undecodable NALU") corrupts nvdec's
 * hwaccel context with avcodec still reporting rc=0 - the next tile packet
 * dereferenced ctx->decoder==NULL and crashed QEMU in nvdec start_frame
 * (Opus's symbolization, 04:23/04:40 rfq8 arms).  D3D12 is the tile decoder;
 * call at every 'dtil' entry (BQL). */
static void vtpv_tile_off_cuda(VtpvSess *se)
{
    if (se->dx_api == VTPV_API_CUDA) {
        if (se->dx) {
            vtpv_dx_discard(se->dx);
            se->dx = NULL;
        }
        se->dx_api = VTPV_API_D3D12;
    }
}

static bool vtpv_dx_decode_sample(AppleVtpv *s, VtpvSess *se, const uint8_t *sample,
                                  uint32_t sl, uint64_t tag)
{
    unsigned cls = vtpv_stream_class(se);
    int64_t t_in = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
    if (!se->dx && se->dx_api == VTPV_API_D3D12) {
        if (vtpv_learned_api[cls] && !se->tile_live) {
            /* a tile session never inherits the class's cuda rung - nvdec
             * does not survive an odd HEIF tile NALU (see vtpv_tile_off_cuda) */
            se->dx_api = vtpv_learned_api[cls] - 1;
        } else if (se->dx_codec == VTPV_DX_HEVC && ((cls >> 2) & 3) == 0) {
            /* monochrome HEVC: no D3D12 or NVDEC profile on either GPU here */
            se->dx_api = VTPV_API_SW;
        }
    }
    for (;;) {
        if (se->dx_api >= VTPV_API_NONE) {
            return false;
        }
        if (!se->dx) {
            const char *gpu = se->dx_api == VTPV_API_CUDA ? "nv" : s->dec_gpu_str ? s->dec_gpu_str : "int";
            se->dx = vtpv_dx_open(gpu, vtpv_api_name[se->dx_api], (VtpvDxCodec)se->dx_codec,
                                 se->dx_cfg_len ? se->dx_cfg : NULL, se->dx_cfg_len, vtpv_dx_put, se);
            if (!se->dx) {
                lab_elog( "apple-vtpv: dx: %s decoder open failed (codec %d)\n",
                        vtpv_api_name[se->dx_api], se->dx_codec);
                do {
                    se->dx_api++;   /* a tile session skips cuda on the walk too */
                } while (se->tile_live && se->dx_api == VTPV_API_CUDA);
                continue;
            }
        }
        VtpvDxSink *k = &se->dx_sink;
        k->written = 0;             /* per-message counters; mode + stage stay */
        k->w = k->h = 0;
        k->pictures = 0;
        int64_t t0 = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
        int rc = vtpv_dx_decode(se->dx, sample, sl, (int64_t)tag);
        int64_t t1 = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
        lab_elog( "apple-vtpv: HW-DECODE(%s) frame=%llu codec=%d %ux%u pictures=%d "
                "-> canvas %llu bytes rc=%d %lld us (open %lld us)\n", vtpv_api_name[se->dx_api],
                (unsigned long long)tag, se->dx_codec, k->w, k->h, k->pictures,
                (unsigned long long)k->written, rc, (long long)(t1 - t0), (long long)(t0 - t_in));
        if (rc == 0 && k->pictures > 0) {
            vtpv_learned_api[cls] = se->dx_api + 1;
            s->stat_dec++;
            return true;
        }
        /* a stream this decoder does not take: the session moves to the next
         * one once, and stays there (HEVC only - the range extensions) */
        if (se->dx_codec != VTPV_DX_HEVC || se->dx_api + 1 >= VTPV_API_NONE) {
            return false;
        }
        vtpv_dx_discard(se->dx);
        se->dx = NULL;
        do {
            se->dx_api++;       /* a tile session skips cuda on the walk too */
        } while (se->tile_live && se->dx_api == VTPV_API_CUDA);
    }
}

/* decode the ',sbf' of a 'dtil'/'dfrm' and write the picture back */
static bool vtpv_dx_decode_sbf(AppleVtpv *s, uint64_t total, uint64_t tag)
{
    uint32_t sl = 0;
    const uint8_t *sample = vtpv_dx_sbf_sample(s, total, tag, &sl);
    return sample && vtpv_dx_decode_sample(s, s->cur, sample, sl, tag);
}

/* the decode thread: frames whose 'Dbuf' is out, in that (decode) order;
 * each picture is kept as the frame's own tight NV12 until the guest
 * answers with a buffer */
static void *vtpv_dec_thread(void *opaque)
{
    AppleVtpv *s = opaque;
    qemu_mutex_lock(&s->dec_lock);
    for (;;) {
        while (!s->dec_n) {
            qemu_cond_wait(&s->dec_cond, &s->dec_lock);
        }
        VtpvSess *se = s->dec_job[s->dec_head].se;
        int idx = s->dec_job[s->dec_head].idx;
        s->dec_head = (s->dec_head + 1) % ARRAY_SIZE(s->dec_job);
        s->dec_n--;
        s->dec_busy = se;
        uint8_t *sample = se->frame[idx].sample;
        uint32_t sl = se->frame[idx].sample_len;
        uint64_t frame_id = se->frame[idx].frame_id;
        se->frame[idx].sample = NULL;
        qemu_mutex_unlock(&s->dec_lock);

        VtpvDxSink *k = &se->dx_sink;
        uint8_t *pic = NULL;
        uint32_t w = 0, h = 0;
        k->to_stage = true;
        if (sample && vtpv_dx_decode_sample(s, se, sample, sl, frame_id)) {
            pic = k->stage;         /* the frame owns this picture now */
            w = k->w;
            h = k->h;
            k->stage = NULL;
            k->stage_cap = 0;
        }
        k->to_stage = false;
        g_free(sample);

        qemu_mutex_lock(&s->dec_lock);
        se->frame[idx].nv12 = pic;
        se->frame[idx].w = w;
        se->frame[idx].h = h;
        se->frame[idx].decoded = true;
        s->dec_busy = NULL;
        qemu_cond_broadcast(&s->dec_cond);
    }
    return NULL;
}

static bool vtpv_dec_queue(AppleVtpv *s, VtpvSess *se, int idx)
{
    bool ok = false;
    qemu_mutex_lock(&s->dec_lock);
    if (s->dec_n < ARRAY_SIZE(s->dec_job)) {
        uint32_t t = (s->dec_head + s->dec_n) % ARRAY_SIZE(s->dec_job);
        s->dec_job[t].se = se;
        s->dec_job[t].idx = idx;
        s->dec_n++;
        se->frame[idx].queued = true;
        qemu_cond_broadcast(&s->dec_cond);
        ok = true;
    }
    qemu_mutex_unlock(&s->dec_lock);
    return ok;
}

/* a queued frame's picture is needed now: wait for the thread; false if the
 * frame was never queued (decode it in place) */
static bool vtpv_dec_wait(AppleVtpv *s, VtpvSess *se, int idx)
{
    qemu_mutex_lock(&s->dec_lock);
    bool queued = se->frame[idx].queued;
    while (queued && !se->frame[idx].decoded) {
        qemu_cond_wait(&s->dec_cond, &s->dec_lock);
    }
    qemu_mutex_unlock(&s->dec_lock);
    return queued;
}

static void vtpv_hwdecode_dfrm(AppleVtpv *s, uint64_t total);
static const VtpvKickSurf *vtpv_kick_surf(AppleVtpv *s, uint32_t id);

/* a staged tile -> its place (ox, oy) in the caller's canvas. The stage is pw x ph
 * tight luma, then CbCr pairs with chroma shifts (csx, csy) (vtpv_dx_put); w x h
 * of it from (sx, sy) is the tile. The canvas is biplanar 4:2:0 ('420v'/'420f'),
 * 4:2:2 ('422v'/'422f') or 4:4:4 ('444v'/'444f'); its chroma takes the stage's
 * nearest sample when the two subsamplings differ, a row copy when they agree.
 * A one-component 'L008' canvas (a HEIF alpha plane) takes the luma alone; a
 * luma-only stage (gray) gives a biplanar canvas neutral chroma. */
static bool vtpv_fill_tile(AppleVtpv *s, const VtpvKickSurf *ks, const uint8_t *stage, bool gray,
                           uint32_t csx, uint32_t csy, uint32_t pw, uint32_t ph, uint32_t sx,
                           uint32_t sy, uint32_t w, uint32_t h, uint32_t ox, uint32_t oy)
{
    uint32_t kx, ky;
    switch (ks->pixfmt) {
    case 0x34323076u: /* '420v' */
    case 0x34323066u: /* '420f' */
        kx = 1, ky = 1;
        break;
    case 0x34323276u: /* '422v' */
    case 0x34323266u: /* '422f' */
        kx = 1, ky = 0;
        break;
    case 0x34343476u: /* '444v' */
    case 0x34343466u: /* '444f' */
        kx = 0, ky = 0;
        break;
    case 0x4c303038u: /* 'L008' */
        kx = ky = 0;
        break;
    default:
        kx = ky = 2;
        break;
    }
    sx &= ~1u;
    sy &= ~1u;
    bool one = ks->pixfmt == 0x4c303038u;
    if (kx > 1 || ks->planes != (one ? 1u : 2u) || ox >= ks->pw[0] || oy >= ks->ph[0] || sx >= pw || sy >= ph) {
        lab_elog( "apple-vtpv: tile: canvas %u is %.4s with %u planes, %ux%u, tile at %u,%u - "
                "not written\n", ks->id, (const char *)&ks->pixfmt, ks->planes, ks->pw[0],
                ks->ph[0], ox, oy);
        return false;
    }
    uint32_t dw = MIN(MIN(w, pw - sx), ks->pw[0] - ox), dh = MIN(MIN(h, ph - sy), ks->ph[0] - oy);
    for (uint32_t y = 0; y < dh; y++) {
        vtpv_ksurf_bytes(s, ks, ks->off[0] + (uint64_t)(oy + y) * ks->bpr[0] + ox,
                         stage + (size_t)(sy + y) * pw + sx, dw);
    }
    if (one) {
        vtpv_tile_dump(s, ks, stage, gray, pw, ph, sx, sy, w, h, ox, oy, dw, dh);
        return true;
    }
    uint32_t cols = (dw + (1u << kx) - 1) >> kx, rows = (dh + (1u << ky) - 1) >> ky;
    if (gray) {
        g_autofree uint8_t *mid = g_malloc((size_t)cols * 2);
        memset(mid, 0x80, (size_t)cols * 2);
        for (uint32_t r = 0; r < rows && (oy >> ky) + r < ks->ph[1]; r++) {
            vtpv_ksurf_bytes(s, ks, ks->off[1] + (uint64_t)((oy >> ky) + r) * ks->bpr[1] +
                             (uint64_t)(ox >> kx) * 2, mid, cols * 2);
        }
        vtpv_tile_dump(s, ks, stage, gray, pw, ph, sx, sy, w, h, ox, oy, dw, dh);
        return true;
    }
    const uint8_t *chroma = stage + (size_t)pw * ph;
    uint32_t scw = (pw + (1u << csx) - 1) >> csx;           /* stage CbCr pairs a row */
    uint32_t sch = (ph + (1u << csy) - 1) >> csy;
    g_autofree uint8_t *row = g_malloc((size_t)cols * 2 + 2);
    for (uint32_t r = 0; r < rows && (oy >> ky) + r < ks->ph[1]; r++) {
        uint32_t srow = MIN((sy + (r << ky)) >> csy, sch - 1);
        const uint8_t *src = chroma + (size_t)srow * scw * 2;
        const uint8_t *put = row;
        if (kx == csx) {
            put = src + (size_t)(sx >> csx) * 2;
        } else {
            for (uint32_t c = 0; c < cols; c++) {
                uint32_t scol = MIN((sx + (c << kx)) >> csx, scw - 1);
                row[2 * c] = src[2 * scol];
                row[2 * c + 1] = src[2 * scol + 1];
            }
        }
        vtpv_ksurf_bytes(s, ks, ks->off[1] + (uint64_t)((oy >> ky) + r) * ks->bpr[1] +
                         (uint64_t)(ox >> kx) * 2, put, cols * 2);
    }
    vtpv_tile_dump(s, ks, stage, gray, pw, ph, sx, sy, w, h, ox, oy, dw, dh);
    return true;
}

/*
 * 'TDem' after a pause. CMPhoto's tile handler reads state its DecodeTile
 * caller sets up after DecodeTile returns: a host's asynchronous decoder
 * emits a decode time later, ours had the picture before the reply and the
 * handler ran beside the returning caller (TV, Weather, Maps, Podcasts: the
 * handler on a garbage pixel buffer or dictionary, ~1 s after launch, with
 * 'TDem' before the reply as well as right after it). A session's waiting
 * 'TDem's go out before anything that ends its tiles ('dtfi', 'dinv',
 * '-dec') is answered, and with its UUID when that goes.
 */
typedef struct {
    int slot;
    int64_t due_us;
    uint8_t uuid16[16];
    uint32_t n;
    uint8_t paras[128];
    uint32_t type;              /* 0 = 'TDem' (the queue's native cargo) */
    uint32_t flags;
    bool id_set;                /* false = host_msg_id at send time */
    uint8_t id8[8];
    uint64_t gate_kick;         /* tdem-kick-gated: send only after kick_count[slot] passes this */
} VtpvTdem;

static void vtpv_tdem_send(AppleVtpv *s, VtpvTdem *t)
{
    uint8_t id8[8];
    if (t->id_set) {
        memcpy(id8, t->id8, 8);
    } else if (t->type) {
        stq_be_p(id8, 0);       /* queued non-'TDem' cargo carries its own id */
    } else {
        /* 'TDem' is a pure event: identifier 0 - a nonzero id made the
         * guest's dispatcher try DeliverReply and spray -19093 (rfq) */
        stq_be_p(id8, 0);
        s->host_msg_id++;
    }
    vtpv_send_h2g2(s, t->slot, t->type ? t->type : 0x5444656du /* 'TDem' */,
                   t->type ? t->flags : 2, t->uuid16, id8, t->paras, t->n);
}

/* a window held by an ack that never comes (the dispatcher sends none when
 * the 'TDem's group lookup fails) opens again after this long.  25ob (Opus):
 * 250 ms -> 5 ms. Real apps DO ack (Maps stress boot: 1152 acks for 1157
 * 'dtil' on its slot), but the few missing acks each held the next 'TDem'
 * 250 ms - far past an app's ~8 ms patience: it gave the tile up and freed
 * its CMPhoto state, and the late emit's handler ran on freed memory (Maps
 * crash reports 10-25 14:10 / 14:48: EmitDecodedTile -> CMPhoto ->
 * CVPixelBufferGetAttributes / dispatch_semaphore_signal; 10-14 such stalls
 * a boot). Acks normally land in milliseconds, so the reply-before-handler
 * ordering the window buys is kept; a lost ack now costs 5 ms, not a tile. */
#define VTPV_TDEM_STALL_US 5000
/* a 'TDem' waits at most this long for its slot's queue to drain (24g): the
 * probe drained its ring in milliseconds, but a real app's reply pump takes
 * 10-300 ms and gives up on the tile after ~8 ms (ins6b: 24 dtil, 0 TDem out,
 * every session -dec'd before a 3 s drain cap could expire). The real host's
 * emit just follows its decode latency (1-5 ms) - it never waits for empty.
 * 3 ms: probe bursts still drain fully (ordering kept), real apps get their
 * emit inside their patience window. */
#define VTPV_TDEM_DRAIN_US 3000

/* entries of this slot the guest has not taken yet?  The real host's emits
 * follow its actual decode latency, so every 'dtil' reply of a session burst
 * enters the guest's serial queue BEFORE the earlier tiles' emits; ours used
 * to leave 0.2 ms after their reply and each later reply queued behind a
 * 10-70 ms CMPhoto handler - past the reply clerk's 1.1 s at a ~17-session
 * burst, and the whole image fell back to a late guest decode (Opus: every
 * timeout is a second-late, CPU-decoded picture).  Holding a 'TDem' while
 * its queue holds anything gives the replies the real host's ordering; a
 * lone tile's emit goes out at once (the queue is empty then). */
static bool vtpv_h2g_busy(AppleVtpv *s, int slot)
{
    AppleVtpvSlot *sl = &s->slot[slot];
    struct io_dq_header h;
    if (!sl->h2g_gpa ||
        dma_memory_read(pci_get_address_space(PCI_DEVICE(s)), sl->h2g_gpa, &h, sizeof(h),
                        MEMTXATTRS_UNSPECIFIED)) {
        return false;
    }
    return le32_to_cpu(h.head) != le32_to_cpu(h.tail);
}

/* send queued 'TDem's while their slot's unacked window is open: the ack
 * (message flags 8) follows EmitDecodedTile - the tile handler included -
 * so a later 'dtil' reply never queues behind more than tdem_window
 * handlers on the guest's serial dispatch queue */
static void vtpv_tdem_pump(AppleVtpv *s)
{
    int64_t now = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
    /* TEMP lab gate diag: the flag/cold state every 2 s, independent of answers */
    if (s->dbuf_gate_live && now >= s->gate_diag_us) {
        s->gate_diag_us = now + 2000000;
        lab_elog( "apple-vtpv: gate-live flag=%d on=%d cold=%u prev=%u seen=%u\n",
                s->dbuf_gate, s->gate_on, s->gate_cold_win, s->gate_cold_prev, s->gate_seen_n);
    }
    VtpvTdem *t;
    static int p;
    while ((t = g_queue_peek_head(&s->tdem_q)) != NULL) {
        if (s->tdem_kick_gated && !t->type &&
            s->kick_count[t->slot] <= t->gate_kick &&
            now < t->due_us + s->tdem_gate_cap_us) {
            /* not before that client's next kick (or the cap / session flush) */
            timer_mod(s->tdem_timer, t->due_us + s->tdem_gate_cap_us);
            return;
        }
        if (t->due_us > now) {
            if (p < 40) { p++; lab_elog( "apple-vtpv: pump hold due (%lld > %lld) qlen=%d\n", (long long)t->due_us, (long long)now, g_queue_get_length(&s->tdem_q)); }
            timer_mod(s->tdem_timer, t->due_us);
            return;
        }
        if (!s->tdem_fast &&
            s->tdem_unacked[t->slot] >= s->tdem_window &&
            now - s->tdem_last_us < VTPV_TDEM_STALL_US) {
            if (p < 40) { p++; lab_elog( "apple-vtpv: pump hold stall unacked=%d\n", s->tdem_unacked[t->slot]); }
            timer_mod(s->tdem_timer, s->tdem_last_us + VTPV_TDEM_STALL_US);
            return;
        }
        if (!s->tdem_fast &&
            vtpv_h2g_busy(s, t->slot) && now < t->due_us + VTPV_TDEM_DRAIN_US) {
            if (p < 40) { p++; lab_elog( "apple-vtpv: pump hold busy drain (dt %lld us)\n", (long long)(now - t->due_us)); }
            timer_mod(s->tdem_timer, now + 3000);
            return;
        }
        if (s->tdem_fast && !vtpv_dq_room(s, t->slot, 4096)) {
            /* ring-free pacing: no fixed holds, but keep room for the next
             * kick's in-kick reply - it must never wait behind outboxed
             * TDems; recheck on a short timer */
            timer_mod(s->tdem_timer, now + 200);
            return;
        }
        if (p < 40) { p++; lab_elog( "apple-vtpv: pump SEND dt %lld us busy=%d\n", (long long)(now - t->due_us), vtpv_h2g_busy(s, t->slot)); }
        g_queue_pop_head(&s->tdem_q);
        vtpv_tdem_send(s, t);
        if (!t->type) {
            s->tdem_unacked[t->slot]++;
            s->tdem_last_us = now;
        }
        g_free(t);
    }
}

static void vtpv_tdem_tick(void *opaque)
{
    static int n;
    if (n < 40) {
        n++;
        lab_elog( "apple-vtpv: tdem tick #%d qlen=%d\n", n,
                g_queue_get_length(&((AppleVtpv *)opaque)->tdem_q));
    }
    vtpv_tdem_pump(opaque);
}

/* send (or with drop, forget) the waiting 'TDem's of one session, oldest first */
static void vtpv_tdem_flush(AppleVtpv *s, const uint8_t *uuid16, bool drop)
{
    for (GList *l = s->tdem_q.head; l;) {
        GList *next = l->next;
        VtpvTdem *t = l->data;
        if (memcmp(t->uuid16, uuid16, 16) == 0) {
            g_queue_delete_link(&s->tdem_q, l);
            if (!drop) {
                vtpv_tdem_send(s, t);
            }
            g_free(t);
        }
        l = next;
    }
}

/* forget everything queued for a slot: its queue was wedged or its client is
 * gone (vtpv_h2g_watch) */
static void vtpv_tdem_purge_slot(AppleVtpv *s, int slot)
{
    for (GList *l = s->tdem_q.head; l;) {
        GList *next = l->next;
        VtpvTdem *t = l->data;
        if (t->slot == slot) {
            g_queue_delete_link(&s->tdem_q, l);
            g_free(t);
        }
        l = next;
    }
    s->tdem_unacked[slot] = 0;
}

/* the 'TDem' of one tile - success or error: never both leaves the guest's
 * tile record queued and the surface group alive.  kVTDecodeInfo_
 * Asynchronous in it too: the real host's decoder finishes a
 * paravirtualized tile asynchronously and passes the same flag to its emit. */
static void vtpv_tile_tdem(AppleVtpv *s, int slot, const uint8_t *uuid16, uint64_t tile_id,
                           int32_t err, const uint8_t *uui)
{
    uint8_t paras[128];
    uint32_t n = 0;
    int32_t be_err = cpu_to_be32(err);
    uint32_t be_inf = cpu_to_be32(1);
    uint64_t be_til = cpu_to_be64(tile_id);
    if (s->tdem_ver) {
        uint32_t be_ver = cpu_to_be32(6);
        n += vtpv_para_put(paras + n, 0x2c766572u /* ',ver' */, "=u32", &be_ver, 4);
    }
    n += vtpv_para_put(paras + n, 0x2c74696cu /* ',til' */, "=s64", &be_til, 8);
    n += vtpv_para_put(paras + n, 0x2c657272u /* ',err' */, "=s32", &be_err, 4);
    n += vtpv_para_put(paras + n, 0x2c696e66u /* ',inf' */, "=u32", &be_inf, 4);
    if (uui && s->tile_uui) {
        n += vtpv_para_put(paras + n, 0x2c757569u /* ',uui' */, "=uui", uui, 16);
    }
    VtpvTdem *t = g_new0(VtpvTdem, 1);
    t->slot = slot;
    memcpy(t->uuid16, uuid16, 16);
    memcpy(t->paras, paras, n);
    t->n = n;
    t->due_us = qemu_clock_get_us(QEMU_CLOCK_REALTIME) + s->tdem_delay_us;
    /* a tile's 'TDem' never passes its reply: EmitDecodedTile before the
     * reply completes DecodeTile is the caller's-head-start crash class */
    if (t->due_us < s->reply_due_us[slot] + 1000) {
        t->due_us = s->reply_due_us[slot] + 1000;
    }
    t->gate_kick = s->kick_count[slot];
    g_queue_push_tail(&s->tdem_q, t);
    vtpv_tdem_pump(s);
}

/* the 'dtil' reply leaves the kick: our in-kick reply reached the guest's
 * dispatcher before the sender had returned from the kick MMIO and the reply
 * clerk's entry was there to receive it (-19093 in DeliverReply, then the
 * wait's timeout, then CMPhoto's CPU fallback).  The queue's timer sends it
 * dtil_reply_delay_us after the kick; stock flags 4 (bit 3 on this wire
 * means EVENT - rfq3/rfq4: with 8 the guest logs "unhandled message dtil"). */
static void vtpv_queue_dtil_reply(AppleVtpv *s, int slot, const uint8_t *paras, uint32_t n)
{
    if (s->dtil_reply_delay_us == 0) {
        /* the one shape never actually tested: the reply leaves INSIDE the
         * kick MMIO - the kext signals the port at kick return and the pump
         * completes the clerk while the sender is still in the syscall */
        s->reply_due_us[slot] = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
        vtpv_send_h2g2(s, slot, 0x6474696cu /* 'dtil' */, s->dtil_reply_flags,
                       s->msg + 0x10, s->msg + 0x20, paras, n);
        return;
    }
    VtpvTdem *t = g_new0(VtpvTdem, 1);
    t->slot = slot;
    t->type = 0x6474696cu /* 'dtil' */;
    t->flags = s->dtil_reply_flags;
    t->id_set = true;
    memcpy(t->id8, s->msg + 0x20, 8);
    memcpy(t->uuid16, s->msg + 0x10, 16);
    memcpy(t->paras, paras, MIN(n, sizeof(t->paras)));
    t->n = n;
    t->due_us = qemu_clock_get_us(QEMU_CLOCK_REALTIME) + s->dtil_reply_delay_us;
    s->reply_due_us[slot] = t->due_us;
    g_queue_push_tail(&s->tdem_q, t);
    vtpv_tdem_pump(s);
}

/* one 'dtil' for the tile thread: everything the kick lent us is copied (the
 * message buffer and the surface descriptor are reused by the next kick; the
 * canvas pages themselves stay pinned by the kext until our done word) */
typedef struct VtpvTileJob {
    int slot;
    VtpvSess *se;
    uint8_t uuid16[16];
    uint64_t tile_id;
    VtpvKickSurf ks;               /* .pages owned */
    uint32_t sx, sy, tw, th, ox, oy;
    bool has_uui;
    uint8_t uui[16];
    uint8_t *sample;
    uint32_t sl;
    uint64_t hold_gpa;
    uint32_t token;
    bool cancel;                   /* its session ended: no fill, no 'TDem' - done still advances */
    bool ok;
    uint8_t *pic;                  /* the decoded stage (vtpv_dx_put layout) */
    uint32_t w, h;
    uint8_t csx, csy;
    bool gray;
} VtpvTileJob;

static void *vtpv_tile_thread(void *opaque)
{
    AppleVtpv *s = opaque;
    qemu_mutex_lock(&s->dec_lock);
    for (;;) {
        VtpvTileJob *j;
        while (!(j = g_queue_pop_head(&s->tile_q))) {
            qemu_cond_wait(&s->dec_cond, &s->dec_lock);
        }
        s->tile_busy = j;
        bool cancel = j->cancel;
        /* one decoder per session: if the VIDEO thread is decoding this
         * session's frames, wait it out (it holds no lock while decoding,
         * so no cycle); the tile side of the pair is the drain below */
        while (s->dec_busy == j->se && !cancel) {
            qemu_cond_wait(&s->dec_cond, &s->dec_lock);
        }
        qemu_mutex_unlock(&s->dec_lock);
        if (!cancel) {
            VtpvDxSink *k = &j->se->dx_sink;
            k->to_stage = true;
            if (vtpv_dx_decode_sample(s, j->se, j->sample, j->sl, j->tile_id)) {
                j->pic = k->stage;  /* the job owns this picture now */
                k->stage = NULL;
                k->stage_cap = 0;
                j->w = k->w;
                j->h = k->h;
                j->csx = k->csx;
                j->csy = k->csy;
                j->gray = k->cgray;
                j->ok = true;
            }
            k->to_stage = false;
        }
        g_free(j->sample);
        j->sample = NULL;
        qemu_mutex_lock(&s->dec_lock);
        s->tile_busy = NULL;
        g_queue_push_tail(&s->tile_done, j);
        qemu_cond_broadcast(&s->dec_cond);
        qemu_bh_schedule(s->tile_bh);
    }
    return NULL;
}

/* bounded serial-log budget for the tile-stage timing prints (kfill/ktile) */
static int g_klog;
static int g_probe;   /* canvas-probe readback budget */

/* decoded tiles -> their canvases, the kext's done word, their 'TDem's (BQL;
 * in decode order, so done only ever moves forward) */
static void vtpv_tile_complete(AppleVtpv *s)
{
    AddressSpace *as = pci_get_address_space(PCI_DEVICE(s));
    for (;;) {
        qemu_mutex_lock(&s->dec_lock);
        VtpvTileJob *j = g_queue_pop_head(&s->tile_done);
        qemu_mutex_unlock(&s->dec_lock);
        if (!j) {
            break;
        }
        int32_t err = -12909;       /* kVTVideoDecoderBadDataErr */
        int64_t f0 = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
        if (!j->cancel && j->ok &&
            vtpv_fill_tile(s, &j->ks, j->pic, j->gray, j->csx, j->csy, j->w, j->h, j->sx, j->sy,
                           j->tw ? j->tw : j->w, j->th ? j->th : j->h, j->ox, j->oy)) {
            vtpv_note_surface(&j->ks);
            err = 0;
            if (s->canvas_probe && j->ks.npages && qatomic_fetch_inc(&g_probe) < 40) {
                /* read back the canvas's first bytes in the SAME context that just
                 * wrote them (this BH's dma writes are the fill itself): zeros here
                 * would mean the fill's writes vanish before the guest can read */
                uint8_t rb[32] = { 0 };
                dma_memory_read(as, (j->ks.pages[0] & ~(uint64_t)0xfff) + (j->ks.org & 0xfff),
                                rb, sizeof(rb), MEMTXATTRS_UNSPECIFIED);
                lab_elog( "apple-vtpv: probe tile=%llu canvas=%u rb0=",
                        (unsigned long long)j->tile_id, j->ks.id);
                for (int k = 0; k < 16; k++) {
                    lab_elog( "%02x", rb[k]);
                }
                lab_elog( " nz=%d\n", rb[0] || rb[1] || rb[2] || rb[3]);
            }
        }
        int64_t f1 = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
        if (qatomic_fetch_inc(&g_klog) < 120 && f1 - f0 > 500) {
            lab_elog( "apple-vtpv: kfill %lld us (canvas %u %ux%u)\n",
                    (long long)(f1 - f0), j->ks.id, j->ks.pw[0], j->ks.ph[0]);
        }
        uint32_t le_done = cpu_to_le32(j->token);
        dma_memory_write(as, j->hold_gpa + 4, &le_done, 4, MEMTXATTRS_UNSPECIFIED);
        if (!j->cancel) {
            vtpv_tile_tdem(s, j->slot, j->uuid16, j->tile_id, err, j->has_uui ? j->uui : NULL);
        }
        if (s->verbose) {
            lab_elog( "apple-vtpv: tile %llu -> canvas %u (%.4s) at %u,%u err=%d token %u%s\n",
                    (unsigned long long)j->tile_id, j->ks.id, (const char *)&j->ks.pixfmt, j->ox,
                    j->oy, err, j->token, j->cancel ? " (session gone)" : ", TDem queued");
        }
        g_free(j->pic);
        g_free(j->ks.pages);
        g_free(j);
    }
}

static void vtpv_tile_bh(void *opaque)
{
    vtpv_tile_complete(opaque);
}

/* wait out a session's tiles on the thread (cancel: they only release their
 * pins), then complete what is done - before the session's decoder goes or a
 * message that ends its tiles is answered */
static void vtpv_tile_drain(AppleVtpv *s, VtpvSess *se, bool cancel)
{
    if (!se) {
        return;
    }
    qemu_mutex_lock(&s->dec_lock);
    for (;;) {
        bool pending = s->tile_busy && s->tile_busy->se == se;
        if (pending && cancel) {
            s->tile_busy->cancel = true;
        }
        for (GList *l = s->tile_q.head; l; l = l->next) {
            VtpvTileJob *j = l->data;
            if (j->se == se) {
                j->cancel |= cancel;
                pending = true;
            }
        }
        if (!pending) {
            break;
        }
        qemu_cond_wait(&s->dec_cond, &s->dec_lock);
    }
    if (cancel) {
        for (GList *l = s->tile_done.head; l; l = l->next) {
            VtpvTileJob *j = l->data;
            if (j->se == se) {
                j->cancel = true;
            }
        }
    }
    qemu_mutex_unlock(&s->dec_lock);
    vtpv_tile_complete(s);
}

/* 'dtil' on the D3D12 decoder - the real host decoder's shape: the reply
 * goes out at once (',err'=0, ',inf'=kVTDecodeInfo_Asynchronous - the tile
 * was ACCEPTED), the decode runs after it and the outcome travels in the
 * 'TDem' (',til', ',err' = 0 or the decode error, ',inf', ',uui'; message
 * flags 2).  The guest's tile handler runs only on 'TDem'.  Replying only
 * after the decode put the 'TDem' 0.2 ms behind the reply - the handler ran
 * beside the caller still returning out of DecodeTile, and CMPhoto's tile
 * state was freed or half-armed (Maps/TV/Weather/Podcasts crashes on the
 * second tile of a session, every build through 24a); the real host's decode
 * latency is the caller's head start.  A 'TDem' with its error also consumes
 * the guest's queued tile record and its surface-group entry on the paths a
 * sync-error reply used to leak them.  With a kext that holds the canvas
 * (VTPV_KICK_HOLD) the decode leaves the kick for the tile thread (25p). */
static void vtpv_dtil_dx(AppleVtpv *s, int slot, uint64_t total, uint64_t tile_id,
                         const VtpvKickSurf *ks, uint32_t sx, uint32_t sy, uint32_t tw,
                         uint32_t th, uint32_t ox, uint32_t oy, const uint8_t *uui)
{
    /* 25rm regression hunt (g0b/g1b: reply p50 ~860 ms with the gate OFF):
     * bounded per-stage timing of the dtil kick, to the serial log. */
    bool ktiming = qatomic_fetch_inc(&g_klog) < 120;
    int64_t kt0 = ktiming ? qemu_clock_get_us(QEMU_CLOCK_REALTIME) : 0;
    int64_t kt1 = 0, kt2 = 0;
    uint8_t paras[96];
    uint32_t n = 0;
    int32_t be_ok = 0;
    uint32_t be_inf_async = cpu_to_be32(1);
    uint32_t be_ver = cpu_to_be32(6);
    vtpv_tile_off_cuda(s->cur);
    if (s->tile_sync) {
        /* 25r A/B: the synchronous contract - decode into the canvas first,
         * then one reply (',err' = the outcome, ',inf' = 0: done, no 'TDem');
         * the guest emits the tile from its canvas itself (11e: what it does
         * on an ',inf' 0 reply - it failed then only for want of a canvas) */
        int32_t err = -12909;
        uint32_t sl = 0;
        const uint8_t *sample = vtpv_dx_sbf_sample(s, total, tile_id, &sl);
        if (sample) {
            vtpv_tile_drain(s, s->cur, false);   /* one sink per session - see the branch below */
            VtpvDxSink *k = &s->cur->dx_sink;
            bool was_stage = k->to_stage;
            k->to_stage = true;
            bool ok = vtpv_dx_decode_sample(s, s->cur, sample, sl, tile_id);
            k->to_stage = was_stage;
            if (ok && vtpv_fill_tile(s, ks, k->stage, k->cgray, k->csx, k->csy, k->w, k->h, sx, sy,
                                     tw ? tw : k->w, th ? th : k->h, ox, oy)) {
                vtpv_note_surface(ks);
                err = 0;
            }
        }
        int32_t be_err = cpu_to_be32(err);
        uint32_t be_inf0 = 0;
        n += vtpv_para_put(paras + n, 0x2c766572u /* ',ver' */, "=u32", &be_ver, 4);
        n += vtpv_para_put(paras + n, 0x2c657272u /* ',err' */, "=s32", &be_err, 4);
        n += vtpv_para_put(paras + n, 0x2c696e66u /* ',inf' */, "=u32", &be_inf0, 4);
        vtpv_queue_dtil_reply(s, slot, paras, n);
        if (s->verbose) {
            lab_elog( "apple-vtpv: tile %llu -> canvas %u sync err=%d\n",
                    (unsigned long long)tile_id, ks->id, err);
        }
        return;
    }
    /* 25q: ',ver' first, as in every reply the guest's real host sends (our
     * template replies carry it) - the 'dtil' reply alone went without */
    n += vtpv_para_put(paras + n, 0x2c766572u /* ',ver' */, "=u32", &be_ver, 4);
    n += vtpv_para_put(paras + n, 0x2c657272u /* ',err' */, "=s32", &be_ok, 4);
    n += vtpv_para_put(paras + n, 0x2c696e66u /* ',inf' */, "=u32", &be_inf_async, 4);
    vtpv_queue_dtil_reply(s, slot, paras, n);
    if (ktiming) {
        kt1 = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
    }

    int32_t err = -12909;           /* kVTVideoDecoderBadDataErr */
    uint32_t sl = 0;
    const uint8_t *sample = s->tile_fake_error ?
        NULL : vtpv_dx_sbf_sample(s, total, tile_id, &sl);
    if (sample && s->tile_async && s->kick_hold_gpa) {
        VtpvTileJob *j = g_new0(VtpvTileJob, 1);
        j->slot = slot;
        j->se = s->cur;
        memcpy(j->uuid16, s->msg + 0x10, 16);
        j->tile_id = tile_id;
        j->ks = *ks;
        j->ks.pages = g_memdup2(ks->pages, (gsize)(ks->npages ? ks->npages : 1) * 8);
        j->sx = sx, j->sy = sy, j->tw = tw, j->th = th, j->ox = ox, j->oy = oy;
        if (uui) {
            j->has_uui = true;
            memcpy(j->uui, uui, 16);
        }
        j->sample = g_memdup2(sample, sl);
        j->sl = sl;
        j->hold_gpa = s->kick_hold_gpa;
        if (!++s->tile_token[slot]) {
            ++s->tile_token[slot];  /* 0 means "not held" */
        }
        j->token = s->tile_token[slot];
        uint32_t le_tok = cpu_to_le32(j->token);
        dma_memory_write(pci_get_address_space(PCI_DEVICE(s)), j->hold_gpa, &le_tok, 4,
                         MEMTXATTRS_UNSPECIFIED);
        qemu_mutex_lock(&s->dec_lock);
        g_queue_push_tail(&s->tile_q, j);
        qemu_cond_broadcast(&s->dec_cond);
        qemu_mutex_unlock(&s->dec_lock);
        if (ktiming) {
            kt2 = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
            lab_elog( "apple-vtpv: ktile #%d async reply=%lld job=%lld us\n",
                    (int)g_klog, (long long)(kt1 - kt0), (long long)(kt2 - kt1));
        }
        return;
    }
    if (sample) {
        /* the session's sink is ONE decoder: the tile thread may still be
         * decoding this session's earlier jobs (the in-kick branch runs when
         * the kext lent no hold slot - its table full or the lookup failed);
         * concurrent sink use tore the ffmpeg/dx state and crashed QEMU
         * (0xc0000005 in the decode path, both 40 ms rfq8 arms).  The jobs
         * hold se, so the wait also keeps j->se alive here. */
        vtpv_tile_drain(s, s->cur, false);
        VtpvDxSink *k = &s->cur->dx_sink;
        bool was_stage = k->to_stage;
        k->to_stage = true;
        bool ok = vtpv_dx_decode_sample(s, s->cur, sample, sl, tile_id);
        k->to_stage = was_stage;
        if (ok && vtpv_fill_tile(s, ks, k->stage, k->cgray, k->csx, k->csy, k->w, k->h, sx, sy, tw ? tw : k->w,
                                 th ? th : k->h, ox, oy)) {
            vtpv_note_surface(ks);
            err = 0;
        }
    }
    vtpv_tile_tdem(s, slot, s->msg + 0x10, tile_id, err, uui);
    if (ktiming) {
        kt2 = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
        lab_elog( "apple-vtpv: ktile #%d inkick reply=%lld dec=%lld us\n",
                (int)g_klog, (long long)(kt1 - kt0), (long long)(kt2 - kt1));
    }
    if (s->verbose) {
        lab_elog( "apple-vtpv: tile %llu -> canvas %u (%.4s) at %u,%u err=%d, TDem queued\n",
                (unsigned long long)tile_id, ks->id, (const char *)&ks->pixfmt, ox, oy, err);
    }
}

static void vtpv_handle_message(AppleVtpv *s, int slot, uint64_t total,
                                uint32_t nsurf)
{
    AppleVtpvSlot *sl = &s->slot[slot];
    s->kick_count[slot]++;      /* tdem-kick-gated: this client kicked again */

    if (total < 0x30 || total > sl->g2h_len || total > VTPV_MSG_MAX) {
        if (s->verbose) {
            lab_elog( "apple-vtpv: slot%d bad message size %llu\n",
                    slot, (unsigned long long)total);
        }
        return;
    }
    vtpv_msg_realloc(s, total);
    if (dma_memory_read(pci_get_address_space(PCI_DEVICE(s)), sl->g2h_gpa, s->msg, total, MEMTXATTRS_UNSPECIFIED)) {
        lab_elog( "apple-vtpv: slot%d g2h dma read failed\n", slot);
        return;
    }
    s->msgs_in++;
    if (vtpv_outbox_flush(s, slot)) {
        vtpv_raise(s, slot);        /* the guest is active: hand it what waited */
    }

    uint32_t be_type, be_flags;
    memcpy(&be_type, s->msg + 8, 4);
    memcpy(&be_flags, s->msg + 12, 4);
    uint32_t type = be32_to_cpu(be_type);
    uint32_t flags = be32_to_cpu(be_flags);
    s->cur = vtpv_sess_get(s, s->msg + 0x10);
    char t4[5];
    vtpv_log4cc(t4, type);
    if (type == 0x64746669u /* 'dtfi' */ || type == 0x64696e76u /* 'dinv' */) {
        vtpv_tile_drain(s, s->cur, false);          /* its decodes finished, */
        vtpv_tdem_flush(s, s->msg + 0x10, false);   /* its tiles out before the answer */
    } else if (type == 0x2d646563u /* '-dec' */) {
        vtpv_tile_drain(s, s->cur, true);
        vtpv_tdem_flush(s, s->msg + 0x10, true);    /* the decoder is gone */
    } else if (type == 0x5444656du /* 'TDem' */ && (flags & 8)) {
        /* the ack of a 'TDem' we sent: EmitDecodedTile - the guest's tile
         * handler included - has returned, a window slot is free */
        if (s->tdem_unacked[slot]) {
            s->tdem_unacked[slot]--;
        }
        vtpv_tdem_pump(s);
    }

    FILE *f = s->msglog ? fopen(s->msglog, "a") : NULL;
    if (f) {
        fprintf(f, "=== IN slot=%d time=%lld size=%llu type=%s flags=%#x surf=[",
                slot, (long long)qemu_clock_get_ns(QEMU_CLOCK_REALTIME),
                (unsigned long long)total, t4, flags);
        for (uint32_t i = 0; i < nsurf; i++) {
            fprintf(f, "%s%u", i ? "," : "", sl->surf[i]);
        }
        fprintf(f, "] uuid=");
        vtpv_hex(f, s->msg + 16, 16);
        fprintf(f, "\n");
        vtpv_hex(f, s->msg, total);
        fclose(f);
    }
    if (s->verbose) {
        lab_elog( "apple-vtpv: slot%d msg size=%llu type=%s flags=%#x "
                "nsurf=%u first=%02x %02x %02x %02x\n", slot,
                (unsigned long long)total, t4, flags, nsurf,
                s->msg[0x20], s->msg[0x21], s->msg[0x22], s->msg[0x23]);
    }

    /* Stage 4: the non-tile 'dfrm' is sent async, but the framework still
     * arms a reply clerk on it (nA33 guest log: "timeout waiting for pending
     * reply with identifier 103, messageType 'dfrm'") - so it MUST get the
     * generic ',err'+',inf' reply or DecodeFrame completes with -19092.
     * The 'Dbuf'/'Demt' buffer dance stays behind mock_dbuf=on for the
     * write-back experiments; reply first, dance after. */
    bool non_tile_dfrm = false;
    if (type == 0x6466726du /* 'dfrm' */) {
        uint32_t vlen = 0;
        const uint8_t *til = vtpv_para_find(s->msg, total, 0x2c74696cu /* ',til' */, &vlen);
        const uint8_t *frm = vtpv_para_find(s->msg, total, 0x2c66726du /* ',frm' */, &vlen);
        non_tile_dfrm = !til && frm && vlen == 8;
        if (non_tile_dfrm) {
            s->last_frame_id = ldq_be_p(frm);
            uint32_t dlen = 0;
            const uint8_t *dfl = vtpv_para_find(s->msg, total, 0x2c64666cu /* ',dfl' */, &dlen);
            s->cur_dfl = dfl && dlen == 4 ? ldl_be_p(dfl) : 0;
            bool fresh = true;
            for (uint32_t k = 0; k < s->dfl_seen_n; k++) {
                fresh &= s->dfl_seen[k] != s->cur_dfl;
            }
            if (fresh && s->dfl_seen_n < ARRAY_SIZE(s->dfl_seen)) {
                s->dfl_seen[s->dfl_seen_n++] = s->cur_dfl;
                lab_elog( "apple-vtpv: dfrm dfl=0x%x first seen (frame %llu, dno=%d)\n",
                        s->cur_dfl, (unsigned long long)s->last_frame_id, s->dno);
            }
            if (s->cur_dfl & (1u << 1)) {
                s->dfl_dno_n++;
                if ((s->dfl_dno_n & 255) == 1) {
                    lab_elog( "apple-vtpv: dfrm DoNotOutputFrame n=%llu shed_total=%llu\n",
                            (unsigned long long)s->dfl_dno_n,
                            (unsigned long long)s->shed_total);
                }
            }
            if (s->outbuf && s->hwdecode && vtpv_use_dx(s)) {
                /* the first picture of a stream is decoded now - its size
                 * goes into 'Dpba' before the first 'Dbuf'; later samples
                 * wait compressed and are decoded straight into the buffer
                 * the guest answers 'Dbuf' with (an app may queue a whole
                 * GOP ahead: ox5b's 1080p/4K probes overflowed a table of
                 * decoded pictures) */
                uint32_t smp_len = 0;
                const uint8_t *smp = vtpv_dx_sbf_sample(s, total, s->last_frame_id, &smp_len);
                /* the session's dx sink belongs to the decode thread: this
                 * thread touches it only after draining (ox16b: zeroing its
                 * picture count here raced a decode, the picture was taken
                 * for none and the frame went out dropped) */
                s->ob_first = false;
                s->ob_sample = NULL;
                s->ob_sample_len = 0;
                if (smp && !s->cur->dpba_w) {
                    vtpv_dec_drain(s, s->cur);
                    s->cur->dx_sink.to_stage = true;
                    s->ob_first = vtpv_dx_decode_sample(s, s->cur, smp, smp_len,
                                                        s->last_frame_id);
                    s->cur->dx_sink.to_stage = false;
                } else if (smp) {
                    /* dec-tid-max: a temporal layer above the cap is answered
                     * "dropped" right here - no decode, no buffer dance, no 4K
                     * copy into the guest; the tid<=cap sublayer stays intact */
                    int tid = vtpv_sample_tid(smp, smp_len);
                    if (s->dec_tid_max >= 0 && tid >= 0 && tid > s->dec_tid_max) {
                        s->stat_tidskip++;
                        if (s->stat_tidskip <= 3 || s->stat_tidskip % 256 == 0) {
                            lab_elog("apple-vtpv: tid-skip #%llu (tid=%d > %d)\n",
                                    (unsigned long long)s->stat_tidskip,
                                    tid, s->dec_tid_max);
                        }
                    } else {
                        s->ob_sample = smp;     /* inside s->msg: copied by vtpv_send_dbuf */
                        s->ob_sample_len = smp_len;
                    }
                }
            } else if (s->hwdecode && s->writeback && s->surfmap_count) {
                if (vtpv_use_dx(s)) {
                    vtpv_dx_decode_sbf(s, total, s->last_frame_id);
                } else {
                    vtpv_hwdecode_dfrm(s, total);
                }
            }
        }
    }
    if (type == 0x6474696cu /* 'dtil' */) {
        /* Tile decode request: sync send with the caller's canvas IOSurface
         * attached (surf[0]); the guest-side sender reads back only
         * ',err'(s32) + ',inf'(u32).  Parse the tile geometry for the wire
         * log; reply directly so the template dance cannot desync it. */
        uint32_t vlen = 0;
        const uint8_t *til = vtpv_para_find(s->msg, total, 0x2c74696cu, &vlen);
        const uint8_t *tco = vtpv_para_find(s->msg, total, 0x2c74636fu, &vlen);
        const uint8_t *tcd = vtpv_para_find(s->msg, total, 0x2c746364u, &vlen);
        const uint8_t *opb = vtpv_para_find(s->msg, total, 0x2c6f7062u, &vlen);
        uint64_t tile_id = til ? ldq_be_p(til) : 0;
        lab_elog( "apple-vtpv: dtil tile=%llu surf=[%u,%u] "
                "tco=%d,%d tcd=%dx%d opb=%d,%d\n",
                (unsigned long long)tile_id,
                sl->surf[0], sl->surf[1],
                tco ? (int32_t)ldl_be_p(tco) : -1,
                tco ? (int32_t)ldl_be_p(tco + 4) : -1,
                tcd ? (int32_t)ldl_be_p(tcd) : -1,
                tcd ? (int32_t)ldl_be_p(tcd + 4) : -1,
                opb ? (int32_t)ldl_be_p(opb) : -1,
                opb ? (int32_t)ldl_be_p(opb + 4) : -1);
        const VtpvKickSurf *canvas = nsurf ? vtpv_kick_surf(s, sl->surf[0]) : NULL;
        if (s->hwdecode && canvas && vtpv_use_dx(s)) {
            uint32_t ulen = 0;
            const uint8_t *uui = vtpv_para_find(s->msg, total, 0x2c757569u /* ',uui' */, &ulen);
            int32_t ox = opb ? (int32_t)ldl_be_p(opb) : 0, oy = opb ? (int32_t)ldl_be_p(opb + 4) : 0;
            int32_t tw = tcd ? (int32_t)ldl_be_p(tcd) : 0, th = tcd ? (int32_t)ldl_be_p(tcd + 4) : 0;
            int32_t sx = tco ? (int32_t)ldl_be_p(tco) : 0, sy = tco ? (int32_t)ldl_be_p(tco + 4) : 0;
            vtpv_dtil_dx(s, slot, total, tile_id, canvas, MAX(sx, 0), MAX(sy, 0), MAX(tw, 0),
                         MAX(th, 0), MAX(ox, 0), MAX(oy, 0), ulen == 16 ? uui : NULL);
            return;
        }
        if (s->hwdecode && s->writeback && s->surfmap_count && vtpv_use_dx(s)) {
            vtpv_dx_decode_sbf(s, total, tile_id);
        } else if (s->hwdecode && s->writeback && s->surfmap_count) {
            /* hardware path: the dtil ',sbf' is a serialized CMSampleBuffer
             * (atoms [BE32 len][4cc]): 'sdat' = sample in AVCC
             * ([BE32 nal_len][NAL]...), 'fdsc' = avcC with SPS/PPS.  Decode
             * on the host GPU and write NV12 into the canvas pages. */
            uint32_t sbf_len = 0;
            const uint8_t *sbf = vtpv_para_find(s->msg, total,
                                                 0x2c736266u /* ',sbf' */,
                                                 &sbf_len);
            bool decoded = false;
            static uint8_t frame[256 * 1024];
            static uint8_t nv12[4096 * 2304 * 3 / 2];
            if (sbf && sbf_len > 8 && sbf_len < sizeof(frame) &&
                getenv("VTPV_DUMP_SBF") && vtpv_memfind(sbf, sbf_len, "fdsc", 4)) {
                FILE *df = fopen(getenv("VTPV_DUMP_SBF"), "wb");
                if (df) {
                    fwrite(sbf, 1, sbf_len, df);
                    fclose(df);
                }
            }
            if (sbf && sbf_len > 16 && sbf_len < sizeof(frame)) {
                const uint8_t *sps = NULL, *pps = NULL;
                uint32_t sps_l = 0, pps_l = 0;
                uint32_t slice_off[64];
                uint32_t nslices = 0;
                uint32_t flen = 0;
                vtpv_sbf_codec_first(s, sbf, sbf_len);
                /* flat tag scan: the payload atoms nest inside 'sbuf', so
                 * walk one level but descend into every well-formed atom */
                uint32_t aoff = 8;   /* skip the outer 'sbuf' header */
                while (aoff + 8 <= sbf_len) {
                    /* find the next tag start at or after aoff whose length
                     * field is sane (handles the nesting without recursion) */
                    uint32_t aln = 0;
                    const uint8_t *tagb = NULL;
                    uint32_t scan = aoff;
                    while (scan + 8 <= sbf_len) {
                        uint32_t l = ((uint32_t)sbf[scan] << 24) |
                                     ((uint32_t)sbf[scan + 1] << 16) |
                                     ((uint32_t)sbf[scan + 2] << 8) |
                                     sbf[scan + 3];
                        if (l >= 8 && scan + l <= sbf_len &&
                            (memcmp(sbf + scan + 4, "sdat", 4) == 0 ||
                             memcmp(sbf + scan + 4, "fdsc", 4) == 0)) {
                            aln = l;
                            tagb = sbf + scan + 4;
                            aoff = scan;
                            break;
                        }
                        scan++;
                    }
                    if (!tagb) {
                        break;
                    }
                    const uint8_t *pl = sbf + aoff + 8;
                    uint32_t pln = aln - 8;
                    if (memcmp(tagb, "sdat", 4) == 0) {
                        uint32_t j = 0;
                        while (j + 4 <= pln && nslices < 64) {
                            uint32_t nl = ((uint32_t)pl[j] << 24) |
                                          ((uint32_t)pl[j + 1] << 16) |
                                          ((uint32_t)pl[j + 2] << 8) |
                                          pl[j + 3];
                            if (nl < 1 || j + 4 + nl > pln) {
                                break;
                            }
                            const uint8_t *nal = pl + j + 4;
                            /* H.264: 1-byte header, type = b&0x1f (1/5);
                             * HEVC: 2-byte header, type = (b>>1)&0x3f (<32) */
                            int t = s->dec_codec ?
                                    (((nal[0] >> 1) & 0x3f) < 32 ? 1 : 0)
                                    : ((nal[0] & 0x1f) == 1 ||
                                       (nal[0] & 0x1f) == 5) ? 1 : 0;
                            if (t &&
                                flen + 3 + nl < sizeof(frame)) {
                                slice_off[nslices] = flen;
                                frame[flen] = 0;
                                frame[flen + 1] = 0;
                                frame[flen + 2] = 1;
                                memcpy(frame + flen + 3, nal, nl);
                                flen += 3 + nl;
                                nslices++;
                            }
                            j += 4 + nl;
                        }
                    } else if (memcmp(tagb, "fdsc", 4) == 0) {
                        /* codec detect: avc1 -> H.264 (avcC record);
                         * hvc1/hev1 -> HEVC (hvcC arrays: VPS 32/SPS 33/PPS 34,
                         * 2-byte NAL lengths) */
                        if (vtpv_memfind(pl, pln, "hvc1", 4) ||
                            vtpv_memfind(pl, pln, "hev1", 4)) {
                            s->dec_codec = 1;
                            const uint8_t *hvcc = vtpv_memfind(pl, pln, "hvcC", 4);
                            if (hvcc && hvcc + 25 < pl + pln) {
                                const uint8_t *hc = hvcc + 4;
                                uint32_t narr = hc[22];  /* numOfArrays */
                                uint32_t o = 23;         /* first array */
                                for (uint32_t a = 0;
                                     a < narr && o + 3 <= pln - (hc - pl); a++) {
                                    uint32_t ntype = hc[o] & 0x3f;
                                    uint32_t nnal = (hc[o + 1] << 8) | hc[o + 2];
                                    o += 3;
                                    for (uint32_t q = 0;
                                         q < nnal && o + 2 <= pln - (hc - pl);
                                         q++) {
                                        uint32_t nl = (hc[o] << 8) | hc[o + 1];
                                        o += 2;
                                        if (nl > 0 &&
                                            o + nl <= pln - (hc - pl)) {
                                            if (ntype == 32 && nl <= sizeof(s->h265_vps)) {
                                                memcpy(s->h265_vps, hc + o, nl);
                                                s->h265_vps_len = nl;
                                            } else if (ntype == 33 && nl <= sizeof(s->h265_sps)) {
                                                memcpy(s->h265_sps, hc + o, nl);
                                                s->h265_sps_len = nl;
                                            } else if (ntype == 34 && nl <= sizeof(s->h265_pps)) {
                                                memcpy(s->h265_pps, hc + o, nl);
                                                s->h265_pps_len = nl;
                                            }
                                        }
                                        o += nl;
                                    }
                                }
                                lab_elog( "apple-vtpv: HEVC hvcC cached "
                                        "vps=%u sps=%u pps=%u\n",
                                        s->h265_vps_len, s->h265_sps_len,
                                        s->h265_pps_len);
                            }
                        }
                        for (uint32_t k = 0; k + 8 <= pln; k++) {
                            if (pl[k] == 1 && pl[k + 4] == 0xff &&
                                (pl[k + 5] & 0x1f) >= 1 &&
                                (pl[k + 5] & 0x1f) <= 3) {
                                uint32_t nsp = pl[k + 5] & 0x1f;
                                uint32_t o = k + 6;
                                for (uint32_t q = 0;
                                     q < nsp && o + 2 <= pln; q++) {
                                    uint32_t spssl = (pl[o] << 8) | pl[o + 1];
                                    o += 2;
                                    if (spssl > 0 && spssl <= 63 &&
                                        o + spssl <= pln && !sps) {
                                        sps = pl + o;
                                        sps_l = spssl;
                                    }
                                    o += spssl;
                                }
                                if (o < pln) {
                                    uint32_t npp = pl[o++];
                                    for (uint32_t q = 0;
                                         q < npp && o + 2 <= pln; q++) {
                                        uint32_t pl2 = (pl[o] << 8) |
                                                       pl[o + 1];
                                        o += 2;
                                        if (pl2 > 0 && pl2 <= 63 &&
                                            o + pl2 <= pln && !pps) {
                                            pps = pl + o;
                                            pps_l = pl2;
                                        }
                                        o += pl2;
                                    }
                                }
                                break;
                            }
                        }
                    }
                    aoff += aln;
                }
                if (sps && sps_l <= sizeof(s->avcC_sps) && !s->avcC_sps_len) {
                    memcpy(s->avcC_sps, sps, sps_l);
                    s->avcC_sps_len = sps_l;
                }
                if (pps && pps_l <= sizeof(s->avcC_pps) && !s->avcC_pps_len) {
                    memcpy(s->avcC_pps, pps, pps_l);
                    s->avcC_pps_len = pps_l;
                }
                if (!sps && s->avcC_sps_len) {
                    sps = s->avcC_sps;
                    sps_l = s->avcC_sps_len;
                }
                if (!pps && s->avcC_pps_len) {
                    pps = s->avcC_pps;
                    pps_l = s->avcC_pps_len;
                }
                if (((sps && pps) || s->dec_codec == 1) && nslices && flen > 0) {
                    /* 617.42 dies on the second decode of one session; the
                     * working mode reopens the decoder per frame (tile frames
                     * are self-contained).  VTPV_DEC_REUSE=1 keeps one session
                     * with the reference ring (for the P-frame line). */
                    bool per_frame = !getenv("VTPV_DEC_REUSE");
                    int fr = -1;
                    uint32_t dw = 0, dh = 0;
                    if (s->dec_codec == 0) {
                        bool need_open = per_frame || !s->dec ||
                            !vtpv_dec_match(s->dec, sps, sps_l, pps, pps_l);
                        if (need_open) {
                            vtpv_dec_close(s->dec); /* same ABI closer for h264 */
                            s->dec = vtpv_dec_open(
                                s->dec_gpu_str ? s->dec_gpu_str : "nv",
                                sps, sps_l, pps, pps_l);
                        }
                        if (s->dec) {
                            s->dec_codec_last = 0;
                            fr = vtpv_dec_frame(s->dec, frame, flen, slice_off,
                                                nslices, nv12);
                            dw = vtpv_dec_width(s->dec);
                            dh = vtpv_dec_height(s->dec);
                        }
                    } else if (s->h265_vps_len && s->h265_sps_len &&
                               s->h265_pps_len) {
                        if (per_frame && s->dec && s->dec_codec_last != 1) {
                            vtpv_dec_close(s->dec);
                            s->dec = NULL;
                        }
                        bool need_open = per_frame || !s->dec ||
                            !vtpv_dec265_match(s->dec, s->h265_vps,
                                               s->h265_vps_len, s->h265_sps,
                                               s->h265_sps_len, s->h265_pps,
                                               s->h265_pps_len);
                        if (need_open) {
                            vtpv_dec265_close(s->dec);
                            s->dec = vtpv_dec265_open(
                                s->dec_gpu_str ? s->dec_gpu_str : "nv",
                                s->h265_vps, s->h265_vps_len,
                                s->h265_sps, s->h265_sps_len,
                                s->h265_pps, s->h265_pps_len);
                        }
                        if (s->dec) {
                            s->dec_codec_last = 1;
                            fr = vtpv_dec265_frame(s->dec, frame, flen,
                                                   slice_off, nslices, nv12);
                            dw = vtpv_dec265_width(s->dec);
                            dh = vtpv_dec265_height(s->dec);
                        }
                    }
                    if (s->dec && fr == 0) {
                        uint32_t w = s->surfmap_meta[0] >> 16;
                        uint32_t h = s->surfmap_meta[0] & 0xffffu;
                        uint32_t bpr0 = s->surfmap_meta[1] ?
                                        s->surfmap_meta[1] : w;
                        uint32_t p1 = s->surfmap_meta[2];
                        uint32_t bpr1 = s->surfmap_meta[3] ?
                                        s->surfmap_meta[3] : w;
                        uint64_t written = 0;
                        if (dw <= w && dh <= h) {
                            for (uint32_t y = 0; y < dh; y++) {
                                uint64_t off = (uint64_t)y * bpr0;
                                uint32_t done = 0;
                                while (done < dw) {
                                    uint64_t sa = off + vtpv_surf_org(s);
                                    uint32_t pg = sa >> 12;
                                    uint32_t in_pg = 4096 - (sa & 0xfff);
                                    uint32_t chunk = dw - done < in_pg ?
                                                     dw - done : in_pg;
                                    if (vtpv_page_ok(s, pg)) {
                                        dma_memory_write(
                                            pci_get_address_space(PCI_DEVICE(s)),
                                            vtpv_surf_page(s, pg) + (sa & 0xfff),
                                            nv12 + (size_t)y * dw + done,
                                            chunk, MEMTXATTRS_UNSPECIFIED);
                                        written += chunk;
                                    }
                                    off += chunk;
                                    done += chunk;
                                }
                            }
                            for (uint32_t y = 0; y < dh / 2; y++) {
                                uint64_t off = p1 + (uint64_t)y * bpr1;
                                uint32_t len = dw;
                                uint32_t done = 0;
                                while (done < len) {
                                    uint64_t sa = off + vtpv_surf_org(s);
                                    uint32_t pg = sa >> 12;
                                    uint32_t in_pg = 4096 - (sa & 0xfff);
                                    uint32_t chunk = len - done < in_pg ?
                                                     len - done : in_pg;
                                    if (vtpv_page_ok(s, pg)) {
                                        dma_memory_write(
                                            pci_get_address_space(PCI_DEVICE(s)),
                                            vtpv_surf_page(s, pg) + (sa & 0xfff),
                                            nv12 + (size_t)dw * dh +
                                                (size_t)y * dw + done,
                                            chunk, MEMTXATTRS_UNSPECIFIED);
                                        written += chunk;
                                    }
                                    off += chunk;
                                    done += chunk;
                                }
                            }
                        }
                        lab_elog( "apple-vtpv: HW-DECODE tile=%llu "
                                "%ux%d -> canvas %llu bytes (chroma=%d)\n",
                                (unsigned long long)tile_id, dw, dh,
                                (unsigned long long)written,
                                (int)vtpv_dec_has_chroma(s->dec));
                        decoded = true;
                    }
                } else {
                    lab_elog( "apple-vtpv: dtil sbf parse: sps=%u pps=%u "
                            "slices=%u flen=%u (len=%u)\n",
                            sps ? sps_l : 0, pps ? pps_l : 0, nslices, flen,
                            sbf_len);
                }
            }
            if (!decoded) {
                lab_elog( "apple-vtpv: hwdecode miss, no writeback\n");
            }
        }
        if (!s->hwdecode && s->writeback && s->surfmap_count) {
            /* proof-of-life pixels: gradient Y + neutral UV written straight
             * into the exported canvas pages (the real path swaps this for
             * NVDEC output later) */
            uint32_t w = s->surfmap_meta[0] >> 16, h = s->surfmap_meta[0] & 0xffffu;
            uint32_t bpr0 = s->surfmap_meta[1] ? s->surfmap_meta[1] : w;
            uint32_t p1 = s->surfmap_meta[2], bpr1 = s->surfmap_meta[3] ? s->surfmap_meta[3] : w;
            uint8_t row[4096];
            uint64_t written = 0;
            for (uint32_t y = 0; y < h && y < 4096; y++) {
                for (uint32_t x = 0; x < w && x < 4096; x++) {
                    row[x] = (uint8_t)(x * 255 / (w ? w : 1));
                }
                uint64_t off = (uint64_t)y * bpr0;
                uint32_t i = 0;
                while (i < w) {
                    uint64_t sa = off + vtpv_surf_org(s);
                    uint32_t pg = sa >> 12;
                    uint32_t in_pg = 4096 - (sa & 0xfff);
                    uint32_t chunk = w - i < in_pg ? w - i : in_pg;
                    if (vtpv_page_ok(s, pg)) {
                        dma_memory_write(pci_get_address_space(PCI_DEVICE(s)),
                                         vtpv_surf_page(s, pg) + (sa & 0xfff),
                                         row + i, chunk, MEMTXATTRS_UNSPECIFIED);
                        written += chunk;
                    }
                    off += chunk;
                    i += chunk;
                }
            }
            for (uint32_t y = 0; y < h / 2; y++) {
                memset(row, 0x80, sizeof(row));
                uint64_t off = p1 + (uint64_t)y * bpr1;
                uint32_t len = w;
                uint32_t i = 0;
                while (i < len) {
                    uint64_t sa = off + vtpv_surf_org(s);
                    uint32_t pg = sa >> 12;
                    uint32_t in_pg = 4096 - (sa & 0xfff);
                    uint32_t chunk = len - i < in_pg ? len - i : in_pg;
                    if (vtpv_page_ok(s, pg)) {
                        dma_memory_write(pci_get_address_space(PCI_DEVICE(s)),
                                         vtpv_surf_page(s, pg) + (sa & 0xfff),
                                         row + i, chunk, MEMTXATTRS_UNSPECIFIED);
                        written += chunk;
                    }
                    off += chunk;
                    i += chunk;
                }
            }
            lab_elog( "apple-vtpv: writeback %llu bytes into %u pages "
                    "(w=%u h=%u bpr0=%u p1@%u)\n", (unsigned long long)written,
                    s->surfmap_count, w, h, bpr0, p1);
        }
        uint8_t paras[64];
        uint32_t n = 0;
        int32_t be_err = 0;
        uint32_t be_inf = 0;
        n += vtpv_para_put(paras + n, 0x2c657272u /* ',err' */, "=s32",
                           &be_err, 4);
        n += vtpv_para_put(paras + n, 0x2c696e66u /* ',inf' */, "=u32",
                           &be_inf, 4);
        vtpv_send_h2g2(s, slot, type, 4, s->msg + 0x10, s->msg + 0x20,
                       paras, n);
        return;
    }
    if (type == 0x44627566u /* 'Dbuf' */) {
        /* two directions share this type: the guest's PV session ASKS us for
         * a buffer (clerk-armed request with its own reply-id - the guest-side
         * CreatePixelBuffer delegates to the host!) and ANSWERS our Dbuf
         * (echoes our id).  The reply-id decides which is which. */
        uint8_t uuid[16] = { 0 };
        int32_t err = 0;
        uint32_t ulen = 0, vlen = 0;   /* separate: ',uui' is 16 bytes, ',err' 4 */
        const uint8_t *u = vtpv_para_find(s->msg, total, 0x2c757569u /* ',uui' */, &ulen);
        const uint8_t *e = vtpv_para_find(s->msg, total, 0x2c657272u /* ',err' */, &vlen);
        int i, hit = -1;
        for (i = 0; i < VTPV_FRAMES; i++) {
            if (s->cur->frame[i].in_use && s->cur->frame[i].dbuf_sent &&
                memcmp(s->cur->frame[i].dbuf_id, s->msg + 0x20, 8) == 0) {
                hit = i;
                break;
            }
        }
        if (hit < 0) {
            /* their buffer REQUEST (must carry ',frm'; a bare 'Dbuf' would be
             * a stray reply and answering it would loop) */
            uint32_t flen = 0;
            if (!vtpv_para_find(s->msg, total, 0x2c66726du /* ',frm' */, &flen)) {
                lab_elog( "apple-vtpv: stray Dbuf (no frame param) dropped\n");
                return;
            }
            uint8_t paras[64];
            uint32_t n = 0;
            int32_t be_err = 0;
            n += vtpv_para_put(paras + n, 0x2c657272u /* ',err' */, "=s32",
                               &be_err, 4);
            s->host_msg_id++;
            stq_be_p(s->mock_uuid, s->host_msg_id);
            memset(s->mock_uuid + 8, 0xa5, 8);
            n += vtpv_para_put(paras + n, 0x2c757569u /* ',uui' */, "=uui",
                               s->mock_uuid, 16);
            if (s->verbose) {
                lab_elog( "apple-vtpv: their Dbuf request -> mock uuid "
                        "(id=%llu)\n", (unsigned long long)s->host_msg_id);
            }
            vtpv_send_h2g2(s, slot, type, 4, s->msg + 0x10, s->msg + 0x20,
                           paras, n);
            return;
        }
        if (e && vlen == 4) {
            memcpy(&err, e, 4);
            err = (int32_t)be32_to_cpu(err);
        }
        if (u && ulen == 16) {
            memcpy(uuid, u, 16);
        }
        uint64_t frame_id = s->cur->frame[hit].frame_id;
        if (nsurf) {
            vtpv_gate_note_answer(s, s->cur, sl->surf[0]);
            vtpv_pace_note_answer(s, s->cur, sl->surf[0]);
        }
        s->cur->frame[hit].dbuf_sent = false;
        if (s->cur->dbuf_out) {
            s->cur->dbuf_out--;
        }
        uint32_t inf = 0;
        int64_t t_wait = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
        bool was_queued = vtpv_dec_wait(s, s->cur, hit);
        t_wait = qemu_clock_get_us(QEMU_CLOCK_REALTIME) - t_wait;
        /* the first picture of a stream, or what the decode thread made */
        const uint8_t *pic = s->cur->frame[hit].nv12;
        uint32_t pw = s->cur->frame[hit].w, ph = s->cur->frame[hit].h;
        if (s->outbuf && was_queued && !pic) {
            inf = 1u << 1;          /* kVTDecodeInfo_FrameDropped: nothing to show */
        } else if (s->outbuf && s->cur->frame[hit].sample) {
            vtpv_dec_drain(s, s->cur);  /* the session's decoder is ours again */
            /* decode now - replies come back in 'Dbuf' order, which is the
             * decode order - and fill the buffer straight from the stage */
            VtpvDxSink *k = &s->cur->dx_sink;
            k->to_stage = true;
            k->pictures = 0;
            vtpv_dx_decode_sample(s, s->cur, s->cur->frame[hit].sample, s->cur->frame[hit].sample_len, frame_id);
            k->to_stage = false;
            g_free(s->cur->frame[hit].sample);
            s->cur->frame[hit].sample = NULL;
            if (k->pictures) {
                pic = k->stage;
                pw = k->w;
                ph = k->h;
            } else {
                inf = 1u << 1;      /* kVTDecodeInfo_FrameDropped: nothing to show */
            }
        }
        if (s->outbuf && pic) {
            /* the guest's pool buffer for this frame: its IOSurface rides
             * on this reply (surf[0]) and the kext described its pages */
            const VtpvKickSurf *ks = sl->surf[0] ? vtpv_kick_surf(s, sl->surf[0]) : NULL;
            int64_t t_fill = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
            bool ok = !err && ks && vtpv_fill_surface(s, ks, pic, pw, ph);
            if (ok) {
                vtpv_note_surface(ks);
            }
            t_fill = qemu_clock_get_us(QEMU_CLOCK_REALTIME) - t_fill;
            if (s->verbose > 1) {
                /* zlib crc32 of the tight NV12, as vtprobe11's .crc truth */
                uLong c = crc32(0L, Z_NULL, 0);
                c = crc32(c, pic, (uInt)((size_t)pw * ph + (size_t)((pw + 1) & ~1u) * ((ph + 1) / 2)));
                lab_elog( "apple-vtpv: frame %llu: picture crc %08lx%s\n",
                        (unsigned long long)frame_id, (unsigned long)c,
                        was_queued ? " (decode thread)" : "");
            }
            if (s->verbose) {
                lab_elog( "apple-vtpv: frame %llu: Dbuf reply err=%d surf=%u %s -> %s "
                        "(wait %lld us, fill %lld us, %u waiting, %u in outbox)\n",
                        (unsigned long long)frame_id, err, sl->surf[0],
                        ks ? "described" : "NOT described", ok ? "filled" : "NOT filled",
                        (long long)t_wait, (long long)t_fill, vtpv_sess_waiting(s->cur),
                        g_queue_get_length(&sl->outbox));
            }
            if (!err && !ok) {
                err = -12911;   /* kVTVideoDecoderMalfunctionErr: no writable buffer */
            }
            g_free(s->cur->frame[hit].nv12);
            s->cur->frame[hit].nv12 = NULL;
        }
        s->cur->frame[hit].in_use = false;
        s->cur->frame[hit].queued = false;
        s->cur->frame[hit].decoded = false;
        vtpv_pace_complete(s, s->cur, slot, frame_id, err, inf, s->msg + 0x10, uuid);
        if (s->outbuf && pic && (pw != s->cur->dpba_w || ph != s->cur->dpba_h)) {
            /* the stream changed size: the next buffers must fit it */
            vtpv_send_dpba(s, slot, s->msg + 0x10, pw, ph);
        }
        vtpv_pump_dbuf(s, slot, s->msg + 0x10);
        return;
    }

    /* Stage 2: 'decs' answered from a host-side template file (reply_dcap=);
     * the file's bytes are patched with the request's UUID (0x10) and reply
     * id (0x20) before enqueue, so reply formats iterate without rebuilds.
     * The template's first u32 must be the 'vtpv' atom size (0x30 in the
     * live capture) - the guest's atom walk skips that many bytes to reach
     * the 'para' boxes, so it must NOT be the whole message length.
     * '+dec' (session create) gets its own template (reply_plusdec=) with
     * ',ver'/',hin'/',err'; 'dcap' needs ',dca' as a DICTIONARY - left
     * unanswered until its capability keys come back from the dlsym probe;
     * a wrong-type reply would strand the caller, a timeout is safer. */
    /* cache SPS/PPS from dsts ',fds': find the avcC record by signature and
     * keep the (start-code-free) parameter-set NALs for the decoder */
    if (type == 0x64696e76u /* 'dinv' */ && vtpv_use_dx(s)) {
        vtpv_dx_reset(s);           /* session invalidated: drop its references */
    }
    /* the format rides in dsts or, for sessions made from a format
     * description with extension atoms, only in dsta (ox3: no dsts at all) */
    if (type == 0x64737473u /* 'dsts' */ || type == 0x64737461u /* 'dsta' */) {
        uint32_t fds_len = 0;
        const uint8_t *fds = vtpv_para_find(s->msg, total,
                                            0x2c666473u /* ',fds' */, &fds_len);
        if (fds && vtpv_use_dx(s)) {
            vtpv_dx_set_fmt(s, fds, fds_len);
        }
        if (fds && fds_len > 16) {
            for (uint32_t k = 0; k + 8 <= fds_len; k++) {
                if (fds[k] == 1 && fds[k + 4] == 0xff &&
                    (fds[k + 5] & 0x1f) >= 1 && (fds[k + 5] & 0x1f) <= 3) {
                    uint32_t nsp = fds[k + 5] & 0x1f;   /* count lives in byte 5 */
                    uint32_t o = k + 6;
                    for (uint32_t q = 0; q < nsp && o + 2 <= fds_len; q++) {
                        uint32_t fdsl = (fds[o] << 8) | fds[o + 1];
                        o += 2;
                        if (fdsl > 0 && fdsl <= 63 && o + fdsl <= fds_len) {
                            memcpy(s->avcC_sps, fds + o, fdsl);
                            s->avcC_sps_len = fdsl;
                        }
                        o += fdsl;
                    }
                    if (o < fds_len) {
                        uint32_t npp = fds[o++];
                        for (uint32_t q = 0; q < npp && o + 2 <= fds_len; q++) {
                            uint32_t pl2 = (fds[o] << 8) | fds[o + 1];
                            o += 2;
                            if (pl2 > 0 && pl2 <= 63 && o + pl2 <= fds_len) {
                                memcpy(s->avcC_pps, fds + o, pl2);
                                s->avcC_pps_len = pl2;
                            }
                            o += pl2;
                        }
                    }
                    break;
                }
            }
            lab_elog( "apple-vtpv: dsts avcC cached sps=%u pps=%u\n",
                    s->avcC_sps_len, s->avcC_pps_len);
        }
    }
    /* each reply comes from its reply_*= file when one is given, else from
     * the built-in template (apple-vtpv-replies.h, the same bytes) */
    const char *reply_file = NULL;
    const uint8_t *reply_tpl = NULL;
    size_t reply_tpl_len = 0;
    bool patch_type = false;
    if (type == 0x64656373u /* 'decs' */) {
        reply_file = s->reply_dcap;
        reply_tpl = vtpv_tpl_decs;
        reply_tpl_len = sizeof(vtpv_tpl_decs);
    } else if (type == 0x2b646563u /* '+dec' */) {
        reply_file = s->reply_plusdec;
        reply_tpl = vtpv_tpl_plusdec;
        reply_tpl_len = sizeof(vtpv_tpl_plusdec);
    } else if (type == 0x64737570u /* 'dsup' */) {
        reply_file = s->reply_dsup ? s->reply_dsup : s->reply_generic;
        patch_type = !s->reply_dsup && s->reply_generic;
        reply_tpl = vtpv_tpl_dsup;
        reply_tpl_len = sizeof(vtpv_tpl_dsup);
    } else if (flags & 1) {
        /* 'dsta' (StartSession validation) and 'dinv' need only ',err' per the
         * disassembly; the generic template's type field gets the request's.
         * Only a request (flags bit 0) waits for a reply. flags 0x8 is the
         * guest's REPLY to one of our requests (its ',err' for 'Demt'/'Dpba');
         * '-dec' (flags 0) is sent as the decoder goes and its UUID is gone
         * from the guest's handler table by the time a reply lands: the
         * guest's host-to-guest pump stops draining at a message it cannot
         * dispatch, so every reply queued behind the stray one waited for the
         * guest's next send - 1.1 s reply-clerk timeouts on 'dtil' (Maps, TV,
         * Weather crashed in CMPhoto after them). */
        reply_file = s->reply_generic;
        reply_tpl = vtpv_tpl_generic;
        reply_tpl_len = sizeof(vtpv_tpl_generic);
        patch_type = true;
    }
    if (reply_tpl) {
        if (s->verbose) {
            char t5[5];
            vtpv_log4cc(t5, type);
            lab_elog( "apple-vtpv: reply %.4s <- %s%s\n", t5,
                    reply_file ? reply_file : "built-in",
                    patch_type ? " (type patched)" : "");
        }
        uint8_t *rep = NULL;
        long rlen = 0;
        FILE *rf = reply_file ? fopen(reply_file, "rb") : NULL;
        if (!reply_file) {
            rep = g_memdup2(reply_tpl, reply_tpl_len);
            rlen = (long)reply_tpl_len;
        } else if (rf) {
            fseek(rf, 0, SEEK_END);
            rlen = ftell(rf);
            fseek(rf, 0, SEEK_SET);
            if (rlen >= 0x30 && rlen <= VTPV_H2G_ENTRY_MAX) {
                rep = g_malloc(rlen);
                if (fread(rep, 1, rlen, rf) != (size_t)rlen) {
                    g_free(rep);
                    rep = NULL;
                    rlen = 0;
                }
            }
            fclose(rf);
        }
        if (!rep) {
            lab_elog( "apple-vtpv: reply template unreadable (%s)\n",
                    reply_file);
            return;
        }
        memcpy(rep + 0x10, s->msg + 0x10, 16);      /* guest UUID */
        memcpy(rep + 0x20, s->msg + 0x20, 8);       /* reply identifier (u64) */
        if (patch_type) {
            memcpy(rep + 8, s->msg + 8, 4);         /* echo the request's 4CC type */
        }
        if (type == 0x64737473u /* 'dsts' */ && reply_tpl == vtpv_tpl_generic &&
            rlen == (long)sizeof(vtpv_tpl_generic)) {
            if (!s->tiles || (s->tiles_one_session && s->tile_sessions > 0)) {
                /* tiles=off, or (diagnostic tiles-one-session=on) another HW
                 * tile session is live: StartTileSession fails
                 * (kVTCouldNotFindVideoDecoderErr, ',err' of the generic
                 * template at 0x54); ImageIO decodes the HEIF itself.
                 * The parallel tile sessions one image runs crashed
                 * CMPhoto's tile handler on every build through 24b - the
                 * switch stays off in production (Mish: parallel sessions
                 * must work), it only isolates the trigger. */
                stl_be_p(rep + 0x54, (uint32_t)-12906);
            } else {
                s->cur->tile_live = true;
                s->tile_sessions++;
            }
        }
        /* dec-pace v7: hold the non-tile dfrm's own reply for a token - the
         * guest's DecodeFrame call waits on this reply (the reply clerk), so
         * the submit loop itself runs at the budget.  The dance below still
         * runs now: the sample is consumed once, completions flow freely. */
        bool rep_held = false;
        if (patch_type && non_tile_dfrm && s->dec_pace && s->outbuf && s->cur->dx &&
            rlen >= 0x30 && rlen <= (long)sizeof(s->pace_rep[0].data) &&
            !vtpv_pace_allow(s, s->cur, true, 0)) {
            for (uint32_t i = 0; i < ARRAY_SIZE(s->pace_rep); i++) {
                if (!s->pace_rep[i].used) {
                    s->pace_rep[i].used = true;
                    s->pace_rep[i].slot = slot;
                    s->pace_rep[i].len = (uint32_t)rlen;
                    memcpy(s->pace_rep[i].data, rep, rlen);
                    if (!timer_pending(s->pace_timer)) {
                        timer_mod(s->pace_timer, qemu_clock_get_us(QEMU_CLOCK_REALTIME) +
                                  s->pace_ivl_us / 4);
                    }
                    rep_held = true;
                    break;
                }
            }
        }
        if (!rep_held) {
            vtpv_dq_push(s, slot, rep, rlen);
            vtpv_raise(s, slot);
        }
        g_free(rep);
    }

    /* after the StartSession reply has been delivered, give the session its
     * output buffer attributes so the pool exists before the first 'Dbuf' */
    if (type == 0x64737461u /* 'dsta' */ && !s->cur->dpba_sent) {
        s->cur->dpba_sent = true;
        vtpv_send_dpba(s, slot, s->msg + 0x10, 0, 0);
    }
    if (non_tile_dfrm && s->outbuf) {
        VtpvDxSink *k = &s->cur->dx_sink;
        s->stat_dfrm++;
        /* dec-pace v3: the REAL ask path is here (the pump only retries);
         * a denied ask still allocates its frame and decodes in order for
         * its references, completed dropped without a buffer (the shed
         * machinery) - what a full real-Mac VT queue answers a pusher. */
        bool pdeny = s->dec_pace && s->outbuf && s->cur->dx && !vtpv_pace_allow(s, s->cur, false, 0);
        if (s->dec_tid_live) {
            int64_t nowt = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
            if (nowt >= s->tid_live_check_us) {
                s->tid_live_check_us = nowt + 250000;
                gchar *txt = NULL;
                if (g_file_get_contents("C:/reims-lab/live/vtpv-dectid", &txt, NULL, NULL) && txt) {
                    long v = atol(g_strchomp(txt));
                    g_free(txt);
                    if (v != s->dec_tid_max) {
                        s->dec_tid_max = v < 0 || v > 6 ? -1 : (int32_t)v;
                        lab_elog("apple-vtpv: dec-tid-max LIVE %d\n", s->dec_tid_max);
                    }
                }
            }
        }
        s->pace_drop_next = pdeny;
        if (s->ob_first) {
            if (k->w != s->cur->dpba_w || k->h != s->cur->dpba_h) {
                vtpv_send_dpba(s, slot, s->msg + 0x10, k->w, k->h);
            }
            vtpv_send_dbuf(s, slot, s->last_frame_id, s->msg + 0x10);
        } else if (s->ob_sample) {
            vtpv_send_dbuf(s, slot, s->last_frame_id, s->msg + 0x10);
        } else {
            /* nothing to show for this sample: complete it as dropped
             * (kVTDecodeInfo_FrameDropped) so the session never waits on it */
            uint8_t zero_uuid[16] = { 0 };
            vtpv_send_demt2(s, slot, s->last_frame_id, 0, 1u << 1, s->msg + 0x10, zero_uuid);
        }
        s->ob_sample = NULL;
    } else if (non_tile_dfrm && s->mock_dbuf) {
        vtpv_send_dbuf(s, slot, s->last_frame_id, s->msg + 0x10);
    }
    /* the PV session has NO local pool by design (pool getter returns NULL when
     * the PV bridge is set -> err 3840, nA37/39) and VTDecoderSessionEmitDecodedFrame
     * accepts a NULL pixel buffer (the 'Demt' handler passes literal 0) - so
     * complete the frame immediately with a null-buffer uuid: the app callback
     * fires with status 0.  mock_dbuf=on replaces this with the real Dbuf dance. */
    if (non_tile_dfrm && s->mock_demt && !s->outbuf) {
        uint8_t zero_uuid[16] = { 0 };
        vtpv_send_demt(s, slot, s->last_frame_id, 0, s->msg + 0x10, zero_uuid);
    }
}

/* the kext's VtpvSurfDesc for this kick: layout + page list of each attached
 * IOSurface (vtpv.cpp pinSurfaces; little endian, 72-byte entries) */
static void vtpv_load_surfdesc(AppleVtpv *s)
{
    AddressSpace *as = pci_get_address_space(PCI_DEVICE(s));
    uint32_t hdr[2];
    s->nksurf = 0;
    if (!s->surfdesc_gpa ||
        dma_memory_read(as, s->surfdesc_gpa, hdr, sizeof(hdr), MEMTXATTRS_UNSPECIFIED) != MEMTX_OK ||
        le32_to_cpu(hdr[0]) != VTPV_SURFDESC_MAGIC) {
        return;
    }
    uint32_t count = MIN(le32_to_cpu(hdr[1]), (uint32_t)ARRAY_SIZE(s->ksurf));
    for (uint32_t i = 0; i < count; i++) {
        uint32_t e[18];
        uint64_t pages_gpa;
        if (dma_memory_read(as, s->surfdesc_gpa + 8 + (uint64_t)i * 72, e, 72,
                            MEMTXATTRS_UNSPECIFIED) != MEMTX_OK) {
            break;
        }
        for (int j = 0; j < 18; j++) {
            e[j] = le32_to_cpu(e[j]);
        }
        VtpvKickSurf *k = &s->ksurf[s->nksurf];
        k->id = e[0];
        k->pixfmt = e[1];
        k->width = e[2];
        k->height = e[3];
        k->planes = e[4];
        k->alloc = e[5];
        for (int p = 0; p < 2; p++) {
            k->off[p] = e[6 + p];
            k->bpr[p] = e[8 + p];
            k->pw[p] = e[10 + p];
            k->ph[p] = e[12 + p];
        }
        k->npages = MIN(e[14], 65536u);
        k->org = e[15] & 0xfff;
        pages_gpa = (uint64_t)e[16] | ((uint64_t)e[17] << 32);
        k->pages = g_new(uint64_t, k->npages ? k->npages : 1);
        if (dma_memory_read(as, pages_gpa, k->pages, (dma_addr_t)k->npages * 8,
                            MEMTXATTRS_UNSPECIFIED) != MEMTX_OK) {
            g_free(k->pages);
            break;
        }
        for (uint32_t j = 0; j < k->npages; j++) {
            k->pages[j] = le64_to_cpu(k->pages[j]);
        }
        if (s->verbose) {
            lab_elog( "apple-vtpv: surface %u %.4s %ux%u planes=%u alloc=%u pages=%u "
                    "org=%u off=%u/%u bpr=%u/%u\n", k->id, (const char *)&k->pixfmt, k->width,
                    k->height, k->planes, k->alloc, k->npages, k->org, k->off[0], k->off[1],
                    k->bpr[0], k->bpr[1]);
        }
        s->nksurf++;
    }
}

static void vtpv_drop_surfdesc(AppleVtpv *s)
{
    for (uint32_t i = 0; i < s->nksurf; i++) {
        g_free(s->ksurf[i].pages);
        s->ksurf[i].pages = NULL;
    }
    s->nksurf = 0;
    s->surfdesc_gpa = 0;
}

static const VtpvKickSurf *vtpv_kick_surf(AppleVtpv *s, uint32_t id)
{
    for (uint32_t i = 0; i < s->nksurf; i++) {
        if (s->ksurf[i].id == id) {
            return &s->ksurf[i];
        }
    }
    return NULL;
}

static void vtpv_selftest(AppleVtpv *s, uint32_t size)
{
    vtpv_msg_realloc(s, size < 0x30 ? 0x30 : size);
    memset(s->msg, 0x5a, size < 0x30 ? 0x30 : size);
    uint32_t be;
    be = cpu_to_be32(0x30);
    memcpy(s->msg, &be, 4);         /* 'para' atom size */
    be = cpu_to_be32(0x70617261);
    memcpy(s->msg + 4, &be, 4);     /* 'para' */
    be = cpu_to_be32(0x74657374);   /* 'tset' */
    memcpy(s->msg + 8, &be, 4);
    be = 0;
    memcpy(s->msg + 12, &be, 4);
    uint32_t n = size < 0x30 ? 0x30 : size;
    if (vtpv_dq_enqueue(s, s->client_id, s->msg, n)) {
        vtpv_raise(s, s->client_id);
    } else {
        lab_elog( "apple-vtpv: selftest enqueue failed\n");
    }
}

/* ------------------------------------------------------------------ */

static uint64_t vtpv_mmio_read(void *opaque, hwaddr addr, unsigned size)
{
    AppleVtpv *s = APPLE_VTPV(opaque);
    uint32_t v = 0;

    switch (addr) {
    case VTPV_MAGIC:
        v = VTPV_MAGIC_VAL;
        break;
    case VTPV_VERSION:
        v = s->kickdesc ? 3 : 2;
        break;
    case VTPV_CLIENT_ID:
        v = s->client_id;
        break;
    case VTPV_IRQ_PEND:
        v = s->irq_pending;
        break;
    case VTPV_CAPS:
        v = 0;    /* stage 2: codec bits */
        break;
    case VTPV_MSG_KICK:
        v = s->slot[s->client_id].last_kick;
        break;
    case VTPV_SURF0:
    case VTPV_SURF0 + 4:
    case VTPV_SURF0 + 8:
    case VTPV_SURF0 + 12:
    case VTPV_SURF0 + 16:
        v = s->slot[s->client_id].surf[(addr - VTPV_SURF0) / 4];
        break;
    case VTPV_MSGS_IN:
        v = (uint32_t)s->msgs_in;
        break;
    case VTPV_MSGS_OUT:
        v = (uint32_t)s->msgs_out;
        break;
    case VTPV_DEBUGVAL:
        v = s->debugval;
        break;
    default: {
        /* slot registers read back their stored values (the kext logs real GPAs) */
        const AppleVtpvSlot *sl2 = &s->slot[s->client_id];
        uint32_t base = addr & ~7u;
        uint64_t cur;
        switch (base) {
        case VTPV_G2H_GPA: cur = sl2->g2h_gpa; break;
        case VTPV_G2H_LEN: cur = sl2->g2h_len; break;
        case VTPV_H2G_GPA: cur = sl2->h2g_gpa; break;
        case VTPV_H2G_LEN: cur = sl2->h2g_len; break;
        case VTPV_UUID_GPA: cur = sl2->uuid_gpa; break;
        default:
            if (s->verbose) {
                lab_elog( "apple-vtpv: read %#llx\n", (unsigned long long)addr);
            }
            return v;
        }
        v = (addr & 4) ? (uint32_t)(cur >> 32) : (uint32_t)cur;
        break;
    }
    }
    return v;
}

static void vtpv_set_reg64(AppleVtpv *s, uint32_t off, uint64_t val)
{
    AppleVtpvSlot *sl = &s->slot[s->client_id];
    switch (off) {
    case VTPV_G2H_GPA: sl->g2h_gpa = val; break;
    case VTPV_G2H_LEN: sl->g2h_len = val; break;
    case VTPV_H2G_GPA: sl->h2g_gpa = val; break;
    case VTPV_H2G_LEN: sl->h2g_len = val; break;
    case VTPV_UUID_GPA: sl->uuid_gpa = val; break;
    default: return;
    }
}

static void vtpv_mmio_write(void *opaque, hwaddr addr, uint64_t val, unsigned size)
{
    AppleVtpv *s = APPLE_VTPV(opaque);
    AppleVtpvSlot *sl = &s->slot[s->client_id];

    switch (addr) {
    case VTPV_CLIENT_ID:
        if (val < VTPV_MAX_SLOT) {
            s->client_id = val;
        }
        s->kick_open_us = qemu_clock_get_us(QEMU_CLOCK_REALTIME);  /* the kext's locked sequence starts */
        break;
    case VTPV_G2H_GPA ... VTPV_G2H_LEN + 4:
    case VTPV_H2G_GPA ... VTPV_H2G_LEN + 4:
    case VTPV_UUID_GPA ... VTPV_UUID_GPA + 4: {
        /* 64-bit registers are written as two adjacent u32s */
        uint32_t base = addr & ~7u;
        if (addr & 4) {
            uint64_t cur;
            switch (base) {
            case VTPV_G2H_GPA: cur = sl->g2h_gpa; break;
            case VTPV_G2H_LEN: cur = sl->g2h_len; break;
            case VTPV_H2G_GPA: cur = sl->h2g_gpa; break;
            case VTPV_H2G_LEN: cur = sl->h2g_len; break;
            case VTPV_UUID_GPA: cur = sl->uuid_gpa; break;
            default: return;
            }
            vtpv_set_reg64(s, base, (cur & 0xffffffffu) | (val << 32));
        } else {
            vtpv_set_reg64(s, base, val);
        }
        break;
    }
    case VTPV_UUID_OP: {
        uint8_t uuid[16];
        if (sl->uuid_gpa &&
            !dma_memory_read(pci_get_address_space(PCI_DEVICE(s)), sl->uuid_gpa, uuid, 16, MEMTXATTRS_UNSPECIFIED)) {
            FILE *f = s->msglog ? fopen(s->msglog, "a") : NULL;
            if (f) {
                fprintf(f, "=== UUID op=%llu uuid=", (unsigned long long)val);
                vtpv_hex(f, uuid, 16);
                fclose(f);
            }
            if (s->verbose) {
                lab_elog( "apple-vtpv: slot%d uuid op=%lld\n",
                        s->client_id, (long long)val);
            }
            if (val == 2) {
                int64_t d0 = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
                vtpv_sess_drop(s, uuid);    /* the guest session is gone */
                int64_t d1 = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
                if (d1 - d0 > 5000) {
                    lab_elog( "apple-vtpv: slot%d long uuid drop %lld us\n", s->client_id,
                            (long long)(d1 - d0));
                }
            }
        }
        break;
    }
    case VTPV_MSG_SIZE:
        /* kept implicit: the kick carries the total */
        break;
    case VTPV_MSG_NSURF:
        /* kept implicit: SURF0..4 are read at kick; nsurf = nonzero count */
        break;
    case VTPV_SURF0 ... VTPV_SURF0 + 16:
        sl->surf[(addr - VTPV_SURF0) / 4] = val;
        break;
    case VTPV_MSG_KICK: {
        uint32_t nsurf = 0;
        sl->last_kick = (uint32_t)val;
        for (int i = 0; i < 5; i++) {
            nsurf += sl->surf[i] != 0;
        }
        int64_t k0 = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
        vtpv_load_surfdesc(s);
        int64_t k1 = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
        vtpv_handle_message(s, s->client_id, val, nsurf);
        int64_t k2 = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
        vtpv_drop_surfdesc(s);
        s->kick_hold_gpa = 0;       /* the kext programs it for every kick it can hold */
        memset(sl->surf, 0, sizeof(sl->surf));
        if (s->kick_open_us && k0 - s->kick_open_us > 20000) {
            /* the kext's register writes before the kick, under its spin lock:
             * each one an MMIO exit that waits for the BQL */
            lab_elog( "apple-vtpv: slot%d slow kick prologue %lld us\n", s->client_id,
                    (long long)(k0 - s->kick_open_us));
        }
        if (k2 - k0 > 20000) {
            /* the guest kext holds a spin lock for the whole kick: every other
             * caller of the device - and its user lock chain - waits that long */
            char t5[5];
            vtpv_log4cc(t5, ldl_be_p(s->msg + 8));
            lab_elog( "apple-vtpv: slot%d long kick %.4s %lld us (surfdesc %lld us)\n",
                    s->client_id, t5, (long long)(k2 - k0), (long long)(k1 - k0));
        }
        break;
    }
    case VTPV_KICK_HOLD ... VTPV_KICK_HOLD + 7:
        if (addr & 4) {
            s->kick_hold_gpa = (s->kick_hold_gpa & 0xffffffffu) | (val << 32);
        } else {
            s->kick_hold_gpa = (s->kick_hold_gpa & ~0xffffffffull) | (val & 0xffffffffu);
        }
        break;
    case VTPV_KICKDESC ... VTPV_KICKDESC + 7: {
        /* kickdesc=on (version 3): the kext's whole register sequence of a
         * send in ONE MMIO write (slot, surfaces, kick size, surfdesc GPA),
         * DMA-read here.  The v2 path cost one MMIO exit per register under
         * the kext's locks - the kext send-path measured p50 0.53 / p90 2.7 /
         * max 6.3 ms, most of it these exits plus the per-send page walk. */
        uint64_t gpa = (addr & 4) ? ((s->kickdesc_gpa & 0xffffffffu) | ((uint64_t)val << 32))
                                  : ((s->kickdesc_gpa & ~0xffffffffull) | (val & 0xffffffffu));
        s->kickdesc_gpa = gpa;
        if (!(addr & 4)) {
            break;      /* the low half alone is not a doorbell */
        }
        struct {
            uint32_t magic, slot, nsurf, total;
            uint32_t surf[5];
            uint32_t pad;
            uint64_t surfdesc;
            uint64_t hold;
        } QEMU_PACKED d;
        if (dma_memory_read(pci_get_address_space(PCI_DEVICE(s)), gpa, &d, sizeof(d),
                            MEMTXATTRS_UNSPECIFIED) ||
            d.magic != VTPV_KICKDESC_MAGIC || d.slot >= VTPV_MAX_SLOT) {
            lab_elog( "apple-vtpv: bad kick descriptor at %#llx\n",
                    (unsigned long long)gpa);
            break;
        }
        uint32_t nsurf = MIN(d.nsurf, 5u);
        s->client_id = d.slot;
        sl = &s->slot[d.slot];
        for (uint32_t i = 0; i < 5; i++) {
            sl->surf[i] = i < nsurf ? d.surf[i] : 0;
        }
        s->surfdesc_gpa = d.surfdesc;
        s->kick_hold_gpa = d.hold;
        sl->last_kick = d.total;
        int64_t k0 = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
        vtpv_load_surfdesc(s);
        int64_t k1 = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
        vtpv_handle_message(s, d.slot, d.total, nsurf);
        int64_t k2 = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
        vtpv_drop_surfdesc(s);
        s->kick_hold_gpa = 0;
        memset(sl->surf, 0, sizeof(sl->surf));
        if (k2 - k0 > 20000) {
            char t5[5];
            vtpv_log4cc(t5, ldl_be_p(s->msg + 8));
            lab_elog( "apple-vtpv: slot%d long kickdesc %.4s %lld us (surfdesc %lld us)\n",
                    d.slot, t5, (long long)(k2 - k0), (long long)(k1 - k0));
        }
        break;
    }
    case VTPV_SURFDESC ... VTPV_SURFDESC + 7:
        if (addr & 4) {
            s->surfdesc_gpa = (s->surfdesc_gpa & 0xffffffffu) | (val << 32);
        } else {
            s->surfdesc_gpa = (s->surfdesc_gpa & ~0xffffffffull) | (val & 0xffffffffu);
        }
        break;
    case VTPV_IRQ_ACK:
        /* clears the written bits only: the kext polls IRQ_PEND (its MSI
         * handler never runs here) and acks what it saw - a bit set after
         * that read must survive for the next poll */
        s->irq_pending &= ~(uint32_t)val;
        if (!s->irq_pending) {
            vtpv_lower(s);
        }
        break;
    case VTPV_SELFTEST:
        vtpv_selftest(s, val);
        break;
    case VTPV_DEBUGVAL:
        s->debugval = val;    /* real scratch: the kext reports its MSI state here */
        if (s->verbose) {
            lab_elog( "apple-vtpv: debugval %#llx\n", (unsigned long long)val);
        }
        break;
    case VTPV_SURFMAP_COUNT:
        s->surfmap_count = val & 0xffffu;
        break;
    case VTPV_SURFMAP_KICK:
        s->surfmap_kick = val;
        g_free(s->surfmap_list);
        s->surfmap_list = NULL;
        s->surfmap_list_n = 0;
        if (s->surfmap_list_gpa && s->surfmap_count) {
            uint32_t n = s->surfmap_count;
            uint64_t *l = g_new(uint64_t, n);
            if (dma_memory_read(pci_get_address_space(PCI_DEVICE(s)), s->surfmap_list_gpa, l,
                                (dma_addr_t)n * 8, MEMTXATTRS_UNSPECIFIED) == MEMTX_OK) {
                for (uint32_t i = 0; i < n; i++) {
                    l[i] = le64_to_cpu(l[i]);
                }
                s->surfmap_list = l;
                s->surfmap_list_n = n;
            } else {
                g_free(l);
                lab_elog( "apple-vtpv: surfmap list read failed (%u pages)\n", n);
            }
        }
        if (s->verbose) {
            lab_elog( "apple-vtpv: surfmap kick=%u pages=%u w=%u h=%u "
                    "bpr=%u p1@%u bpr1=%u gpa0=%llx\n", s->surfmap_kick,
                    s->surfmap_count, s->surfmap_meta[0] >> 16,
                    s->surfmap_meta[0] & 0xffffu, s->surfmap_meta[1],
                    s->surfmap_meta[2], s->surfmap_meta[3],
                    s->surfmap_count ? (unsigned long long)s->surfmap_gpa[0] : 0ull);
            for (uint32_t i = 0; i < 8 && vtpv_page_ok(s, i); i++) {
                lab_elog( "apple-vtpv:   page %u gpa %llx\n", i,
                        (unsigned long long)vtpv_page_gpa(s, i));
            }
        }
        break;
    case VTPV_SURFMAP_GPA0 ... VTPV_SURFMAP_GPA0 + 16 * 8 - 1: {
        uint32_t idx = (addr - VTPV_SURFMAP_GPA0) >> 3;
        if (addr & 4) {
            s->surfmap_gpa[idx] = (s->surfmap_gpa[idx] & 0xffffffffu) | (val << 32);
        } else {
            s->surfmap_gpa[idx] = (s->surfmap_gpa[idx] & ~0xffffffffu) | val;
        }
        break;
    }
    case VTPV_SURFMAP_LIST ... VTPV_SURFMAP_LIST + 7:
        if (addr & 4) {
            s->surfmap_list_gpa = (s->surfmap_list_gpa & 0xffffffffu) | (val << 32);
        } else {
            s->surfmap_list_gpa = (s->surfmap_list_gpa & ~0xffffffffull) | (uint32_t)val;
        }
        break;
    case VTPV_SURFMAP_META ... VTPV_SURFMAP_META + 12:
        s->surfmap_meta[(addr - VTPV_SURFMAP_META) / 4] = val;
        break;
    default:
        if (s->verbose) {
            lab_elog( "apple-vtpv: write %#llx <- %#llx\n",
                    (unsigned long long)addr, (unsigned long long)val);
        }
        break;
    }
}

static const MemoryRegionOps vtpv_mmio_ops = {
    .read = vtpv_mmio_read,
    .write = vtpv_mmio_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = { .min_access_size = 4, .max_access_size = 4 },
    .valid = { .min_access_size = 4, .max_access_size = 8 },
};

static void vtpv_hwdecode_dfrm(AppleVtpv *s, uint64_t total)
{
    if (s->hwdecode && s->writeback && s->surfmap_count) {
        /* hardware path: the dtil ',sbf' is a serialized CMSampleBuffer
         * (atoms [BE32 len][4cc]): 'sdat' = sample in AVCC
         * ([BE32 nal_len][NAL]...), 'fdsc' = avcC with SPS/PPS.  Decode
         * on the host GPU and write NV12 into the canvas pages. */
        uint32_t sbf_len = 0;
        const uint8_t *sbf = vtpv_para_find(s->msg, total,
                                             0x2c736266u /* ',sbf' */,
                                             &sbf_len);
        bool decoded = false;
        static uint8_t frame[256 * 1024];
        static uint8_t nv12[4096 * 2304 * 3 / 2];
        if (sbf && sbf_len > 8 && sbf_len < sizeof(frame) &&
            getenv("VTPV_DUMP_SBF") && vtpv_memfind(sbf, sbf_len, "fdsc", 4)) {
            FILE *df = fopen(getenv("VTPV_DUMP_SBF"), "wb");
            if (df) {
                fwrite(sbf, 1, sbf_len, df);
                fclose(df);
            }
        }
        if (sbf && sbf_len > 16 && sbf_len < sizeof(frame)) {
            const uint8_t *sps = NULL, *pps = NULL;
            uint32_t sps_l = 0, pps_l = 0;
            uint32_t slice_off[64];
            uint32_t nslices = 0;
            uint32_t flen = 0;
            vtpv_sbf_codec_first(s, sbf, sbf_len);
            /* flat tag scan: the payload atoms nest inside 'sbuf', so
             * walk one level but descend into every well-formed atom */
            uint32_t aoff = 8;   /* skip the outer 'sbuf' header */
            while (aoff + 8 <= sbf_len) {
                /* find the next tag start at or after aoff whose length
                 * field is sane (handles the nesting without recursion) */
                uint32_t aln = 0;
                const uint8_t *tagb = NULL;
                uint32_t scan = aoff;
                while (scan + 8 <= sbf_len) {
                    uint32_t l = ((uint32_t)sbf[scan] << 24) |
                                 ((uint32_t)sbf[scan + 1] << 16) |
                                 ((uint32_t)sbf[scan + 2] << 8) |
                                 sbf[scan + 3];
                    if (l >= 8 && scan + l <= sbf_len &&
                        (memcmp(sbf + scan + 4, "sdat", 4) == 0 ||
                         memcmp(sbf + scan + 4, "fdsc", 4) == 0)) {
                        aln = l;
                        tagb = sbf + scan + 4;
                        aoff = scan;
                        break;
                    }
                    scan++;
                }
                if (!tagb) {
                    break;
                }
                const uint8_t *pl = sbf + aoff + 8;
                uint32_t pln = aln - 8;
                if (memcmp(tagb, "sdat", 4) == 0) {
                    uint32_t j = 0;
                    while (j + 4 <= pln && nslices < 64) {
                        uint32_t nl = ((uint32_t)pl[j] << 24) |
                                      ((uint32_t)pl[j + 1] << 16) |
                                      ((uint32_t)pl[j + 2] << 8) |
                                      pl[j + 3];
                        if (nl < 1 || j + 4 + nl > pln) {
                            break;
                        }
                        const uint8_t *nal = pl + j + 4;
                        /* H.264: 1-byte header, type = b&0x1f (1/5);
                         * HEVC: 2-byte header, type = (b>>1)&0x3f (<32) */
                        int t = s->dec_codec ?
                                (((nal[0] >> 1) & 0x3f) < 32 ? 1 : 0)
                                : ((nal[0] & 0x1f) == 1 ||
                                   (nal[0] & 0x1f) == 5) ? 1 : 0;
                        if (t &&
                            flen + 3 + nl < sizeof(frame)) {
                            slice_off[nslices] = flen;
                            frame[flen] = 0;
                            frame[flen + 1] = 0;
                            frame[flen + 2] = 1;
                            memcpy(frame + flen + 3, nal, nl);
                            flen += 3 + nl;
                            nslices++;
                        }
                        j += 4 + nl;
                    }
                } else if (memcmp(tagb, "fdsc", 4) == 0) {
                    /* codec detect: avc1 -> H.264 (avcC record);
                     * hvc1/hev1 -> HEVC (hvcC arrays: VPS 32/SPS 33/PPS 34,
                     * 2-byte NAL lengths) */
                    if (vtpv_memfind(pl, pln, "hvc1", 4) ||
                        vtpv_memfind(pl, pln, "hev1", 4)) {
                        s->dec_codec = 1;
                        const uint8_t *hvcc = vtpv_memfind(pl, pln, "hvcC", 4);
                        if (hvcc && hvcc + 25 < pl + pln) {
                            const uint8_t *hc = hvcc + 4;
                            uint32_t narr = hc[22];  /* numOfArrays */
                            uint32_t o = 23;         /* first array */
                            for (uint32_t a = 0;
                                 a < narr && o + 3 <= pln - (hc - pl); a++) {
                                uint32_t ntype = hc[o] & 0x3f;
                                uint32_t nnal = (hc[o + 1] << 8) | hc[o + 2];
                                o += 3;
                                for (uint32_t q = 0;
                                     q < nnal && o + 2 <= pln - (hc - pl);
                                     q++) {
                                    uint32_t nl = (hc[o] << 8) | hc[o + 1];
                                    o += 2;
                                    if (nl > 0 &&
                                        o + nl <= pln - (hc - pl)) {
                                        if (ntype == 32 && nl <= sizeof(s->h265_vps)) {
                                            memcpy(s->h265_vps, hc + o, nl);
                                            s->h265_vps_len = nl;
                                        } else if (ntype == 33 && nl <= sizeof(s->h265_sps)) {
                                            memcpy(s->h265_sps, hc + o, nl);
                                            s->h265_sps_len = nl;
                                        } else if (ntype == 34 && nl <= sizeof(s->h265_pps)) {
                                            memcpy(s->h265_pps, hc + o, nl);
                                            s->h265_pps_len = nl;
                                        }
                                    }
                                    o += nl;
                                }
                            }
                            lab_elog( "apple-vtpv: HEVC hvcC cached "
                                    "vps=%u sps=%u pps=%u\n",
                                    s->h265_vps_len, s->h265_sps_len,
                                    s->h265_pps_len);
                        }
                    }
                    for (uint32_t k = 0; k + 8 <= pln; k++) {
                        if (pl[k] == 1 && pl[k + 4] == 0xff &&
                            (pl[k + 5] & 0x1f) >= 1 &&
                            (pl[k + 5] & 0x1f) <= 3) {
                            uint32_t nsp = pl[k + 5] & 0x1f;
                            uint32_t o = k + 6;
                            for (uint32_t q = 0;
                                 q < nsp && o + 2 <= pln; q++) {
                                uint32_t spssl = (pl[o] << 8) | pl[o + 1];
                                o += 2;
                                if (spssl > 0 && spssl <= 63 &&
                                    o + spssl <= pln && !sps) {
                                    sps = pl + o;
                                    sps_l = spssl;
                                }
                                o += spssl;
                            }
                            if (o < pln) {
                                uint32_t npp = pl[o++];
                                for (uint32_t q = 0;
                                     q < npp && o + 2 <= pln; q++) {
                                    uint32_t pl2 = (pl[o] << 8) |
                                                   pl[o + 1];
                                    o += 2;
                                    if (pl2 > 0 && pl2 <= 63 &&
                                        o + pl2 <= pln && !pps) {
                                        pps = pl + o;
                                        pps_l = pl2;
                                    }
                                    o += pl2;
                                }
                            }
                            break;
                        }
                    }
                }
                aoff += aln;
            }
            if (sps && sps_l <= sizeof(s->avcC_sps) && !s->avcC_sps_len) {
                memcpy(s->avcC_sps, sps, sps_l);
                s->avcC_sps_len = sps_l;
            }
            if (pps && pps_l <= sizeof(s->avcC_pps) && !s->avcC_pps_len) {
                memcpy(s->avcC_pps, pps, pps_l);
                s->avcC_pps_len = pps_l;
            }
            if (!sps && s->avcC_sps_len) {
                sps = s->avcC_sps;
                sps_l = s->avcC_sps_len;
            }
            if (!pps && s->avcC_pps_len) {
                pps = s->avcC_pps;
                pps_l = s->avcC_pps_len;
            }
            if (((sps && pps) || s->dec_codec == 1) && nslices && flen > 0) {
                /* 617.42 dies on the second decode of one session; the
                 * working mode reopens the decoder per frame (tile frames
                 * are self-contained).  VTPV_DEC_REUSE=1 keeps one session
                 * with the reference ring (for the P-frame line). */
                bool per_frame = !getenv("VTPV_DEC_REUSE");
                int fr = -1;
                uint32_t dw = 0, dh = 0;
                if (s->dec_codec == 0) {
                    bool need_open = per_frame || !s->dec ||
                        !vtpv_dec_match(s->dec, sps, sps_l, pps, pps_l);
                    if (need_open) {
                        vtpv_dec_close(s->dec); /* same ABI closer for h264 */
                        s->dec = vtpv_dec_open(
                            s->dec_gpu_str ? s->dec_gpu_str : "nv",
                            sps, sps_l, pps, pps_l);
                    }
                    if (s->dec) {
                        s->dec_codec_last = 0;
                        fr = vtpv_dec_frame(s->dec, frame, flen, slice_off,
                                            nslices, nv12);
                        dw = vtpv_dec_width(s->dec);
                        dh = vtpv_dec_height(s->dec);
                    }
                } else if (s->h265_vps_len && s->h265_sps_len &&
                           s->h265_pps_len) {
                    if (per_frame && s->dec && s->dec_codec_last != 1) {
                        vtpv_dec_close(s->dec);
                        s->dec = NULL;
                    }
                    bool need_open = per_frame || !s->dec ||
                        !vtpv_dec265_match(s->dec, s->h265_vps,
                                           s->h265_vps_len, s->h265_sps,
                                           s->h265_sps_len, s->h265_pps,
                                           s->h265_pps_len);
                    if (need_open) {
                        vtpv_dec265_close(s->dec);
                        s->dec = vtpv_dec265_open(
                            s->dec_gpu_str ? s->dec_gpu_str : "nv",
                            s->h265_vps, s->h265_vps_len,
                            s->h265_sps, s->h265_sps_len,
                            s->h265_pps, s->h265_pps_len);
                    }
                    if (s->dec) {
                        s->dec_codec_last = 1;
                        fr = vtpv_dec265_frame(s->dec, frame, flen,
                                               slice_off, nslices, nv12);
                        dw = vtpv_dec265_width(s->dec);
                        dh = vtpv_dec265_height(s->dec);
                    }
                }
                if (s->dec && fr == 0) {
                    uint32_t w = s->surfmap_meta[0] >> 16;
                    uint32_t h = s->surfmap_meta[0] & 0xffffu;
                    uint32_t bpr0 = s->surfmap_meta[1] ?
                                    s->surfmap_meta[1] : w;
                    uint32_t p1 = s->surfmap_meta[2];
                    uint32_t bpr1 = s->surfmap_meta[3] ?
                                    s->surfmap_meta[3] : w;
                    uint64_t written = 0;
                    if (dw <= w && dh <= h) {
                        for (uint32_t y = 0; y < dh; y++) {
                            uint64_t off = (uint64_t)y * bpr0;
                            uint32_t done = 0;
                            while (done < dw) {
                                uint64_t sa = off + vtpv_surf_org(s);
                                uint32_t pg = sa >> 12;
                                uint32_t in_pg = 4096 - (sa & 0xfff);
                                uint32_t chunk = dw - done < in_pg ?
                                                 dw - done : in_pg;
                                if (vtpv_page_ok(s, pg)) {
                                    dma_memory_write(
                                        pci_get_address_space(PCI_DEVICE(s)),
                                        vtpv_surf_page(s, pg) + (sa & 0xfff),
                                        nv12 + (size_t)y * dw + done,
                                        chunk, MEMTXATTRS_UNSPECIFIED);
                                    written += chunk;
                                }
                                off += chunk;
                                done += chunk;
                            }
                        }
                        for (uint32_t y = 0; y < dh / 2; y++) {
                            uint64_t off = p1 + (uint64_t)y * bpr1;
                            uint32_t len = dw;
                            uint32_t done = 0;
                            while (done < len) {
                                uint64_t sa = off + vtpv_surf_org(s);
                                uint32_t pg = sa >> 12;
                                uint32_t in_pg = 4096 - (sa & 0xfff);
                                uint32_t chunk = len - done < in_pg ?
                                                 len - done : in_pg;
                                if (vtpv_page_ok(s, pg)) {
                                    dma_memory_write(
                                        pci_get_address_space(PCI_DEVICE(s)),
                                        vtpv_surf_page(s, pg) + (sa & 0xfff),
                                        nv12 + (size_t)dw * dh +
                                            (size_t)y * dw + done,
                                        chunk, MEMTXATTRS_UNSPECIFIED);
                                    written += chunk;
                                }
                                off += chunk;
                                done += chunk;
                            }
                        }
                    }
                    lab_elog( "apple-vtpv: HW-DECODE tile=%llu "
                            "%ux%d -> canvas %llu bytes (chroma=%d)\n",
                            (unsigned long long)s->last_frame_id, dw, dh,
                            (unsigned long long)written,
                            (int)vtpv_dec_has_chroma(s->dec));
                    decoded = true;
                }
            } else {
                lab_elog( "apple-vtpv: dtil sbf parse: sps=%u pps=%u "
                        "slices=%u flen=%u (len=%u)\n",
                        sps ? sps_l : 0, pps ? pps_l : 0, nslices, flen,
                        sbf_len);
            }
        }
        if (!decoded) {
            lab_elog( "apple-vtpv: hwdecode miss, no writeback\n");
        }
    }
}

static void vtpv_realize(PCIDevice *dev, Error **errp)
{
    AppleVtpv *s = APPLE_VTPV(dev);

    memory_region_init_io(&s->mmio, OBJECT(s), &vtpv_mmio_ops, s,
                          "apple-vtpv.regs", VTPV_REG_SIZE);
    pci_register_bar(dev, 0, PCI_BASE_ADDRESS_SPACE_MEMORY, &s->mmio);
    msi_init(dev, 0, 1, true, false, errp);         /* 1 MSI vector: h2g notify */
    /* INTx pin A as the fallback path; the guest's MSI-enable bit decides which
     * one vtpv_raise actually fires. */
    pci_config_set_interrupt_pin(dev->config, 1);
    for (int i = 0; i < VTPV_SESS; i++) {
        s->sess[i].dx_codec = -1;
        s->sess[i].owner = s;
    }
    qemu_mutex_init(&s->dec_lock);
    qemu_cond_init(&s->dec_cond);
    qemu_thread_create(&s->dec_thread, "vtpv-decode", vtpv_dec_thread, s,
                       QEMU_THREAD_DETACHED);
    s->outbox_timer = timer_new_ms(QEMU_CLOCK_REALTIME, vtpv_outbox_tick, s);
    s->tdem_timer = timer_new_us(QEMU_CLOCK_REALTIME, vtpv_tdem_tick, s);
    s->pace_timer = timer_new_us(QEMU_CLOCK_REALTIME, vtpv_pace_tick, s);
    s->pace_ivl_us = VTPV_PACE_IVL_US;
    s->pace_toks_us = (int64_t)VTPV_PACE_BURST * s->pace_ivl_us;
    s->pace_last_us = 0;
    {
        const char *pv = getenv("REIMS_VGPU_LAB_DEC_PACE_US");
        if (pv) {
            long ivl = atol(pv);
            if (ivl >= 1000 && ivl <= 1000000) {
                s->pace_ivl_us = ivl;
            }
        }
    }
    g_queue_init(&s->tdem_q);
    g_queue_init(&s->tile_q);
    g_queue_init(&s->tile_done);
    s->tile_bh = qemu_bh_new(vtpv_tile_bh, s);
    qemu_thread_create(&s->tile_thread, "vtpv-tiles", vtpv_tile_thread, s,
                       QEMU_THREAD_DETACHED);

    lab_elog( "apple-vtpv: realized v3 transport (msi+intx), msglog=%s verbose=%u "
            "hwbackend=%s\n", s->msglog ? s->msglog : "(stderr)", s->verbose,
            vtpv_use_dx(s) ? "dx" : "vk");
}

static void vtpv_reset_hold(Object *obj, ResetType type)
{
    AppleVtpv *s = APPLE_VTPV(obj);
    for (int i = 0; i < VTPV_MAX_SLOT; i++) {
        g_queue_clear_full(&s->slot[i].outbox, g_free);
    }
    memset(s->slot, 0, sizeof(s->slot));
    s->client_id = 0;
    s->irq_pending = 0;
    s->msgs_in = s->msgs_out = 0;
    for (int i = 0; i < VTPV_SESS; i++) {
        vtpv_sess_free(s, &s->sess[i]);
    }
    s->cur = NULL;
    g_free(s->surfmap_list);
    s->surfmap_list = NULL;
    s->surfmap_list_n = 0;
    s->surfmap_list_gpa = 0;
}

static const Property vtpv_properties[] = {
    DEFINE_PROP_STRING("msglog", AppleVtpv, msglog),
    DEFINE_PROP_STRING("reply_dcap", AppleVtpv, reply_dcap),
    DEFINE_PROP_STRING("reply_plusdec", AppleVtpv, reply_plusdec),
    DEFINE_PROP_STRING("reply_generic", AppleVtpv, reply_generic),
    DEFINE_PROP_STRING("reply_dsup", AppleVtpv, reply_dsup),
    DEFINE_PROP_STRING("dpba_file", AppleVtpv, dpba_file),
    DEFINE_PROP_BOOL("mock_dbuf", AppleVtpv, mock_dbuf, false),
    DEFINE_PROP_BOOL("mock_demt", AppleVtpv, mock_demt, false),
    DEFINE_PROP_BOOL("outbuf", AppleVtpv, outbuf, true),
    DEFINE_PROP_BOOL("decthread", AppleVtpv, decthread, true),
    DEFINE_PROP_UINT32("backlog", AppleVtpv, backlog, 0),
    DEFINE_PROP_BOOL("dno", AppleVtpv, dno, false),
    DEFINE_PROP_BOOL("tiles", AppleVtpv, tiles, true),
    DEFINE_PROP_BOOL("writeback", AppleVtpv, writeback, true),
    DEFINE_PROP_BOOL("hwdecode", AppleVtpv, hwdecode, true),
    DEFINE_PROP_BOOL("tile_fake_error", AppleVtpv, tile_fake_error, false),
    DEFINE_PROP_BOOL("tile-async", AppleVtpv, tile_async, true),
    DEFINE_PROP_BOOL("tdem-ver", AppleVtpv, tdem_ver, false),
    DEFINE_PROP_BOOL("tile-sync", AppleVtpv, tile_sync, false),
    DEFINE_PROP_UINT32("dtil-reply-flags", AppleVtpv, dtil_reply_flags, 4),
    DEFINE_PROP_UINT32("dtil-reply-delay-us", AppleVtpv, dtil_reply_delay_us, 0),
    DEFINE_PROP_BOOL("tdem-kick-gated", AppleVtpv, tdem_kick_gated, false),
    DEFINE_PROP_BOOL("kickdesc", AppleVtpv, kickdesc, false),
    DEFINE_PROP_UINT32("tile-dump", AppleVtpv, tile_dump_left, 0),
    DEFINE_PROP_BOOL("tile-dump-read", AppleVtpv, tile_dump_read, false),
    DEFINE_PROP_BOOL("tdem-fast", AppleVtpv, tdem_fast, true),   /* A/B won 10-26: tail p99 124->72-86ms, 0 errors, 108/108; off = rollback */
    DEFINE_PROP_BOOL("tile-uui", AppleVtpv, tile_uui, true),
    DEFINE_PROP_BOOL("canvas-probe", AppleVtpv, canvas_probe, false),
    DEFINE_PROP_BOOL("dbuf-gate", AppleVtpv, dbuf_gate, true),  /* A/B WON 10-27 (yt24+yt29: hidden kernel 9-12 % vs 47-117 % off; visible clean; appsmoke all-alive) - DEFAULT ON; off = rollback */
    DEFINE_PROP_BOOL("dbuf-gate-live", AppleVtpv, dbuf_gate_live, false), /* A/B-in-one-boot */
    DEFINE_PROP_BOOL("dec-pace", AppleVtpv, dec_pace, false), /* TEMP lab: pool-as-the-clock Dbuf credit */
    DEFINE_PROP_INT32("dec-tid-max", AppleVtpv, dec_tid_max, -1), /* TEMP lab: drop HEVC temporal layers above this tid */
    DEFINE_PROP_BOOL("dec-tid-max-live", AppleVtpv, dec_tid_live, false), /* A/B-in-one-boot */
    DEFINE_PROP_BOOL("dec-pace-live", AppleVtpv, dec_pace_live, false), /* A/B-in-one-boot */
    DEFINE_PROP_UINT64("tdem-gate-cap-us", AppleVtpv, tdem_gate_cap_us, 100000),
    DEFINE_PROP_STRING("decgpu", AppleVtpv, dec_gpu_str),
    DEFINE_PROP_STRING("hwbackend", AppleVtpv, hwbackend_str),
    DEFINE_PROP_UINT32("verbose", AppleVtpv, verbose, 1),
    DEFINE_PROP_UINT32("tdem-delay-us", AppleVtpv, tdem_delay_us, 0),
    DEFINE_PROP_UINT32("tdem-window", AppleVtpv, tdem_window, 1),
    DEFINE_PROP_BOOL("tiles-one-session", AppleVtpv, tiles_one_session, false),
};

static void vtpv_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);
    ResettableClass *rc = RESETTABLE_CLASS(klass);

    k->realize = vtpv_realize;
    k->vendor_id = PCI_VENDOR_ID_APPLE;
    k->device_id = PCI_DEVICE_ID_APPLE_VTPV;
    k->revision = 1;
    k->class_id = PCI_CLASS_MULTIMEDIA_VIDEO;
    set_bit(DEVICE_CATEGORY_DISPLAY, dc->categories);
    device_class_set_props(dc, vtpv_properties);
    rc->phases.hold = vtpv_reset_hold;
    dc->desc = "Apple paravirtual VideoToolbox transport (lab)";
}

static const TypeInfo vtpv_info = {
    .name = TYPE_APPLE_VTPV,
    .parent = TYPE_PCI_DEVICE,
    .instance_size = sizeof(AppleVtpv),
    .class_init = vtpv_class_init,
    .interfaces = (InterfaceInfo[]) {
        { INTERFACE_CONVENTIONAL_PCI_DEVICE },
        { },
    },
};

static void vtpv_register_types(void)
{
    type_register_static(&vtpv_info);
}

type_init(vtpv_register_types)
