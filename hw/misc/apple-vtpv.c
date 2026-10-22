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

#define TYPE_APPLE_VTPV "apple-vtpv-pci"
OBJECT_DECLARE_SIMPLE_TYPE(AppleVtpv, APPLE_VTPV)

/* what the dx decoder's callbacks wrote during one message */
typedef struct {
    uint64_t written;
    uint32_t w, h;
    int pictures;
    bool to_stage;          /* outbuf: keep the picture (tight NV12) in stage */
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

#define VTPV_FRAMES 256     /* frames waiting for their 'Dbuf' reply (samples are small) */
#define VTPV_SESS 16        /* decoder sessions alive at once (wallpaper, Safari tabs...) */
#define VTPV_DBUF_INFLIGHT 4 /* 'Dbuf' requests outstanding per session */
#define VTPV_BACKLOG 6      /* waiting frames before 'dfrm' replies are held */
#define VTPV_HELD 64        /* held 'dfrm' replies per session */

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
    struct {
        uint64_t frame_id;
        uint64_t seq;               /* arrival order = decode order */
        uint8_t dbuf_id[8];         /* raw reply-id bytes of our 'Dbuf' */
        bool in_use;
        bool dbuf_sent;
        uint8_t *nv12;              /* outbuf: decoded picture waiting for its buffer */
        uint32_t w, h;
        uint8_t *sample;            /* outbuf: or the compressed sample, decoded at the reply */
        uint32_t sample_len;
    } frame[VTPV_FRAMES];
    uint64_t next_seq;
    uint32_t dbuf_out;              /* 'Dbuf's sent and not answered yet */
    /* backpressure: past VTPV_BACKLOG waiting frames the 'dfrm' reply is
     * held until frames complete (ox10: the wallpaper player ran 260 frames
     * ahead of its 4-buffer pool and filled the table) */
    struct {
        uint8_t rep[128];
        uint32_t len;
        int slot;
        uint64_t frame_id;
    } held[VTPV_HELD];
    uint32_t held_head, held_n;
    bool dpba_sent;
    uint32_t dpba_w, dpba_h;        /* size the last 'Dpba' announced */
    VtpvDx *dx;
    int dx_codec;                   /* VtpvDxCodec of the open decoder, -1 none */
    uint8_t dx_cfg[4096];           /* avcC/hvcC/vpcC/av1C record it was opened with */
    uint32_t dx_cfg_len;
    VtpvDxSink dx_sink;
} VtpvSess;

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
    uint64_t sess_clock;
    const uint8_t *ob_sample;       /* this dfrm's sample (inside msg) for vtpv_send_dbuf */
    uint32_t ob_sample_len;
    uint64_t host_msg_id;
    /* outbuf=on: the app's own output buffers - each decoded picture is held
     * until the guest answers our 'Dbuf' with a pool buffer (its IOSurface
     * attached), then written into that surface and completed by 'Demt' */
    bool outbuf;
    uint64_t surfdesc_gpa;          /* VTPV_SURFDESC of the next kick */
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
        fprintf(stderr, "apple-vtpv: enq fail: no h2g buffer\n");
        return false;
    }
    if (entry > VTPV_H2G_ENTRY_MAX) {
        fprintf(stderr, "apple-vtpv: enq fail: entry %#x too big\n", entry);
        return false;
    }
    if (dma_memory_read(pci_get_address_space(PCI_DEVICE(s)), sl->h2g_gpa, &h, sizeof(h), MEMTXATTRS_UNSPECIFIED)) {
        fprintf(stderr, "apple-vtpv: enq fail: header read\n");
        return false;
    }
    h.queue_size = le32_to_cpu(h.queue_size);
    h.head = le32_to_cpu(h.head);
    h.tail = le32_to_cpu(h.tail);
    if (h.queue_size == 0 || h.queue_size + sizeof(h) > sl->h2g_len) {
        h.queue_size = sl->h2g_len - sizeof(h);
    }

    if (h.head == h.tail) {
        /* empty in Apple semantics - head==tail is ALWAYS empty (the old
         * "!=0 means full" reading froze the queue after the first consumed
         * reply: n995 "reply enqueue failed" for every message after #3).
         * If the entry cannot fit contiguously at tail, reset head to 0 and
         * restart there - safe because the consumer only wakes on our notify,
         * which happens after the enqueue. */
        if (entry <= h.queue_size - h.tail) {
            tail_off = h.tail;
            freesz = h.queue_size - h.tail;
        } else {
            uint32_t zero = 0;
            dma_memory_write(pci_get_address_space(PCI_DEVICE(s)),
                             sl->h2g_gpa + 4, &zero, 4, MEMTXATTRS_UNSPECIFIED);
            h.head = 0;
            tail_off = 0;
            freesz = h.queue_size;
        }
    } else if (h.tail > h.head) {
        freesz = h.queue_size - h.tail;
        if (entry > freesz) {
            /* wrap marker (zero size entry) then restart at 0 */
            uint32_t zero = 0;
            if (dma_memory_write(pci_get_address_space(PCI_DEVICE(s)),
                                 sl->h2g_gpa + sizeof(h) + h.tail,
                                 &zero, 4, MEMTXATTRS_UNSPECIFIED)) {
                return false;
            }
            freesz = h.head;
            tail_off = 0;
        } else {
            tail_off = h.tail;
        }
    } else {
        freesz = h.head - h.tail;
        tail_off = h.tail;
    }
    if (entry >= freesz) {
        fprintf(stderr, "apple-vtpv: enq fail: entry %#x >= free %#x (head=%u tail=%u qsize=%u)\n",
                entry, freesz, h.head, h.tail, h.queue_size);
        return false;
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
    h.tail = tail_off + entry;
    if (h.tail == h.queue_size) {
        h.tail = 0;    /* exactly at the end: wrap now */
    }
    uint32_t le_tail = cpu_to_le32(h.tail);
    dma_memory_write(pci_get_address_space(PCI_DEVICE(s)), sl->h2g_gpa + 8, &le_tail, 4,
                     MEMTXATTRS_UNSPECIFIED);

    s->msgs_out++;
    return true;
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
        MSIMessage msg = msi_get_message(PCI_DEVICE(s), 0);
        fprintf(stderr, "apple-vtpv: msi_notify addr=%#llx data=%#x\n",
                (unsigned long long)msg.address, msg.data);
        msi_notify(PCI_DEVICE(s), 0);
    } else {
        fprintf(stderr, "apple-vtpv: intx assert (msi off)\n");
        pci_irq_assert(PCI_DEVICE(s));
    }
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
    bool ok = vtpv_dq_enqueue(s, slot, buf, 0x30 + paras_len);
    if (ok) {
        vtpv_raise(s, slot);
    } else {
        fprintf(stderr, "apple-vtpv: h2g %.4s enqueue failed\n", (char *)&type);
    }
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
        fprintf(stderr, "apple-vtpv: frame table full, dropping frame %llu\n",
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
    g_free(s->cur->frame[free_i].nv12);
    s->cur->frame[free_i].nv12 = NULL;
    g_free(s->cur->frame[free_i].sample);
    s->cur->frame[free_i].sample = NULL;
    if (s->outbuf && s->ob_sample) {
        s->cur->frame[free_i].sample = g_memdup2(s->ob_sample, s->ob_sample_len);
        s->cur->frame[free_i].sample_len = s->ob_sample_len;
    } else if (s->outbuf && s->cur->dx_sink.pictures) {
        VtpvDxSink *k = &s->cur->dx_sink;
        size_t sz = (size_t)k->w * k->h + (size_t)((k->w + 1) & ~1u) * ((k->h + 1) / 2);
        s->cur->frame[free_i].nv12 = g_memdup2(k->stage, sz);
        s->cur->frame[free_i].w = k->w;
        s->cur->frame[free_i].h = k->h;
    }
    vtpv_pump_dbuf(s, slot, uuid16);
}

/* ask the guest for buffers for the oldest waiting frames, at most
 * VTPV_DBUF_INFLIGHT at a time (the next goes out when one is answered) */
static void vtpv_pump_dbuf(AppleVtpv *s, int slot, const uint8_t *uuid16)
{
    VtpvSess *se = s->cur;
    while (se->dbuf_out < VTPV_DBUF_INFLIGHT) {
        int next = -1;
        for (int i = 0; i < VTPV_FRAMES; i++) {
            if (se->frame[i].in_use && !se->frame[i].dbuf_sent &&
                (next < 0 || se->frame[i].seq < se->frame[next].seq)) {
                next = i;
            }
        }
        if (next < 0) {
            return;
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
        if (!vtpv_send_h2g(s, slot, 0x44627566u /* 'Dbuf' */, uuid16, id8, paras, n)) {
            return;                 /* queue full: retried at the next reply */
        }
        se->frame[next].dbuf_sent = true;
        se->dbuf_out++;
        if (s->verbose) {
            fprintf(stderr, "apple-vtpv: frame %llu: Dbuf sent (id=%llu, %u out)\n",
                    (unsigned long long)se->frame[next].frame_id,
                    (unsigned long long)s->host_msg_id, se->dbuf_out);
        }
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
        fprintf(stderr, "apple-vtpv: frame %llu: Demt sent err=%d "
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
            fprintf(stderr, "apple-vtpv: Dpba sent %ux%u 420v (%u plist bytes)\n", w, h, pl);
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
            fprintf(stderr, "apple-vtpv: dpba template unreadable (%s)\n", s->dpba_file);
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
        fprintf(stderr, "apple-vtpv: Dpba sent (%ld plist bytes)\n", rlen);
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

static void vtpv_dx_reset(AppleVtpv *s)
{
    vtpv_dx_close(s->cur->dx);
    s->cur->dx = NULL;
    s->cur->dx_codec = -1;
    s->cur->dx_cfg_len = 0;
    s->cur->dpba_w = s->cur->dpba_h = 0;      /* outbuf: the next stream announces its size again */
}

/* a session ends (its uuid deregistered, or evicted): decoder, waiting
 * frames and stage go with it */
static void vtpv_sess_free(VtpvSess *se)
{
    vtpv_dx_close(se->dx);
    for (int i = 0; i < VTPV_FRAMES; i++) {
        g_free(se->frame[i].nv12);
        g_free(se->frame[i].sample);
    }
    g_free(se->dx_sink.stage);
    memset(se, 0, sizeof(*se));
    se->dx_codec = -1;
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
        } else if (!lru || se->last_use < lru->last_use) {
            lru = se;
        }
    }
    VtpvSess *se = free_se ? free_se : lru;
    if (se->used) {
        fprintf(stderr, "apple-vtpv: session table full, evicting the oldest\n");
    }
    vtpv_sess_free(se);
    se->used = true;
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

/* deliver held 'dfrm' replies: all of them (flush/invalidate), else those of
 * frames up to `upto` and the rest once the backlog is below VTPV_BACKLOG */
static void vtpv_release_held(AppleVtpv *s, uint64_t upto, bool all)
{
    VtpvSess *se = s->cur;
    bool below = vtpv_sess_waiting(se) < VTPV_BACKLOG;
    while (se->held_n) {
        typeof(se->held[0]) *h = &se->held[se->held_head];
        if (!all && !below && h->frame_id > upto) {
            break;
        }
        if (vtpv_dq_enqueue(s, h->slot, h->rep, h->len)) {
            vtpv_raise(s, h->slot);
        } else {
            fprintf(stderr, "apple-vtpv: held reply enqueue failed\n");
        }
        se->held_head = (se->held_head + 1) % VTPV_HELD;
        se->held_n--;
    }
}

static void vtpv_sess_drop(AppleVtpv *s, const uint8_t *uuid16)
{
    for (int i = 0; i < VTPV_SESS; i++) {
        VtpvSess *se = &s->sess[i];
        if (se->used && memcmp(se->uuid, uuid16, 16) == 0) {
            if (s->cur == se) {
                s->cur = NULL;
            }
            vtpv_sess_free(se);
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
    fprintf(stderr, "apple-vtpv: dx stream codec=%d cfg=%u bytes\n", codec, cfg_len);
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
            dma_memory_write(pci_get_address_space(PCI_DEVICE(s)),
                             (ks->pages[pg] & ~(uint64_t)0xfff) + (sa & 0xfff), src, chunk,
                             MEMTXATTRS_UNSPECIFIED);
        }
        off += chunk;
        src += chunk;
        len -= chunk;
    }
}

/* a staged tight NV12 picture -> the guest pool buffer the app will receive */
static bool vtpv_fill_surface(AppleVtpv *s, const VtpvKickSurf *ks, const uint8_t *nv12,
                              uint32_t w, uint32_t h)
{
    if ((ks->pixfmt != 0x34323076u /* '420v' */ && ks->pixfmt != 0x34323066u /* '420f' */) ||
        ks->planes != 2) {
        fprintf(stderr, "apple-vtpv: outbuf: surface %u is %.4s with %u planes - not NV12\n",
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
    AppleVtpv *s = opaque;
    VtpvDxSink *k = &s->cur->dx_sink;
    uint32_t w = s->surfmap_meta[0] >> 16;
    uint32_t h = s->surfmap_meta[0] & 0xffffu;
    uint32_t bpr0 = s->surfmap_meta[1] ? s->surfmap_meta[1] : w;
    uint32_t p1 = s->surfmap_meta[2];
    uint32_t bpr1 = s->surfmap_meta[3] ? s->surfmap_meta[3] : w;
    uint32_t dw = MIN(f->width, w), dh = MIN(f->height, h);

    if (k->to_stage) {
        /* outbuf: the picture waits for the guest's own buffer ('Dbuf') */
        /* tight NV12: luma w x h, then CbCr rows of the even-rounded width */
        uint32_t cw = (f->width + 1) & ~1u, ch = (f->height + 1) / 2;
        size_t need = (size_t)f->width * f->height + (size_t)cw * ch;
        if (k->stage_cap < need) {
            k->stage = g_realloc(k->stage, need);
            k->stage_cap = need;
        }
        for (uint32_t y = 0; y < f->height; y++) {
            memcpy(k->stage + (size_t)y * f->width, f->y + (size_t)y * f->y_pitch, f->width);
        }
        uint8_t *uv = k->stage + (size_t)f->width * f->height;
        for (uint32_t y = 0; y < ch; y++) {
            memcpy(uv + (size_t)y * cw, f->uv + (size_t)y * f->uv_pitch, cw);
        }
        k->w = f->width;
        k->h = f->height;
        k->pictures++;
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
            fprintf(stderr, "apple-vtpv: dx put %ux%u org=%llu y0 src %02x%02x%02x%02x "
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
        fprintf(stderr, "apple-vtpv: dx: no ',sbf' in frame %llu\n",
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
        fprintf(stderr, "apple-vtpv: dx: frame %llu sample=%u codec=%d - not decodable\n",
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

static bool vtpv_dx_decode_sample(AppleVtpv *s, const uint8_t *sample, uint32_t sl,
                                  uint64_t tag)
{
    if (!s->cur->dx) {
        s->cur->dx = vtpv_dx_open(s->dec_gpu_str ? s->dec_gpu_str : "int", "d3d12",
                             (VtpvDxCodec)s->cur->dx_codec, s->cur->dx_cfg_len ? s->cur->dx_cfg : NULL,
                             s->cur->dx_cfg_len, vtpv_dx_put, s);
        if (!s->cur->dx) {
            fprintf(stderr, "apple-vtpv: dx: decoder open failed (codec %d)\n", s->cur->dx_codec);
            return false;
        }
    }
    VtpvDxSink *k = &s->cur->dx_sink;
    k->written = 0;                 /* per-message counters; mode + stage stay */
    k->w = k->h = 0;
    k->pictures = 0;
    int64_t t0 = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
    int rc = vtpv_dx_decode(s->cur->dx, sample, sl, (int64_t)tag);
    fprintf(stderr, "apple-vtpv: HW-DECODE(dx) frame=%llu codec=%d %ux%u pictures=%d "
            "-> canvas %llu bytes rc=%d %lld us\n", (unsigned long long)tag, s->cur->dx_codec,
            k->w, k->h, k->pictures, (unsigned long long)k->written, rc,
            (long long)(qemu_clock_get_us(QEMU_CLOCK_REALTIME) - t0));
    return rc == 0 && k->pictures > 0;
}

/* decode the ',sbf' of a 'dtil'/'dfrm' and write the picture back */
static bool vtpv_dx_decode_sbf(AppleVtpv *s, uint64_t total, uint64_t tag)
{
    uint32_t sl = 0;
    const uint8_t *sample = vtpv_dx_sbf_sample(s, total, tag, &sl);
    return sample && vtpv_dx_decode_sample(s, sample, sl, tag);
}

static void vtpv_hwdecode_dfrm(AppleVtpv *s, uint64_t total);
static const VtpvKickSurf *vtpv_kick_surf(AppleVtpv *s, uint32_t id);

static void vtpv_handle_message(AppleVtpv *s, int slot, uint64_t total,
                                uint32_t nsurf)
{
    AppleVtpvSlot *sl = &s->slot[slot];

    if (total < 0x30 || total > sl->g2h_len || total > VTPV_MSG_MAX) {
        if (s->verbose) {
            fprintf(stderr, "apple-vtpv: slot%d bad message size %llu\n",
                    slot, (unsigned long long)total);
        }
        return;
    }
    vtpv_msg_realloc(s, total);
    if (dma_memory_read(pci_get_address_space(PCI_DEVICE(s)), sl->g2h_gpa, s->msg, total, MEMTXATTRS_UNSPECIFIED)) {
        fprintf(stderr, "apple-vtpv: slot%d g2h dma read failed\n", slot);
        return;
    }
    s->msgs_in++;

    uint32_t be_type, be_flags;
    memcpy(&be_type, s->msg + 8, 4);
    memcpy(&be_flags, s->msg + 12, 4);
    uint32_t type = be32_to_cpu(be_type);
    uint32_t flags = be32_to_cpu(be_flags);
    s->cur = vtpv_sess_get(s, s->msg + 0x10);
    char t4[5];
    vtpv_log4cc(t4, type);

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
        fprintf(stderr, "apple-vtpv: slot%d msg size=%llu type=%s flags=%#x "
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
            if (s->outbuf && s->hwdecode && vtpv_use_dx(s)) {
                /* the first picture of a stream is decoded now - its size
                 * goes into 'Dpba' before the first 'Dbuf'; later samples
                 * wait compressed and are decoded straight into the buffer
                 * the guest answers 'Dbuf' with (an app may queue a whole
                 * GOP ahead: ox5b's 1080p/4K probes overflowed a table of
                 * decoded pictures) */
                uint32_t smp_len = 0;
                const uint8_t *smp = vtpv_dx_sbf_sample(s, total, s->last_frame_id, &smp_len);
                s->cur->dx_sink.pictures = 0;
                s->ob_sample = NULL;
                s->ob_sample_len = 0;
                if (smp && !s->cur->dpba_w) {
                    s->cur->dx_sink.to_stage = true;
                    vtpv_dx_decode_sample(s, smp, smp_len, s->last_frame_id);
                    s->cur->dx_sink.to_stage = false;
                } else if (smp) {
                    s->ob_sample = smp;     /* inside s->msg: copied by vtpv_send_dbuf */
                    s->ob_sample_len = smp_len;
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
        fprintf(stderr, "apple-vtpv: dtil tile=%llu surf=[%u,%u] "
                "tco=%d,%d tcd=%dx%d opb=%d,%d\n",
                (unsigned long long)tile_id,
                sl->surf[0], sl->surf[1],
                tco ? (int32_t)ldl_be_p(tco) : -1,
                tco ? (int32_t)ldl_be_p(tco + 4) : -1,
                tcd ? (int32_t)ldl_be_p(tcd) : -1,
                tcd ? (int32_t)ldl_be_p(tcd + 4) : -1,
                opb ? (int32_t)ldl_be_p(opb) : -1,
                opb ? (int32_t)ldl_be_p(opb + 4) : -1);
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
                                fprintf(stderr, "apple-vtpv: HEVC hvcC cached "
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
                        fprintf(stderr, "apple-vtpv: HW-DECODE tile=%llu "
                                "%ux%d -> canvas %llu bytes (chroma=%d)\n",
                                (unsigned long long)tile_id, dw, dh,
                                (unsigned long long)written,
                                (int)vtpv_dec_has_chroma(s->dec));
                        decoded = true;
                    }
                } else {
                    fprintf(stderr, "apple-vtpv: dtil sbf parse: sps=%u pps=%u "
                            "slices=%u flen=%u (len=%u)\n",
                            sps ? sps_l : 0, pps ? pps_l : 0, nslices, flen,
                            sbf_len);
                }
            }
            if (!decoded) {
                fprintf(stderr, "apple-vtpv: hwdecode miss, no writeback\n");
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
            fprintf(stderr, "apple-vtpv: writeback %llu bytes into %u pages "
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
                fprintf(stderr, "apple-vtpv: stray Dbuf (no frame param) dropped\n");
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
                fprintf(stderr, "apple-vtpv: their Dbuf request -> mock uuid "
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
        s->cur->frame[hit].in_use = false;
        s->cur->frame[hit].dbuf_sent = false;
        if (s->cur->dbuf_out) {
            s->cur->dbuf_out--;
        }
        uint32_t inf = 0;
        const uint8_t *pic = s->cur->frame[hit].nv12;    /* first picture of a stream */
        uint32_t pw = s->cur->frame[hit].w, ph = s->cur->frame[hit].h;
        if (s->outbuf && s->cur->frame[hit].sample) {
            /* decode now - replies come back in 'Dbuf' order, which is the
             * decode order - and fill the buffer straight from the stage */
            VtpvDxSink *k = &s->cur->dx_sink;
            k->to_stage = true;
            k->pictures = 0;
            vtpv_dx_decode_sample(s, s->cur->frame[hit].sample, s->cur->frame[hit].sample_len, frame_id);
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
            bool ok = !err && ks && vtpv_fill_surface(s, ks, pic, pw, ph);
            if (s->verbose) {
                fprintf(stderr, "apple-vtpv: frame %llu: Dbuf reply err=%d surf=%u %s -> %s\n",
                        (unsigned long long)frame_id, err, sl->surf[0],
                        ks ? "described" : "NOT described", ok ? "filled" : "NOT filled");
            }
            if (!err && !ok) {
                err = -12911;   /* kVTVideoDecoderMalfunctionErr: no writable buffer */
            }
            g_free(s->cur->frame[hit].nv12);
            s->cur->frame[hit].nv12 = NULL;
        }
        vtpv_release_held(s, frame_id, false);     /* its dfrm reply before its Demt */
        vtpv_send_demt2(s, slot, frame_id, err, inf, s->msg + 0x10, uuid);
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
    if (type == 0x6466696eu /* 'dfin' */ || type == 0x64696e76u /* 'dinv' */) {
        vtpv_release_held(s, 0, true);  /* the guest flushes: nothing may stay held */
    }
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
            fprintf(stderr, "apple-vtpv: dsts avcC cached sps=%u pps=%u\n",
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
    } else if (!(flags & 0x8)) {
        /* 'dsta' (StartSession validation) and 'dinv' need only ',err' per the
         * disassembly; the generic template's type field gets the request's.
         * flags 0x8 = the guest's REPLY to one of our requests (its ',err' for
         * 'Demt'/'Dpba') - answering it would only add a stray message */
        reply_file = s->reply_generic;
        reply_tpl = vtpv_tpl_generic;
        reply_tpl_len = sizeof(vtpv_tpl_generic);
        patch_type = true;
    }
    if (reply_tpl) {
        if (s->verbose) {
            char t5[5];
            vtpv_log4cc(t5, type);
            fprintf(stderr, "apple-vtpv: reply %.4s <- %s%s\n", t5,
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
            fprintf(stderr, "apple-vtpv: reply template unreadable (%s)\n",
                    reply_file);
            return;
        }
        memcpy(rep + 0x10, s->msg + 0x10, 16);      /* guest UUID */
        memcpy(rep + 0x20, s->msg + 0x20, 8);       /* reply identifier (u64) */
        if (patch_type) {
            memcpy(rep + 8, s->msg + 8, 4);         /* echo the request's 4CC type */
        }
        VtpvSess *se = s->cur;
        bool queued = s->ob_sample || se->dx_sink.pictures;    /* this dfrm gets a frame entry */
        if (non_tile_dfrm && s->outbuf && queued && rlen <= sizeof(se->held[0].rep) &&
            se->held_n < VTPV_HELD && vtpv_sess_waiting(se) >= VTPV_BACKLOG) {
            uint32_t h = (se->held_head + se->held_n) % VTPV_HELD;
            memcpy(se->held[h].rep, rep, rlen);
            se->held[h].len = rlen;
            se->held[h].slot = slot;
            se->held[h].frame_id = s->last_frame_id;
            se->held_n++;
        } else if (vtpv_dq_enqueue(s, slot, rep, rlen)) {
            vtpv_raise(s, slot);
        } else {
            fprintf(stderr, "apple-vtpv: reply enqueue failed\n");
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
        if (k->pictures) {
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
            fprintf(stderr, "apple-vtpv: surface %u %.4s %ux%u planes=%u alloc=%u pages=%u "
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
        fprintf(stderr, "apple-vtpv: selftest enqueue failed\n");
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
        v = 2;
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
                fprintf(stderr, "apple-vtpv: read %#llx\n", (unsigned long long)addr);
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
                fprintf(stderr, "apple-vtpv: slot%d uuid op=%lld\n",
                        s->client_id, (long long)val);
            }
            if (val == 2) {
                vtpv_sess_drop(s, uuid);    /* the guest session is gone */
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
        vtpv_load_surfdesc(s);
        vtpv_handle_message(s, s->client_id, val, nsurf);
        vtpv_drop_surfdesc(s);
        memset(sl->surf, 0, sizeof(sl->surf));
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
        s->irq_pending = 0;
        vtpv_lower(s);
        break;
    case VTPV_SELFTEST:
        vtpv_selftest(s, val);
        break;
    case VTPV_DEBUGVAL:
        s->debugval = val;    /* real scratch: the kext reports its MSI state here */
        if (s->verbose) {
            fprintf(stderr, "apple-vtpv: debugval %#llx\n", (unsigned long long)val);
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
                fprintf(stderr, "apple-vtpv: surfmap list read failed (%u pages)\n", n);
            }
        }
        if (s->verbose) {
            fprintf(stderr, "apple-vtpv: surfmap kick=%u pages=%u w=%u h=%u "
                    "bpr=%u p1@%u bpr1=%u gpa0=%llx\n", s->surfmap_kick,
                    s->surfmap_count, s->surfmap_meta[0] >> 16,
                    s->surfmap_meta[0] & 0xffffu, s->surfmap_meta[1],
                    s->surfmap_meta[2], s->surfmap_meta[3],
                    s->surfmap_count ? (unsigned long long)s->surfmap_gpa[0] : 0ull);
            for (uint32_t i = 0; i < 8 && vtpv_page_ok(s, i); i++) {
                fprintf(stderr, "apple-vtpv:   page %u gpa %llx\n", i,
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
            fprintf(stderr, "apple-vtpv: write %#llx <- %#llx\n",
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
                            fprintf(stderr, "apple-vtpv: HEVC hvcC cached "
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
                    fprintf(stderr, "apple-vtpv: HW-DECODE tile=%llu "
                            "%ux%d -> canvas %llu bytes (chroma=%d)\n",
                            (unsigned long long)s->last_frame_id, dw, dh,
                            (unsigned long long)written,
                            (int)vtpv_dec_has_chroma(s->dec));
                    decoded = true;
                }
            } else {
                fprintf(stderr, "apple-vtpv: dtil sbf parse: sps=%u pps=%u "
                        "slices=%u flen=%u (len=%u)\n",
                        sps ? sps_l : 0, pps ? pps_l : 0, nslices, flen,
                        sbf_len);
            }
        }
        if (!decoded) {
            fprintf(stderr, "apple-vtpv: hwdecode miss, no writeback\n");
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
    }

    fprintf(stderr, "apple-vtpv: realized v3 transport (msi+intx), msglog=%s verbose=%u "
            "hwbackend=%s\n", s->msglog ? s->msglog : "(stderr)", s->verbose,
            vtpv_use_dx(s) ? "dx" : "vk");
}

static void vtpv_reset_hold(Object *obj, ResetType type)
{
    AppleVtpv *s = APPLE_VTPV(obj);
    memset(s->slot, 0, sizeof(s->slot));
    s->client_id = 0;
    s->irq_pending = 0;
    s->msgs_in = s->msgs_out = 0;
    for (int i = 0; i < VTPV_SESS; i++) {
        vtpv_sess_free(&s->sess[i]);
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
    DEFINE_PROP_BOOL("writeback", AppleVtpv, writeback, true),
    DEFINE_PROP_BOOL("hwdecode", AppleVtpv, hwdecode, true),
    DEFINE_PROP_STRING("decgpu", AppleVtpv, dec_gpu_str),
    DEFINE_PROP_STRING("hwbackend", AppleVtpv, hwbackend_str),
    DEFINE_PROP_UINT32("verbose", AppleVtpv, verbose, 1),
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
