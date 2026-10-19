/*
 * Linux host USB redirector
 *
 * Copyright (c) 2005 Fabrice Bellard
 *
 * Copyright (c) 2008 Max Krasnyansky
 *      Support for host device auto connect & disconnect
 *      Major rewrite to support fully async operation
 *
 * Copyright 2008 TJ <linux@tjworld.net>
 *      Added flexible support for /dev/bus/usb /sys/bus/usb/devices in addition
 *      to the legacy /proc/bus/usb USB device discovery and handling
 *
 * (c) 2012 Gerd Hoffmann <kraxel@redhat.com>
 *      Completely rewritten to use libusb instead of usbfs ioctls.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "qemu/osdep.h"
#include "qom/object.h"
#ifndef CONFIG_WIN32
#include <poll.h>
#endif
#include <libusb.h>

#ifdef CONFIG_LINUX
#include <sys/ioctl.h>
#include <linux/usbdevice_fs.h>
#endif

#include "qapi/error.h"
#include "migration/vmstate.h"
#include "monitor/monitor.h"
#include "qemu/error-report.h"
#include "qemu/main-loop.h"
#include "qemu/module.h"
#include "qemu/thread.h"
#ifdef CONFIG_WIN32
#include <setupapi.h>
#endif
#include "system/runstate.h"
#include "system/system.h"
#include "trace.h"

#include "hw/core/qdev-properties.h"
#include "hw/usb/usb.h"

/* ------------------------------------------------------------------------ */

#define TYPE_USB_HOST_DEVICE "usb-host"
OBJECT_DECLARE_SIMPLE_TYPE(USBHostDevice, USB_HOST_DEVICE)

typedef struct USBHostRequest USBHostRequest;
typedef struct USBHostIsoXfer USBHostIsoXfer;
typedef struct USBHostIsoRing USBHostIsoRing;

struct USBAutoFilter {
    uint32_t bus_num;
    uint32_t addr;
    char     *port;
    uint32_t vendor_id;
    uint32_t product_id;
};

enum USBHostDeviceOptions {
    USB_HOST_OPT_PIPELINE,
};

/*
 * Where a keyboard's input report keeps its keys, learned from the HID
 * report descriptor of the interface (or the boot layout when there is
 * none): bitmaps of 1-bit variable fields (the modifiers, an NKRO key map)
 * and arrays of usage codes, each under its report ID.
 */
typedef struct USBHostKbdField {
    uint8_t                          report_id;   /* 0: no report IDs */
    bool                             bitmap;
    uint16_t                         bit;         /* after the ID byte */
    uint16_t                         count;
    uint8_t                          size;        /* bits per array entry */
    uint16_t                         first;       /* usage of bit 0 / value lmin */
    int32_t                          lmin;
} USBHostKbdField;

#define USB_HOST_KBD_FIELDS 12

/*
 * Read-ahead for the interrupt IN endpoints of HID interfaces (mice,
 * keyboards): a few transfers stay in flight on the device all the time and
 * their reports queue here, so the device is polled at its own rate instead
 * of once per guest round trip (event thread -> main loop -> xHCI -> guest ->
 * next TD -> submit), which a fast-moving mouse felt as stutter (2026-10-07).
 * A guest packet takes the oldest queued report at once or waits for the
 * next one. QEMU_USB_HOST_READAHEAD=0 turns it off (A/B).
 */
#define USB_HOST_AHEAD_XFERS 4
#define USB_HOST_AHEAD_QUEUE 128
#define USB_HOST_AHEAD_MAXPKT 64

typedef struct USBHostAhead {
    bool                             active;
    int                              inflight;
    int                              status;      /* sticky stall/nodev */
    USBPacket                        *parked;
    uint16_t                         head, count;
    uint64_t                         dropped;
    uint8_t                          len[USB_HOST_AHEAD_QUEUE];
    uint8_t                          data[USB_HOST_AHEAD_QUEUE][USB_HOST_AHEAD_MAXPKT];
} USBHostAhead;

typedef struct USBHostKbdLayout {
    uint8_t                          nfields;
    bool                             ids;         /* reports carry an ID */
    USBHostKbdField                  f[USB_HOST_KBD_FIELDS];
} USBHostKbdLayout;

struct USBHostDevice {
    USBDevice parent_obj;

    /* properties */
    struct USBAutoFilter             match;
    char                             *hostdevice;
    int32_t                          bootindex;
    uint32_t                         iso_urb_count;
    uint32_t                         iso_urb_frames;
    uint32_t                         options;
    uint32_t                         loglevel;
    bool                             needs_autoscan;
    bool                             allow_one_guest_reset;
    bool                             allow_all_guest_resets;
    bool                             suppress_remote_wake;
    bool                             grab;
    bool                             switch_pending; /* Ctrl+Alt+M seen */
    bool                             switch_armed;   /* M seen up since open */

    /* state */
    QTAILQ_ENTRY(USBHostDevice)      next;
    int                              seen, errcount;
    int                              bus_num;
    int                              addr;
    char                             port[16];
    uint16_t                         kbd_in_eps;  /* keyboard int IN eps */
    USBHostKbdLayout                 kbd[16];     /* by endpoint number */
    int64_t                          int_done[16]; /* lab stats */
    uint16_t                         hid_in_eps;   /* HID int IN eps */
    USBHostAhead                     *ahead[16];
    bool                             win32_refused;

    int                              hostfd;
    libusb_device                    *dev;
    libusb_device_handle             *dh;
    struct libusb_device_descriptor  ddesc;

    struct {
        bool                         detached;
        bool                         claimed;
    } ifs[USB_MAX_INTERFACES];

    /* callbacks & friends */
    QEMUBH                           *bh_nodev;
    QEMUBH                           *bh_postld;
    bool                             bh_postld_pending;
    Notifier                         exit;

    /*
     * TEMP lab (QEMU_USB_HOST_SLOW_ASYNC): the guest's device resets and endpoint clear-halts run on this device's
     * own thread instead of the main loop. `slow_busy` (main loop only) holds from the request until the bottom
     * half has taken the result; `slow_running` (under `slow_lock`) while the thread is inside libusb.
     */
    QemuThread                       slow_thread;
    QemuMutex                        slow_lock;
    QemuCond                         slow_cond;
    QEMUBH                           *slow_bh;
    bool                             slow_init;
    bool                             slow_quit;
    bool                             slow_running;
    bool                             slow_busy;
    int                              slow_op;
    int                              slow_arg;
    int                              slow_rc;
    USBPacket                        *slow_pkt;
    int64_t                          slow_t0;
    /* A control request that came while busy: held (ep0 has one at a time) and handled after. */
    struct {
        USBPacket                    *p;
        int                          request, value, index, length;
        uint8_t                      *data;
    } slow_ctl;

    /* request queues */
    QTAILQ_HEAD(, USBHostRequest)    requests;
    QTAILQ_HEAD(, USBHostIsoRing)    isorings;
};

struct USBHostRequest {
    USBHostDevice                    *host;
    USBPacket                        *p;
    bool                             in;
    struct libusb_transfer           *xfer;
    unsigned char                    *buffer;
    unsigned char                    *cbuf;
    unsigned int                     clen;
    bool                             usb3ep0quirk;
    int64_t                          t_submit;    /* lab stats */
    int8_t                           ahead_ep;    /* read-ahead ring, or -1 */
    QTAILQ_ENTRY(USBHostRequest)     next;
};

struct USBHostIsoXfer {
    USBHostIsoRing                   *ring;
    struct libusb_transfer           *xfer;
    bool                             copy_complete;
    unsigned int                     packet;
    QTAILQ_ENTRY(USBHostIsoXfer)     next;
};

struct USBHostIsoRing {
    USBHostDevice                    *host;
    USBEndpoint                      *ep;
    QTAILQ_HEAD(, USBHostIsoXfer)    unused;
    QTAILQ_HEAD(, USBHostIsoXfer)    inflight;
    QTAILQ_HEAD(, USBHostIsoXfer)    copy;
    QTAILQ_ENTRY(USBHostIsoRing)     next;
};

static QTAILQ_HEAD(, USBHostDevice) hostdevs =
    QTAILQ_HEAD_INITIALIZER(hostdevs);

static void usb_host_auto_check(void *unused);
static void usb_host_release_all(void);
static int usb_host_close(USBHostDevice *s);
static void usb_host_release_interfaces(USBHostDevice *s);
static void usb_host_nodev(USBHostDevice *s);
static void usb_host_detach_kernel(USBHostDevice *s);
static void usb_host_attach_kernel(USBHostDevice *s);

/* ------------------------------------------------------------------------ */

#ifndef LIBUSB_LOG_LEVEL_WARNING /* older libusb didn't define these */
#define LIBUSB_LOG_LEVEL_WARNING 2
#endif

/* ------------------------------------------------------------------------ */

#define CONTROL_TIMEOUT  10000        /* 10 sec    */
#define BULK_TIMEOUT         0        /* unlimited */
#define INTR_TIMEOUT         0        /* unlimited */

#ifndef LIBUSB_API_VERSION
# define LIBUSB_API_VERSION LIBUSBX_API_VERSION
#endif
#if LIBUSB_API_VERSION >= 0x01000103
# define HAVE_STREAMS 1
#endif
#if LIBUSB_API_VERSION >= 0x01000106
# define HAVE_SUPER_PLUS 1
#endif

static const char *speed_name[] = {
    [LIBUSB_SPEED_UNKNOWN] = "?",
    [LIBUSB_SPEED_LOW]     = "1.5",
    [LIBUSB_SPEED_FULL]    = "12",
    [LIBUSB_SPEED_HIGH]    = "480",
    [LIBUSB_SPEED_SUPER]   = "5000",
#ifdef HAVE_SUPER_PLUS
    [LIBUSB_SPEED_SUPER_PLUS] = "5000+",
#endif
};

static const unsigned int speed_map[] = {
    [LIBUSB_SPEED_LOW]     = USB_SPEED_LOW,
    [LIBUSB_SPEED_FULL]    = USB_SPEED_FULL,
    [LIBUSB_SPEED_HIGH]    = USB_SPEED_HIGH,
    [LIBUSB_SPEED_SUPER]   = USB_SPEED_SUPER,
#ifdef HAVE_SUPER_PLUS
    [LIBUSB_SPEED_SUPER_PLUS] = USB_SPEED_SUPER,
#endif
};

static const unsigned int status_map[] = {
    [LIBUSB_TRANSFER_COMPLETED] = USB_RET_SUCCESS,
    [LIBUSB_TRANSFER_ERROR]     = USB_RET_IOERROR,
    [LIBUSB_TRANSFER_TIMED_OUT] = USB_RET_IOERROR,
    [LIBUSB_TRANSFER_CANCELLED] = USB_RET_IOERROR,
    [LIBUSB_TRANSFER_STALL]     = USB_RET_STALL,
    [LIBUSB_TRANSFER_NO_DEVICE] = USB_RET_NODEV,
    [LIBUSB_TRANSFER_OVERFLOW]  = USB_RET_BABBLE,
};

static const char *err_names[] = {
    [-LIBUSB_ERROR_IO]               = "IO",
    [-LIBUSB_ERROR_INVALID_PARAM]    = "INVALID_PARAM",
    [-LIBUSB_ERROR_ACCESS]           = "ACCESS",
    [-LIBUSB_ERROR_NO_DEVICE]        = "NO_DEVICE",
    [-LIBUSB_ERROR_NOT_FOUND]        = "NOT_FOUND",
    [-LIBUSB_ERROR_BUSY]             = "BUSY",
    [-LIBUSB_ERROR_TIMEOUT]          = "TIMEOUT",
    [-LIBUSB_ERROR_OVERFLOW]         = "OVERFLOW",
    [-LIBUSB_ERROR_PIPE]             = "PIPE",
    [-LIBUSB_ERROR_INTERRUPTED]      = "INTERRUPTED",
    [-LIBUSB_ERROR_NO_MEM]           = "NO_MEM",
    [-LIBUSB_ERROR_NOT_SUPPORTED]    = "NOT_SUPPORTED",
    [-LIBUSB_ERROR_OTHER]            = "OTHER",
};

static libusb_context *ctx;
static uint32_t loglevel;

#ifndef CONFIG_WIN32

static void usb_host_handle_fd(void *opaque)
{
    struct timeval tv = { 0, 0 };
    libusb_handle_events_timeout(ctx, &tv);
}

static void usb_host_add_fd(int fd, short events, void *user_data)
{
    qemu_set_fd_handler(fd,
                        (events & POLLIN)  ? usb_host_handle_fd : NULL,
                        (events & POLLOUT) ? usb_host_handle_fd : NULL,
                        ctx);
}

static void usb_host_del_fd(int fd, void *user_data)
{
    qemu_set_fd_handler(fd, NULL, NULL, NULL);
}

#else

/*
 * Windows has no pollable libusb fds. A thread sits in libusb's own event
 * loop and hands every completion to the main loop through a bottom half,
 * so completions run under the BQL exactly like the fd handlers above, at
 * the moment the host controller finishes them (a poll timer quantised
 * them to 10 ms: a passed-through mouse was capped at 100 Hz and iso
 * streams starved). The libusb callbacks only queue: taking the BQL inside
 * libusb's event handling would deadlock against libusb_close() and the
 * synchronous transfers the main loop issues under the BQL.
 */
/* When libusb called back the completion now running (lab stats). */
static int64_t usb_host_done_t;

typedef struct USBHostDone {
    struct libusb_transfer           *xfer;
    libusb_transfer_cb_fn            fn;
    int64_t                          t;
    QSIMPLEQ_ENTRY(USBHostDone)      next;
} USBHostDone;

static QemuThread event_thread;
static QemuMutex done_lock;
static QSIMPLEQ_HEAD(, USBHostDone) done_queue =
    QSIMPLEQ_HEAD_INITIALIZER(done_queue);
static QEMUBH *done_bh;

/* Main loop, BQL held: run the queued completions in completion order. */
static void usb_host_done_run(void)
{
    USBHostDone *d;

    for (;;) {
        qemu_mutex_lock(&done_lock);
        d = QSIMPLEQ_FIRST(&done_queue);
        if (d) {
            QSIMPLEQ_REMOVE_HEAD(&done_queue, next);
        }
        qemu_mutex_unlock(&done_lock);
        if (!d) {
            return;
        }
        usb_host_done_t = d->t;
        d->fn(d->xfer);
        g_free(d);
    }
}

static void usb_host_done_bh(void *opaque)
{
    usb_host_done_run();
}

/* Event thread: queue one completion for the main loop. */
static void usb_host_defer(struct libusb_transfer *xfer,
                           libusb_transfer_cb_fn fn)
{
    USBHostDone *d = g_new(USBHostDone, 1);

    d->xfer = xfer;
    d->fn = fn;
    d->t = get_clock();
    qemu_mutex_lock(&done_lock);
    QSIMPLEQ_INSERT_TAIL(&done_queue, d, next);
    qemu_mutex_unlock(&done_lock);
    qemu_bh_schedule(done_bh);
}

static void *usb_host_event_thread(void *opaque)
{
    for (;;) {
        libusb_handle_events(ctx);
    }
    return NULL;
}

#endif /* !CONFIG_WIN32 */

static int usb_host_init(void)
{
#ifndef CONFIG_WIN32
    const struct libusb_pollfd **poll;
#endif
    int rc;

    if (ctx) {
        return 0;
    }
    rc = libusb_init(&ctx);
    if (rc != 0) {
        return -1;
    }
#if LIBUSB_API_VERSION >= 0x01000106
    libusb_set_option(ctx, LIBUSB_OPTION_LOG_LEVEL, loglevel);
#else
    libusb_set_debug(ctx, loglevel);
#endif
#ifdef CONFIG_WIN32
    /*
     * UsbDk, when installed, detaches any device from its Windows driver
     * for as long as QEMU holds it; without it only devices bound to
     * WinUSB (Zadig) can be opened. libusb wants the backend chosen
     * before the context is used.
     */
    rc = libusb_set_option(ctx, LIBUSB_OPTION_USE_USBDK);
    info_report("usb-host: libusb %s backend",
                rc == LIBUSB_SUCCESS ? "UsbDk" : "WinUSB");
    qemu_mutex_init(&done_lock);
    done_bh = qemu_bh_new(usb_host_done_bh, NULL);
    qemu_thread_create(&event_thread, "usb-host-events",
                       usb_host_event_thread, NULL, QEMU_THREAD_DETACHED);
#else
    libusb_set_pollfd_notifiers(ctx, usb_host_add_fd,
                                usb_host_del_fd,
                                ctx);
    poll = libusb_get_pollfds(ctx);
    if (poll) {
        int i;
        for (i = 0; poll[i] != NULL; i++) {
            usb_host_add_fd(poll[i]->fd, poll[i]->events, ctx);
        }
    }
    free(poll);
#endif
    return 0;
}

static int usb_host_get_port(libusb_device *dev, char *port, size_t len)
{
    uint8_t path[7];
    size_t off;
    int rc, i;

#if LIBUSB_API_VERSION >= 0x01000102
    rc = libusb_get_port_numbers(dev, path, 7);
#else
    rc = libusb_get_port_path(ctx, dev, path, 7);
#endif
    if (rc < 0) {
        return 0;
    }
    off = snprintf(port, len, "%d", path[0]);
    for (i = 1; i < rc; i++) {
        off += snprintf(port+off, len-off, ".%d", path[i]);
    }
    return off;
}

static void usb_host_libusb_error(const char *func, int rc)
{
    const char *errname;

    if (rc >= 0) {
        return;
    }

    if (-rc < ARRAY_SIZE(err_names) && err_names[-rc]) {
        errname = err_names[-rc];
    } else {
        errname = "?";
    }
    error_report("%s: %d [%s]", func, rc, errname);
}

/* ------------------------------------------------------------------------ */

static bool usb_host_use_combining(USBEndpoint *ep)
{
    int type;

    if (!ep->pipeline) {
        return false;
    }
    if (ep->pid != USB_TOKEN_IN) {
        return false;
    }
    type = usb_ep_get_type(ep->dev, ep->pid, ep->nr);
    if (type != USB_ENDPOINT_XFER_BULK) {
        return false;
    }
    return true;
}

/* ------------------------------------------------------------------------ */

static USBHostRequest *usb_host_req_alloc(USBHostDevice *s, USBPacket *p,
                                          bool in, size_t bufsize)
{
    USBHostRequest *r = g_new0(USBHostRequest, 1);

    r->host = s;
    r->p = p;
    r->in = in;
    r->ahead_ep = -1;
    r->xfer = libusb_alloc_transfer(0);
    if (bufsize) {
        r->buffer = g_malloc(bufsize);
    }
    QTAILQ_INSERT_TAIL(&s->requests, r, next);
    return r;
}

static void usb_host_req_free(USBHostRequest *r)
{
    QTAILQ_REMOVE(&r->host->requests, r, next);
    libusb_free_transfer(r->xfer);
    g_free(r->buffer);
    g_free(r);
}

static USBHostRequest *usb_host_req_find(USBHostDevice *s, USBPacket *p)
{
    USBHostRequest *r;

    QTAILQ_FOREACH(r, &s->requests, next) {
        if (r->p == p) {
            return r;
        }
    }
    return NULL;
}

/*
 * The release combo: Ctrl+Alt+Shift+Esc on a passed-through keyboard hands
 * every grabbed device back to the host. The keyboard and mouse may be the
 * host's only ones, and a guest that hangs must not keep them. Seen in boot
 * protocol input reports (modifier byte, reserved, six key codes), which is
 * the layout a boot keyboard interface sends without report IDs; the key
 * still reaches the guest. QMP qom-set ... grab=true takes them back.
 */
#define USB_HOST_COMBO_KEY  0x29                 /* Escape */
/*
 * The switch combo (Mish, 2026-10-19): Ctrl+Alt+M on a passed-through
 * keyboard hands that keyboard alone back to the host - the same keys the
 * host's VM switcher (vmswitch.py) takes to go to Windows and back, so one
 * chord moves both the screen and the keyboard. Every other passed-through
 * device (Bluetooth, disks, the trackpad) stays with the guest.
 */
#define USB_HOST_SWITCH_KEY 0x10                 /* M */

static QEMUBH *release_bh;
static bool release_all_pending;

static void usb_host_release_bh(void *opaque)
{
    USBHostDevice *s;

    if (release_all_pending) {
        release_all_pending = false;
        usb_host_release_all();
    }
    QTAILQ_FOREACH(s, &hostdevs, next) {
        if (s->switch_pending) {
            s->switch_pending = false;
            object_property_set_bool(OBJECT(s), "grab", false, NULL);
        }
    }
}

/* The boot keyboard report: modifier bitmap, reserved byte, six key codes. */
static void usb_host_kbd_boot_layout(USBHostKbdLayout *l)
{
    memset(l, 0, sizeof(*l));
    l->nfields = 2;
    l->f[0] = (USBHostKbdField) { .bitmap = true, .bit = 0, .count = 8,
                                  .first = 0xe0 };
    l->f[1] = (USBHostKbdField) { .bit = 16, .count = 6, .size = 8 };
}

/*
 * The keyboard fields of a HID report descriptor's input reports (usage
 * page 7). Push/pop and long items are not followed; a descriptor that
 * needs them simply yields fewer fields. Returns whether any were found.
 */
static bool usb_host_kbd_parse(const uint8_t *d, int len, USBHostKbdLayout *l)
{
    uint32_t page = 0, size = 0, count = 0, id = 0, umin = 0, umax = 0;
    int32_t lmin = 0;
    uint32_t usage0 = 0;
    bool have_usage = false, have_range = false;
    uint16_t off[256] = { 0 };
    int i = 0;

    memset(l, 0, sizeof(*l));
    while (i < len) {
        uint8_t b = d[i], tag = b >> 4, type = (b >> 2) & 3;
        int n = (b & 3) == 3 ? 4 : (b & 3);
        uint32_t v = 0;
        int k;

        if (b == 0xfe) {                       /* long item */
            if (i + 1 >= len) {
                break;
            }
            i += 3 + d[i + 1];
            continue;
        }
        if (i + 1 + n > len) {
            break;
        }
        for (k = 0; k < n; k++) {
            v |= (uint32_t)d[i + 1 + k] << (8 * k);
        }
        i += 1 + n;
        if (type == 1) {                       /* global */
            switch (tag) {
            case 0: page = v; break;
            case 1: lmin = n == 1 ? (int8_t)v : n == 2 ? (int16_t)v : (int32_t)v;
                    break;
            case 7: size = v; break;
            case 8: id = v & 0xff; l->ids = true; break;
            case 9: count = v; break;
            }
        } else if (type == 2) {                /* local */
            uint32_t u = n == 4 ? (v & 0xffff) : v;
            switch (tag) {
            case 0: if (!have_usage) { usage0 = u; have_usage = true; } break;
            case 1: umin = u; have_range = true; break;
            case 2: umax = u; break;
            }
        } else if (type == 0) {                /* main */
            if (tag == 8) {                    /* input */
                bool constant = v & 1, variable = v & 2;
                if (!constant && page == 7 && l->nfields < USB_HOST_KBD_FIELDS) {
                    USBHostKbdField *f = &l->f[l->nfields];
                    f->report_id = id;
                    f->bit = off[id];
                    f->count = count;
                    f->size = size;
                    f->lmin = lmin;
                    if (variable && size == 1) {
                        f->bitmap = true;
                        f->first = have_range ? umin : usage0;
                        l->nfields++;
                    } else if (!variable) {
                        f->first = have_range ? umin : 0;
                        l->nfields++;
                    }
                }
                off[id] += size * count;
            }
            have_usage = have_range = false;
            umin = umax = 0;
        }
    }
    (void)umax;
    return l->nfields > 0;
}

/*
 * The release combo: Ctrl+Alt+Shift+Esc on a passed-through keyboard hands
 * every grabbed device back to the host. Decoded through the interface's
 * own report layout - modifiers held (no GUI key) and Escape the only key -
 * in whatever form the keyboard reports: a boot report, an NKRO bitmap, or
 * a report with an ID. Raw bytes are not enough: the first field test
 * (2026-10-07) released everything on part of a 64-byte NKRO report, and
 * missed the real combo on the same keyboard.
 */
static void usb_host_check_release_combo(USBHostDevice *s, int ep,
                                         uint8_t *rep, int len)
{
    const USBHostKbdLayout *l = &s->kbd[ep];
    uint8_t mods = 0, rid = 0;
    int keys = 0, esc = 0, em = 0, i, j;
    uint8_t *p = rep;

    if (l->ids) {
        if (len < 1) {
            return;
        }
        rid = rep[0];
        p++;
        len--;
    }
    for (i = 0; i < l->nfields; i++) {
        const USBHostKbdField *f = &l->f[i];
        if (f->report_id != rid) {
            continue;
        }
        for (j = 0; j < f->count; j++) {
            uint32_t bit = f->bit + (f->bitmap ? j : j * f->size);
            uint32_t width = f->bitmap ? 1 : f->size, val = 0, k, usage;
            if (bit + width > (uint32_t)len * 8 || width > 16) {
                break;
            }
            for (k = 0; k < width; k++) {
                if (p[(bit + k) / 8] & (1 << ((bit + k) % 8))) {
                    val |= 1u << k;
                }
            }
            if (f->bitmap) {
                if (!val) {
                    continue;
                }
                usage = f->first + j;
            } else {
                if (val == 0 || (int32_t)val < f->lmin) {
                    continue;
                }
                usage = f->first + (val - f->lmin);
            }
            if (usage >= 0xe0 && usage <= 0xe7) {
                mods |= 1 << (usage - 0xe0);
            } else if (usage == USB_HOST_COMBO_KEY) {
                esc++;
            } else if (usage == USB_HOST_SWITCH_KEY) {
                em++;
            } else if (usage > 3) {              /* 1-3: rollover/errors */
                keys++;
            }
        }
    }
    /*
     * The chord that sent the keyboard here (Ctrl+Alt+M on the host) is
     * often still held when the guest gets it: armed only once a report
     * shows M up, so that press does not bounce the keyboard straight back.
     */
    if (!em) {
        s->switch_armed = true;
    }
    if (keys || !(mods & 0x11) || !(mods & 0x44) || (mods & 0x88)) {
        return;
    }
    if (esc == 1 && !em && (mods & 0x22)) {
        info_report("usb-host %04x:%04x: release combo (ep %d, report id %u, "
                    "%d bytes, modifiers %02x)", s->ddesc.idVendor,
                    s->ddesc.idProduct, ep, rid, len, mods);
        release_all_pending = true;
    } else if (em == 1 && !esc && !(mods & 0x22) && s->switch_armed) {
        /*
         * The guest gets this report with every key up instead: it saw
         * Ctrl and Alt go down, and the keyboard is about to vanish, so a
         * report with them (and M) held would be the last it ever got -
         * stuck modifiers and an M repeating until the keyboard returns.
         */
        memset(p, 0, len);
        info_report("usb-host %04x:%04x: switch combo Ctrl+Alt+M (ep %d, "
                    "report id %u, %d bytes): keyboard back to the host",
                    s->ddesc.idVendor, s->ddesc.idProduct, ep, rid, len);
        s->switch_pending = true;
        s->switch_armed = false;
    } else {
        return;
    }
    if (!release_bh) {
        release_bh = qemu_bh_new(usb_host_release_bh, NULL);
    }
    qemu_bh_schedule(release_bh);
}

/*
 * Learn the keyboard layout of every HID interface's interrupt IN endpoint
 * from its report descriptor (a standard GET_DESCRIPTOR to the interface);
 * a boot keyboard interface whose descriptor cannot be read keeps the boot
 * layout. Runs at open, before the guest owns the device.
 */
static void usb_host_kbd_learn(USBHostDevice *s)
{
    struct libusb_config_descriptor *conf;
    int i, e;

    s->kbd_in_eps = 0;
    s->hid_in_eps = 0;
    memset(s->kbd, 0, sizeof(s->kbd));
    if (libusb_get_active_config_descriptor(s->dev, &conf) != 0) {
        return;
    }
    for (i = 0; i < conf->bNumInterfaces; i++) {
        const struct libusb_interface_descriptor *intf =
            &conf->interface[i].altsetting[0];
        int dlen = 0, x, rc;
        uint8_t *buf;
        USBHostKbdLayout l;
        bool boot = intf->bInterfaceSubClass == 1 &&
                    intf->bInterfaceProtocol == 1;
        bool found;

        if (intf->bInterfaceClass != LIBUSB_CLASS_HID) {
            continue;
        }
        /* HID descriptor: bLength 0x21 bcdHID country n {type len16}... */
        for (x = 0; x + 1 < intf->extra_length; x += intf->extra[x]) {
            const uint8_t *h = intf->extra + x;
            int k;
            if (h[0] < 2) {
                break;
            }
            if (h[1] != 0x21 || h[0] < 9) {
                continue;
            }
            for (k = 0; k < h[5] && 6 + 3 * k + 2 < h[0]; k++) {
                if (h[6 + 3 * k] == 0x22) {
                    dlen = h[7 + 3 * k] | (h[8 + 3 * k] << 8);
                }
            }
        }
        found = false;
        if (dlen > 0) {
            buf = g_malloc(dlen);
            rc = libusb_control_transfer(s->dh, 0x81, 6, 0x2200,
                                         intf->bInterfaceNumber, buf, dlen,
                                         1000);
            if (rc > 0) {
                found = usb_host_kbd_parse(buf, rc, &l);
            }
            g_free(buf);
        }
        if (!found && boot) {
            usb_host_kbd_boot_layout(&l);
            found = true;
        }
        for (e = 0; e < intf->bNumEndpoints; e++) {
            const struct libusb_endpoint_descriptor *endp = &intf->endpoint[e];
            if ((endp->bEndpointAddress & USB_DIR_IN) &&
                (endp->bmAttributes & 3) == USB_ENDPOINT_XFER_INT &&
                endp->wMaxPacketSize <= USB_HOST_AHEAD_MAXPKT) {
                s->hid_in_eps |= 1 << (endp->bEndpointAddress & 0xf);
            }
        }
        if (!found) {
            continue;
        }
        for (e = 0; e < intf->bNumEndpoints; e++) {
            const struct libusb_endpoint_descriptor *endp = &intf->endpoint[e];
            int ep = endp->bEndpointAddress & 0xf;
            if ((endp->bEndpointAddress & USB_DIR_IN) &&
                (endp->bmAttributes & 3) == USB_ENDPOINT_XFER_INT) {
                s->kbd[ep] = l;
                s->kbd_in_eps |= 1 << ep;
                info_report("usb-host %04x:%04x: keyboard on interface %d "
                            "ep %d (%u fields%s)", s->ddesc.idVendor,
                            s->ddesc.idProduct, intf->bInterfaceNumber, ep,
                            l.nfields, l.ids ? ", report IDs" : "");
            }
        }
    }
    libusb_free_config_descriptor(conf);
}

/*
 * TEMP lab: QEMU_USB_HOST_STATS=1 prints, once a second while interrupt IN
 * transfers complete, where a report's time goes: submit -> libusb callback
 * (the device; at most its polling interval when it has data), callback ->
 * completion in the main loop (the event thread's hand-off: BH and BQL),
 * and completion -> the guest's next submit on that endpoint (the guest's
 * round trip, during which a device's next report waits).
 */
static int usb_host_stats = -1;
static struct {
    int64_t t0;
    uint64_t n, wait_sum, wait_max, bh_sum, bh_max, rt_n, rt_sum, rt_max;
    uint64_t taken, parked, q_max, dropped;   /* read-ahead */
} ist;

static bool usb_host_stats_on(void)
{
    if (usb_host_stats < 0) {
        const char *e = getenv("QEMU_USB_HOST_STATS");
        usb_host_stats = e && *e && *e != '0';
    }
    return usb_host_stats;
}

static void usb_host_stats_add(uint64_t *sum, uint64_t *max, int64_t v)
{
    if (v < 0) {
        return;
    }
    *sum += v;
    if ((uint64_t)v > *max) {
        *max = v;
    }
}

static void usb_host_stats_flush(int64_t now)
{
    if (!ist.t0) {
        ist.t0 = now;
        return;
    }
    if (now - ist.t0 < NANOSECONDS_PER_SECOND) {
        return;
    }
    if (ist.n) {
        info_report("usb-host-stats int_in n=%" PRIu64 " dev_wait_us avg=%" PRIu64
                    " max=%" PRIu64 " handoff_us avg=%" PRIu64 " max=%" PRIu64
                    " guest_rt_us avg=%" PRIu64 " max=%" PRIu64
                    " ahead taken=%" PRIu64 " parked=%" PRIu64 " q_max=%" PRIu64
                    " dropped=%" PRIu64,
                    ist.n, ist.wait_sum / ist.n / 1000, ist.wait_max / 1000,
                    ist.bh_sum / ist.n / 1000, ist.bh_max / 1000,
                    ist.rt_n ? ist.rt_sum / ist.rt_n / 1000 : 0,
                    ist.rt_max / 1000, ist.taken, ist.parked, ist.q_max,
                    ist.dropped);
    }
    memset(&ist, 0, sizeof(ist));
    ist.t0 = now;
}

/*
 * USB_HOST_CB(fn) is what a transfer gets as its callback: fn itself where
 * libusb completes in the main loop, a queueing trampoline on Windows.
 */
#ifdef CONFIG_WIN32
#define USB_HOST_CB(fn) fn##_deferred
#define USB_HOST_DEFERRED(fn)                                           \
    static void LIBUSB_CALL fn##_deferred(struct libusb_transfer *xfer) \
    {                                                                   \
        usb_host_defer(xfer, fn);                                       \
    }
#else
#define USB_HOST_CB(fn) fn
#define USB_HOST_DEFERRED(fn)
#endif

static void LIBUSB_CALL usb_host_req_complete_ctrl(struct libusb_transfer *xfer)
{
    USBHostRequest *r = xfer->user_data;
    USBHostDevice  *s = r->host;
    bool disconnect = (xfer->status == LIBUSB_TRANSFER_NO_DEVICE);

    if (r->p == NULL) {
        goto out; /* request was canceled */
    }

    r->p->status = status_map[xfer->status];
    r->p->actual_length = xfer->actual_length;
    if (r->in && xfer->actual_length) {
        USBDevice *udev = USB_DEVICE(s);
        struct libusb_config_descriptor *conf = (void *)r->cbuf;
        memcpy(r->cbuf, r->buffer + 8, xfer->actual_length);

        /* Fix up USB-3 ep0 maxpacket size to allow superspeed connected devices
         * to work redirected to a not superspeed capable hcd */
        if (r->usb3ep0quirk && xfer->actual_length >= 18 &&
            r->cbuf[7] == 9) {
            r->cbuf[7] = 64;
        }
        /*
         *If this is GET_DESCRIPTOR request for configuration descriptor,
         * remove 'remote wakeup' flag from it to prevent idle power down
         * in Windows guest
         */
        if (s->suppress_remote_wake &&
            udev->setup_buf[0] == USB_DIR_IN &&
            udev->setup_buf[1] == USB_REQ_GET_DESCRIPTOR &&
            udev->setup_buf[3] == USB_DT_CONFIG && udev->setup_buf[2] == 0 &&
            xfer->actual_length >
                offsetof(struct libusb_config_descriptor, bmAttributes) &&
            (conf->bmAttributes & USB_CFG_ATT_WAKEUP)) {
                trace_usb_host_remote_wakeup_removed(s->bus_num, s->addr);
                conf->bmAttributes &= ~USB_CFG_ATT_WAKEUP;
        }
    }
    trace_usb_host_req_complete(s->bus_num, s->addr, r->p,
                                r->p->status, r->p->actual_length);
    usb_generic_async_ctrl_complete(USB_DEVICE(s), r->p);

out:
    usb_host_req_free(r);
    if (disconnect) {
        usb_host_nodev(s);
    }
}
USB_HOST_DEFERRED(usb_host_req_complete_ctrl)

static void LIBUSB_CALL usb_host_req_complete_data(struct libusb_transfer *xfer)
{
    USBHostRequest *r = xfer->user_data;
    USBHostDevice  *s = r->host;
    bool disconnect = (xfer->status == LIBUSB_TRANSFER_NO_DEVICE);

    if (r->p == NULL) {
        goto out; /* request was canceled */
    }

    r->p->status = status_map[xfer->status];
    /*
     * TEMP lab (Opus 2026-10-19, Mish: the Bluetooth trackpad keeps dropping): every failed data transfer of a
     * passed-through device, with its endpoint and libusb status (2 timeout, 4 stall, 5 no device, 6 overflow,
     * 1 error); the first 50, then every 1000th.
     */
    if (xfer->status != LIBUSB_TRANSFER_COMPLETED && xfer->status != LIBUSB_TRANSFER_CANCELLED) {
        static uint64_t errs;
        uint64_t n = ++errs;
        if (n <= 50 || n % 1000 == 0) {
            fprintf(stderr, "usb-host-err #%" PRIu64 " %04x:%04x ep %#x status %d len %d/%d epoch_ms=%" PRId64 "\n",
                    n, s->ddesc.idVendor, s->ddesc.idProduct, xfer->endpoint, xfer->status, xfer->actual_length,
                    xfer->length, g_get_real_time() / 1000);
        }
    }
    if (r->in && xfer->actual_length) {
        /* Before the copy: the switch combo rewrites what the guest gets. */
        if (s->kbd_in_eps & (1 << r->p->ep->nr)) {
            usb_host_check_release_combo(s, r->p->ep->nr, r->buffer,
                                         xfer->actual_length);
        }
        usb_packet_copy(r->p, r->buffer, xfer->actual_length);
        if (usb_host_stats_on() && xfer->type == LIBUSB_TRANSFER_TYPE_INTERRUPT) {
            int64_t now = get_clock();
#ifdef CONFIG_WIN32
            int64_t cb = usb_host_done_t;
#else
            int64_t cb = now;
#endif
            ist.n++;
            usb_host_stats_add(&ist.wait_sum, &ist.wait_max, cb - r->t_submit);
            usb_host_stats_add(&ist.bh_sum, &ist.bh_max, now - cb);
            s->int_done[r->p->ep->nr & 15] = now;
            usb_host_stats_flush(now);
        }
    }
    trace_usb_host_req_complete(s->bus_num, s->addr, r->p,
                                r->p->status, r->p->actual_length);
    if (usb_host_use_combining(r->p->ep)) {
        usb_combined_input_packet_complete(USB_DEVICE(s), r->p);
    } else {
        usb_packet_complete(USB_DEVICE(s), r->p);
    }

out:
    usb_host_req_free(r);
    if (disconnect) {
        usb_host_nodev(s);
    }
}
USB_HOST_DEFERRED(usb_host_req_complete_data)

/* ---- read-ahead for HID interrupt IN endpoints (see USBHostAhead) ---- */

static bool usb_host_readahead_on(void)
{
    static int on = -1;

    if (on < 0) {
        const char *e = getenv("QEMU_USB_HOST_READAHEAD");
        on = !(e && *e == '0');
    }
    return on;
}

/* The oldest queued report into the guest's packet. */
static void usb_host_ahead_pop(USBHostAhead *a, USBPacket *p)
{
    size_t len = MIN((size_t)a->len[a->head], p->iov.size);

    usb_packet_copy(p, a->data[a->head], len);
    if (usb_lab_trace_on()) {
        usb_lab_trace("T %04x ep%d q%d",
                      USB_HOST_DEVICE(p->ep->dev)->match.product_id,
                      p->ep->nr, a->count);
    }
    a->head = (a->head + 1) % USB_HOST_AHEAD_QUEUE;
    a->count--;
    p->status = USB_RET_SUCCESS;
}

static void LIBUSB_CALL usb_host_ahead_complete(struct libusb_transfer *xfer);
USB_HOST_DEFERRED(usb_host_ahead_complete)

static int usb_host_ahead_submit(USBHostDevice *s, int ep)
{
    USBEndpoint *uep = usb_ep_get(USB_DEVICE(s), USB_TOKEN_IN, ep);
    int mps = MIN(uep->max_packet_size, USB_HOST_AHEAD_MAXPKT);
    USBHostRequest *r = usb_host_req_alloc(s, NULL, true, mps);
    int rc;

    r->ahead_ep = ep;
    libusb_fill_interrupt_transfer(r->xfer, s->dh, ep | USB_DIR_IN, r->buffer,
                                   mps, USB_HOST_CB(usb_host_ahead_complete),
                                   r, INTR_TIMEOUT);
    rc = libusb_submit_transfer(r->xfer);
    if (rc != 0) {
        usb_host_req_free(r);
        return rc;
    }
    s->ahead[ep]->inflight++;
    return 0;
}

static void LIBUSB_CALL usb_host_ahead_complete(struct libusb_transfer *xfer)
{
    USBHostRequest *r = xfer->user_data;
    USBHostDevice *s = r->host;
    int ep = r->ahead_ep;
    USBHostAhead *a = s->ahead[ep];
    bool disconnect = (xfer->status == LIBUSB_TRANSFER_NO_DEVICE);

    a->inflight--;
    if (!a->active || xfer->status == LIBUSB_TRANSFER_CANCELLED) {
        usb_host_req_free(r);
        goto out;
    }
    if (xfer->status == LIBUSB_TRANSFER_COMPLETED) {
        int len = MIN(xfer->actual_length, USB_HOST_AHEAD_MAXPKT);
        if (usb_host_stats_on()) {
            int64_t now = get_clock();
#ifdef CONFIG_WIN32
            int64_t cb = usb_host_done_t;
#else
            int64_t cb = now;
#endif
            ist.n++;
            usb_host_stats_add(&ist.bh_sum, &ist.bh_max, now - cb);
            usb_host_stats_flush(now);
        }
        if (len > 0) {
            int slot;
            if (s->kbd_in_eps & (1 << ep)) {
                usb_host_check_release_combo(s, ep, r->buffer, len);
            }
            if (a->count == USB_HOST_AHEAD_QUEUE) {
                a->head = (a->head + 1) % USB_HOST_AHEAD_QUEUE;
                a->count--;
                a->dropped++;
                ist.dropped++;
            }
            if (usb_lab_trace_on()) {
                const uint8_t *b = r->buffer;
                usb_lab_trace("R %04x ep%d len%d %02x %02x %02x %02x %02x %02x %02x %02x",
                              s->match.product_id, ep, len, b[0], len > 1 ? b[1] : 0,
                              len > 2 ? b[2] : 0, len > 3 ? b[3] : 0,
                              len > 4 ? b[4] : 0, len > 5 ? b[5] : 0,
                              len > 6 ? b[6] : 0, len > 7 ? b[7] : 0);
            }
            slot = (a->head + a->count) % USB_HOST_AHEAD_QUEUE;
            a->len[slot] = len;
            memcpy(a->data[slot], r->buffer, len);
            a->count++;
            if (a->count > ist.q_max) {
                ist.q_max = a->count;
            }
        }
        if (libusb_submit_transfer(r->xfer) == 0) {
            a->inflight++;
        } else {
            usb_host_req_free(r);
        }
    } else {
        a->status = disconnect ? USB_RET_NODEV :
            xfer->status == LIBUSB_TRANSFER_STALL ? USB_RET_STALL : USB_RET_IOERROR;
        a->active = false;
        usb_host_req_free(r);
    }
    if (a->parked) {
        USBPacket *p = a->parked;
        if (a->count) {
            a->parked = NULL;
            usb_host_ahead_pop(a, p);
            usb_packet_complete(USB_DEVICE(s), p);
        } else if (!a->active && a->status) {
            a->parked = NULL;
            p->status = a->status;
            a->status = 0;
            usb_packet_complete(USB_DEVICE(s), p);
        }
    }
out:
    if (disconnect) {
        usb_host_nodev(s);
    }
}

/*
 * The ring of ep, running (started on the guest's first poll), or NULL: the
 * endpoint then goes the plain way - one transfer per guest packet - e.g.
 * while the transfers of a stopped ring are still draining.
 */
static USBHostAhead *usb_host_ahead_start(USBHostDevice *s, int ep)
{
    USBHostAhead *a = s->ahead[ep];
    int i;

    if (!a) {
        a = s->ahead[ep] = g_new0(USBHostAhead, 1);
    }
    if (!a->active && a->inflight == 0) {
        a->active = true;
        a->status = 0;
        a->head = a->count = 0;
        for (i = 0; i < USB_HOST_AHEAD_XFERS; i++) {
            if (usb_host_ahead_submit(s, ep) != 0) {
                break;
            }
        }
        if (a->inflight == 0) {
            a->active = false;
        }
    }
    return a->active ? a : NULL;
}

/*
 * Stop every ring: in-flight transfers are cancelled and free themselves as
 * they come back; a parked guest packet completes with status.
 */
static void usb_host_ahead_stop_all(USBHostDevice *s, int status)
{
    USBHostRequest *r;
    int ep;

    for (ep = 0; ep < 16; ep++) {
        USBHostAhead *a = s->ahead[ep];
        if (!a) {
            continue;
        }
        a->active = false;
        a->head = a->count = 0;
        a->status = 0;
        if (a->parked) {
            USBPacket *p = a->parked;
            a->parked = NULL;
            p->status = status;
            usb_packet_complete(USB_DEVICE(s), p);
        }
    }
    QTAILQ_FOREACH(r, &s->requests, next) {
        if (r->ahead_ep >= 0) {
            libusb_cancel_transfer(r->xfer);
        }
    }
}

/* After usb_host_abort_xfers drained every transfer. */
static void usb_host_ahead_free_all(USBHostDevice *s)
{
    int ep;

    for (ep = 0; ep < 16; ep++) {
        g_free(s->ahead[ep]);
        s->ahead[ep] = NULL;
    }
}

static void usb_host_req_abort(USBHostRequest *r)
{
    USBHostDevice  *s = r->host;
    bool inflight = (r->p && r->p->state == USB_PACKET_ASYNC);

    if (r->ahead_ep >= 0) {
        libusb_cancel_transfer(r->xfer);
        return;
    }

    if (inflight) {
        r->p->status = USB_RET_NODEV;
        trace_usb_host_req_complete(s->bus_num, s->addr, r->p,
                                    r->p->status, r->p->actual_length);
        if (r->p->ep->nr == 0) {
            usb_generic_async_ctrl_complete(USB_DEVICE(s), r->p);
        } else {
            usb_packet_complete(USB_DEVICE(s), r->p);
        }
        r->p = NULL;

        libusb_cancel_transfer(r->xfer);
    }
}

/* ------------------------------------------------------------------------ */

static void LIBUSB_CALL
usb_host_req_complete_iso(struct libusb_transfer *transfer)
{
    USBHostIsoXfer *xfer = transfer->user_data;

    if (!xfer) {
        /* USBHostIsoXfer released while inflight */
        g_free(transfer->buffer);
        libusb_free_transfer(transfer);
        return;
    }

    QTAILQ_REMOVE(&xfer->ring->inflight, xfer, next);
    if (QTAILQ_EMPTY(&xfer->ring->inflight)) {
        USBHostDevice *s = xfer->ring->host;
        trace_usb_host_iso_stop(s->bus_num, s->addr, xfer->ring->ep->nr);
    }
    if (xfer->ring->ep->pid == USB_TOKEN_IN) {
        QTAILQ_INSERT_TAIL(&xfer->ring->copy, xfer, next);
        usb_wakeup(xfer->ring->ep, 0);
    } else {
        QTAILQ_INSERT_TAIL(&xfer->ring->unused, xfer, next);
    }
}
USB_HOST_DEFERRED(usb_host_req_complete_iso)

static USBHostIsoRing *usb_host_iso_alloc(USBHostDevice *s, USBEndpoint *ep)
{
    USBHostIsoRing *ring = g_new0(USBHostIsoRing, 1);
    USBHostIsoXfer *xfer;
    /* FIXME: check interval (for now assume one xfer per frame) */
    int packets = s->iso_urb_frames;
    int i;

    ring->host = s;
    ring->ep = ep;
    QTAILQ_INIT(&ring->unused);
    QTAILQ_INIT(&ring->inflight);
    QTAILQ_INIT(&ring->copy);
    QTAILQ_INSERT_TAIL(&s->isorings, ring, next);

    for (i = 0; i < s->iso_urb_count; i++) {
        xfer = g_new0(USBHostIsoXfer, 1);
        xfer->ring = ring;
        xfer->xfer = libusb_alloc_transfer(packets);
        xfer->xfer->dev_handle = s->dh;
        xfer->xfer->type = LIBUSB_TRANSFER_TYPE_ISOCHRONOUS;

        xfer->xfer->endpoint = ring->ep->nr;
        if (ring->ep->pid == USB_TOKEN_IN) {
            xfer->xfer->endpoint |= USB_DIR_IN;
        }
        xfer->xfer->callback = USB_HOST_CB(usb_host_req_complete_iso);
        xfer->xfer->user_data = xfer;

        xfer->xfer->num_iso_packets = packets;
        xfer->xfer->length = ring->ep->max_packet_size * packets;
        xfer->xfer->buffer = g_malloc0(xfer->xfer->length);

        QTAILQ_INSERT_TAIL(&ring->unused, xfer, next);
    }

    return ring;
}

static USBHostIsoRing *usb_host_iso_find(USBHostDevice *s, USBEndpoint *ep)
{
    USBHostIsoRing *ring;

    QTAILQ_FOREACH(ring, &s->isorings, next) {
        if (ring->ep == ep) {
            return ring;
        }
    }
    return NULL;
}

static void usb_host_iso_reset_xfer(USBHostIsoXfer *xfer)
{
    libusb_set_iso_packet_lengths(xfer->xfer,
                                  xfer->ring->ep->max_packet_size);
    xfer->packet = 0;
    xfer->copy_complete = false;
}

static void usb_host_iso_free_xfer(USBHostIsoXfer *xfer, bool inflight)
{
    if (inflight) {
        xfer->xfer->user_data = NULL;
    } else {
        g_free(xfer->xfer->buffer);
        libusb_free_transfer(xfer->xfer);
    }
    g_free(xfer);
}

static void usb_host_iso_free(USBHostIsoRing *ring)
{
    USBHostIsoXfer *xfer;

    while ((xfer = QTAILQ_FIRST(&ring->inflight)) != NULL) {
        QTAILQ_REMOVE(&ring->inflight, xfer, next);
        usb_host_iso_free_xfer(xfer, true);
    }
    while ((xfer = QTAILQ_FIRST(&ring->unused)) != NULL) {
        QTAILQ_REMOVE(&ring->unused, xfer, next);
        usb_host_iso_free_xfer(xfer, false);
    }
    while ((xfer = QTAILQ_FIRST(&ring->copy)) != NULL) {
        QTAILQ_REMOVE(&ring->copy, xfer, next);
        usb_host_iso_free_xfer(xfer, false);
    }

    QTAILQ_REMOVE(&ring->host->isorings, ring, next);
    g_free(ring);
}

static void usb_host_iso_free_all(USBHostDevice *s)
{
    USBHostIsoRing *ring;

    while ((ring = QTAILQ_FIRST(&s->isorings)) != NULL) {
        usb_host_iso_free(ring);
    }
}

static bool usb_host_iso_data_copy(USBHostIsoXfer *xfer, USBPacket *p)
{
    unsigned int psize;
    unsigned char *buf;

    buf = libusb_get_iso_packet_buffer_simple(xfer->xfer, xfer->packet);
    if (p->pid == USB_TOKEN_OUT) {
        psize = p->iov.size;
        if (psize > xfer->ring->ep->max_packet_size) {
            /* should not happen (guest bug) */
            psize = xfer->ring->ep->max_packet_size;
        }
        xfer->xfer->iso_packet_desc[xfer->packet].length = psize;
    } else {
        psize = xfer->xfer->iso_packet_desc[xfer->packet].actual_length;
        if (psize > p->iov.size) {
            /* should not happen (guest bug) */
            psize = p->iov.size;
        }
    }
    usb_packet_copy(p, buf, psize);
    xfer->packet++;
    xfer->copy_complete = (xfer->packet == xfer->xfer->num_iso_packets);
    return xfer->copy_complete;
}

static void usb_host_iso_data_in(USBHostDevice *s, USBPacket *p)
{
    USBHostIsoRing *ring;
    USBHostIsoXfer *xfer;
    bool disconnect = false;
    int rc;

    ring = usb_host_iso_find(s, p->ep);
    if (ring == NULL) {
        ring = usb_host_iso_alloc(s, p->ep);
    }

    /* copy data to guest */
    xfer = QTAILQ_FIRST(&ring->copy);
    if (xfer != NULL) {
        if (usb_host_iso_data_copy(xfer, p)) {
            QTAILQ_REMOVE(&ring->copy, xfer, next);
            QTAILQ_INSERT_TAIL(&ring->unused, xfer, next);
        }
    }

    /* submit empty bufs to host */
    while ((xfer = QTAILQ_FIRST(&ring->unused)) != NULL) {
        QTAILQ_REMOVE(&ring->unused, xfer, next);
        usb_host_iso_reset_xfer(xfer);
        rc = libusb_submit_transfer(xfer->xfer);
        if (rc != 0) {
            usb_host_libusb_error("libusb_submit_transfer [iso]", rc);
            QTAILQ_INSERT_TAIL(&ring->unused, xfer, next);
            if (rc == LIBUSB_ERROR_NO_DEVICE) {
                disconnect = true;
            }
            break;
        }
        if (QTAILQ_EMPTY(&ring->inflight)) {
            trace_usb_host_iso_start(s->bus_num, s->addr, p->ep->nr);
        }
        QTAILQ_INSERT_TAIL(&ring->inflight, xfer, next);
    }

    if (disconnect) {
        usb_host_nodev(s);
    }
}

static void usb_host_iso_data_out(USBHostDevice *s, USBPacket *p)
{
    USBHostIsoRing *ring;
    USBHostIsoXfer *xfer;
    bool disconnect = false;
    int rc, filled = 0;

    ring = usb_host_iso_find(s, p->ep);
    if (ring == NULL) {
        ring = usb_host_iso_alloc(s, p->ep);
    }

    /* copy data from guest */
    xfer = QTAILQ_FIRST(&ring->copy);
    while (xfer != NULL && xfer->copy_complete) {
        filled++;
        xfer = QTAILQ_NEXT(xfer, next);
    }
    if (xfer == NULL) {
        xfer = QTAILQ_FIRST(&ring->unused);
        if (xfer == NULL) {
            trace_usb_host_iso_out_of_bufs(s->bus_num, s->addr, p->ep->nr);
            return;
        }
        QTAILQ_REMOVE(&ring->unused, xfer, next);
        usb_host_iso_reset_xfer(xfer);
        QTAILQ_INSERT_TAIL(&ring->copy, xfer, next);
    }
    usb_host_iso_data_copy(xfer, p);

    if (QTAILQ_EMPTY(&ring->inflight)) {
        /* wait until half of our buffers are filled
           before kicking the iso out stream */
        if (filled*2 < s->iso_urb_count) {
            return;
        }
    }

    /* submit filled bufs to host */
    while ((xfer = QTAILQ_FIRST(&ring->copy)) != NULL &&
           xfer->copy_complete) {
        QTAILQ_REMOVE(&ring->copy, xfer, next);
        rc = libusb_submit_transfer(xfer->xfer);
        if (rc != 0) {
            usb_host_libusb_error("libusb_submit_transfer [iso]", rc);
            QTAILQ_INSERT_TAIL(&ring->unused, xfer, next);
            if (rc == LIBUSB_ERROR_NO_DEVICE) {
                disconnect = true;
            }
            break;
        }
        if (QTAILQ_EMPTY(&ring->inflight)) {
            trace_usb_host_iso_start(s->bus_num, s->addr, p->ep->nr);
        }
        QTAILQ_INSERT_TAIL(&ring->inflight, xfer, next);
    }

    if (disconnect) {
        usb_host_nodev(s);
    }
}

/* ------------------------------------------------------------------------ */

static void usb_host_speed_compat(USBHostDevice *s)
{
    USBDevice *udev = USB_DEVICE(s);
    struct libusb_config_descriptor *conf;
    const struct libusb_interface_descriptor *intf;
    const struct libusb_endpoint_descriptor *endp;
#ifdef HAVE_STREAMS
    struct libusb_ss_endpoint_companion_descriptor *endp_ss_comp;
#endif
    bool compat_high = true;
    bool compat_full = true;
    uint8_t type;
    int rc, c, i, a, e;

    for (c = 0;; c++) {
        rc = libusb_get_config_descriptor(s->dev, c, &conf);
        if (rc != 0) {
            break;
        }
        for (i = 0; i < conf->bNumInterfaces; i++) {
            for (a = 0; a < conf->interface[i].num_altsetting; a++) {
                intf = &conf->interface[i].altsetting[a];

                if (intf->bInterfaceClass == LIBUSB_CLASS_MASS_STORAGE &&
                    intf->bInterfaceSubClass == 6) { /* SCSI */
                    udev->flags |= (1 << USB_DEV_FLAG_IS_SCSI_STORAGE);
                    break;
                }

                for (e = 0; e < intf->bNumEndpoints; e++) {
                    endp = &intf->endpoint[e];
                    type = endp->bmAttributes & 0x3;
                    switch (type) {
                    case 0x01: /* ISO */
                        compat_full = false;
                        compat_high = false;
                        break;
                    case 0x02: /* BULK */
#ifdef HAVE_STREAMS
                        rc = libusb_get_ss_endpoint_companion_descriptor
                            (ctx, endp, &endp_ss_comp);
                        if (rc == LIBUSB_SUCCESS) {
                            int streams = endp_ss_comp->bmAttributes & 0x1f;
                            if (streams) {
                                compat_full = false;
                                compat_high = false;
                            }
                            libusb_free_ss_endpoint_companion_descriptor
                                (endp_ss_comp);
                        }
#endif
                        break;
                    case 0x03: /* INTERRUPT */
                        if (endp->wMaxPacketSize > 64) {
                            compat_full = false;
                        }
                        if (endp->wMaxPacketSize > 1024) {
                            compat_high = false;
                        }
                        break;
                    }
                }
            }
        }
        libusb_free_config_descriptor(conf);
    }

    udev->speedmask = (1 << udev->speed);
    if (udev->speed == USB_SPEED_SUPER && compat_high) {
        udev->speedmask |= USB_SPEED_MASK_HIGH;
    }
    if (udev->speed == USB_SPEED_SUPER && compat_full) {
        udev->speedmask |= USB_SPEED_MASK_FULL;
    }
    if (udev->speed == USB_SPEED_HIGH && compat_full) {
        udev->speedmask |= USB_SPEED_MASK_FULL;
    }
}

static void usb_host_ep_update(USBHostDevice *s)
{
    static const char *tname[] = {
        [USB_ENDPOINT_XFER_CONTROL] = "control",
        [USB_ENDPOINT_XFER_ISOC]    = "isoc",
        [USB_ENDPOINT_XFER_BULK]    = "bulk",
        [USB_ENDPOINT_XFER_INT]     = "int",
    };
    USBDevice *udev = USB_DEVICE(s);
    struct libusb_config_descriptor *conf;
    const struct libusb_interface_descriptor *intf;
    const struct libusb_endpoint_descriptor *endp;
#ifdef HAVE_STREAMS
    struct libusb_ss_endpoint_companion_descriptor *endp_ss_comp;
#endif
    uint8_t devep, type;
    int pid, ep, alt;
    int rc, i, e;

    usb_ep_reset(udev);
    rc = libusb_get_active_config_descriptor(s->dev, &conf);
    if (rc != 0) {
        return;
    }
    trace_usb_host_parse_config(s->bus_num, s->addr,
                                conf->bConfigurationValue, true);

    for (i = 0; i < conf->bNumInterfaces; i++) {
        /*
         * The udev->altsetting array indexes alternate settings
         * by the interface number. Get the 0th alternate setting
         * first so that we can grab the interface number, and
         * then correct the alternate setting value if necessary.
         */
        intf = &conf->interface[i].altsetting[0];
        alt = udev->altsetting[intf->bInterfaceNumber];

        if (alt != 0) {
            assert(alt < conf->interface[i].num_altsetting);
            intf = &conf->interface[i].altsetting[alt];
        }

        trace_usb_host_parse_interface(s->bus_num, s->addr,
                                       intf->bInterfaceNumber,
                                       intf->bAlternateSetting, true);
        for (e = 0; e < intf->bNumEndpoints; e++) {
            endp = &intf->endpoint[e];

            devep = endp->bEndpointAddress;
            pid = (devep & USB_DIR_IN) ? USB_TOKEN_IN : USB_TOKEN_OUT;
            ep = devep & 0xf;
            type = endp->bmAttributes & 0x3;

            if (ep == 0) {
                trace_usb_host_parse_error(s->bus_num, s->addr,
                                           "invalid endpoint address");
                return;
            }
            if (usb_ep_get_type(udev, pid, ep) != USB_ENDPOINT_XFER_INVALID) {
                trace_usb_host_parse_error(s->bus_num, s->addr,
                                           "duplicate endpoint address");
                return;
            }

            trace_usb_host_parse_endpoint(s->bus_num, s->addr, ep,
                                          (devep & USB_DIR_IN) ? "in" : "out",
                                          tname[type], true);
            usb_ep_set_max_packet_size(udev, pid, ep,
                                       endp->wMaxPacketSize);
            usb_ep_set_type(udev, pid, ep, type);
            usb_ep_set_ifnum(udev, pid, ep, i);
            usb_ep_set_halted(udev, pid, ep, 0);
#ifdef HAVE_STREAMS
            if (type == LIBUSB_TRANSFER_TYPE_BULK &&
                    libusb_get_ss_endpoint_companion_descriptor(ctx, endp,
                        &endp_ss_comp) == LIBUSB_SUCCESS) {
                usb_ep_set_max_streams(udev, pid, ep,
                                       endp_ss_comp->bmAttributes);
                libusb_free_ss_endpoint_companion_descriptor(endp_ss_comp);
            }
#endif
        }
    }

    libusb_free_config_descriptor(conf);
}

#ifdef CONFIG_WIN32
/*
 * libusb opens more than it can drive on Windows: a device still on its
 * Windows driver (HidUsb, usbccgp, BTHUSB) opens through libusb's HID or
 * composite shims, yet its interfaces cannot be claimed, and the guest got
 * a device whose every configuration failed. Only a device whose every
 * interface can be claimed - bound to WinUSB - is given to the guest; the
 * others are refused once in the log and retried by the auto scan.
 */
/*
 * Whether Windows has the device (the whole device node, not an interface
 * of it) on the WinUSB service. A claim is no proof: libusb claims the HID
 * interfaces of a composite device that is still on usbccgp + HidUsb
 * (2026-10-07: the main keyboard and mouse, already handed back to Windows,
 * were reopened by a restarted QEMU and fed both Windows and the guest).
 * SetupAPI is loaded at run time: the probe must not change what QEMU links.
 */
static bool usb_host_win32_on_winusb(uint16_t vid, uint16_t pid)
{
    typedef HDEVINFO (WINAPI *get_class_devs_fn)(const GUID *, PCSTR, HWND,
                                                 DWORD);
    typedef BOOL (WINAPI *enum_fn)(HDEVINFO, DWORD, PSP_DEVINFO_DATA);
    typedef BOOL (WINAPI *instance_id_fn)(HDEVINFO, PSP_DEVINFO_DATA, PSTR,
                                          DWORD, PDWORD);
    typedef BOOL (WINAPI *property_fn)(HDEVINFO, PSP_DEVINFO_DATA, DWORD,
                                       PDWORD, PBYTE, DWORD, PDWORD);
    typedef BOOL (WINAPI *destroy_fn)(HDEVINFO);
    static HMODULE lib;
    get_class_devs_fn get_class_devs;
    enum_fn enum_info;
    instance_id_fn instance_id;
    property_fn property;
    destroy_fn destroy;
    SP_DEVINFO_DATA data = { .cbSize = sizeof(data) };
    char want[32], id[512], service[64];
    HDEVINFO set;
    DWORD i;
    bool found = false;

    if (!lib) {
        lib = LoadLibraryA("setupapi.dll");
    }
    if (!lib) {
        return false;
    }
    get_class_devs = (get_class_devs_fn)GetProcAddress(lib, "SetupDiGetClassDevsA");
    enum_info = (enum_fn)GetProcAddress(lib, "SetupDiEnumDeviceInfo");
    instance_id = (instance_id_fn)GetProcAddress(lib, "SetupDiGetDeviceInstanceIdA");
    property = (property_fn)GetProcAddress(lib, "SetupDiGetDeviceRegistryPropertyA");
    destroy = (destroy_fn)GetProcAddress(lib, "SetupDiDestroyDeviceInfoList");
    if (!get_class_devs || !enum_info || !instance_id || !property || !destroy) {
        return false;
    }
    /* "USB\VID_xxxx&PID_yyyy\": the device node; interfaces add "&MI_nn". */
    snprintf(want, sizeof(want), "USB\\VID_%04X&PID_%04X\\", vid, pid);
    set = get_class_devs(NULL, "USB", NULL, DIGCF_PRESENT | DIGCF_ALLCLASSES);
    if (set == INVALID_HANDLE_VALUE) {
        return false;
    }
    for (i = 0; !found && enum_info(set, i, &data); i++) {
        if (!instance_id(set, &data, id, sizeof(id), NULL) ||
            g_ascii_strncasecmp(id, want, strlen(want)) != 0) {
            continue;
        }
        if (property(set, &data, SPDRP_SERVICE, NULL, (PBYTE)service,
                     sizeof(service), NULL) &&
            g_ascii_strcasecmp(service, "WinUSB") == 0) {
            found = true;
        }
    }
    destroy(set);
    return found;
}

static bool usb_host_win32_owned(USBHostDevice *s)
{
    struct libusb_config_descriptor *conf;
    int rc, i, n, bad = -1;

    if (!usb_host_win32_on_winusb(s->ddesc.idVendor, s->ddesc.idProduct)) {
        if (!s->win32_refused) {
            warn_report("usb-host %04x:%04x: Windows has it on its own driver, "
                        "not WinUSB - not given to the guest",
                        s->ddesc.idVendor, s->ddesc.idProduct);
            s->win32_refused = true;
        }
        return false;
    }
    rc = libusb_get_active_config_descriptor(s->dev, &conf);
    if (rc != 0) {
        return true; /* address state: nothing to claim yet */
    }
    n = conf->bNumInterfaces;
    for (i = 0; i < n && bad < 0; i++) {
        int nr = conf->interface[i].altsetting[0].bInterfaceNumber;
        rc = libusb_claim_interface(s->dh, nr);
        if (rc != 0) {
            bad = nr;
            break;
        }
    }
    while (--i >= 0) {
        libusb_release_interface(s->dh,
                                 conf->interface[i].altsetting[0].bInterfaceNumber);
    }
    libusb_free_config_descriptor(conf);
    if (bad >= 0 && !s->win32_refused) {
        warn_report("usb-host %04x:%04x: interface %d cannot be claimed (%s) - "
                    "not bound to WinUSB, not given to the guest",
                    s->ddesc.idVendor, s->ddesc.idProduct, bad,
                    libusb_error_name(rc));
        s->win32_refused = true;
    }
    if (bad < 0 && s->win32_refused) {
        info_report("usb-host %04x:%04x: on WinUSB now", s->ddesc.idVendor,
                    s->ddesc.idProduct);
        s->win32_refused = false;
    }
    return bad < 0;
}
#endif

static int usb_host_open(USBHostDevice *s, libusb_device *dev, int hostfd)
{
    USBDevice *udev = USB_DEVICE(s);
    int libusb_speed;
    int bus_num = 0;
    int addr = 0;
    int rc;
    Error *local_err = NULL;

    if (s->bh_postld_pending) {
        return -1;
    }
    if (s->dh != NULL) {
        goto fail;
    }

    if (dev) {
        bus_num = libusb_get_bus_number(dev);
        addr = libusb_get_device_address(dev);
        trace_usb_host_open_started(bus_num, addr);

        rc = libusb_open(dev, &s->dh);
        if (rc != 0) {
            goto fail;
        }
    } else {
#if LIBUSB_API_VERSION >= 0x01000107 && !defined(CONFIG_WIN32)
        trace_usb_host_open_hostfd(hostfd);

        rc = libusb_wrap_sys_device(ctx, hostfd, &s->dh);
        if (rc != 0) {
            goto fail;
        }
        s->hostfd  = hostfd;
        dev = libusb_get_device(s->dh);
        bus_num = libusb_get_bus_number(dev);
        addr = libusb_get_device_address(dev);
#else
        g_assert_not_reached();
#endif
    }

    s->dev     = dev;
    s->bus_num = bus_num;
    s->addr    = addr;

#ifdef CONFIG_WIN32
    libusb_get_device_descriptor(dev, &s->ddesc);
    if (!usb_host_win32_owned(s)) {
        libusb_close(s->dh);
        s->dh = NULL;
        s->dev = NULL;
        return -1;
    }
#endif
    usb_host_detach_kernel(s);

    libusb_get_device_descriptor(dev, &s->ddesc);
    usb_host_get_port(s->dev, s->port, sizeof(s->port));

    usb_ep_init(udev);
    usb_host_ep_update(s);
    usb_host_kbd_learn(s);
    s->switch_armed = false;

    libusb_speed = libusb_get_device_speed(dev);
#if LIBUSB_API_VERSION >= 0x01000107 && defined(CONFIG_LINUX) && \
        defined(USBDEVFS_GET_SPEED)
    if (hostfd && libusb_speed == 0) {
        /*
         * Workaround libusb bug: libusb_get_device_speed() does not
         * work for libusb_wrap_sys_device() devices in v1.0.23.
         *
         * Speeds are defined in linux/usb/ch9.h, file not included
         * due to name conflicts.
         */
        rc = ioctl(hostfd, USBDEVFS_GET_SPEED, NULL);
        switch (rc) {
        case 1: /* low */
            libusb_speed = LIBUSB_SPEED_LOW;
            break;
        case 2: /* full */
            libusb_speed = LIBUSB_SPEED_FULL;
            break;
        case 3: /* high */
        case 4: /* wireless */
            libusb_speed = LIBUSB_SPEED_HIGH;
            break;
        case 5: /* super */
            libusb_speed = LIBUSB_SPEED_SUPER;
            break;
        case 6: /* super plus */
#ifdef HAVE_SUPER_PLUS
            libusb_speed = LIBUSB_SPEED_SUPER_PLUS;
#else
            libusb_speed = LIBUSB_SPEED_SUPER;
#endif
            break;
        }
    }
#endif
    udev->speed = speed_map[libusb_speed];
    usb_host_speed_compat(s);

    if (s->ddesc.iProduct) {
        libusb_get_string_descriptor_ascii(s->dh, s->ddesc.iProduct,
                                           (unsigned char *)udev->product_desc,
                                           sizeof(udev->product_desc));
    } else {
        snprintf(udev->product_desc, sizeof(udev->product_desc),
                 "host:%d.%d", bus_num, addr);
    }

    usb_device_attach(udev, &local_err);
    if (local_err) {
        error_report_err(local_err);
        goto fail;
    }

    trace_usb_host_open_success(bus_num, addr);
    return 0;

fail:
    trace_usb_host_open_failure(bus_num, addr);
    if (s->dh != NULL) {
        usb_host_release_interfaces(s);
        libusb_reset_device(s->dh);
        usb_host_attach_kernel(s);
        libusb_close(s->dh);
        s->dh = NULL;
        s->dev = NULL;
    }
    return -1;
}

/*
 * TEMP lab (QEMU_USB_HOST_SLOW_ASYNC=on, 2026-10-14): a wedged Bluetooth adapter (AX201) froze the whole work VM - the
 * guest re-reset it every ~5 s and each libusb_reset_device() through WinUSB held the main loop ~4.5 s under the BQL
 * (QMP query-status 4.5 s, vCPU BQL waits 5-20 s per 5 s; QMP device_del of the adapter brought the loop back at
 * once). With the switch on, a guest reset and an endpoint clear-halt are handed to this device's thread; until the
 * bottom half has taken the result, every packet for the device is answered NAK and the controller is woken
 * afterwards, so the guest sees a device that is busy for a while and the rest of the VM keeps running.
 */
enum { USB_HOST_SLOW_NONE, USB_HOST_SLOW_RESET, USB_HOST_SLOW_CLEAR_HALT };

static void usb_host_handle_control(USBDevice *udev, USBPacket *p, int request, int value, int index, int length,
                                    uint8_t *data);

static bool usb_host_slow_async_on(void)
{
    static int on = -1;

    if (on < 0) {
        const char *e = getenv("QEMU_USB_HOST_SLOW_ASYNC");
        on = e && (!strcmp(e, "on") || !strcmp(e, "1"));
        info_report("usb-host: guest resets and clear-halts %s (QEMU_USB_HOST_SLOW_ASYNC)",
                    on ? "on a per-device thread" : "in the main loop");
    }
    return on;
}

static void *usb_host_slow_thread(void *opaque)
{
    USBHostDevice *s = opaque;

    qemu_mutex_lock(&s->slow_lock);
    for (;;) {
        int op, arg, rc = 0;

        while (!s->slow_quit && !s->slow_running) {
            qemu_cond_wait(&s->slow_cond, &s->slow_lock);
        }
        if (s->slow_quit) {
            break;
        }
        op = s->slow_op;
        arg = s->slow_arg;
        qemu_mutex_unlock(&s->slow_lock);
        if (op == USB_HOST_SLOW_RESET) {
            rc = libusb_reset_device(s->dh);
        } else if (op == USB_HOST_SLOW_CLEAR_HALT) {
            rc = libusb_clear_halt(s->dh, arg);
        }
        qemu_mutex_lock(&s->slow_lock);
        s->slow_rc = rc;
        s->slow_running = false;
        qemu_cond_broadcast(&s->slow_cond);
        qemu_bh_schedule(s->slow_bh);
    }
    qemu_mutex_unlock(&s->slow_lock);
    return NULL;
}

/* Main loop: wake every endpoint the guest was NAKed on while the operation ran. */
static void usb_host_slow_wake(USBHostDevice *s)
{
    USBDevice *udev = USB_DEVICE(s);
    int i;

    usb_wakeup(&udev->ep_ctl, 0);
    for (i = 1; i < USB_MAX_ENDPOINTS; i++) {
        usb_wakeup(usb_ep_get(udev, USB_TOKEN_IN, i), 0);
        usb_wakeup(usb_ep_get(udev, USB_TOKEN_OUT, i), 0);
    }
}

static void usb_host_slow_bh(void *opaque)
{
    USBHostDevice *s = opaque;
    USBDevice *udev = USB_DEVICE(s);
    USBPacket *p = s->slow_pkt;
    int op, arg, rc;
    int64_t ms;

    qemu_mutex_lock(&s->slow_lock);
    if (s->slow_running || !s->slow_busy) {
        qemu_mutex_unlock(&s->slow_lock);
        return;
    }
    op = s->slow_op;
    arg = s->slow_arg;
    rc = s->slow_rc;
    s->slow_op = USB_HOST_SLOW_NONE;
    qemu_mutex_unlock(&s->slow_lock);
    s->slow_busy = false;
    s->slow_pkt = NULL;
    ms = (qemu_clock_get_ns(QEMU_CLOCK_REALTIME) - s->slow_t0) / SCALE_MS;
    if (ms >= 100 || rc != 0) {
        info_report("usb-host %d.%d: %s took %" PRId64 " ms off the main loop (rc %d)", s->bus_num, s->addr,
                    op == USB_HOST_SLOW_RESET ? "reset" : "clear-halt", ms, rc);
    }
    if (op == USB_HOST_SLOW_CLEAR_HALT && p) {
        int pid = (arg & USB_DIR_IN) ? USB_TOKEN_IN : USB_TOKEN_OUT;

        usb_ep_set_halted(udev, pid, arg & 0x0f, 0);
        p->status = USB_RET_SUCCESS;
        usb_generic_async_ctrl_complete(udev, p);
    }
    if (op == USB_HOST_SLOW_RESET && rc != 0) {
        usb_host_nodev(s);
        return;
    }
    usb_host_slow_wake(s);
    if (s->slow_ctl.p) {
        USBPacket *cp = s->slow_ctl.p;

        s->slow_ctl.p = NULL;
        usb_host_handle_control(udev, cp, s->slow_ctl.request, s->slow_ctl.value, s->slow_ctl.index,
                                s->slow_ctl.length, s->slow_ctl.data);
        if (cp->status != USB_RET_ASYNC) {
            usb_generic_async_ctrl_complete(udev, cp);
        }
    }
}

/* Main loop: start `op` on the device's thread. False when the switch is off (the caller does it inline). */
static bool usb_host_slow_start(USBHostDevice *s, int op, int arg, USBPacket *p)
{
    if (!usb_host_slow_async_on() || s->dh == NULL) {
        return false;
    }
    if (!s->slow_init) {
        qemu_mutex_init(&s->slow_lock);
        qemu_cond_init(&s->slow_cond);
        s->slow_bh = qemu_bh_new_guarded(usb_host_slow_bh, s, &DEVICE(s)->mem_reentrancy_guard);
        s->slow_quit = false;
        qemu_thread_create(&s->slow_thread, "usb-host-slow", usb_host_slow_thread, s, QEMU_THREAD_JOINABLE);
        s->slow_init = true;
    }
    s->slow_busy = true;
    s->slow_pkt = p;
    s->slow_t0 = qemu_clock_get_ns(QEMU_CLOCK_REALTIME);
    qemu_mutex_lock(&s->slow_lock);
    s->slow_op = op;
    s->slow_arg = arg;
    s->slow_running = true;
    qemu_cond_broadcast(&s->slow_cond);
    qemu_mutex_unlock(&s->slow_lock);
    return true;
}

/*
 * Before the handle goes away: wait for an operation still inside libusb (the device is leaving anyway) and drop
 * whatever the bottom half would have done with it. A packet it held is failed.
 */
static void usb_host_slow_drain(USBHostDevice *s)
{
    if (!s->slow_init) {
        return;
    }
    qemu_mutex_lock(&s->slow_lock);
    while (s->slow_running) {
        qemu_cond_wait(&s->slow_cond, &s->slow_lock);
    }
    s->slow_op = USB_HOST_SLOW_NONE;
    qemu_mutex_unlock(&s->slow_lock);
    qemu_bh_cancel(s->slow_bh);
    if (s->slow_busy && s->slow_pkt) {
        s->slow_pkt->status = USB_RET_NODEV;
        usb_generic_async_ctrl_complete(USB_DEVICE(s), s->slow_pkt);
    }
    if (s->slow_ctl.p) {
        s->slow_ctl.p->status = USB_RET_NODEV;
        usb_generic_async_ctrl_complete(USB_DEVICE(s), s->slow_ctl.p);
        s->slow_ctl.p = NULL;
    }
    s->slow_pkt = NULL;
    s->slow_busy = false;
}

static void usb_host_abort_xfers(USBHostDevice *s)
{
    USBHostRequest *r, *rtmp;
    int limit = 100;

    QTAILQ_FOREACH_SAFE(r, &s->requests, next, rtmp) {
        usb_host_req_abort(r);
    }

    while (QTAILQ_FIRST(&s->requests) != NULL) {
        struct timeval tv;
        memset(&tv, 0, sizeof(tv));
        tv.tv_usec = 2500;
        libusb_handle_events_timeout(ctx, &tv);
#ifdef CONFIG_WIN32
        usb_host_done_run();
#endif
        if (--limit == 0) {
            /*
             * Don't wait forever for libusb calling the complete
             * callback (which will unlink and free the request).
             *
             * Leaking memory here, to make sure libusb will not
             * access memory which we have released already.
             */
            QTAILQ_FOREACH_SAFE(r, &s->requests, next, rtmp) {
                QTAILQ_REMOVE(&s->requests, r, next);
            }
            return;
        }
    }
}

static int usb_host_close(USBHostDevice *s)
{
    USBDevice *udev = USB_DEVICE(s);

    if (s->dh == NULL) {
        return -1;
    }

    trace_usb_host_close(s->bus_num, s->addr);

    usb_host_slow_drain(s);   /* TEMP lab (QEMU_USB_HOST_SLOW_ASYNC) */
    usb_host_ahead_stop_all(s, USB_RET_NODEV);
    usb_host_abort_xfers(s);
    usb_host_ahead_free_all(s);
    usb_host_iso_free_all(s);

    if (udev->attached) {
        usb_device_detach(udev);
    }

    usb_host_release_interfaces(s);
    libusb_reset_device(s->dh);
    usb_host_attach_kernel(s);
    libusb_close(s->dh);
    s->dh = NULL;
    s->dev = NULL;

    if (s->hostfd != -1) {
        close(s->hostfd);
        s->hostfd = -1;
    }

    usb_host_auto_check(NULL);
    return 0;
}

static void usb_host_nodev_bh(void *opaque)
{
    USBHostDevice *s = opaque;
    usb_host_close(s);
}

static void usb_host_nodev(USBHostDevice *s)
{
    if (!s->bh_nodev) {
        s->bh_nodev = qemu_bh_new_guarded(usb_host_nodev_bh, s,
                                          &DEVICE(s)->mem_reentrancy_guard);
    }
    qemu_bh_schedule(s->bh_nodev);
}

static void usb_host_exit_notifier(struct Notifier *n, void *data)
{
    USBHostDevice *s = container_of(n, USBHostDevice, exit);

    if (s->dh) {
        if (s->slow_init) {   /* TEMP lab (QEMU_USB_HOST_SLOW_ASYNC): no reset beside the thread's */
            qemu_mutex_lock(&s->slow_lock);
            while (s->slow_running) {
                qemu_cond_wait(&s->slow_cond, &s->slow_lock);
            }
            qemu_mutex_unlock(&s->slow_lock);
        }
        usb_host_ahead_stop_all(s, USB_RET_NODEV);
        usb_host_abort_xfers(s);
        usb_host_release_interfaces(s);
        libusb_reset_device(s->dh);
        usb_host_attach_kernel(s);
        libusb_close(s->dh);
    }
}

static libusb_device *usb_host_find_ref(int bus, int addr)
{
    libusb_device **devs = NULL;
    libusb_device *ret = NULL;
    int i, n;

    n = libusb_get_device_list(ctx, &devs);
    for (i = 0; i < n; i++) {
        if (libusb_get_bus_number(devs[i]) == bus &&
            libusb_get_device_address(devs[i]) == addr) {
            ret = libusb_ref_device(devs[i]);
            break;
        }
    }
    libusb_free_device_list(devs, 1);
    return ret;
}

static void usb_host_realize(USBDevice *udev, Error **errp)
{
    USBHostDevice *s = USB_HOST_DEVICE(udev);
    libusb_device *ldev;
    int rc;

    if (usb_host_init() != 0) {
        error_setg(errp, "failed to init libusb");
        return;
    }
    if (s->match.vendor_id > 0xffff) {
        error_setg(errp, "vendorid out of range");
        return;
    }
    if (s->match.product_id > 0xffff) {
        error_setg(errp, "productid out of range");
        return;
    }
    if (s->match.addr > 127) {
        error_setg(errp, "hostaddr out of range");
        return;
    }

    loglevel = s->loglevel;
    udev->flags |= (1 << USB_DEV_FLAG_IS_HOST);
    udev->auto_attach = 0;
    QTAILQ_INIT(&s->requests);
    QTAILQ_INIT(&s->isorings);
    s->hostfd = -1;

#if LIBUSB_API_VERSION >= 0x01000107 && !defined(CONFIG_WIN32)
    if (s->hostdevice) {
        int fd;
        s->needs_autoscan = false;
        fd = qemu_open(s->hostdevice, O_RDWR, errp);
        if (fd < 0) {
            return;
        }
        rc = usb_host_open(s, NULL, fd);
        if (rc < 0) {
            error_setg(errp, "failed to open host usb device %s", s->hostdevice);
            return;
        }
    } else
#endif
    if (s->match.addr && s->match.bus_num &&
        !s->match.vendor_id &&
        !s->match.product_id &&
        !s->match.port) {
        s->needs_autoscan = false;
        ldev = usb_host_find_ref(s->match.bus_num,
                                 s->match.addr);
        if (!ldev) {
            error_setg(errp, "failed to find host usb device %d:%d",
                       s->match.bus_num, s->match.addr);
            return;
        }
        rc = usb_host_open(s, ldev, 0);
        libusb_unref_device(ldev);
        if (rc < 0) {
            error_setg(errp, "failed to open host usb device %d:%d",
                       s->match.bus_num, s->match.addr);
            return;
        }
    } else {
        s->needs_autoscan = true;
        QTAILQ_INSERT_TAIL(&hostdevs, s, next);
        usb_host_auto_check(NULL);
    }

    s->exit.notify = usb_host_exit_notifier;
    qemu_add_exit_notifier(&s->exit);
}

/*
 * grab: whether this device belongs to the guest. False closes it and
 * keeps the auto scan off it, so the host driver gets it back; true lets
 * the next scan open it again. Settable at run time (QMP qom-set).
 */
static bool usb_host_get_grab(Object *obj, Error **errp)
{
    return USB_HOST_DEVICE(obj)->grab;
}

static void usb_host_set_grab(Object *obj, bool value, Error **errp)
{
    USBHostDevice *s = USB_HOST_DEVICE(obj);

    if (s->grab == value) {
        return;
    }
    s->grab = value;
    if (!DEVICE(obj)->realized) {
        return;
    }
    if (value) {
        s->errcount = 0;
        usb_host_auto_check(NULL);
    } else {
        usb_host_close(s);
    }
    info_report("usb-host %04x:%04x: %s", s->match.vendor_id,
                s->match.product_id, value ? "grabbed" : "released to host");
}

static void usb_host_release_all(void)
{
    USBHostDevice *s;

    QTAILQ_FOREACH(s, &hostdevs, next) {
        object_property_set_bool(OBJECT(s), "grab", false, NULL);
    }
}

static void usb_host_instance_init(Object *obj)
{
    USBDevice *udev = USB_DEVICE(obj);
    USBHostDevice *s = USB_HOST_DEVICE(udev);

    device_add_bootindex_property(obj, &s->bootindex,
                                  "bootindex", NULL,
                                  &udev->qdev);
    s->grab = true;
    object_property_add_bool(obj, "grab", usb_host_get_grab,
                             usb_host_set_grab);
}

static void usb_host_unrealize(USBDevice *udev)
{
    USBHostDevice *s = USB_HOST_DEVICE(udev);

    qemu_remove_exit_notifier(&s->exit);
    if (s->needs_autoscan) {
        QTAILQ_REMOVE(&hostdevs, s, next);
    }
    usb_host_close(s);
    if (s->slow_init) {   /* TEMP lab (QEMU_USB_HOST_SLOW_ASYNC): close drained it; stop the thread */
        qemu_mutex_lock(&s->slow_lock);
        s->slow_quit = true;
        qemu_cond_broadcast(&s->slow_cond);
        qemu_mutex_unlock(&s->slow_lock);
        qemu_thread_join(&s->slow_thread);
        qemu_bh_delete(s->slow_bh);
        qemu_cond_destroy(&s->slow_cond);
        qemu_mutex_destroy(&s->slow_lock);
        s->slow_init = false;
    }
}

static void usb_host_cancel_packet(USBDevice *udev, USBPacket *p)
{
    USBHostDevice *s = USB_HOST_DEVICE(udev);
    USBHostRequest *r;

    if (p->combined) {
        usb_combined_packet_cancel(udev, p);
        return;
    }

    trace_usb_host_req_canceled(s->bus_num, s->addr, p);

    if (p == s->slow_pkt) {   /* TEMP lab (QEMU_USB_HOST_SLOW_ASYNC): the operation runs on, nothing to complete */
        s->slow_pkt = NULL;
        return;
    }
    if (p == s->slow_ctl.p) {
        s->slow_ctl.p = NULL;
        return;
    }
    if (s->ahead[p->ep->nr & 15] && s->ahead[p->ep->nr & 15]->parked == p) {
        s->ahead[p->ep->nr & 15]->parked = NULL;
        return;
    }
    r = usb_host_req_find(s, p);
    if (r && r->p) {
        r->p = NULL; /* mark as dead */
        libusb_cancel_transfer(r->xfer);
    }
}

static void usb_host_detach_kernel(USBHostDevice *s)
{
    struct libusb_config_descriptor *conf;
    int rc, i;

#ifdef CONFIG_WIN32
    /* Windows has no driver detach: the device's driver binding decides. */
    return;
#endif

    rc = libusb_get_active_config_descriptor(s->dev, &conf);
    if (rc != 0) {
        return;
    }
    for (i = 0; i < USB_MAX_INTERFACES; i++) {
        rc = libusb_kernel_driver_active(s->dh, i);
        usb_host_libusb_error("libusb_kernel_driver_active", rc);
        if (rc != 1) {
            if (rc == 0) {
                s->ifs[i].detached = true;
            }
            continue;
        }
        trace_usb_host_detach_kernel(s->bus_num, s->addr, i);
        rc = libusb_detach_kernel_driver(s->dh, i);
        usb_host_libusb_error("libusb_detach_kernel_driver", rc);
        s->ifs[i].detached = true;
    }
    libusb_free_config_descriptor(conf);
}

static void usb_host_attach_kernel(USBHostDevice *s)
{
    struct libusb_config_descriptor *conf;
    int rc, i;

#ifdef CONFIG_WIN32
    return;
#endif

    rc = libusb_get_active_config_descriptor(s->dev, &conf);
    if (rc != 0) {
        return;
    }
    for (i = 0; i < USB_MAX_INTERFACES; i++) {
        if (!s->ifs[i].detached) {
            continue;
        }
        trace_usb_host_attach_kernel(s->bus_num, s->addr, i);
        libusb_attach_kernel_driver(s->dh, i);
        s->ifs[i].detached = false;
    }
    libusb_free_config_descriptor(conf);
}

static int usb_host_claim_interfaces(USBHostDevice *s, int configuration)
{
    USBDevice *udev = USB_DEVICE(s);
    struct libusb_config_descriptor *conf;
    int rc, i, claimed;

    for (i = 0; i < USB_MAX_INTERFACES; i++) {
        udev->altsetting[i] = 0;
    }
    udev->ninterfaces   = 0;
    udev->configuration = 0;

    usb_host_detach_kernel(s);

    rc = libusb_get_active_config_descriptor(s->dev, &conf);
    if (rc != 0) {
        if (rc == LIBUSB_ERROR_NOT_FOUND) {
            /* address state - ignore */
            return USB_RET_SUCCESS;
        }
        return USB_RET_STALL;
    }

    claimed = 0;
    for (i = 0; i < USB_MAX_INTERFACES; i++) {
        trace_usb_host_claim_interface(s->bus_num, s->addr, configuration, i);
        rc = libusb_claim_interface(s->dh, i);
        if (rc == 0) {
            s->ifs[i].claimed = true;
            if (++claimed == conf->bNumInterfaces) {
                break;
            }
        }
    }
    if (claimed != conf->bNumInterfaces) {
        return USB_RET_STALL;
    }

    udev->ninterfaces   = conf->bNumInterfaces;
    udev->configuration = configuration;

    libusb_free_config_descriptor(conf);
    return USB_RET_SUCCESS;
}

static void usb_host_release_interfaces(USBHostDevice *s)
{
    int i, rc;

    for (i = 0; i < USB_MAX_INTERFACES; i++) {
        if (!s->ifs[i].claimed) {
            continue;
        }
        trace_usb_host_release_interface(s->bus_num, s->addr, i);
        rc = libusb_release_interface(s->dh, i);
        usb_host_libusb_error("libusb_release_interface", rc);
        s->ifs[i].claimed = false;
    }
}

static void usb_host_set_address(USBHostDevice *s, int addr)
{
    USBDevice *udev = USB_DEVICE(s);

    trace_usb_host_set_address(s->bus_num, s->addr, addr);
    udev->addr = addr;
}

static void usb_host_set_config(USBHostDevice *s, int config, USBPacket *p)
{
    int rc = 0;

    trace_usb_host_set_config(s->bus_num, s->addr, config);

    usb_host_ahead_stop_all(s, USB_RET_IOERROR);
    usb_host_release_interfaces(s);
    if (s->ddesc.bNumConfigurations != 1) {
        rc = libusb_set_configuration(s->dh, config);
        if (rc != 0) {
            usb_host_libusb_error("libusb_set_configuration", rc);
            p->status = USB_RET_STALL;
            if (rc == LIBUSB_ERROR_NO_DEVICE) {
                usb_host_nodev(s);
            }
            return;
        }
    }
    p->status = usb_host_claim_interfaces(s, config);
    if (p->status != USB_RET_SUCCESS) {
        return;
    }
    usb_host_ep_update(s);
}

static void usb_host_set_interface(USBHostDevice *s, int iface, int alt,
                                   USBPacket *p)
{
    USBDevice *udev = USB_DEVICE(s);
    int rc;

    trace_usb_host_set_interface(s->bus_num, s->addr, iface, alt);

    usb_host_iso_free_all(s);
    usb_host_ahead_stop_all(s, USB_RET_IOERROR);

    if (iface >= USB_MAX_INTERFACES) {
        p->status = USB_RET_STALL;
        return;
    }

    rc = libusb_set_interface_alt_setting(s->dh, iface, alt);
    if (rc != 0) {
        usb_host_libusb_error("libusb_set_interface_alt_setting", rc);
        p->status = USB_RET_STALL;
        if (rc == LIBUSB_ERROR_NO_DEVICE) {
            usb_host_nodev(s);
        }
        return;
    }

    udev->altsetting[iface] = alt;
    usb_host_ep_update(s);
}

static void usb_host_handle_control(USBDevice *udev, USBPacket *p,
                                    int request, int value, int index,
                                    int length, uint8_t *data)
{
    USBHostDevice *s = USB_HOST_DEVICE(udev);
    USBHostRequest *r;
    int rc;

    trace_usb_host_req_control(s->bus_num, s->addr, p, request, value, index);

    if (s->dh == NULL) {
        p->status = USB_RET_NODEV;
        trace_usb_host_req_emulated(s->bus_num, s->addr, p, p->status);
        return;
    }
    if (s->slow_busy && !s->slow_ctl.p) {
        /* TEMP lab (QEMU_USB_HOST_SLOW_ASYNC): held until the reset or clear-halt in flight is done. */
        s->slow_ctl.p = p;
        s->slow_ctl.request = request;
        s->slow_ctl.value = value;
        s->slow_ctl.index = index;
        s->slow_ctl.length = length;
        s->slow_ctl.data = data;
        p->status = USB_RET_ASYNC;
        return;
    }

    switch (request) {
    case DeviceOutRequest | USB_REQ_SET_ADDRESS:
        usb_host_set_address(s, value);
        trace_usb_host_req_emulated(s->bus_num, s->addr, p, p->status);
        return;

    case DeviceOutRequest | USB_REQ_SET_CONFIGURATION:
        usb_host_set_config(s, value & 0xff, p);
        trace_usb_host_req_emulated(s->bus_num, s->addr, p, p->status);
        return;

    case InterfaceOutRequest | USB_REQ_SET_INTERFACE:
        usb_host_set_interface(s, index, value, p);
        trace_usb_host_req_emulated(s->bus_num, s->addr, p, p->status);
        return;

    case EndpointOutRequest | USB_REQ_CLEAR_FEATURE:
        if (value == 0) { /* clear halt */
            int pid = (index & USB_DIR_IN) ? USB_TOKEN_IN : USB_TOKEN_OUT;
            if (usb_host_slow_start(s, USB_HOST_SLOW_CLEAR_HALT, index, p)) {
                p->status = USB_RET_ASYNC;   /* TEMP lab (QEMU_USB_HOST_SLOW_ASYNC) */
                return;
            }
            libusb_clear_halt(s->dh, index);
            usb_ep_set_halted(udev, pid, index & 0x0f, 0);
            trace_usb_host_req_emulated(s->bus_num, s->addr, p, p->status);
            return;
        }
    }

    r = usb_host_req_alloc(s, p, (request >> 8) & USB_DIR_IN, length + 8);
    r->cbuf = data;
    r->clen = length;
    memcpy(r->buffer, udev->setup_buf, 8);
    if (!r->in) {
        memcpy(r->buffer + 8, r->cbuf, r->clen);
    }

    /* Fix up USB-3 ep0 maxpacket size to allow superspeed connected devices
     * to work redirected to a not superspeed capable hcd */
    if ((udev->speedmask & USB_SPEED_MASK_SUPER) &&
        !(udev->port->speedmask & USB_SPEED_MASK_SUPER) &&
        request == 0x8006 && value == 0x100 && index == 0) {
        r->usb3ep0quirk = true;
    }

    libusb_fill_control_transfer(r->xfer, s->dh, r->buffer,
                                 USB_HOST_CB(usb_host_req_complete_ctrl), r,
                                 CONTROL_TIMEOUT);
    rc = libusb_submit_transfer(r->xfer);
    if (rc != 0) {
        p->status = USB_RET_NODEV;
        trace_usb_host_req_complete(s->bus_num, s->addr, p,
                                    p->status, p->actual_length);
        if (rc == LIBUSB_ERROR_NO_DEVICE) {
            usb_host_nodev(s);
        }
        return;
    }

    p->status = USB_RET_ASYNC;
}

static void usb_host_handle_data(USBDevice *udev, USBPacket *p)
{
    USBHostDevice *s = USB_HOST_DEVICE(udev);
    USBHostRequest *r;
    size_t size;
    int ep, rc;

    if (usb_host_use_combining(p->ep) && p->state == USB_PACKET_SETUP) {
        p->status = USB_RET_ADD_TO_QUEUE;
        return;
    }

    trace_usb_host_req_data(s->bus_num, s->addr, p,
                            p->pid == USB_TOKEN_IN,
                            p->ep->nr, p->iov.size);

    if (s->dh == NULL) {
        p->status = USB_RET_NODEV;
        trace_usb_host_req_emulated(s->bus_num, s->addr, p, p->status);
        return;
    }
    if (p->ep->halted) {
        p->status = USB_RET_STALL;
        trace_usb_host_req_emulated(s->bus_num, s->addr, p, p->status);
        return;
    }
    if (s->slow_busy && !usb_host_use_combining(p->ep) &&
        usb_ep_get_type(udev, p->pid, p->ep->nr) != USB_ENDPOINT_XFER_ISOC) {
        /* TEMP lab (QEMU_USB_HOST_SLOW_ASYNC): busy with a reset or clear-halt; the bottom half wakes the ep. */
        p->status = USB_RET_NAK;
        return;
    }

    switch (usb_ep_get_type(udev, p->pid, p->ep->nr)) {
    case USB_ENDPOINT_XFER_BULK:
        size = usb_packet_size(p);
        r = usb_host_req_alloc(s, p, p->pid == USB_TOKEN_IN, size);
        if (!r->in) {
            usb_packet_copy(p, r->buffer, size);
        }
        ep = p->ep->nr | (r->in ? USB_DIR_IN : 0);
        if (p->stream) {
#ifdef HAVE_STREAMS
            libusb_fill_bulk_stream_transfer(r->xfer, s->dh, ep, p->stream,
                                             r->buffer, size,
                                             USB_HOST_CB(usb_host_req_complete_data), r,
                                             BULK_TIMEOUT);
#else
            usb_host_req_free(r);
            p->status = USB_RET_STALL;
            return;
#endif
        } else {
            libusb_fill_bulk_transfer(r->xfer, s->dh, ep,
                                      r->buffer, size,
                                      USB_HOST_CB(usb_host_req_complete_data), r,
                                      BULK_TIMEOUT);
        }
        break;
    case USB_ENDPOINT_XFER_INT:
        if (p->pid == USB_TOKEN_IN && (s->hid_in_eps & (1 << p->ep->nr)) &&
            usb_host_readahead_on()) {
            USBHostAhead *a = s->ahead[p->ep->nr];
            if (a && !a->active && a->status) {
                p->status = a->status;           /* a stall the ring saw */
                a->status = 0;
                return;
            }
            a = usb_host_ahead_start(s, p->ep->nr);
            if (a) {
                if (a->count) {
                    usb_host_ahead_pop(a, p);
                    ist.taken++;
                } else {
                    a->parked = p;
                    p->status = USB_RET_ASYNC;
                    ist.parked++;
                    if (usb_lab_trace_on()) {
                        usb_lab_trace("Q %04x ep%d", s->match.product_id, p->ep->nr);
                    }
                }
                return;
            }
        }
        r = usb_host_req_alloc(s, p, p->pid == USB_TOKEN_IN, p->iov.size);
        if (!r->in) {
            usb_packet_copy(p, r->buffer, p->iov.size);
        }
        if (r->in && usb_host_stats_on()) {
            int64_t now = get_clock();
            int64_t last = s->int_done[p->ep->nr & 15];
            r->t_submit = now;
            if (last) {
                ist.rt_n++;
                usb_host_stats_add(&ist.rt_sum, &ist.rt_max, now - last);
                s->int_done[p->ep->nr & 15] = 0;
            }
        }
        ep = p->ep->nr | (r->in ? USB_DIR_IN : 0);
        libusb_fill_interrupt_transfer(r->xfer, s->dh, ep,
                                       r->buffer, p->iov.size,
                                       USB_HOST_CB(usb_host_req_complete_data), r,
                                       INTR_TIMEOUT);
        break;
    case USB_ENDPOINT_XFER_ISOC:
        if (p->pid == USB_TOKEN_IN) {
            usb_host_iso_data_in(s, p);
        } else {
            usb_host_iso_data_out(s, p);
        }
        trace_usb_host_req_complete(s->bus_num, s->addr, p,
                                    p->status, p->actual_length);
        return;
    default:
        p->status = USB_RET_STALL;
        trace_usb_host_req_complete(s->bus_num, s->addr, p,
                                    p->status, p->actual_length);
        return;
    }

    rc = libusb_submit_transfer(r->xfer);
    if (rc != 0) {
        p->status = USB_RET_NODEV;
        trace_usb_host_req_complete(s->bus_num, s->addr, p,
                                    p->status, p->actual_length);
        if (rc == LIBUSB_ERROR_NO_DEVICE) {
            usb_host_nodev(s);
        }
        return;
    }

    p->status = USB_RET_ASYNC;
}

static void usb_host_flush_ep_queue(USBDevice *dev, USBEndpoint *ep)
{
    if (usb_host_use_combining(ep)) {
        usb_ep_combine_input_packets(ep);
    }
}

static void usb_host_handle_reset(USBDevice *udev)
{
    USBHostDevice *s = USB_HOST_DEVICE(udev);
    int rc;

    if (!s->allow_one_guest_reset && !s->allow_all_guest_resets) {
        return;
    }
    if (!s->allow_all_guest_resets && udev->addr == 0) {
        return;
    }

    trace_usb_host_reset(s->bus_num, s->addr);
    /* TEMP lab (Opus 2026-10-19): guest resets of passed-through devices (a Bluetooth controller reset drops links). */
    fprintf(stderr, "usb-host-reset %04x:%04x addr %d epoch_ms=%" PRId64 "\n", s->ddesc.idVendor,
            s->ddesc.idProduct, udev->addr, g_get_real_time() / 1000);

    usb_host_ahead_stop_all(s, USB_RET_NODEV);
    if (s->slow_busy) {
        return;   /* TEMP lab (QEMU_USB_HOST_SLOW_ASYNC): the reset or clear-halt in flight covers this one */
    }
    if (usb_host_slow_start(s, USB_HOST_SLOW_RESET, 0, NULL)) {
        return;
    }
    rc = libusb_reset_device(s->dh);
    if (rc != 0) {
        usb_host_nodev(s);
    }
}

static int usb_host_alloc_streams(USBDevice *udev, USBEndpoint **eps,
                                  int nr_eps, int streams)
{
#ifdef HAVE_STREAMS
    USBHostDevice *s = USB_HOST_DEVICE(udev);
    unsigned char endpoints[30];
    int i, rc;

    for (i = 0; i < nr_eps; i++) {
        endpoints[i] = eps[i]->nr;
        if (eps[i]->pid == USB_TOKEN_IN) {
            endpoints[i] |= 0x80;
        }
    }
    rc = libusb_alloc_streams(s->dh, streams, endpoints, nr_eps);
    if (rc < 0) {
        usb_host_libusb_error("libusb_alloc_streams", rc);
    } else if (rc != streams) {
        error_report("libusb_alloc_streams: got less streams "
                     "then requested %d < %d", rc, streams);
    }

    return (rc == streams) ? 0 : -1;
#else
    error_report("libusb_alloc_streams: error not implemented");
    return -1;
#endif
}

static void usb_host_free_streams(USBDevice *udev, USBEndpoint **eps,
                                  int nr_eps)
{
#ifdef HAVE_STREAMS
    USBHostDevice *s = USB_HOST_DEVICE(udev);
    unsigned char endpoints[30];
    int i;

    for (i = 0; i < nr_eps; i++) {
        endpoints[i] = eps[i]->nr;
        if (eps[i]->pid == USB_TOKEN_IN) {
            endpoints[i] |= 0x80;
        }
    }
    libusb_free_streams(s->dh, endpoints, nr_eps);
#endif
}

/*
 * This is *NOT* about restoring state.  We have absolutely no idea
 * what state the host device is in at the moment and whenever it is
 * still present in the first place.  Attempting to continue where we
 * left off is impossible.
 *
 * What we are going to do here is emulate a surprise removal of
 * the usb device passed through, then kick host scan so the device
 * will get re-attached (and re-initialized by the guest) in case it
 * is still present.
 *
 * As the device removal will change the state of other devices (usb
 * host controller, most likely interrupt controller too) we have to
 * wait with it until *all* vmstate is loaded.  Thus post_load just
 * kicks a bottom half which then does the actual work.
 */
static void usb_host_post_load_bh(void *opaque)
{
    USBHostDevice *dev = opaque;
    USBDevice *udev = USB_DEVICE(dev);

    if (dev->dh != NULL) {
        usb_host_close(dev);
    }
    if (udev->attached) {
        usb_device_detach(udev);
    }
    dev->bh_postld_pending = false;
    usb_host_auto_check(NULL);
}

static int usb_host_post_load(void *opaque, int version_id)
{
    USBHostDevice *dev = opaque;

    if (!dev->bh_postld) {
        dev->bh_postld = qemu_bh_new_guarded(usb_host_post_load_bh, dev,
                                             &DEVICE(dev)->mem_reentrancy_guard);
    }
    qemu_bh_schedule(dev->bh_postld);
    dev->bh_postld_pending = true;
    return 0;
}

static const VMStateDescription vmstate_usb_host = {
    .name = "usb-host",
    .version_id = 1,
    .minimum_version_id = 1,
    .post_load = usb_host_post_load,
    .fields = (const VMStateField[]) {
        VMSTATE_USB_DEVICE(parent_obj, USBHostDevice),
        VMSTATE_END_OF_LIST()
    }
};

static const Property usb_host_dev_properties[] = {
    DEFINE_PROP_UINT32("hostbus",  USBHostDevice, match.bus_num,    0),
    DEFINE_PROP_UINT32("hostaddr", USBHostDevice, match.addr,       0),
    DEFINE_PROP_STRING("hostport", USBHostDevice, match.port),
    DEFINE_PROP_UINT32("vendorid",  USBHostDevice, match.vendor_id,  0),
    DEFINE_PROP_UINT32("productid", USBHostDevice, match.product_id, 0),
#if LIBUSB_API_VERSION >= 0x01000107
    DEFINE_PROP_STRING("hostdevice", USBHostDevice, hostdevice),
#endif
    DEFINE_PROP_UINT32("isobufs",  USBHostDevice, iso_urb_count,    4),
    DEFINE_PROP_UINT32("isobsize", USBHostDevice, iso_urb_frames,   32),
    DEFINE_PROP_BOOL("guest-reset", USBHostDevice,
                     allow_one_guest_reset, true),
    DEFINE_PROP_BOOL("guest-resets-all", USBHostDevice,
                     allow_all_guest_resets, false),
    DEFINE_PROP_UINT32("loglevel",  USBHostDevice, loglevel,
                       LIBUSB_LOG_LEVEL_WARNING),
    DEFINE_PROP_BIT("pipeline",    USBHostDevice, options,
                    USB_HOST_OPT_PIPELINE, true),
    DEFINE_PROP_BOOL("suppress-remote-wake", USBHostDevice,
                     suppress_remote_wake, true),
};

static void usb_host_class_initfn(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    USBDeviceClass *uc = USB_DEVICE_CLASS(klass);

    uc->realize        = usb_host_realize;
    uc->product_desc   = "USB Host Device";
    uc->cancel_packet  = usb_host_cancel_packet;
    uc->handle_data    = usb_host_handle_data;
    uc->handle_control = usb_host_handle_control;
    uc->handle_reset   = usb_host_handle_reset;
    uc->unrealize      = usb_host_unrealize;
    uc->flush_ep_queue = usb_host_flush_ep_queue;
    uc->alloc_streams  = usb_host_alloc_streams;
    uc->free_streams   = usb_host_free_streams;
    dc->vmsd = &vmstate_usb_host;
    device_class_set_props(dc, usb_host_dev_properties);
    set_bit(DEVICE_CATEGORY_BRIDGE, dc->categories);
}

static const TypeInfo usb_host_dev_info = {
    .name          = TYPE_USB_HOST_DEVICE,
    .parent        = TYPE_USB_DEVICE,
    .instance_size = sizeof(USBHostDevice),
    .class_init    = usb_host_class_initfn,
    .instance_init = usb_host_instance_init,
};
module_obj(TYPE_USB_HOST_DEVICE);
module_kconfig(USB);

static void usb_host_register_types(void)
{
    type_register_static(&usb_host_dev_info);
    monitor_register_hmp("usbhost", true, hmp_info_usbhost);
}

type_init(usb_host_register_types)

/* ------------------------------------------------------------------------ */

static QEMUTimer *usb_auto_timer;
static VMChangeStateEntry *usb_vmstate;

static void usb_host_vm_state(void *unused, bool running, RunState state)
{
    if (running) {
        usb_host_auto_check(unused);
    }
}

/* Main loop: match a device list against the auto-scanned devices; frees it. */
static void usb_host_auto_match(libusb_device **devs, ssize_t n)
{
    struct USBHostDevice *s;
    struct USBAutoFilter *f;
    struct libusb_device_descriptor ddesc;
    ssize_t i;

    if (devs == NULL || n < 0) {
        return;
    }
    if (runstate_is_running()) {
        for (i = 0; i < n; i++) {
            if (libusb_get_device_descriptor(devs[i], &ddesc) != 0) {
                continue;
            }
            if (ddesc.bDeviceClass == LIBUSB_CLASS_HUB) {
                continue;
            }
            QTAILQ_FOREACH(s, &hostdevs, next) {
                f = &s->match;
                if (f->bus_num > 0 &&
                    f->bus_num != libusb_get_bus_number(devs[i])) {
                    continue;
                }
                if (f->addr > 0 &&
                    f->addr != libusb_get_device_address(devs[i])) {
                    continue;
                }
                if (f->port != NULL) {
                    char port[16] = "-";
                    usb_host_get_port(devs[i], port, sizeof(port));
                    if (strcmp(f->port, port) != 0) {
                        continue;
                    }
                }
                if (f->vendor_id > 0 &&
                    f->vendor_id != ddesc.idVendor) {
                    continue;
                }
                if (f->product_id > 0 &&
                    f->product_id != ddesc.idProduct) {
                    continue;
                }

                /* We got a match */
                s->seen++;
                if (!s->grab) {
                    continue;
                }
                if (s->errcount >= 3) {
#ifdef CONFIG_WIN32
                    /* a rebind to WinUSB changes nothing libusb can see */
                    if (++s->errcount < 3 + 5) {
                        continue;
                    }
                    s->errcount = 2;
#else
                    continue;
#endif
                }
                if (s->dh != NULL) {
                    continue;
                }
                if (usb_host_open(s, devs[i], 0) < 0) {
                    s->errcount++;
                    continue;
                }
                break;
            }
        }

        QTAILQ_FOREACH(s, &hostdevs, next) {
            if (s->seen == 0) {
                if (s->dh) {
                    usb_host_close(s);
                }
                s->errcount = 0;
            }
            s->seen = 0;
        }
    }
    libusb_free_device_list(devs, 1);
}

static void usb_host_auto_timer(void *unused);

static void usb_host_auto_rearm(void)
{
    if (!usb_vmstate) {
        usb_vmstate = qemu_add_vm_change_state_handler(usb_host_vm_state, NULL);
    }
    if (!usb_auto_timer) {
        usb_auto_timer = timer_new_ms(QEMU_CLOCK_REALTIME, usb_host_auto_timer, NULL);
        if (!usb_auto_timer) {
            return;
        }
        trace_usb_host_auto_scan_enabled();
    }
    timer_mod(usb_auto_timer, qemu_clock_get_ms(QEMU_CLOCK_REALTIME) + 2000);
}

/* A scan now, in the main loop: realize, close, a grab, the VM resuming. */
static void usb_host_auto_check(void *unused)
{
    libusb_device **devs = NULL;
    ssize_t n;

    if (usb_host_init() != 0) {
        return;
    }
    if (runstate_is_running()) {
        n = libusb_get_device_list(ctx, &devs);
        usb_host_auto_match(devs, n);
    }
    usb_host_auto_rearm();
}

#ifdef CONFIG_WIN32
/*
 * Listing the host's USB devices costs 5-9 ms on Windows (SetupAPI walks
 * every node). The periodic scan therefore lists on its own thread and
 * only the match runs in the main loop: a 2 s beat of BQL-held stalls is a
 * dropped frame for the vCPUs every 2 s while anything is passed through.
 */
static QemuThread scan_thread;
static QemuSemaphore scan_sem;
static QEMUBH *scan_bh;
static libusb_device **scan_devs;
static ssize_t scan_n;
static bool scan_busy;

static void *usb_host_scan_thread(void *opaque)
{
    for (;;) {
        qemu_sem_wait(&scan_sem);
        scan_n = libusb_get_device_list(ctx, &scan_devs);
        qemu_bh_schedule(scan_bh);
    }
    return NULL;
}

static void usb_host_scan_bh(void *opaque)
{
    libusb_device **devs = scan_devs;

    scan_devs = NULL;
    scan_busy = false;
    usb_host_auto_match(devs, scan_n);
    usb_host_auto_rearm();
}
#endif

static void usb_host_auto_timer(void *unused)
{
#ifdef CONFIG_WIN32
    if (!runstate_is_running()) {
        usb_host_auto_rearm();
        return;
    }
    if (!scan_bh) {
        qemu_sem_init(&scan_sem, 0);
        scan_bh = qemu_bh_new(usb_host_scan_bh, NULL);
        qemu_thread_create(&scan_thread, "usb-host-scan",
                           usb_host_scan_thread, NULL, QEMU_THREAD_DETACHED);
    }
    if (!scan_busy) {
        scan_busy = true;
        qemu_sem_post(&scan_sem);
    }
    /* the scan's bottom half re-arms the timer */
#else
    usb_host_auto_check(NULL);
#endif
}

void hmp_info_usbhost(Monitor *mon, const QDict *qdict)
{
    libusb_device **devs = NULL;
    struct libusb_device_descriptor ddesc;
    char port[16];
    int i, n;

    if (usb_host_init() != 0) {
        return;
    }

    n = libusb_get_device_list(ctx, &devs);
    for (i = 0; i < n; i++) {
        if (libusb_get_device_descriptor(devs[i], &ddesc) != 0) {
            continue;
        }
        if (ddesc.bDeviceClass == LIBUSB_CLASS_HUB) {
            continue;
        }
        usb_host_get_port(devs[i], port, sizeof(port));
        monitor_printf(mon, "  Bus %d, Addr %d, Port %s, Speed %s Mb/s\n",
                       libusb_get_bus_number(devs[i]),
                       libusb_get_device_address(devs[i]),
                       port,
                       speed_name[libusb_get_device_speed(devs[i])]);
        monitor_printf(mon, "    Class %02x:", ddesc.bDeviceClass);
        monitor_printf(mon, " USB device %04x:%04x",
                       ddesc.idVendor, ddesc.idProduct);
        if (ddesc.iProduct) {
            libusb_device_handle *handle;
            if (libusb_open(devs[i], &handle) == 0) {
                unsigned char name[64] = "";
                libusb_get_string_descriptor_ascii(handle,
                                                   ddesc.iProduct,
                                                   name, sizeof(name));
                libusb_close(handle);
                monitor_printf(mon, ", %s", name);
            }
        }
        monitor_printf(mon, "\n");
    }
    libusb_free_device_list(devs, 1);
}
