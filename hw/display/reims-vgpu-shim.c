/*
 * reims-vgpu: bus-independent shim services. See reims-vgpu-shim.h.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/timer.h"
#include "qemu/rcu.h"
#include "hw/core/cpu.h"
#include "exec/cpu-common.h"
#include "system/address-spaces.h"
#include "system/hw_accel.h"
#include "system/memory.h"
#include "system/physmem.h"
#include "system/ram_addr.h"
#include "ui/console.h"
#include "ui/input.h"
#include "reims_vgpu_qemu_abi.h"
#include "reims-vgpu-shim.h"

uint64_t reims_vgpu_shim_mono_ns(void *ctx)
{
    (void)ctx;
    return (uint64_t)qemu_clock_get_ns(QEMU_CLOCK_HOST);
}

/*
 * RAM-only attrs. `memory = 1` makes the address space reject a translation
 * that resolves to a device rather than RAM (MEMTX_ACCESS_ERROR) instead of
 * performing it. Every GPA reaching read/write_gpa came from a guest-supplied
 * page-entry list, so an entry pointing at one of our own BARs would otherwise
 * re-enter this device's MMIO handler from inside a Rust call that already
 * holds the device lock. Failing closed turns that into a visible decline.
 */
static const MemTxAttrs reims_vgpu_shim_ram_attrs = {
    .memory = true,
};

/*
 * TEMP lab REIMS_VGPU_LAB_READ_FAST=on (opt-in): the drain reads guest RAM
 * mostly a word at a time (page-table walks, object lists), and a 5K drag
 * profile put address_space_read's translation (RCU, flatview lookup, dispatch)
 * at ~8 % of the drain thread. Each thread remembers the last plain-RAM section
 * it read from — its GPA bounds and host pointer, valid for one FlatView — so a
 * read inside it is a memcpy. Only RAM sections of at least 1 MiB are kept
 * (the guest's main memory); everything else, and any read crossing the
 * section's end, takes address_space_read exactly as before.
 */
static int reims_vgpu_shim_read_fast_mode = -1;

typedef struct ReimsVgpuReadCache {
    FlatView *fv;
    uint64_t lo, hi;        /* [lo, hi) of guest physical address space */
    uint8_t *host;          /* host address of `lo` */
    bool ok;                /* false: a section to read through address_space_read */
} ReimsVgpuReadCache;

/*
 * Four sections per thread: main memory is two (below and above 4 GiB) and the
 * drain's reads alternate between them (page tables vs. data). One entry
 * missed on nearly every read and paid memory_region_find each time, which
 * made the drag 10 % slower than the uncached path (frf*, 2026-10-01).
 */
#define REIMS_VGPU_READ_CACHE_N 4
static __thread ReimsVgpuReadCache reims_vgpu_read_cache[REIMS_VGPU_READ_CACHE_N];
static __thread unsigned reims_vgpu_read_cache_next;

static bool reims_vgpu_shim_read_cached(uint64_t gpa, uint8_t *buf, size_t len)
{
    ReimsVgpuReadCache *c;
    MemoryRegionSection sec;
    FlatView *fv;
    uint64_t size;
    int i;

    if (gpa + len < gpa) {
        return false;
    }
    RCU_READ_LOCK_GUARD();
    fv = address_space_to_flatview(&address_space_memory);
    for (i = 0; i < REIMS_VGPU_READ_CACHE_N; i++) {
        c = &reims_vgpu_read_cache[i];
        if (c->fv == fv && gpa >= c->lo && gpa + len <= c->hi) {
            if (!c->ok) {
                return false;
            }
            memcpy(buf, c->host + (gpa - c->lo), len);
            return true;
        }
    }
    sec = memory_region_find(get_system_memory(), gpa, 1);
    if (!sec.mr) {
        return false;
    }
    size = int128_get64(sec.size);
    c = &reims_vgpu_read_cache[reims_vgpu_read_cache_next++ % REIMS_VGPU_READ_CACHE_N];
    c->lo = sec.offset_within_address_space;
    c->hi = c->lo + size;
    c->ok = memory_region_is_ram(sec.mr) && !memory_region_is_ram_device(sec.mr) &&
            !sec.mr->rom_device && size >= (1u << 20);
    c->host = c->ok ? (uint8_t *)memory_region_get_ram_ptr(sec.mr) + sec.offset_within_region
                    : NULL;
    c->fv = fv;
    memory_region_unref(sec.mr);
    if (!c->ok || gpa < c->lo || gpa + len > c->hi) {
        return false;
    }
    memcpy(buf, c->host + (gpa - c->lo), len);
    return true;
}

int reims_vgpu_shim_read_gpa(void *ctx, uint64_t gpa, uint8_t *buf, size_t len)
{
    MemTxResult r;

    (void)ctx;
    if (!buf || len == 0) {
        return 0;
    }
    if (reims_vgpu_shim_read_fast_mode < 0) {
        const char *e = getenv("REIMS_VGPU_LAB_READ_FAST");

        /* Opt-in: on, the 5K drag ran ~10 % slower in two A/Bs (frf*, fa3*). */
        reims_vgpu_shim_read_fast_mode = e && strcmp(e, "on") == 0;
    }
    if (reims_vgpu_shim_read_fast_mode && reims_vgpu_shim_read_cached(gpa, buf, len)) {
        return 0;
    }
    r = address_space_read(&address_space_memory, gpa,
                           reims_vgpu_shim_ram_attrs, buf, len);
    return r == MEMTX_OK ? 0 : -1;
}

/*
 * The device's own stores must not reach the DIRTY_MEMORY_VGA bitmap.
 * reims-vgpu-dirty.c reads that bitmap as "the guest CPU wrote these pages",
 * and the gather witness refuses (and re-reads) every sampled window whose
 * pages it finds set there. address_space_write() marks every enabled dirty
 * client, so each writeback the device made read back as a guest store.
 *
 * Hosts with the packed page alias send the device's bulk stores through host
 * pointers and never reached this function with them; without the alias
 * (Windows) every store came here, and the witness refused almost every bind.
 * Guest RAM is therefore written directly and dirtied for every client except
 * VGA. Anything that is not plain RAM, and any range TCG must invalidate code
 * for, keeps the checked address_space path.
 */
int reims_vgpu_shim_write_gpa(void *ctx, uint64_t gpa, const uint8_t *buf,
                              size_t len)
{
    (void)ctx;
    if (!buf || len == 0) {
        return 0;
    }

    RCU_READ_LOCK_GUARD();
    while (len > 0) {
        hwaddr xlat, plen = len;
        MemoryRegion *mr;
        uint8_t mask;

        mr = address_space_translate(&address_space_memory, gpa, &xlat, &plen,
                                     true, reims_vgpu_shim_ram_attrs);
        if (!mr || plen == 0) {
            return -1;
        }
        mask = memory_region_get_dirty_log_mask(mr);
        if (memory_region_is_ram(mr) && !mr->readonly &&
            !memory_region_is_romd(mr) &&
            memory_region_get_ram_addr(mr) != RAM_ADDR_INVALID &&
            !(mask & (1 << DIRTY_MEMORY_CODE))) {
            memcpy((uint8_t *)memory_region_get_ram_ptr(mr) + xlat, buf, plen);
            physical_memory_set_dirty_range(memory_region_get_ram_addr(mr) + xlat,
                                            plen,
                                            mask & ~(1 << DIRTY_MEMORY_VGA));
        } else if (address_space_write(&address_space_memory, gpa,
                                       reims_vgpu_shim_ram_attrs, buf,
                                       plen) != MEMTX_OK) {
            return -1;
        }
        gpa += plen;
        buf += plen;
        len -= plen;
    }
    return 0;
}

void reims_vgpu_shim_thread_attach(void *ctx)
{
    (void)ctx;
    rcu_register_thread();
}

void reims_vgpu_shim_thread_detach(void *ctx)
{
    (void)ctx;
    rcu_unregister_thread();
}

int reims_vgpu_shim_is_ram_gpa(void *ctx, uint64_t gpa)
{
    hwaddr xlat, plen = 1;
    MemoryRegion *mr;
    int ok;

    (void)ctx;
    rcu_read_lock();
    mr = address_space_translate(&address_space_memory, gpa, &xlat, &plen, true,
                                 MEMTXATTRS_UNSPECIFIED);
    ok = mr && memory_region_is_ram(mr) && plen >= 1;
    rcu_read_unlock();
    return ok ? 1 : 0;
}

/*
 * Coalescing state for one guest_ram_regions walk.
 *
 * `found` counts every span the walk produced, including the ones past the
 * caller's `max`. That is the difference between a caller that learns its array
 * was short and one that imports part of the guest's RAM and never finds out.
 */
typedef struct ReimsVgpuRamScan {
    ReimsVgpuGuestRamRegion *out;
    size_t max;
    size_t found;
    bool pending;
    uint64_t gpa;
    uint64_t hva;
    uint64_t len;
} ReimsVgpuRamScan;

static void reims_vgpu_ram_scan_flush(ReimsVgpuRamScan *scan)
{
    if (!scan->pending) {
        return;
    }
    if (scan->found < scan->max) {
        scan->out[scan->found].gpa_base = scan->gpa;
        scan->out[scan->found].host_va = scan->hva;
        scan->out[scan->found].len = scan->len;
    }
    scan->found++;
    scan->pending = false;
}

static bool reims_vgpu_ram_scan_range(Int128 start, Int128 len,
                                      const MemoryRegion *mr,
                                      hwaddr offset_in_region, void *opaque)
{
    ReimsVgpuRamScan *scan = opaque;
    uint64_t gpa, bytes, hva;
    void *ptr;

    /*
     * ROM and ROMD ranges are RAM-backed but the guest cannot store into them,
     * and a ram_device is somebody else's mapping (a VFIO BAR) rather than
     * guest memory. Reporting any of the three would hand the GPU write access
     * to bytes the guest believes are fixed, or does not own at all.
     */
    if (!mr || !memory_region_is_ram(mr) || mr->readonly ||
        memory_region_is_romd(mr) || memory_region_is_ram_device(mr)) {
        reims_vgpu_ram_scan_flush(scan);
        return false;
    }
    /*
     * An Int128 that does not fit a uint64_t cannot be a mapping in this
     * process's address space, so it is not a span anything could import.
     */
    if (int128_gethi(start) != 0 || int128_gethi(len) != 0) {
        reims_vgpu_ram_scan_flush(scan);
        return false;
    }
    gpa = int128_get64(start);
    bytes = int128_get64(len);
    if (bytes == 0) {
        return false;
    }
    ptr = memory_region_get_ram_ptr(mr);
    if (!ptr) {
        reims_vgpu_ram_scan_flush(scan);
        return false;
    }
    hva = (uint64_t)(uintptr_t)ptr + (uint64_t)offset_in_region;

    /*
     * Merge only when the next range continues this one in *both* address
     * spaces. A board that splits one RAMBlock across several flat ranges then
     * costs one span, while two blocks that abut in GPA but not in host VA stay
     * separate — merging those would make an offset inside the span name the
     * wrong host bytes, which is exactly the failure importing a RAMBlock at a
     * time exists to avoid.
     */
    if (scan->pending && scan->gpa + scan->len == gpa &&
        scan->hva + scan->len == hva && scan->len <= UINT64_MAX - bytes) {
        scan->len += bytes;
        return false;
    }
    reims_vgpu_ram_scan_flush(scan);
    scan->pending = true;
    scan->gpa = gpa;
    scan->hva = hva;
    scan->len = bytes;
    return false;
}

int reims_vgpu_shim_guest_ram_regions(void *ctx, ReimsVgpuGuestRamRegion *out,
                                      size_t max)
{
    ReimsVgpuRamScan scan = { 0 };

    (void)ctx;
    if (!out || max == 0) {
        return REIMS_VGPU_GUEST_RAM_ERR_ARGS;
    }
    scan.out = out;
    scan.max = max;

    rcu_read_lock();
    flatview_for_each_range(address_space_to_flatview(&address_space_memory),
                            reims_vgpu_ram_scan_range, &scan);
    rcu_read_unlock();
    reims_vgpu_ram_scan_flush(&scan);

    if (scan.found == 0) {
        return REIMS_VGPU_GUEST_RAM_ERR_NO_RAM;
    }
    /*
     * A span count that does not fit the return type is not a machine anyone
     * runs. Reporting it as "no RAM" would be a lie, so it is the argument
     * refusal: the caller asked for an answer this return value cannot carry.
     */
    if (scan.found > (size_t)INT_MAX) {
        return REIMS_VGPU_GUEST_RAM_ERR_ARGS;
    }
    return (int)scan.found;
}

int reims_vgpu_shim_read_kva(void *ctx, uint64_t kva, uint8_t *buf, size_t len)
{
    CPUState *cs = current_cpu;

    (void)ctx;
    if (!buf || len == 0) {
        return 0;
    }
    if (!cs) {
        return -2;
    }
    cpu_synchronize_state(cs);
    return cpu_memory_rw_debug(cs, kva, buf, len, false) == 0 ? 0 : -1;
}

static InputButton reims_vgpu_shim_button(uint32_t code, bool *ok)
{
    *ok = true;
    switch (code) {
    case REIMS_VGPU_BUTTON_LEFT:
        return INPUT_BUTTON_LEFT;
    case REIMS_VGPU_BUTTON_MIDDLE:
        return INPUT_BUTTON_MIDDLE;
    case REIMS_VGPU_BUTTON_RIGHT:
        return INPUT_BUTTON_RIGHT;
    case REIMS_VGPU_BUTTON_WHEEL_UP:
        return INPUT_BUTTON_WHEEL_UP;
    case REIMS_VGPU_BUTTON_WHEEL_DOWN:
        return INPUT_BUTTON_WHEEL_DOWN;
    case REIMS_VGPU_BUTTON_SIDE:
        return INPUT_BUTTON_SIDE;
    case REIMS_VGPU_BUTTON_EXTRA:
        return INPUT_BUTTON_EXTRA;
    case REIMS_VGPU_BUTTON_WHEEL_LEFT:
        return INPUT_BUTTON_WHEEL_LEFT;
    case REIMS_VGPU_BUTTON_WHEEL_RIGHT:
        return INPUT_BUTTON_WHEEL_RIGHT;
    default:
        *ok = false;
        return INPUT_BUTTON_LEFT;
    }
}

bool reims_vgpu_shim_scanout_may_paint(uint64_t rust_handle, uint32_t mapping_id)
{
    uint32_t may = 0;

    if (rust_handle == 0) {
        return false;
    }
    if (reims_vgpu_qemu_scanout_may_paint(rust_handle, mapping_id, &may) !=
        REIMS_VGPU_QEMU_OK) {
        return false;
    }
    return may != 0;
}

uint32_t reims_vgpu_shim_console_feed(uint64_t rust_handle, uint32_t *out_mid,
                                      uint32_t *out_w, uint32_t *out_h,
                                      uint32_t *out_gen)
{
    uint32_t kind = REIMS_VGPU_CONSOLE_FEED_FIRMWARE;

    if (rust_handle == 0) {
        return REIMS_VGPU_CONSOLE_FEED_FIRMWARE;
    }
    if (reims_vgpu_qemu_console_feed(rust_handle, &kind, out_mid, out_w, out_h,
                                     out_gen) != REIMS_VGPU_QEMU_OK) {
        return REIMS_VGPU_CONSOLE_FEED_FIRMWARE;
    }
    return kind;
}

void reims_vgpu_shim_input_key(QemuConsole *con, uint32_t evdev, bool down)
{
    if (!con) {
        return;
    }
    qemu_input_event_send_key_linux(con, evdev, down);
}

void reims_vgpu_shim_input_pointer_move(QemuConsole *con, uint32_t x,
                                        uint32_t y, uint32_t w, uint32_t h)
{
    if (!con || w == 0 || h == 0) {
        return;
    }
    qemu_input_queue_abs(con, INPUT_AXIS_X, (int)x, 0, (int)w);
    qemu_input_queue_abs(con, INPUT_AXIS_Y, (int)y, 0, (int)h);
    qemu_input_event_sync();
}

void reims_vgpu_shim_input_button(QemuConsole *con, uint32_t code, bool down)
{
    InputButton button;
    bool ok;

    if (!con) {
        return;
    }
    button = reims_vgpu_shim_button(code, &ok);
    if (!ok) {
        return;
    }
    qemu_input_queue_btn(con, button, down);
    qemu_input_event_sync();
}
