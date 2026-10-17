/*
 * Thin PCI QEMU shim for the product paravirtualized GPU device.
 *
 * Type name: reims-vgpu-pci (PCI IDs vendor 0x106B / device 0xEEEE).
 * Sibling of reims-vgpu-mmio (sysbus/vmapple): same HostOps + Rust staticlib
 * boundary. C owns only PCI/QOM/BAR/MSI/console/BH — no protocol or GPU logic
 * Product behaviour lives in crates/reims-vgpu.
 *
 * Topology: -vga none + pcie-root-port (non-zero bus) + this device;
 * class VGA 0x030000; MSI vector 0; BAR0 16 KiB (control +0x1000).
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#if defined(_WIN32)
#include "system/whpx-wp.h"
#endif
#include "qemu/log.h"
#include "qemu/error-report.h"
#include "qemu/module.h"
#include "qemu/main-loop.h"
#include "qemu/aio.h"
#include "qemu/thread.h"
#include "qemu/rcu.h"
#include "qapi/error.h"
#include "hw/pci/pci_device.h"
#include "hw/pci/msi.h"
#include "hw/pci/pci.h"
#include "hw/pci/pcie.h"
#include "qom/object.h"
#include "system/address-spaces.h"
#include "system/memory.h"
#include "system/ramblock.h"
#include "system/runstate.h"
#include "system/whpx.h"
#include "ui/console.h"
#include "ui/surface.h"
#include "trace.h"
#include "hw/input/hid.h"
#include "reims_vgpu_qemu_abi.h"
#include "reims-vgpu-dirty.h"
#include "reims-vgpu-shim.h"

#define TYPE_REIMS_VGPU_PCI "reims-vgpu-pci"
OBJECT_DECLARE_SIMPLE_TYPE(ReimsVGPUPCIState, REIMS_VGPU_PCI)

typedef struct ReimsVGPUPCIPageView {
    void *ptr;
    size_t len;
#ifdef _WIN32
    /* TEMP (lab, WP_VIEW): the alias's run table, owned with the view. */
    QemuWinRamAliasRun *runs;
    size_t runs_n;
#endif
} ReimsVGPUPCIPageView;

/* TEMP lab (b610): alias traffic counters behind page_views_lock, one line a second. */
static struct {
    uint64_t builds, build_ns, build_max_ns, fails;
    uint64_t frees, free_ns;
    uint64_t runs, run_pages;
    int64_t last;
} reims_pagealias;

/*
 * BAR0 = full gfx window; control block at +0x1000. The size is the shared
 * REIMS_VGPU_GFX_MMIO_SIZE — Rust bounds its register store against the same
 * number, and a private copy here is a window the guest can address past it.
 */
/* BAR1 = linear UEFI GOP framebuffer (BGRA8 at the EFI boot mode + headroom). */
#define REIMS_VGPU_PCI_FB_SIZE   (16u * 1024u * 1024u)
/*
 * These two IDs are what the guest's paravirt GPU driver matches on, and the
 * match decides which Metal plugin the guest loads. That has a consequence on
 * macOS 26 that is worth stating here, because it is invisible from this file
 * and it is not a defect anything on this side can repair.
 *
 * The guest ships two personalities for one paravirt GPU:
 *
 *   - one matching an IOPCIDevice on exactly these IDs, whose Metal plugin
 *     descends from Metal's IOAccelerator device base class;
 *   - one matching a *device-tree* node by name, on an ARM platform provider,
 *     whose Metal plugin descends from the IOGPUFamily device base class.
 *
 * Only the second plugin implements the device-architecture property that
 * Metal's binary-archive loader asserts on. On macOS 26 the icon-rendering
 * path loads a precompiled binary archive, so on this pathway that assert
 * fires and the icon renderers abort — repeatedly, since each icon re-render
 * retries. The other personality cannot be reached from here at any PCI
 * identity: it requires an ARM platform provider that does not exist on an
 * x86_64 guest, and its kext declares a dependency on the ARM platform driver.
 *
 * So this is structural to the PCI pathway rather than a value to tune. Do not
 * spend a boot probing device-info keys for it — the capability table is not in
 * that code path at all. Changing these IDs does not reach the other
 * personality; it only stops the driver matching us.
 */
#define REIMS_VGPU_PCI_VENDOR    0x106B
#define REIMS_VGPU_PCI_DEVICE    0xEEEE
/* BAR1's own pixel size, not a shared constant: it describes this shim's GOP
 * layout, which Rust reaches only through efi_console_copy's explicit stride. */
#define REIMS_VGPU_PCI_EFI_BPP   4u

struct ReimsVGPUPCIState {
    PCIDevice parent_obj;

    MemoryRegion iomem_gfx;
    /*
     * BAR1 linear FB (same PCI device — not a second display). UEFI GOP option
     * ROM + OpenCore paint here until the product present path latches.
     */
    MemoryRegion fb_vram;
    QemuConsole *con;
    DisplaySurface *surface;
    bool new_frame_ready;

    ReimsVgpuHostOps host_ops;
    /* Hypervisor dirty-bitmap adapter: the only witness for a write to a
     * surface's guest pages that no device operation made. */
    ReimsVgpuDirty *dirty;
    uint64_t rust_handle;
    /* Caller-owned packed aliases over file-backed guest pages. Direct
     * RAMBlock pointers never enter this list; constructed views leave through
     * unmap_pages when the Rust mapping lifetime ends, with device exit as the
     * final safety net. */
    GArray *page_views;
    QemuMutex page_views_lock;
    uint64_t page_view_live_bytes;
    uint64_t page_view_live_pages;
    uint64_t page_views_created;
    uint64_t page_views_destroyed;

    /* Ordered Rust FIFO/render drain owner. The AIO BH only applies completed
     * HostActions; it never translates shaders or waits for GPU work. */
    QemuThread drain_thread;
    QemuMutex drain_mutex;
    QemuCond drain_cond;
    QEMUBH *action_bh;
    /*
     * Steady vblank + Dekker-rescue heartbeat. Keep its wait on a dedicated
     * thread: main-loop timers are delayed by input/QMP/display work during
     * window drags, which starves the guest display time base.
     */
    QemuThread heartbeat_thread;
    QemuMutex heartbeat_mutex;
    QemuCond heartbeat_cond;
    bool drain_pending;
    bool drain_stopping;
    bool drain_started;
    bool heartbeat_stopping;
    bool heartbeat_started;
    /*
     * Dirty-log harvest off the vCPU. The harvest has to finish before the
     * drain acts on the work a doorbell handed over, but nothing requires the
     * guest to wait for it: a vCPU that rang the doorbell used to sit in the
     * MMIO exit, holding the BQL, for the whole accelerator sync.
     */
    QemuThread harvest_thread;
    QemuMutex harvest_mutex;
    QemuCond harvest_cond;
    QemuCond harvest_done_cond;
    uint64_t harvest_asked;
    /* Harvest without the BQL (lab A/B: REIMS_VGPU_HARVEST_BQL=on restores it). */
    bool harvest_nobql;
    /* On-demand dirty sync (default; REIMS_VGPU_DIRTY_ONDEMAND=off restores the harvest). */
    bool dirty_ondemand;
    /* TEMP lab diagnostic: harvest cost and the drain's wait on it, per second. */
    uint64_t lab_hv_n, lab_hv_ns, lab_hv_max_ns, lab_wait_n, lab_wait_ns, lab_wait_max_ns;
    int64_t lab_hv_last;
    uint64_t harvest_done;
    bool harvest_stopping;
    bool harvest_started;
    /* The drain was told "not settled" and left work; wake it on catch-up. */
    bool harvest_rearm;
    /*
     * TEMP lab REIMS_VGPU_LAB_DIRTY_PEEK_MS=<1..16> (with DIRTY_HOLE=tail): a thread reads every
     * child ring's tail that often and, for one the guest moved without ringing (it appends to a
     * ring the device has not drained, ~80 times a second under load), does what that doorbell
     * would have: reads the tail, then opens an epoch, then wakes the prefetch. The drain, still
     * busy with older packets, later finds the tail covered (DIRTY_HOLE=tail opens no epoch of its
     * own) and the sets those packets read already synced for it (DIRTY_FRONTIER).
     */
    QemuThread peek_thread;
    QemuMutex peek_mutex;
    QemuCond peek_cond;
    int peek_ms;
    bool peek_stopping;
    bool peek_started;
    uint64_t lab_peeks;
    /* HostOps.irq_pulse may raise MSIs off the BQL (lab A/B: REIMS_VGPU_IRQ_DIRECT=on). */
    bool irq_direct;
    Notifier shutdown_notifier;
    bool shutdown_notifier_registered;
};

/* ---------- HostOps (GPA R/W + BH; Linux has no map_pages / xreg) ---------- */

static int reims_vgpu_pci_read_xreg(void *ctx, uint32_t index, uint64_t *out)
{
    (void)ctx;
    (void)index;
    (void)out;
    /* x86 product path: no arm xreg handoff. */
    return -1;
}

static int reims_vgpu_pci_map_pages(void *ctx, const uint64_t *gpas, size_t count,
                             void **out_ptr, ReimsVgpuMapPagesFailure *failure)
{
    ReimsVGPUPCIState *s = ctx;
    /* x86 guest page granularity (host-pointer import view stride). */
    const hwaddr page = (hwaddr)1 << REIMS_VGPU_GUEST_PAGE_SHIFT_X86_64;
    uint8_t *base = NULL;
    MemoryRegion *base_mr = NULL;
    size_t i;

    if (failure) {
        *failure = (ReimsVgpuMapPagesFailure) {
            .stage = REIMS_VGPU_MAP_PAGES_FAILURE_NONE,
        };
    }
    if (!s || !gpas || count == 0 || !out_ptr || !failure ||
        count > SIZE_MAX / page) {
        if (failure) {
            failure->stage = REIMS_VGPU_MAP_PAGES_FAILURE_INVALID_GUEST_PAGE;
            failure->page_index = UINT64_MAX;
        }
        return -1;
    }

    /*
     * Packed contiguous host-VA view: guest RAM is already mmap'd in the QEMU
     * process. Callers (Rust map_pages consumers) treat page i as
     * `base + i * page` — so every GPA in the list must alias host VA at that
     * packed offset (same RAMBlock, sequential host pages). Non-sequential /
     * cross-MR / non-RAM lists return -1; callers multi-import maximal runs
     * or fail closed (`not_contig`).
     *
     * Note: do **not** accept `hva == base + (gpas[i] - gpas[0])` for sparse
     * GPAs — that would return success while page i is not at base+i*page,
     * silently mis-aliasing multi-page views.
     */
    rcu_read_lock();

    /*
     * Maximal GPA runs are the common large-surface path.  One address-space
     * translation covering the complete sequential run proves the same
     * RAMBlock/HVA contract as the page loop below.  Avoiding thousands of
     * identical translations is important for full-frame staging/writeback;
     * retain the page loop for aliases and region-boundary cases.
     */
    if (count <= UINT64_MAX / page &&
        gpas[0] <= UINT64_MAX - (count - 1) * page) {
        hwaddr total = count * page;
        bool sequential = true;

        for (i = 1; i < count; i++) {
            if (gpas[i] != gpas[0] + i * page) {
                sequential = false;
                break;
            }
        }
        if (sequential) {
            hwaddr xlat, plen = total;
            MemoryRegion *mr;
            uint8_t *hva;

            mr = address_space_translate(&address_space_memory, gpas[0],
                                         &xlat, &plen, true,
                                         MEMTXATTRS_UNSPECIFIED);
            if (mr && memory_region_is_ram(mr) && plen >= total) {
                hva = (uint8_t *)memory_region_get_ram_ptr(mr) + xlat;
                if (((uintptr_t)hva & (page - 1)) == 0) {
                    rcu_read_unlock();
                    *out_ptr = hva;
                    return 0;
                }
            }
        }
    }

    for (i = 0; i < count; i++) {
        hwaddr xlat, plen = page;
        MemoryRegion *mr;
        uint8_t *hva;

        mr = address_space_translate(&address_space_memory, gpas[i], &xlat,
                                     &plen, true, MEMTXATTRS_UNSPECIFIED);
        if (!mr || !memory_region_is_ram(mr) || plen < page) {
            goto fail;
        }
        hva = (uint8_t *)memory_region_get_ram_ptr(mr) + xlat;
        if (i == 0) {
            base = hva;
            base_mr = mr;
            if (((uintptr_t)base & (page - 1)) != 0) {
                goto fail; /* base must be page-aligned for import */
            }
        } else if (mr != base_mr || hva != base + i * page) {
            goto fail; /* page i not packed-contig with the run */
        }
    }
    rcu_read_unlock();

    *out_ptr = base;
    return 0;

fail:
    rcu_read_unlock();

#if defined(_WIN32)
    /*
     * The same packed alias as the mmap arm below, built from views of the
     * WP_VIEW guest-RAM section (qemu_win_ram_alias_pages). Until 2026-10-15
     * this returned -1 for every scattered list, so each one took the copying
     * rails: 652 qemu_map_pages_failure_unspecified and 357 GPU-writeback
     * declines in one Claude.app boot.
     */
    {
        size_t total = count * page;
        void **hosts;

        /*
         * TEMP lab: off unless QEMU_WIN_PAGE_ALIAS=on. b608-b610 (5K Safari canvas): with the aliases the
         * whole VM stalled - vCPUs >1 s outside the hypervisor in memory-access exits, canvas 0.4 fps, Maps
         * 0 - against 61 fps on the copying rails. Each alias was a placeholder reservation plus one
         * MapViewOfFile3 per 4 KiB page (3+ address-space calls per page, a 5K surface is 14400 pages) on
         * the one section the WP dirty machinery reprotects; since the run-coalescing rebuild (one view per
         * maximal section-consecutive run) `=on` is the arm this switch A/Bs against.
         */
        static int alias_off = -1;
        if (alias_off < 0) {
            const char *e = getenv("QEMU_WIN_PAGE_ALIAS");
            alias_off = !(e && strcmp(e, "on") == 0);
        }
        if (alias_off) {
            failure->stage = REIMS_VGPU_MAP_PAGES_FAILURE_ALIAS;
            return -1;
        }
        hosts = g_new(void *, count);
        unsigned long err = 0;
        size_t failed = SIZE_MAX;
        ReimsVGPUPCIPageView held;
        QemuWinRamAliasRun *runs = NULL;
        size_t runs_n = 0;
        int64_t t_build = get_clock();
        void *view;

        rcu_read_lock();
        for (i = 0; i < count; i++) {
            hwaddr xlat, plen = page;
            MemoryRegion *mr;

            mr = address_space_translate(&address_space_memory, gpas[i],
                                         &xlat, &plen, true,
                                         MEMTXATTRS_UNSPECIFIED);
            if (!mr || !memory_region_is_ram(mr) || plen < page) {
                rcu_read_unlock();
                g_free(hosts);
                failure->stage = REIMS_VGPU_MAP_PAGES_FAILURE_INVALID_GUEST_PAGE;
                failure->page_index = i;
                return -1;
            }
            hosts[i] = (uint8_t *)memory_region_get_ram_ptr(mr) + xlat;
        }
        rcu_read_unlock();
        /*
         * The kernel calls run for the whole window (runs of pages, since the
         * rebuild): under the RCU read lock they made every grace period wait
         * out a surface build (b610 had tens of milliseconds per list), and
         * nothing below reads the memory topology - the host pointers are
         * stable for the RAM block's lifetime.
         *
         * TEMP lab (storm fix (a), 2026-10-17 forensics): drop the BQL around
         * the build itself. The build is pure Win32 address-space work over
         * hosts[] and the section; page_views is under its own mutex below.
         * In storm boots the same calls stall 0.17-2.4 s each - with the BQL
         * held that froze every vCPU behind the builder (mmio_bql_us up to
         * 1.4 s/s, hundreds of whpx-vcpu-stuck, 4 fps boots). Without the BQL
         * a stalled build no longer stops the machine; the tripwire below
         * retires the rail entirely once a storm is proven.
         */
        {
            bool bql_held = bql_locked();
            if (bql_held) {
                bql_unlock();
            }
            view = qemu_win_ram_alias_pages(hosts, count, page, &err, &failed,
                                            &runs, &runs_n);
            if (bql_held) {
                bql_lock();
            }
        }
        g_free(hosts);
        /*
         * TEMP lab (storm fix (b)): a build this slow is the bistable
         * AS-lock/WsLock storm, not work - clean builds are linear at 2-3 us
         * per page (20 ms for a 6k+ page surface), storm builds pay a fixed
         * 0.17-2.4 s regardless of size. Trip once and the rest of the boot
         * takes the copying rails (the default arm: canvas 61 fps against
         * 0.4 in b610) instead of gambling on the next stall.
         * QEMU_WIN_PAGE_ALIAS_TRIPWIRE_NS overrides the threshold; 0 disables.
         */
        {
            int64_t build_ns_now = get_clock() - t_build;
            static int64_t tripwire_ns = -1;
            if (tripwire_ns < 0) {
                const char *twe = getenv("QEMU_WIN_PAGE_ALIAS_TRIPWIRE_NS");
                tripwire_ns = twe ? strtoll(twe, NULL, 10) : 100000000LL;
            }
            if (tripwire_ns > 0 && build_ns_now > tripwire_ns && !alias_off) {
                alias_off = 1;
                fprintf(stderr, "reims-pagealias storm tripped: build_us=%"
                        PRId64 " pages=%zu -> copying rails for this boot\n",
                        build_ns_now / 1000, count);
            }
        }
        if (!view) {
            failure->stage = failed == SIZE_MAX
                ? REIMS_VGPU_MAP_PAGES_FAILURE_RESERVATION
                : REIMS_VGPU_MAP_PAGES_FAILURE_ALIAS;
            failure->host_errno = (int)err;
            failure->page_index = failed == SIZE_MAX ? 0 : failed;
            qemu_mutex_lock(&s->page_views_lock);
            reims_pagealias.fails++;
            qemu_mutex_unlock(&s->page_views_lock);
            return -1;
        }
        held.ptr = view;
        held.len = total;
        held.runs = runs;
        held.runs_n = runs_n;
        qemu_mutex_lock(&s->page_views_lock);
        g_array_append_val(s->page_views, held);
        s->page_view_live_bytes += total;
        s->page_view_live_pages += count;
        s->page_views_created++;
        {   /* TEMP lab (b610): one line a second while aliases are being built. */
            int64_t now = get_clock(), build_ns = now - t_build;

            reims_pagealias.builds++;
            reims_pagealias.build_ns += build_ns;
            if (build_ns > reims_pagealias.build_max_ns) {
                reims_pagealias.build_max_ns = build_ns;
            }
            reims_pagealias.runs += runs_n;
            reims_pagealias.run_pages += count;
            if (!reims_pagealias.last || now - reims_pagealias.last >= NANOSECONDS_PER_SECOND) {
                fprintf(stderr, "reims-pagealias epoch_ms=%" PRId64 " builds=%" PRIu64
                        " build_us=%" PRIu64 " build_max_us=%" PRIu64 " frees=%" PRIu64
                        " free_us=%" PRIu64 " runs=%" PRIu64 " run_pages=%" PRIu64
                        " live=%u fails=%" PRIu64 "\n",
                        g_get_real_time() / 1000,
                        reims_pagealias.builds, reims_pagealias.build_ns / 1000,
                        reims_pagealias.build_max_ns / 1000,
                        reims_pagealias.frees, reims_pagealias.free_ns / 1000,
                        reims_pagealias.runs, reims_pagealias.run_pages,
                        s->page_views->len, reims_pagealias.fails);
                memset(&reims_pagealias, 0, sizeof(reims_pagealias));
                reims_pagealias.last = now;
            }
        }
        qemu_mutex_unlock(&s->page_views_lock);
        *out_ptr = view;
        return 0;
    }
#else
    /*
     * A task GVA is linear even when its guest-physical pages are not. When
     * guest RAM is shared file-backed memory, recreate that virtual shape with
     * one MAP_SHARED view. This is an alias, not a copy: guest CPU writes and
     * GPU reads still address the same pages.
     *
     * Reserve the complete range first, then replace it page by page. A failed
     * page tears down the whole reservation, so callers never receive a
     * partially packed view.
     */
    {
        size_t total = count * page;
        uint8_t *view = mmap(NULL, total, PROT_NONE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        ReimsVGPUPCIPageView held;

        if (view == MAP_FAILED) {
            failure->stage = REIMS_VGPU_MAP_PAGES_FAILURE_RESERVATION;
            failure->host_errno = errno;
            return -1;
        }
        rcu_read_lock();
        for (i = 0; i < count; i++) {
            hwaddr xlat, plen = page;
            MemoryRegion *mr;
            uint8_t *hva;
            RAMBlock *rb;
            ram_addr_t rb_offset;
            ram_addr_t fd_offset;
            int fd;
            void *mapped;

            mr = address_space_translate(&address_space_memory, gpas[i],
                                         &xlat, &plen, true,
                                         MEMTXATTRS_UNSPECIFIED);
            if (!mr || !memory_region_is_ram(mr) || plen < page) {
                failure->stage = REIMS_VGPU_MAP_PAGES_FAILURE_INVALID_GUEST_PAGE;
                failure->page_index = i;
                goto alias_fail_locked;
            }
            hva = (uint8_t *)memory_region_get_ram_ptr(mr) + xlat;
            rb = qemu_ram_block_from_host(hva, false, &rb_offset);
            if (!rb || !qemu_ram_is_shared(rb)) {
                failure->stage = REIMS_VGPU_MAP_PAGES_FAILURE_ALIAS;
                failure->page_index = i;
                goto alias_fail_locked;
            }
            fd = qemu_ram_get_fd(rb);
            if (fd < 0 || rb_offset > RAM_ADDR_MAX - qemu_ram_get_fd_offset(rb)) {
                failure->stage = REIMS_VGPU_MAP_PAGES_FAILURE_ALIAS;
                failure->page_index = i;
                goto alias_fail_locked;
            }
            fd_offset = qemu_ram_get_fd_offset(rb) + rb_offset;
            if ((fd_offset & (page - 1)) != 0) {
                failure->stage = REIMS_VGPU_MAP_PAGES_FAILURE_ALIAS;
                failure->page_index = i;
                goto alias_fail_locked;
            }
            mapped = mmap(view + i * page, page, PROT_READ | PROT_WRITE,
                          MAP_SHARED | MAP_FIXED, fd, fd_offset);
            if (mapped == MAP_FAILED) {
                failure->stage = REIMS_VGPU_MAP_PAGES_FAILURE_ALIAS;
                failure->host_errno = errno;
                failure->page_index = i;
                goto alias_fail_locked;
            }
        }
        rcu_read_unlock();

        held.ptr = view;
        held.len = total;
        qemu_mutex_lock(&s->page_views_lock);
        g_array_append_val(s->page_views, held);
        s->page_view_live_bytes += total;
        s->page_view_live_pages += count;
        s->page_views_created++;
        qemu_mutex_unlock(&s->page_views_lock);
        *out_ptr = view;
        return 0;

alias_fail_locked:
        rcu_read_unlock();
        munmap(view, total);
        return -1;
    }
#endif
}

static void reims_vgpu_pci_unmap_pages(void *ctx, void *ptr, size_t len)
{
    ReimsVGPUPCIState *s = ctx;
    size_t i;

    if (!s || !s->page_views || !ptr || len == 0) {
        return;
    }

    qemu_mutex_lock(&s->page_views_lock);
    for (i = 0; i < s->page_views->len; i++) {
        ReimsVGPUPCIPageView *view =
            &g_array_index(s->page_views, ReimsVGPUPCIPageView, i);

        if (view->ptr != ptr || view->len != len) {
            continue;
        }
        {
#ifdef _WIN32
            int64_t t_free = get_clock();

            qemu_win_ram_alias_free(view->ptr,
                                    view->len >> REIMS_VGPU_GUEST_PAGE_SHIFT_X86_64,
                                    (size_t)1 << REIMS_VGPU_GUEST_PAGE_SHIFT_X86_64,
                                    view->runs, view->runs_n);
            g_free(view->runs);
            reims_pagealias.frees++;   /* TEMP lab (b610) */
            reims_pagealias.free_ns += get_clock() - t_free;
#else
            munmap(view->ptr, view->len);
#endif
        }
        s->page_view_live_bytes -= view->len;
        s->page_view_live_pages -= view->len >> REIMS_VGPU_GUEST_PAGE_SHIFT_X86_64;
        s->page_views_destroyed++;
        g_array_remove_index_fast(s->page_views, i);
        qemu_mutex_unlock(&s->page_views_lock);
        return;
    }
    qemu_mutex_unlock(&s->page_views_lock);
    /* A direct RAMBlock pointer is borrowed from QEMU and has no entry. */
}

static int reims_vgpu_pci_page_alias_census(
    void *ctx, ReimsVgpuPageAliasCensus *out)
{
    ReimsVGPUPCIState *s = ctx;

    if (!s || !s->page_views || !out) {
        return -1;
    }
    qemu_mutex_lock(&s->page_views_lock);
    *out = (ReimsVgpuPageAliasCensus) {
        .live = s->page_views->len,
        .live_bytes = s->page_view_live_bytes,
        .live_pages = s->page_view_live_pages,
        .created = s->page_views_created,
        .destroyed = s->page_views_destroyed,
    };
    qemu_mutex_unlock(&s->page_views_lock);
    return 0;
}

/*
 * Guest-write tracking. A surface's storage is plain guest RAM: the guest CPU
 * stores into it with no device operation, so nothing the Rust device counts
 * can witness such a store and every host-side copy of those pages is stale
 * from that instant. These three forward to the shared dirty-bitmap adapter.
 */
static uint64_t reims_vgpu_pci_track_guest_writes(void *ctx, const uint64_t *gpas,
                                                  size_t count, size_t page_size)
{
    ReimsVGPUPCIState *s = ctx;

    return reims_vgpu_dirty_track(s->dirty, gpas, count, page_size);
}

static void reims_vgpu_pci_untrack_guest_writes(void *ctx, uint64_t token)
{
    ReimsVGPUPCIState *s = ctx;

    reims_vgpu_dirty_untrack(s->dirty, token);
}

static uint64_t reims_vgpu_pci_guest_write_gen(void *ctx, uint64_t token)
{
    ReimsVGPUPCIState *s = ctx;

    return reims_vgpu_dirty_gen(s->dirty, token);
}

static int64_t reims_vgpu_pci_guest_written_pages(void *ctx, uint64_t token,
                                         uint64_t since_gen, uint64_t *out,
                                         size_t max)
{
    ReimsVGPUPCIState *s = ctx;

    return reims_vgpu_dirty_written_since(s->dirty, token, since_gen, out, max);
}

static void reims_vgpu_pci_bh(void *opaque);
static void reims_vgpu_pci_deliver_actions(ReimsVGPUPCIState *s);
static void reims_vgpu_pci_apply_action(ReimsVGPUPCIState *s, const ReimsVgpuHostAction *a);

static void reims_vgpu_pci_schedule_bh(void *ctx)
{
    ReimsVGPUPCIState *s = ctx;

    qemu_mutex_lock(&s->drain_mutex);
    s->drain_pending = true;
    qemu_cond_signal(&s->drain_cond);
    qemu_mutex_unlock(&s->drain_mutex);
}

/*
 * Prompt HostAction delivery (IRQ pulses / cursor moves): schedule the
 * action BH from any thread so acks reach the guest while the drain worker
 * is still executing a tranche. qemu_bh_schedule is thread-safe.
 */
static void reims_vgpu_pci_notify_actions(void *ctx)
{
    ReimsVGPUPCIState *s = ctx;

    if (s->action_bh) {
        qemu_bh_schedule(s->action_bh);
    }
}

static void reims_vgpu_pci_deliver_actions(ReimsVGPUPCIState *s)
{
    ReimsVgpuHostAction action;
    int rc;

    if (s->rust_handle == 0) {
        return;
    }
    while ((rc = reims_vgpu_qemu_device_pop_action(s->rust_handle, &action)) ==
           REIMS_VGPU_QEMU_OK) {
        reims_vgpu_pci_apply_action(s, &action);
    }
}

/* ---------- Console ---------- */

static void reims_vgpu_pci_set_mode(ReimsVGPUPCIState *s, uint32_t width,
                             uint32_t height)
{
    if (width == 0 || height == 0 ||
        width > REIMS_VGPU_MAX_SCANOUT_DIM || height > REIMS_VGPU_MAX_SCANOUT_DIM) {
        return;
    }
    if (s->surface &&
        surface_width(s->surface) == width &&
        surface_height(s->surface) == height) {
        return;
    }

    s->surface = qemu_create_displaysurface(width, height);
    if (s->con) {
        qemu_console_set_surface(s->con, s->surface);
    }
    trace_reims_vgpu_pci_mode_change(width, height);
}

/*
 * Copy BAR1 linear BGRA8 into the QEMU DisplaySurface (UEFI GOP / OpenCore).
 * Used when the Metal/product present path has not yet latched a front buffer.
 */
static bool reims_vgpu_pci_copy_gop_fb(ReimsVGPUPCIState *s)
{
    uint8_t *src;
    uint8_t *dst;
    uint32_t dst_stride;
    uint32_t src_stride;
    uint32_t w = REIMS_VGPU_EFI_BOOT_WIDTH;
    uint32_t h = REIMS_VGPU_EFI_BOOT_HEIGHT;
    uint32_t y;

    if (!s->surface || !memory_region_is_ram(&s->fb_vram)) {
        return false;
    }
    src = memory_region_get_ram_ptr(&s->fb_vram);
    if (!src) {
        return false;
    }
    if (surface_width(s->surface) != w || surface_height(s->surface) != h) {
        reims_vgpu_pci_set_mode(s, w, h);
        if (!s->surface) {
            return false;
        }
    }
    dst = surface_data(s->surface);
    dst_stride = surface_stride(s->surface);
    src_stride = w * REIMS_VGPU_PCI_EFI_BPP;
    if (dst_stride == src_stride) {
        memcpy(dst, src, (size_t)src_stride * h);
    } else {
        for (y = 0; y < h; y++) {
            memcpy(dst + (size_t)y * dst_stride,
                   src + (size_t)y * src_stride, src_stride);
        }
    }
    return true;
}

/*
 * Pre-boundary early console: prefer guest-programmed EFI FB (MMIO 0x1210)
 * when the kernel relocates the video console off BAR1 into guest RAM
 * (serial: "console relocated to 0xf1000000"). Fall back to BAR1 GOP.
 * Contract: efi_fb_start / stride from Apple EFI block — not serial scrape.
 */
static bool reims_vgpu_pci_copy_early_console(ReimsVGPUPCIState *s)
{
    uint8_t *dst;
    uint32_t dst_stride;
    uint32_t w = REIMS_VGPU_EFI_BOOT_WIDTH;
    uint32_t h = REIMS_VGPU_EFI_BOOT_HEIGHT;
    int rc;

    if (!s->surface) {
        reims_vgpu_pci_set_mode(s, w, h);
        if (!s->surface) {
            return false;
        }
    }
    if (surface_width(s->surface) != w || surface_height(s->surface) != h) {
        reims_vgpu_pci_set_mode(s, w, h);
        if (!s->surface) {
            return false;
        }
    }
    if (s->rust_handle != 0) {
        dst = surface_data(s->surface);
        dst_stride = surface_stride(s->surface);
        rc = reims_vgpu_qemu_efi_console_copy(s->rust_handle, dst, dst_stride, w, h);
        if (rc == REIMS_VGPU_QEMU_OK) {
            return true;
        }
    }
    return reims_vgpu_pci_copy_gop_fb(s);
}

/*
 * Paint a named guest mapping into the QEMU surface (pre- or post-boundary).
 * Pre-boundary early writebacks and post-boundary DisplaySwap both use this
 * path. Return true when the surface was updated.
 */
static bool reims_vgpu_pci_paint_scanout(ReimsVGPUPCIState *s, uint32_t mapping_id,
                                  uint32_t width, uint32_t height,
                                  uint32_t generation)
{
    uint8_t *dst;
    uint32_t stride;
    int rc;

    if (!s->surface || s->rust_handle == 0) {
        return false;
    }
    if (width == 0 || height == 0) {
        return false;
    }
    if (surface_width(s->surface) != width ||
        surface_height(s->surface) != height) {
        reims_vgpu_pci_set_mode(s, width, height);
        if (!s->surface) {
            return false;
        }
    }
    dst = surface_data(s->surface);
    stride = surface_stride(s->surface);
    rc = reims_vgpu_qemu_scanout_copy(s->rust_handle, mapping_id, dst, stride,
                               width, height, generation);
    if (rc == REIMS_VGPU_QEMU_OK) {
        s->new_frame_ready = true;
        trace_reims_vgpu_pci_scanout(mapping_id, width, height);
        /* Push retain immediately — waiting only for the next gfx_update can
         * leave a black surface if a later EMPTY/Unchanged path races. */
        if (s->con) {
            qemu_console_update_full(s->con);
            s->new_frame_ready = false;
        }
        return true;
    }
    return false;
}

static void reims_vgpu_pci_apply_scanout(ReimsVGPUPCIState *s, uint32_t mapping_id,
                                  uint32_t width, uint32_t height,
                                  uint32_t generation)
{
    if (!reims_vgpu_shim_scanout_may_paint(s->rust_handle, mapping_id)) {
        return;
    }
    (void)reims_vgpu_pci_paint_scanout(s, mapping_id, width, height, generation);
}

/*
 * Whether the host-owned window (kb host-window) is requested via REIMS_VGPU_WINDOW.
 * Presence enables it; 0/off/no/false explicitly disable so REIMS_VGPU_WINDOW=0 opts
 * out (vm/boot-x86.sh defaults it on for this device, unset to disable). Env
 * plumbing only — the window itself lives entirely in the staticlib.
 */
static bool reims_vgpu_pci_window_requested(void)
{
    const char *v = getenv("REIMS_VGPU_WINDOW");

    if (!v || v[0] == '\0') {
        return false;
    }
    if (!strcmp(v, "0") || !strcasecmp(v, "off") || !strcasecmp(v, "no") ||
        !strcasecmp(v, "false")) {
        return false;
    }
    return true;
}

static void reims_vgpu_pci_apply_action(ReimsVGPUPCIState *s, const ReimsVgpuHostAction *a)
{
    PCIDevice *pdev = PCI_DEVICE(s);

    switch (a->kind) {
    case REIMS_VGPU_HOST_ACTION_IRQ_GFX:
        if (msi_enabled(pdev)) {
            msi_notify(pdev, 0);
        }
        trace_reims_vgpu_pci_irq_gfx();
        break;
    case REIMS_VGPU_HOST_ACTION_IRQ_IOSFC:
        /* Single MSI vector; pulse same vector (guest demuxes by status). */
        if (msi_enabled(pdev)) {
            msi_notify(pdev, 0);
        }
        break;
    case REIMS_VGPU_HOST_ACTION_SCANOUT:
        reims_vgpu_pci_apply_scanout(s, (uint32_t)a->a0, (uint32_t)a->a1,
                              (uint32_t)a->a2, (uint32_t)a->a3);
        break;
    case REIMS_VGPU_HOST_ACTION_CURSOR:
        if (s->con) {
            qemu_console_set_mouse(s->con, (int)a->a0, (int)a->a1, a->a2 != 0);
        }
        break;
    case REIMS_VGPU_HOST_ACTION_INPUT_KEY:
        reims_vgpu_shim_input_key(s->con, (uint32_t)a->a0, a->a1 != 0);
        break;
    case REIMS_VGPU_HOST_ACTION_INPUT_POINTER_MOVE:
        reims_vgpu_shim_input_pointer_move(s->con, (uint32_t)a->a0, (uint32_t)a->a1,
                                   (uint32_t)a->a2, (uint32_t)a->a3);
        break;
    case REIMS_VGPU_HOST_ACTION_INPUT_POINTER_BUTTON:
        reims_vgpu_shim_input_button(s->con, (uint32_t)a->a0, a->a1 != 0);
        break;
    case REIMS_VGPU_HOST_ACTION_WINDOW_CLOSED:
        /* The host window is the VM's display; closing it shuts the VM down. */
        qemu_system_shutdown_request(SHUTDOWN_CAUSE_HOST_UI);
        break;
    case REIMS_VGPU_HOST_ACTION_CURSOR_GLYPH: {
        ReimsVgpuCursorGlyphInfo info;
        g_autofree uint32_t *pixels = NULL;
        QEMUCursor *c;
        int rc;

        if (!s->con || s->rust_handle == 0) {
            break;
        }
        rc = reims_vgpu_qemu_cursor_glyph_info(s->rust_handle, &info);
        if (rc != REIMS_VGPU_QEMU_OK || info.width == 0 || info.height == 0 ||
            info.pixel_count == 0 ||
            info.pixel_count != info.width * info.height) {
            break;
        }
        pixels = g_new(uint32_t, info.pixel_count);
        rc = reims_vgpu_qemu_cursor_glyph_copy(s->rust_handle, pixels,
                                        info.pixel_count);
        if (rc != REIMS_VGPU_QEMU_OK) {
            break;
        }
        c = cursor_alloc(info.width, info.height);
        if (!c) {
            break;
        }
        c->hot_x = info.hot_x;
        c->hot_y = info.hot_y;
        memcpy(c->data, pixels, (size_t)info.pixel_count * sizeof(uint32_t));
        qemu_console_set_cursor(s->con, c);
        cursor_unref(c);
        break;
    }
    case REIMS_VGPU_HOST_ACTION_TRACE:
    case REIMS_VGPU_HOST_ACTION_NONE:
    default:
        break;
    }
}

static void reims_vgpu_pci_bh(void *opaque)
{
    ReimsVGPUPCIState *s = opaque;

    if (s->rust_handle == 0) {
        return;
    }
    reims_vgpu_pci_deliver_actions(s);
}

/*
 * Run the dirty-log harvest a doorbell asked for, under the BQL, then release
 * the drain. Holds only the BQL and the tracker's own lock — never the device —
 * so the drain waiting on it cannot close a cycle.
 */
/* TEMP lab DIRTY_PEEK: one pass over the child rings' tails. */
static void reims_vgpu_pci_peek_tails(ReimsVGPUPCIState *s)
{
    bool want = false;
    uint32_t ch;

    for (ch = 1; ch < 16; ch++) {
        uint64_t gpa = reims_vgpu_qemu_child_tail_gpa(s->rust_handle, ch);
        uint32_t tail;

        if (!gpa || address_space_read(&address_space_memory, gpa, MEMTXATTRS_UNSPECIFIED,
                                       &tail, 4) != MEMTX_OK) {
            continue;
        }
        tail = le32_to_cpu(tail);
        if (!reims_vgpu_dirty_tail_unnoted(s->dirty, ch, tail)) {
            continue;
        }
        /* The doorbell's order: the tail first, then the epoch that covers it. */
        want |= reims_vgpu_dirty_note_doorbell(s->dirty, (int)ch);
        reims_vgpu_dirty_note_child_tail(s->dirty, (int)ch, tail, true);
        qatomic_inc(&s->lab_peeks);
    }
    if (want && s->harvest_started) {
        qemu_mutex_lock(&s->harvest_mutex);
        s->harvest_asked++;
        qemu_cond_signal(&s->harvest_cond);
        qemu_mutex_unlock(&s->harvest_mutex);
    }
}

static void *reims_vgpu_pci_peek_thread(void *opaque)
{
    ReimsVGPUPCIState *s = opaque;
    int64_t last = get_clock();

    reims_vgpu_lab_device_prio("peek");

    rcu_register_thread();
    qemu_mutex_lock(&s->peek_mutex);
    while (!s->peek_stopping) {
        qemu_cond_timedwait(&s->peek_cond, &s->peek_mutex, s->peek_ms);
        if (s->peek_stopping) {
            break;
        }
        qemu_mutex_unlock(&s->peek_mutex);
        if (s->rust_handle != 0) {
            reims_vgpu_pci_peek_tails(s);
        }
        if (get_clock() - last >= NANOSECONDS_PER_SECOND * 5) {
            fprintf(stderr, "reims-peek peeks=%" PRIu64 "\n", qatomic_xchg(&s->lab_peeks, 0));
            last = get_clock();
        }
        qemu_mutex_lock(&s->peek_mutex);
    }
    qemu_mutex_unlock(&s->peek_mutex);
    rcu_unregister_thread();
    return NULL;
}

static void *reims_vgpu_pci_harvest_thread(void *opaque)
{
    ReimsVGPUPCIState *s = opaque;

    reims_vgpu_lab_device_prio("harvest");

    /*
     * This thread reads the FlatView under RCU (the dirty tracker's RAM walk,
     * address-space lookups through HostOps); an unregistered reader is not
     * waited for by synchronize_rcu() and can see a freed view.
     */
    rcu_register_thread();
    for (;;) {
        uint64_t target;

        qemu_mutex_lock(&s->harvest_mutex);
        while (s->harvest_done >= s->harvest_asked && !s->harvest_stopping) {
            qemu_cond_wait(&s->harvest_cond, &s->harvest_mutex);
        }
        if (s->harvest_stopping) {
            qemu_mutex_unlock(&s->harvest_mutex);
            break;
        }
        target = s->harvest_asked;
        qemu_mutex_unlock(&s->harvest_mutex);

        {
            int64_t t0 = get_clock(), dt;

            if (s->harvest_nobql) {
                /* RCU and atomic bitmap operations only; the dirty tracker takes
                 * the BQL itself for the one MemoryRegion transaction it may
                 * need. Held for the whole walk, the BQL stalled every vCPU exit
                 * that needed it for ~6 ms a harvest, ~75 harvests a second.
                 * In on-demand mode this is the prefetch of the hot sets. */
                reims_vgpu_dirty_background(s->dirty);
            } else {
                bql_lock();
                reims_vgpu_dirty_harvest(s->dirty);
                bql_unlock();
            }
            dt = get_clock() - t0;
            qemu_mutex_lock(&s->harvest_mutex);
            s->lab_hv_n++;
            s->lab_hv_ns += dt;
            s->lab_hv_max_ns = MAX(s->lab_hv_max_ns, (uint64_t)dt);
            qemu_mutex_unlock(&s->harvest_mutex);
        }

        qemu_mutex_lock(&s->harvest_mutex);
        s->harvest_done = target;
        qemu_cond_broadcast(&s->harvest_done_cond);
        if (s->harvest_rearm && s->harvest_done >= s->harvest_asked) {
            /* The wakeup an unsettled answer promised; see harvests_settled. */
            s->harvest_rearm = false;
            qemu_mutex_unlock(&s->harvest_mutex);
            reims_vgpu_pci_schedule_bh(s);
            qemu_mutex_lock(&s->harvest_mutex);
        }
        qemu_mutex_unlock(&s->harvest_mutex);
    }
    rcu_unregister_thread();
    return NULL;
}

/*
 * HostOps.irq_pulse: raise the device's MSI from the calling thread, which may
 * be the drain holding the device lock without the BQL.
 *
 * Only where the delivery takes no lock at all. Under WHPX with the in-kernel
 * irqchip, msi_notify ends in a write to the WHPX APIC's MSI region, which is
 * lockless and injects with WHvRequestInterrupt; anywhere else the MMIO
 * dispatch would take the BQL inside this call, and a caller holding the device
 * lock would then deadlock against a vCPU that holds the BQL and waits for it.
 * Those hosts answer 0 and the pulse is queued for the action BH as before.
 * No per-vector masking is advertised (msi_init above), so msi_notify never
 * touches pending bits here.
 */
/* HostOps.note_ring_tail: the drain read a child ring's tail (DIRTY_HOLE=tail). */
static void reims_vgpu_pci_note_ring_tail(void *ctx, uint32_t channel, uint32_t tail)
{
    ReimsVGPUPCIState *s = ctx;

    if (s->dirty_ondemand && reims_vgpu_dirty_note_ring_tail(s->dirty, channel, tail) &&
        s->harvest_started) {
        /* Lab DIRTY_TAILWAKE: the epoch the drain just opened, prefetched as a doorbell's is. */
        qemu_mutex_lock(&s->harvest_mutex);
        s->harvest_asked++;
        qemu_cond_signal(&s->harvest_cond);
        qemu_mutex_unlock(&s->harvest_mutex);
    }
}

/* HostOps.guest_write_sync_many (TEMP lab PREP_SYNC): one merged sync of the named sets. */
static void reims_vgpu_pci_guest_write_sync_many(void *ctx, const uint64_t *tokens, size_t count)
{
    ReimsVGPUPCIState *s = ctx;

    if (s->dirty_ondemand) {
        reims_vgpu_dirty_sync_tokens(s->dirty, tokens, count);
    }
}

/* HostOps.guest_write_sync_ahead (TEMP lab SYNC_AHEAD): queue the named sets for the workers. */
static void reims_vgpu_pci_guest_write_sync_ahead(void *ctx, const uint64_t *tokens, size_t count)
{
    ReimsVGPUPCIState *s = ctx;

    if (s->dirty_ondemand) {
        reims_vgpu_dirty_sync_ahead(s->dirty, tokens, count);
    }
}

/* HostOps.guest_write_rebase: the device's own stores into a set's pages (a hashed set re-takes them). */
static void reims_vgpu_pci_guest_write_rebase(void *ctx, uint64_t token, const uint64_t *gpas,
                                              size_t count)
{
    ReimsVGPUPCIState *s = ctx;

    if (s->dirty) {
        reims_vgpu_dirty_rebase(s->dirty, token, gpas, count);
    }
}

/* HostOps.note_work_scope: which doorbells bind the reads that follow. */
static void reims_vgpu_pci_note_work_scope(void *ctx, uint32_t scope)
{
    ReimsVGPUPCIState *s = ctx;

    if (s->dirty_ondemand) {
        reims_vgpu_dirty_note_work_scope(s->dirty, scope);
    }
}

static int reims_vgpu_pci_irq_pulse(void *ctx, uint32_t kind)
{
    ReimsVGPUPCIState *s = ctx;
    PCIDevice *pdev = PCI_DEVICE(s);

    if (!s->irq_direct || !whpx_enabled() || !whpx_irqchip_in_kernel() ||
        !msi_enabled(pdev)) {
        return 0;
    }
    if (kind != REIMS_VGPU_HOST_ACTION_IRQ_GFX &&
        kind != REIMS_VGPU_HOST_ACTION_IRQ_IOSFC) {
        return 0;
    }
    msi_notify(pdev, 0);
    return 1;
}

/* HostOps.harvests_settled: never blocks; the harvest mutex is a leaf lock. */
static int reims_vgpu_pci_harvests_settled(void *ctx)
{
    ReimsVGPUPCIState *s = ctx;
    int settled;

    if (!s->harvest_started || s->dirty_ondemand) {
        /* On demand, every generation read syncs its own set: nothing to wait for. */
        return 1;
    }
    qemu_mutex_lock(&s->harvest_mutex);
    settled = s->harvest_done >= s->harvest_asked;
    if (!settled) {
        /* The write behind this may not have been a doorbell: nothing else
         * would wake the drain for the work it leaves. */
        s->harvest_rearm = true;
    }
    qemu_mutex_unlock(&s->harvest_mutex);
    return settled;
}

/* Every harvest a doorbell asked for before this call has finished. */
static void reims_vgpu_pci_wait_harvest(ReimsVGPUPCIState *s)
{
    uint64_t target;

    if (!s->harvest_started || s->dirty_ondemand) {
        return;
    }
    qemu_mutex_lock(&s->harvest_mutex);
    target = s->harvest_asked;
    if (s->harvest_done < target) {
        int64_t t0 = get_clock(), dt;

        while (s->harvest_done < target && !s->harvest_stopping) {
            qemu_cond_wait(&s->harvest_done_cond, &s->harvest_mutex);
        }
        dt = get_clock() - t0;
        s->lab_wait_n++;
        s->lab_wait_ns += dt;
        s->lab_wait_max_ns = MAX(s->lab_wait_max_ns, (uint64_t)dt);
    }
    {
        int64_t now = get_clock();

        if (now - s->lab_hv_last >= NANOSECONDS_PER_SECOND) {
            fprintf(stderr, "reims-harvest epoch_ms=%" PRId64 " harvests=%" PRIu64
                    " harvest_us=%" PRIu64 " harvest_max_us=%" PRIu64
                    " waits=%" PRIu64 " wait_us=%" PRIu64 " wait_max_us=%" PRIu64
                    " nobql=%d\n", g_get_real_time() / 1000, s->lab_hv_n,
                    s->lab_hv_ns / 1000, s->lab_hv_max_ns / 1000, s->lab_wait_n,
                    s->lab_wait_ns / 1000, s->lab_wait_max_ns / 1000,
                    s->harvest_nobql);
            s->lab_hv_n = s->lab_hv_ns = s->lab_hv_max_ns = 0;
            s->lab_wait_n = s->lab_wait_ns = s->lab_wait_max_ns = 0;
            s->lab_hv_last = now;
        }
    }
    qemu_mutex_unlock(&s->harvest_mutex);
}

/*
 * TEMP lab (REIMS_VGPU_LAB_HIQOS=on): opt this process out of Windows power
 * throttling (EcoQoS: E-cores at low clocks, coarse timers) and give the drain
 * thread high QoS and above-normal priority. The drain is the device's serial
 * critical path; QEMU launched from a script owns no foreground window, which
 * is what the scheduler throttles on a hybrid (P/E core) CPU. Resolved at run
 * time so the build does not depend on the SDK's Windows-version guards.
 */
#if defined(_WIN32)
static bool reims_vgpu_lab_hiqos(void)
{
    static int on = -1;

    if (on < 0) {
        const char *e = getenv("REIMS_VGPU_LAB_HIQOS");

        on = e && strcmp(e, "on") == 0;
    }
    return on;
}

typedef struct ReimsVgpuPowerThrottling {
    ULONG Version;
    ULONG ControlMask;
    ULONG StateMask;
} ReimsVgpuPowerThrottling;

static void reims_vgpu_hiqos_apply(bool process)
{
    typedef BOOL (WINAPI *SetInfoFn)(HANDLE, int, LPVOID, DWORD);
    /* ProcessPowerThrottling = 4, ThreadPowerThrottling = 3 (processthreadsapi.h) */
    ReimsVgpuPowerThrottling st = { 1, process ? 0x5u : 0x1u, 0 };
    HMODULE k32 = GetModuleHandleA("kernel32.dll");
    SetInfoFn fn;
    BOOL ok = FALSE, prio = TRUE;

    if (!reims_vgpu_lab_hiqos() || !k32) {
        return;
    }
    fn = (SetInfoFn)(void *)GetProcAddress(k32, process ? "SetProcessInformation"
                                                        : "SetThreadInformation");
    if (fn) {
        ok = fn(process ? GetCurrentProcess() : GetCurrentThread(), process ? 4 : 3,
                &st, sizeof(st));
    }
    if (!process) {
        prio = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    }
    fprintf(stderr, "reims-vgpu-hiqos %s throttling_off=%d priority=%d\n",
            process ? "process" : "drain", ok ? 1 : 0, prio ? 1 : 0);
}
#endif

static void *reims_vgpu_pci_drain_thread(void *opaque)
{
    ReimsVGPUPCIState *s = opaque;

    reims_vgpu_lab_device_prio("drain");

#if defined(_WIN32)
    reims_vgpu_hiqos_apply(false);
    whpx_lab_pin_pcores("drain");
    whpx_lab_excl_core_drain();
#endif

    /*
     * This thread reads the FlatView under RCU (the dirty tracker's RAM walk,
     * address-space lookups through HostOps); an unregistered reader is not
     * waited for by synchronize_rcu() and can see a freed view.
     */
    rcu_register_thread();
    for (;;) {
        int rc;

        qemu_mutex_lock(&s->drain_mutex);
        while (!s->drain_pending && !s->drain_stopping) {
            qemu_cond_wait(&s->drain_cond, &s->drain_mutex);
        }
        if (s->drain_stopping) {
            qemu_mutex_unlock(&s->drain_mutex);
            break;
        }
        s->drain_pending = false;
        qemu_mutex_unlock(&s->drain_mutex);

        reims_vgpu_pci_wait_harvest(s);
        rc = reims_vgpu_qemu_device_drain(s->rust_handle);
        if (rc != REIMS_VGPU_QEMU_OK) {
            qemu_log_mask(LOG_GUEST_ERROR, "%s: worker drain failed rc=%d\n",
                          TYPE_REIMS_VGPU_PCI, rc);
        }
        qemu_bh_schedule(s->action_bh);
    }
    rcu_unregister_thread();
    return NULL;
}

/*
 * Oversample the Rust-owned VBL limiter (DISPLAY_VBL_MIN_INTERVAL_MS = 8 ms for
 * the 120 Hz advertised mode). Polling at 4 ms guarantees the 8 ms limiter is
 * the gate even when the main loop stalls under input/QMP/display work. This
 * thread only supplies poll opportunities and schedules the existing main-loop
 * BH; Rust owns pacing and protocol state, while the BH remains the sole
 * HostAction applier.
 */
/*
 * 1 ms, not 4: the limiter's catch-up grid only phase-locks when polls land
 * well inside the 8.33 ms interval, and a Windows host's 4 ms condvar waits
 * (plus the poll's own maintenance) drifted past it often enough that CSS
 * animation sat at ~95 fps median; at 1 ms it reads ~106-112 on the same host.
 */
#define REIMS_VGPU_PCI_HEARTBEAT_MS 1

#ifdef _WIN32
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
/*
 * TEMP (lab): REIMS_VGPU_LAB_HB_US=N (100..2000) polls every N us on a
 * high-resolution waitable timer instead of a 1 ms condvar wait. A condvar
 * timeout on Windows lands anywhere in 1-2 ms, and a VBL is claimed only on
 * the first poll past its grid point, so the guest's frame start inherits up
 * to ~2 ms of that jitter (lab_vbl_late). Returns false, having polled
 * nothing, when the switch is off or the timer cannot be created; the caller
 * then runs the condvar loop.
 */
static bool reims_vgpu_pci_heartbeat_hires(ReimsVGPUPCIState *s)
{
    const char *v = getenv("REIMS_VGPU_LAB_HB_US");
    int us = v ? atoi(v) : 0;
    HANDLE t;

    if (us < 100 || us > 2000) {
        return false;
    }
    t = CreateWaitableTimerExW(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                               TIMER_ALL_ACCESS);
    if (!t) {
        return false;
    }
    /*
     * REIMS_VGPU_LAB_HB_GRID=1: when the next VBL grid point is closer than
     * the period, sleep to ~150 us before it and spin the rest, so the poll
     * that claims it lands on the grid rather than up to a period late.
     */
    const char *g = getenv("REIMS_VGPU_LAB_HB_GRID");
    bool grid = g && g[0] == '1';

    warn_report("reims-vgpu heartbeat: high-resolution timer every %d us grid=%d", us, grid);
    while (!qatomic_read(&s->heartbeat_stopping)) {
        LARGE_INTEGER due;
        uint32_t vbl_in = 0;
        int wait_us = us;

        if (grid && s->rust_handle != 0) {
            vbl_in = reims_vgpu_qemu_vbl_due_us(s->rust_handle);
            if (vbl_in > 0 && vbl_in <= (uint32_t)us) {
                wait_us = (int)vbl_in - 150;
            }
        }
        if (wait_us > 0) {
            due.QuadPart = -(LONGLONG)wait_us * 10;   /* relative, 100 ns units */
            if (!SetWaitableTimerEx(t, &due, 0, NULL, NULL, NULL, 0)) {
                Sleep(1);
            } else {
                WaitForSingleObject(t, INFINITE);
            }
        }
        if (grid && vbl_in > 0 && vbl_in <= (uint32_t)us && s->rust_handle != 0) {
            /* Spin the last stretch: a timer wake is ~0.1-0.5 ms imprecise. */
            int spins = 0;

            while (reims_vgpu_qemu_vbl_due_us(s->rust_handle) > 0 && spins < 200000) {
                YieldProcessor();
                spins++;
            }
        }
        if (qatomic_read(&s->heartbeat_stopping)) {
            break;
        }
        if (s->rust_handle != 0 &&
            reims_vgpu_qemu_device_poll(s->rust_handle) == REIMS_VGPU_QEMU_OK &&
            s->action_bh) {
            qemu_bh_schedule(s->action_bh);
        }
    }
    CloseHandle(t);
    return true;
}
#endif

static void *reims_vgpu_pci_heartbeat_thread(void *opaque)
{
    ReimsVGPUPCIState *s = opaque;

    reims_vgpu_lab_device_prio("heartbeat");

    int period_ms = REIMS_VGPU_PCI_HEARTBEAT_MS;
    const char *hb = getenv("REIMS_VGPU_HEARTBEAT_MS");   /* lab A/B */

#ifdef _WIN32
    if (reims_vgpu_pci_heartbeat_hires(s)) {
        return NULL;
    }
#endif
    if (hb && atoi(hb) >= 1 && atoi(hb) <= 16) {
        period_ms = atoi(hb);
    }
    qemu_mutex_lock(&s->heartbeat_mutex);
    while (!s->heartbeat_stopping) {
        qemu_cond_timedwait(&s->heartbeat_cond, &s->heartbeat_mutex,
                            period_ms);
        if (s->heartbeat_stopping) {
            break;
        }
        qemu_mutex_unlock(&s->heartbeat_mutex);

        if (s->rust_handle != 0 &&
            reims_vgpu_qemu_device_poll(s->rust_handle) == REIMS_VGPU_QEMU_OK &&
            s->action_bh) {
            qemu_bh_schedule(s->action_bh);
        }

        qemu_mutex_lock(&s->heartbeat_mutex);
    }
    qemu_mutex_unlock(&s->heartbeat_mutex);
    return NULL;
}

static bool reims_vgpu_pci_fb_update(void *opaque)
{
    ReimsVGPUPCIState *s = opaque;
    uint32_t kind, mid = 0, w = 0, h = 0, gen = 0;

    if (!s->con) {
        return true;
    }

    if (s->rust_handle != 0 &&
        reims_vgpu_qemu_device_poll(s->rust_handle) == REIMS_VGPU_QEMU_OK) {
        reims_vgpu_pci_deliver_actions(s);
    }

    /* Host-console ownership is Rust's call; this only paints what it names. */
    kind = reims_vgpu_shim_console_feed(s->rust_handle, &mid, &w, &h, &gen);

    if (kind == REIMS_VGPU_CONSOLE_FEED_EARLY) {
        /* Re-pull the latched front (archive fb_update early path).
         *
         * Return either way, for the same reason the firmware arm below states:
         * a failed paint must NOT fall through to the re-push, whose pending
         * frame is a product one. _EARLY means Rust says the early console owns
         * the screen, so pushing a product frame here is the same pre-boundary
         * steal, and this arm used to do it while the arm64 shim returned.
         *
         * No console update here: unlike the arm64 shim's, this
         * `paint_scanout` already calls `qemu_console_update_full` and clears
         * `new_frame_ready` itself on the path that returns true. */
        (void)reims_vgpu_pci_paint_scanout(s, mid, w, h, gen);
        return true;
    } else if (kind == REIMS_VGPU_CONSOLE_FEED_FIRMWARE) {
        if (reims_vgpu_pci_copy_early_console(s)) {
            qemu_console_update_full(s->con);
            s->new_frame_ready = false;
        }
        /* Return either way. A failed firmware copy must NOT fall through to
         * the re-push below: the pending frame there is a product one, and
         * pushing it while Rust says the firmware console owns the screen is
         * the pre-boundary steal this feed exists to prevent. */
        return true;
    }

    /* Nothing painted this tick — re-push the last frame if one is pending.
     * _PRODUCT reaches here every tick: apply_scanout does that painting. */
    if (s->new_frame_ready && s->surface) {
        qemu_console_update_full(s->con);
        s->new_frame_ready = false;
    }
    return true;
}

static const GraphicHwOps reims_vgpu_pci_fb_ops = {
    .gfx_update = reims_vgpu_pci_fb_update,
};

/* ---------- MMIO (forward only) ---------- */

/*
 * Runs without the BQL: the region is lockless (see realize), and a read only
 * asks Rust, whose gfx read path is lock-free for the ISR/spin registers and
 * try-locks the device otherwise. Nothing here touches QEMU device state.
 */
static uint64_t reims_vgpu_pci_gfx_read(void *opaque, hwaddr offset, unsigned size)
{
    ReimsVGPUPCIState *s = opaque;
    uint64_t val = 0;

    if (s->rust_handle == 0) {
        return 0;
    }
    if (reims_vgpu_qemu_gfx_read(s->rust_handle, offset, size, &val) != REIMS_VGPU_QEMU_OK) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: gfx read failed offset=0x%" HWADDR_PRIx " size=%u\n",
                      TYPE_REIMS_VGPU_PCI, offset, size);
        return 0;
    }
    trace_reims_vgpu_pci_gfx_read(offset, val);
    return val;
}

static void reims_vgpu_pci_gfx_write(void *opaque, hwaddr offset, uint64_t data,
                              unsigned size)
{
    ReimsVGPUPCIState *s = opaque;

    /*
     * Writes keep the BQL: the dirty harvest syncs the accelerator's dirty
     * log and action delivery raises interrupts, both main-loop state. The
     * region is lockless for the reads' sake, so take it here when the
     * dispatcher did not.
     */
    BQL_LOCK_GUARD();

    if (s->rust_handle == 0) {
        return;
    }
    trace_reims_vgpu_pci_gfx_write(offset, data);
    {
        /* TEMP lab: which gfx registers the guest writes, one line a second. */
        static uint64_t off_seen[16], off_n[16];
        static int64_t last;
        int64_t now = get_clock();
        int k;

        /* Child doorbells keyed by channel: 0x1020_00cc. */
        uint64_t key = offset == 0x1020 ? (0x10200000u | (data & 0xff)) : offset;

        for (k = 0; k < 16; k++) {
            if (off_n[k] == 0 || off_seen[k] == key) {
                off_seen[k] = key;
                off_n[k]++;
                break;
            }
        }
        if (now - last >= NANOSECONDS_PER_SECOND) {
            char buf[512];
            int len = 0;

            for (k = 0; k < 16 && off_n[k]; k++) {
                len += snprintf(buf + len, sizeof(buf) - len, " 0x%" PRIx64 "=%" PRIu64,
                                off_seen[k], off_n[k]);
                off_n[k] = 0;
            }
            fprintf(stderr, "reims-gfxw epoch_ms=%" PRId64 "%s\n",
                    g_get_real_time() / 1000, len ? buf : "");
            last = now;
        }
    }
    /*
     * Before the register write, not after: this is the guest handing the
     * device work, so every guest store ordered before the handoff must be
     * observed before anything that work does can reuse a host-side copy of
     * those pages. Harvesting here is also the only place it can happen — the
     * accelerator's dirty-log sync needs the BQL, which a vCPU MMIO write
     * holds and the drain thread must never take.
     *
     * Cheap when nothing is tracked or when nothing has read a generation
     * since the last harvest, so a burst of register writes costs one sync.
     */
    if (s->dirty_ondemand) {
        /*
         * Before the register write, as below: the epoch this bumps is what
         * makes the next generation read of every set sync it first. The
         * background harvest is asked only while logging is still to be turned
         * on — its one BQL step — and nothing waits for it.
         */
        int db_channel = reims_vgpu_qemu_gfx_write_doorbell_channel(offset, data, size);
        uint32_t db_tail = 0;
        bool db_tail_ok = false;

        /*
         * A child ring's tail lives in guest memory, so the drain can read
         * packets a doorbell has not been noted for. Read the tail *before*
         * the doorbell opens its epoch: every packet up to it was published
         * before that, so the epoch covers them (DIRTY_HOLE=tail). Rust names
         * the register; 0 is a ring it has not set up yet.
         */
        if (db_channel > 0) {
            uint64_t tail_gpa = reims_vgpu_qemu_child_tail_gpa(s->rust_handle,
                                                               (uint32_t)db_channel);

            if (tail_gpa &&
                address_space_read(&address_space_memory, tail_gpa,
                                   MEMTXATTRS_UNSPECIFIED, &db_tail, 4) == MEMTX_OK) {
                db_tail = le32_to_cpu(db_tail);
                db_tail_ok = true;
            }
        }
        bool db_want = reims_vgpu_dirty_note_doorbell(s->dirty, db_channel);

        if (db_channel > 0) {
            reims_vgpu_dirty_note_child_tail(s->dirty, db_channel, db_tail, db_tail_ok);
        }
        if (db_want && s->harvest_started) {
            qemu_mutex_lock(&s->harvest_mutex);
            s->harvest_asked++;
            qemu_cond_signal(&s->harvest_cond);
            qemu_mutex_unlock(&s->harvest_mutex);
        }
    } else if (s->harvest_started) {
        /* Asked before the write, so the drain the write wakes waits for it. */
        qemu_mutex_lock(&s->harvest_mutex);
        s->harvest_asked++;
        qemu_cond_signal(&s->harvest_cond);
        qemu_mutex_unlock(&s->harvest_mutex);
    } else {
        reims_vgpu_dirty_harvest(s->dirty);
    }
    if (reims_vgpu_qemu_gfx_write(s->rust_handle, offset, data, size) != REIMS_VGPU_QEMU_OK) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: gfx write failed offset=0x%" HWADDR_PRIx
                      " data=0x%" PRIx64 " size=%u\n",
                      TYPE_REIMS_VGPU_PCI, offset, data, size);
        return;
    }
    /*
     * Deliver residual HostActions on the MMIO path, but do not run heavy
     * GPU work under BQL here — Rust schedules BH for pure-host work after
     * snapshotting guest inputs.
     */
    reims_vgpu_pci_deliver_actions(s);
}

static const MemoryRegionOps reims_vgpu_pci_gfx_ops = {
    .read = reims_vgpu_pci_gfx_read,
    .write = reims_vgpu_pci_gfx_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = {
        .min_access_size = 4,
        .max_access_size = 8,
    },
    .impl = {
        .min_access_size = 4,
        .max_access_size = 8,
    },
};

/* ---------- Lifecycle ---------- */

/*
 * Stop and destroy the Rust/Vulkan backend before QEMU enters process/driver
 * teardown.  PCI exit is not guaranteed to run on the main-loop shutdown
 * path: a guest panic previously left reims-vgpu-pci-drain inside Vulkan execution
 * while libc exit unloaded NVIDIA EGL, crashing QEMU in libnvidia-eglcore.
 */
static void reims_vgpu_pci_stop_backend(ReimsVGPUPCIState *s)
{
    if (s->heartbeat_started) {
        qemu_mutex_lock(&s->heartbeat_mutex);
        s->heartbeat_stopping = true;
        qemu_cond_signal(&s->heartbeat_cond);
        qemu_mutex_unlock(&s->heartbeat_mutex);
        qemu_thread_join(&s->heartbeat_thread);
        s->heartbeat_started = false;
    }
    if (s->peek_started) {   /* TEMP lab DIRTY_PEEK */
        qemu_mutex_lock(&s->peek_mutex);
        s->peek_stopping = true;
        qemu_cond_signal(&s->peek_cond);
        qemu_mutex_unlock(&s->peek_mutex);
        qemu_thread_join(&s->peek_thread);
        s->peek_started = false;
    }
    if (s->harvest_started) {
        qemu_mutex_lock(&s->harvest_mutex);
        s->harvest_stopping = true;
        qemu_cond_broadcast(&s->harvest_cond);
        qemu_cond_broadcast(&s->harvest_done_cond);
        qemu_mutex_unlock(&s->harvest_mutex);
    }
    if (s->drain_started) {
        qemu_mutex_lock(&s->drain_mutex);
        s->drain_stopping = true;
        qemu_cond_signal(&s->drain_cond);
        qemu_mutex_unlock(&s->drain_mutex);
        qemu_thread_join(&s->drain_thread);
        s->drain_started = false;
    }
    if (s->harvest_started) {
        bool locked = bql_locked();

        /* The thread may be waiting for the BQL this caller holds. */
        if (locked) {
            bql_unlock();
        }
        qemu_thread_join(&s->harvest_thread);
        if (locked) {
            bql_lock();
        }
        s->harvest_started = false;
    }
    if (s->action_bh) {
        qemu_bh_delete(s->action_bh);
        s->action_bh = NULL;
    }
    if (s->rust_handle != 0) {
        /*
         * Close + join the host window first (no-op if none): its Vulkan
         * objects tear down on the window thread before we proceed, so the
         * process/driver-unload teardown never races live Vulkan work (the
         * libnvidia-eglcore crash class the drain teardown already guards).
         */
        reims_vgpu_qemu_window_stop(s->rust_handle);
        reims_vgpu_qemu_device_destroy(s->rust_handle);
        s->rust_handle = 0;
    }
}

static void reims_vgpu_pci_shutdown_notifier(Notifier *n, void *data)
{
    ReimsVGPUPCIState *s = container_of(n, ReimsVGPUPCIState,
                                         shutdown_notifier);

    (void)data;
    reims_vgpu_pci_stop_backend(s);
}

static void reims_vgpu_pci_realize(PCIDevice *pdev, Error **errp)
{
    ReimsVGPUPCIState *s = REIMS_VGPU_PCI(pdev);
    ReimsVgpuQemuCreateInfo info;
    ReimsVgpuQemuDevice out = {
        .abi_version = 0,
        .struct_size = 0,
        .handle = 0,
    };
    char backend[32];
    int rc;

    /* TEMP (lab): HID latency marks beside the window's present probe. */
    if (getenv("REIMS_VGPU_LAB_PROBE")) {
        hid_lab_mark = reims_vgpu_qemu_lab_mark;
    }

    if (reims_vgpu_qemu_abi_version() != REIMS_VGPU_QEMU_ABI_VERSION) {
        error_setg(errp,
                   "%s: ABI version mismatch (header %u, staticlib %u)",
                   TYPE_REIMS_VGPU_PCI, REIMS_VGPU_QEMU_ABI_VERSION,
                   reims_vgpu_qemu_abi_version());
        return;
    }

    memory_region_init_io(&s->iomem_gfx, OBJECT(s), &reims_vgpu_pci_gfx_ops, s,
                          TYPE_REIMS_VGPU_PCI ".gfx",
                          REIMS_VGPU_GFX_MMIO_SIZE);
    /*
     * The guest's interrupt handler polls this BAR (0x102c, 0x100c, 0x1014/
     * 0x1018) tens of thousands of times a second. Let those reads skip the
     * BQL, so an accelerator can serve them from the vCPU thread directly;
     * the write handler takes the BQL itself.
     */
    memory_region_enable_lockless_io(&s->iomem_gfx);
    /* 32-bit non-prefetch BAR (16 KiB control window). Live 2026-07-13: 64-bit
     * BAR behind pcie-root-port caused Apple efiboot STOP 0x15; keep 32-bit. */
    pci_register_bar(pdev, 0, PCI_BASE_ADDRESS_SPACE_MEMORY, &s->iomem_gfx);

    /*
     * BAR1: host RAM linear framebuffer for UEFI Graphics Output Protocol.
     * OpenCore/OVMF have no built-in driver for 0x106B:0xEEEE; our PCI option
     * ROM (or OC Drivers entry) installs GOP with FrameBufferBase = BAR1.
     * Prefetchable 32-bit MEM so OVMF assigns under 4G (same STOP 0x15 caution).
     */
    memory_region_init_ram(&s->fb_vram, OBJECT(s), TYPE_REIMS_VGPU_PCI ".fb",
                           REIMS_VGPU_PCI_FB_SIZE, &error_fatal);
    /*
     * Non-prefetchable 32-bit MEM (same as BAR0 caution). Prefetchable BAR1
     * was a live suspect for macOS "console relocated" off the display BAR
     * into system RAM (0xf1000000) while VMware SVGA keeps console on its
     * VRAM BAR at the same GPA. Host can only follow system-RAM consoles via
     * EFI FB start (0x1210); pure XNU relocate does not rewrite that reg.
     */
    pci_register_bar(pdev, 1, PCI_BASE_ADDRESS_SPACE_MEMORY, &s->fb_vram);

    /*
     * Behind pcie-root-port the endpoint must be a real PCIe function with an
     * Express capability. Conventional-only devices leave the secondary bus
     * empty under macOS (guest ioreg: bridge 0:5 only, no 0x106B:0xEEEE).
     * Mirror bochs-display / virtio-pci.
     */
    if (pci_bus_is_express(pci_get_bus(pdev))) {
        rc = pcie_endpoint_cap_init(pdev, 0x80);
        if (rc < 0) {
            error_setg(errp, "%s: pcie_endpoint_cap_init failed",
                       TYPE_REIMS_VGPU_PCI);
            return;
        }
    } else {
        pdev->cap_present &= ~QEMU_PCI_CAP_EXPRESS;
    }

    /* Guest enables MSI count 1; always raise vector 0. */
    rc = msi_init(pdev, 0x0, 1, true /* 64-bit */, false /* per-vector mask */,
                  errp);
    if (rc < 0) {
        return;
    }

    s->host_ops = (ReimsVgpuHostOps){
        .abi_version = REIMS_VGPU_QEMU_ABI_VERSION,
        .struct_size = sizeof(ReimsVgpuHostOps),
        .ctx = s,
        .read_gpa = reims_vgpu_shim_read_gpa,
        .write_gpa = reims_vgpu_shim_write_gpa,
        .mono_ns = reims_vgpu_shim_mono_ns,
        .schedule_bh = reims_vgpu_pci_schedule_bh,
        .read_kva = reims_vgpu_shim_read_kva,
        .read_xreg = reims_vgpu_pci_read_xreg,
        .map_pages = reims_vgpu_pci_map_pages,
        .unmap_pages = reims_vgpu_pci_unmap_pages,
        /*
         * Where guest RAM lives. Shared with the sysbus shim and identical on
         * both: it reads the system address space, which the attach type does
         * not change. Unlike map_pages it never builds a view, so there is no
         * bus-specific lifetime for a shim to own.
         */
        .guest_ram_regions = reims_vgpu_shim_guest_ram_regions,
        .is_ram_gpa = reims_vgpu_shim_is_ram_gpa,
        .notify_actions = reims_vgpu_pci_notify_actions,
        /*
         * Every result remains valid until the caller explicitly retires it:
         * direct RAMBlock pointers need no release, while packed aliases leave
         * through unmap_pages after their Vulkan import has retired.
         */
        .map_pages_stable = 1,
        .harvests_settled = reims_vgpu_pci_harvests_settled,
        .irq_pulse = reims_vgpu_pci_irq_pulse,
        .note_work_scope = reims_vgpu_pci_note_work_scope,
        /*
         * Lets the device's own worker threads read guest memory: RCU is per
         * process and the registration touches no device state, so the shared
         * pair is the whole answer. Without it the device starts no worker.
         */
        .thread_attach = reims_vgpu_shim_thread_attach,
        .thread_detach = reims_vgpu_shim_thread_detach,
        .note_ring_tail = reims_vgpu_pci_note_ring_tail,
        .guest_write_sync_many = reims_vgpu_pci_guest_write_sync_many,
        .guest_write_sync_ahead = reims_vgpu_pci_guest_write_sync_ahead,
        .guest_write_rebase = reims_vgpu_pci_guest_write_rebase,
        .track_guest_writes = reims_vgpu_pci_track_guest_writes,
        .untrack_guest_writes = reims_vgpu_pci_untrack_guest_writes,
        .guest_write_gen = reims_vgpu_pci_guest_write_gen,
        .guest_written_pages = reims_vgpu_pci_guest_written_pages,
        .page_alias_census = reims_vgpu_pci_page_alias_census,
    };
    s->dirty = reims_vgpu_dirty_new();

    qemu_mutex_init(&s->drain_mutex);
    qemu_cond_init(&s->drain_cond);
    qemu_mutex_init(&s->heartbeat_mutex);
    qemu_cond_init(&s->heartbeat_cond);
    qemu_mutex_init(&s->harvest_mutex);
    qemu_cond_init(&s->harvest_cond);
    qemu_cond_init(&s->harvest_done_cond);
    s->action_bh = aio_bh_new(qemu_get_aio_context(), reims_vgpu_pci_bh, s);

    info = (ReimsVgpuQemuCreateInfo){
        .abi_version = REIMS_VGPU_QEMU_ABI_VERSION,
        .struct_size = sizeof(ReimsVgpuQemuCreateInfo),
        .host_ops = &s->host_ops,
        /* x86 Tahoe guest: 4 KiB pages. */
        .guest_page_shift = REIMS_VGPU_GUEST_PAGE_SHIFT_X86_64,
    };

    rc = reims_vgpu_qemu_device_create(&info, &out);
    if (rc != REIMS_VGPU_QEMU_OK || out.handle == 0) {
        error_setg(errp, "%s: reims_vgpu_qemu_device_create failed (rc=%d)",
                   TYPE_REIMS_VGPU_PCI, rc);
        msi_uninit(pdev);
        qemu_bh_delete(s->action_bh);
        s->action_bh = NULL;
        qemu_cond_destroy(&s->heartbeat_cond);
        qemu_mutex_destroy(&s->heartbeat_mutex);
        qemu_cond_destroy(&s->drain_cond);
        qemu_mutex_destroy(&s->drain_mutex);
        return;
    }
    s->rust_handle = out.handle;
    {
        const char *d = getenv("REIMS_VGPU_IRQ_DIRECT");   /* lab A/B */

        /*
         * Opt-in: with the 1 ms heartbeat the direct pulse lost to the action
         * BH on the same host (vCPUs waited ~70% longer for the BQL in prerun
         * and the guest's kernel_task doubled), so it stays a lab arm.
         */
        s->irq_direct = d && strcmp(d, "on") == 0;
    }
    {
        const char *h = getenv("REIMS_VGPU_ASYNC_HARVEST");   /* lab A/B */
        const char *hb = getenv("REIMS_VGPU_HARVEST_BQL");     /* lab A/B */

        const char *od = getenv("REIMS_VGPU_DIRTY_ONDEMAND");  /* lab A/B */

        s->harvest_nobql = !(hb && strcmp(hb, "on") == 0);
        /*
         * Default (`off` restores the per-doorbell harvest): each generation
         * read syncs its own set, so the drain stops waiting for a harvest per
         * doorbell (lab: CSS scroll ~90 -> 115-120 fps, Safari scroll ~66 ->
         * ~90). The stale tile pages this mode once left on a full-screen page
         * were lost dirty bits, not ordering: physical_memory_set_dirty_lebitmap
         * mis-strode unaligned ranges on LLP64 hosts, and the per-set ranges
         * here are rarely aligned.
         */
        s->dirty_ondemand = !(od && strcmp(od, "off") == 0) && s->harvest_nobql;
        reims_vgpu_dirty_set_ondemand(s->dirty, s->dirty_ondemand);

        if (!(h && strcmp(h, "off") == 0)) {
            qemu_thread_create(&s->harvest_thread, "reims-vgpu-pci-harvest",
                               reims_vgpu_pci_harvest_thread, s,
                               QEMU_THREAD_JOINABLE);
            s->harvest_started = true;
        }
        {
            const char *pk = getenv("REIMS_VGPU_LAB_DIRTY_PEEK_MS");   /* TEMP lab DIRTY_PEEK */

            s->peek_ms = pk ? atoi(pk) : 0;
            if (s->dirty_ondemand && s->peek_ms >= 1 && s->peek_ms <= 16) {
                qemu_mutex_init(&s->peek_mutex);
                qemu_cond_init(&s->peek_cond);
                qemu_thread_create(&s->peek_thread, "reims-vgpu-peek",
                                   reims_vgpu_pci_peek_thread, s, QEMU_THREAD_JOINABLE);
                s->peek_started = true;
            }
        }
    }
#if defined(_WIN32)
    reims_vgpu_hiqos_apply(true);
#endif
    qemu_thread_create(&s->drain_thread, "reims-vgpu-pci-drain",
                       reims_vgpu_pci_drain_thread, s, QEMU_THREAD_JOINABLE);
    s->drain_started = true;
    qemu_thread_create(&s->heartbeat_thread, "reims-vgpu-pci-heartbeat",
                       reims_vgpu_pci_heartbeat_thread, s, QEMU_THREAD_JOINABLE);
    s->heartbeat_started = true;
    s->shutdown_notifier.notify = reims_vgpu_pci_shutdown_notifier;
    qemu_register_shutdown_notifier(&s->shutdown_notifier);
    s->shutdown_notifier_registered = true;

    s->con = qemu_graphic_console_create(DEVICE(pdev), 0, &reims_vgpu_pci_fb_ops, s);
    reims_vgpu_pci_set_mode(s, REIMS_VGPU_EFI_BOOT_WIDTH, REIMS_VGPU_EFI_BOOT_HEIGHT);
    if (s->surface) {
        memset(surface_data(s->surface), 0,
               (size_t)surface_stride(s->surface) * REIMS_VGPU_EFI_BOOT_HEIGHT);
        qemu_console_update_full(s->con);
    }
    if (s->con) {
        qemu_console_set_cursor(s->con, cursor_builtin_hidden());
        qemu_console_set_mouse(s->con, 0, 0, false);
    }

    /*
     * Host-owned presentation window (kb host-window): on unless REIMS_VGPU_WINDOW
     * disables it (reims_vgpu_pci_window_requested; vm/boot-x86.sh defaults it on for
     * this device). Rust spawns a winit + VkSurfaceKHR window on its own thread,
     * the drain publishes finished present frames to it, and window input is
     * injected
     * through the neutral Input* prompt-action rail (qemu_input_*, which works
     * under -display none). All window/present logic lives in the staticlib;
     * the shim only flips it on. A staticlib built without the host-window
     * feature returns ERR_STATE here and QEMU's own display stays in charge.
     */
    if (reims_vgpu_pci_window_requested()) {
        int wrc = reims_vgpu_qemu_window_start(s->rust_handle, REIMS_VGPU_EFI_BOOT_WIDTH,
                                        REIMS_VGPU_EFI_BOOT_HEIGHT);
        if (wrc != REIMS_VGPU_QEMU_OK) {
            qemu_log_mask(LOG_GUEST_ERROR,
                          "%s: host window unavailable (rc=%d); "
                          "using QEMU display\n",
                          TYPE_REIMS_VGPU_PCI, wrc);
        } else if (memory_region_is_ram(&s->fb_vram)) {
            /*
             * Hand the window the BAR1 GOP framebuffer (host RAM) so it shows
             * UEFI/OpenCore/boot.efi output before the product present path
             * latches. The pointer is a stable RAMBlock host VA valid for the
             * device lifetime; the staticlib reads it on the poll path (gated
             * by protocol console-ownership state, never content).
             */
            uint8_t *fb = memory_region_get_ram_ptr(&s->fb_vram);

            if (fb) {
                reims_vgpu_qemu_window_set_early_fb(
                    s->rust_handle, fb,
                    REIMS_VGPU_EFI_BOOT_WIDTH * REIMS_VGPU_PCI_EFI_BPP,
                    REIMS_VGPU_EFI_BOOT_WIDTH, REIMS_VGPU_EFI_BOOT_HEIGHT);
            }
        }
    }

    backend[0] = '\0';
    reims_vgpu_qemu_backend_name(backend, sizeof(backend));
    trace_reims_vgpu_pci_realize(s->rust_handle, backend);
}

static void reims_vgpu_pci_exit(PCIDevice *pdev)
{
    ReimsVGPUPCIState *s = REIMS_VGPU_PCI(pdev);

    if (s->shutdown_notifier_registered) {
        notifier_remove(&s->shutdown_notifier);
        s->shutdown_notifier_registered = false;
    }
    reims_vgpu_pci_stop_backend(s);
    if (s->page_views) {
        size_t i;

        qemu_mutex_lock(&s->page_views_lock);
        for (i = 0; i < s->page_views->len; i++) {
            ReimsVGPUPCIPageView *view =
                &g_array_index(s->page_views, ReimsVGPUPCIPageView, i);
#ifdef _WIN32
            qemu_win_ram_alias_free(view->ptr,
                                    view->len >> REIMS_VGPU_GUEST_PAGE_SHIFT_X86_64,
                                    (size_t)1 << REIMS_VGPU_GUEST_PAGE_SHIFT_X86_64,
                                    view->runs, view->runs_n);
            g_free(view->runs);
#else
            munmap(view->ptr, view->len);
#endif
            s->page_views_destroyed++;
        }
        g_array_free(s->page_views, true);
        s->page_views = NULL;
        qemu_mutex_unlock(&s->page_views_lock);
    }
    /* After the drain thread is joined: reims_vgpu_dirty_free turns region
     * logging back off, and no tracked set may outlive the Rust device that
     * holds its token. */
    reims_vgpu_dirty_free(s->dirty);
    s->dirty = NULL;
    qemu_cond_destroy(&s->heartbeat_cond);
    qemu_mutex_destroy(&s->heartbeat_mutex);
    qemu_cond_destroy(&s->drain_cond);
    qemu_mutex_destroy(&s->drain_mutex);
    qemu_mutex_destroy(&s->page_views_lock);
    msi_uninit(pdev);
    if (s->con) {
        qemu_graphic_console_close(s->con);
        s->con = NULL;
    }
    s->surface = NULL;
}

static void reims_vgpu_pci_reset(DeviceState *dev)
{
    ReimsVGPUPCIState *s = REIMS_VGPU_PCI(dev);

    if (s->rust_handle != 0) {
        reims_vgpu_qemu_device_reset(s->rust_handle);
    }
    qemu_mutex_lock(&s->drain_mutex);
    s->drain_pending = false;
    qemu_mutex_unlock(&s->drain_mutex);
    s->new_frame_ready = false;
    reims_vgpu_pci_set_mode(s, REIMS_VGPU_EFI_BOOT_WIDTH, REIMS_VGPU_EFI_BOOT_HEIGHT);
    if (s->surface && s->con) {
        memset(surface_data(s->surface), 0,
               (size_t)surface_stride(s->surface) * REIMS_VGPU_EFI_BOOT_HEIGHT);
        qemu_console_set_cursor(s->con, cursor_builtin_hidden());
        qemu_console_set_mouse(s->con, 0, 0, false);
        s->new_frame_ready = true;
        qemu_console_update_full(s->con);
        s->new_frame_ready = false;
    }
}

static void reims_vgpu_pci_instance_init(Object *obj)
{
    ReimsVGPUPCIState *s = REIMS_VGPU_PCI(obj);
    PCIDevice *pdev = PCI_DEVICE(obj);

    /* Required before realize so config space is PCIe-sized (4 KiB). */
    pdev->cap_present |= QEMU_PCI_CAP_EXPRESS;
    s->page_views = g_array_new(false, false, sizeof(ReimsVGPUPCIPageView));
    /*
     * map_pages/unmap_pages run on the drain and pool threads at once; the
     * per-device GArray they append to and scan was unsynchronized (a pool
     * job's unmap racing a drain append reallocs the element array under the
     * reader - both Windows and POSIX arms).
     */
    qemu_mutex_init(&s->page_views_lock);
}

static void reims_vgpu_pci_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);

    (void)data;
    k->realize = reims_vgpu_pci_realize;
    k->exit = reims_vgpu_pci_exit;
    k->vendor_id = REIMS_VGPU_PCI_VENDOR;
    k->device_id = REIMS_VGPU_PCI_DEVICE;
    k->revision = 0x00;
    k->class_id = PCI_CLASS_DISPLAY_VGA;
    k->subsystem_vendor_id = REIMS_VGPU_PCI_VENDOR;
    k->subsystem_id = REIMS_VGPU_PCI_DEVICE;

    dc->desc = "Reims vGPU (PCI thin shim -> reims-vgpu)";
    set_bit(DEVICE_CATEGORY_DISPLAY, dc->categories);
    dc->hotpluggable = false;
    dc->user_creatable = true;
    device_class_set_legacy_reset(dc, reims_vgpu_pci_reset);
}

static const TypeInfo reims_vgpu_pci_info = {
    .name = TYPE_REIMS_VGPU_PCI,
    .parent = TYPE_PCI_DEVICE,
    .instance_size = sizeof(ReimsVGPUPCIState),
    .instance_init = reims_vgpu_pci_instance_init,
    .class_init = reims_vgpu_pci_class_init,
    .interfaces = (const InterfaceInfo[]) {
        { INTERFACE_PCIE_DEVICE },
        { INTERFACE_CONVENTIONAL_PCI_DEVICE },
        { },
    },
};

static void reims_vgpu_pci_register_types(void)
{
    type_register_static(&reims_vgpu_pci_info);
}

type_init(reims_vgpu_pci_register_types)
