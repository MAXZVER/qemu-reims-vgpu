/*
 * reims-vgpu: guest-write tracking over the hypervisor dirty bitmap.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/main-loop.h"
#include "qemu/thread.h"
#include "qemu/rcu.h"
#include "qemu/bitmap.h"
#include "exec/target_page.h"
#include "system/address-spaces.h"
#include "system/memory.h"
#include "system/physmem.h"
#include "system/ram_addr.h"
#include "system/ramlist.h"

#include "reims-vgpu-dirty.h"

/*
 * One guest-RAM range of the system address space. A tracked page is resolved
 * against this table by address, so one entry serves every page that lands in
 * it — the same reason reims_vgpu_*_map_pages walks maximal runs.
 */
typedef struct ReimsVgpuDirtySlice {
    MemoryRegion *mr;
    uint64_t gpa;           /* first GPA covered */
    uint64_t len;           /* bytes covered */
    ram_addr_t ram_addr;    /* ram_addr_t of `gpa` */
    hwaddr offset;          /* offset of `gpa` within mr */
    /* Whether this device was already recording writes to the range when this
     * harvest's sync ran, and whether a tracked page asked it to start. Held
     * per slice rather than in parallel arrays because they are read in the
     * page loop, which is the harvest's inner loop. */
    bool logged;
    bool needs_log;
} ReimsVgpuDirtySlice;

/* A tracked page set: one mapping incarnation's guest storage. */
typedef struct ReimsVgpuDirtySet {
    uint64_t *pages;        /* sorted, deduplicated, page-aligned */
    /*
     * Parallel to `pages`: the value `gen` took at the harvest that last saw
     * this page written, or 0 for a page never seen written. A set-level
     * generation can only say "something in here moved", which forces a reader
     * holding a whole-surface copy to discard all of it; this is what lets it
     * discard exactly the pages that moved.
     */
    uint64_t *page_gen;
    size_t count;
    uint64_t page_size;
    uint64_t gen;           /* 0 until armed */
    uint64_t arm_at;        /* harvest count at which gen may leave 0 */
    /*
     * On-demand sync (see reims_vgpu_dirty_gen): the doorbell epoch this set
     * was last brought up to date at, and the global write sequence its
     * generation last accounted for. A page stamped past `seen_seq` was
     * written since.
     */
    uint64_t synced_epoch;
    uint64_t seen_seq;
    uint64_t last_query_epoch;  /* epoch of the last generation or page read */
    uint64_t claim_epoch;       /* epoch a sync is running for, or 0 */
} ReimsVgpuDirtySet;

struct ReimsVgpuDirty {
    QemuMutex lock;
    GHashTable *sets;       /* token -> ReimsVgpuDirtySet* */
    uint64_t next_token;
    /*
     * Regions this device turned DIRTY_MEMORY_VGA logging on for, referenced
     * so the pointers stay valid until we turn it back off. Logging stays on
     * for the device's life: toggling it per surface re-protects the whole
     * region each time, and the guest's entire working set would refault for a
     * saving this device never collects.
     */
    GHashTable *logged;     /* MemoryRegion* -> itself */
    /*
     * The guest-RAM ranges of the system address space, rebuilt at the top of
     * each harvest and owned across harvests only so the steady state does not
     * allocate. Grown to whatever the flat view reports and never bounded: a
     * range this table dropped would read as written for every page in it, for
     * the life of the VM, with nothing to say so — which is precisely the
     * failure the flat view walk replaced.
     *
     * Measured at 5 ranges on x86 q35 with this device, unchanged across a
     * driven boot including the load phase. So this is not a table under
     * pressure, and the growth is not there to serve a reach anyone has seen.
     * It is here because the alternative to growing is dropping, and dropping
     * is unobservable from the guest side and permanent.
     */
    ReimsVgpuDirtySlice *slices;
    int slices_cap;
    uint64_t harvests;
    /*
     * Generation reads since the last harvest. A harvest nothing has consumed
     * since the previous one cannot tell any reader something it does not
     * already know, so it is skipped — which collapses a burst of doorbell
     * writes into one sync.
     */
    uint64_t reads_since_harvest;
    /*
     * Scratch for reims_vgpu_dirty_sync_tracked, owned across harvests so the
     * steady state does not allocate: one bit per target page of guest-physical
     * space up to the highest RAM range, and the ranges handed to the sync.
     */
    unsigned long *sync_pages;
    uint64_t sync_pages_bits;
    GArray *sync_ranges;    /* MemoryRegionRange */
    /*
     * Where a consumed dirty bit goes: the global write sequence stamped on the
     * target page it was read for. Consuming (reading and clearing) a bit is
     * then safe whichever set asked — a set sharing the page reads the stamp —
     * so one set can be brought up to date without harvesting every set.
     * Indexed by target page frame; grown to the highest RAM page seen.
     */
    uint64_t global_seq;
    uint64_t *page_seq;
    uint64_t page_seq_n;
    /*
     * On-demand mode: doorbells only advance `epoch`; a generation read of a
     * set not synced since the last doorbell syncs that set's pages first. The
     * background harvest is then asked for only to turn logging on
     * (`log_wanted`), the one step that needs the BQL.
     */
    bool ondemand;
    bool log_wanted;
    uint64_t epoch;
    /*
     * Prefetch (REIMS_VGPU_DIRTY_OD_PREFETCH=off turns it off): at each
     * doorbell the harvest thread syncs the sets read in the last two epochs,
     * in parallel with the drain, which then finds most of them up to date.
     */
    bool od_prefetch;
    /* Signalled when an on-demand sync finishes; see claim_epoch. */
    QemuCond synced_cond;
};

static void reims_vgpu_dirty_set_free(gpointer p)
{
    ReimsVgpuDirtySet *s = p;

    g_free(s->pages);
    g_free(s->page_gen);
    g_free(s);
}

ReimsVgpuDirty *reims_vgpu_dirty_new(void)
{
    ReimsVgpuDirty *d = g_new0(ReimsVgpuDirty, 1);

    qemu_mutex_init(&d->lock);
    qemu_cond_init(&d->synced_cond);
    d->sets = g_hash_table_new_full(g_int64_hash, g_int64_equal, g_free,
                                    reims_vgpu_dirty_set_free);
    d->logged = g_hash_table_new(NULL, NULL);
    return d;
}

void reims_vgpu_dirty_free(ReimsVgpuDirty *d)
{
    GHashTableIter it;
    gpointer key, val;

    if (!d) {
        return;
    }
    /* BQL: memory_region_set_log is a MemoryRegion transaction. */
    g_hash_table_iter_init(&it, d->logged);
    while (g_hash_table_iter_next(&it, &key, &val)) {
        MemoryRegion *mr = key;

        memory_region_set_log(mr, false, DIRTY_MEMORY_VGA);
        memory_region_unref(mr);
    }
    g_hash_table_destroy(d->logged);
    g_hash_table_destroy(d->sets);
    g_free(d->slices);
    g_free(d->sync_pages);
    if (d->sync_ranges) {
        g_array_free(d->sync_ranges, TRUE);
    }
    g_free(d->page_seq);
    qemu_cond_destroy(&d->synced_cond);
    qemu_mutex_destroy(&d->lock);
    g_free(d);
}

static int reims_vgpu_dirty_cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a;
    uint64_t y = *(const uint64_t *)b;

    return x < y ? -1 : (x > y ? 1 : 0);
}

/*
 * One guest page a harvest found written, with the geometry it was tracked at.
 * The size travels with the page because the clear must cover the whole guest
 * page and the two pathways do not share a page shift.
 */
typedef struct ReimsVgpuDirtyWritten {
    uint64_t gpa;
    uint64_t page_size;
} ReimsVgpuDirtyWritten;

static int reims_vgpu_dirty_cmp_written(gconstpointer a, gconstpointer b)
{
    const ReimsVgpuDirtyWritten *x = a;
    const ReimsVgpuDirtyWritten *y = b;

    if (x->gpa != y->gpa) {
        return x->gpa < y->gpa ? -1 : 1;
    }
    /* Larger page first, so a duplicate of the same base collapses into the
     * run the wider page opens rather than truncating it. */
    return x->page_size > y->page_size ? -1 : (x->page_size < y->page_size);
}

uint64_t reims_vgpu_dirty_track(ReimsVgpuDirty *d, const uint64_t *gpas,
                                size_t count, size_t page_size)
{
    ReimsVgpuDirtySet *s;
    uint64_t *pages;
    uint64_t token;
    size_t i, n;

    if (!d || !gpas || count == 0 || page_size == 0 ||
        (page_size & (page_size - 1)) != 0) {
        return 0;
    }
    /*
     * The dirty bitmap is indexed in target pages, so a guest page smaller
     * than one cannot be asked about. Both product pathways are the other way
     * round (4 KiB x86 == target, 16 KiB arm64e > target), and answering for a
     * granularity the bitmap does not have would be a guess.
     */
    if (page_size < qemu_target_page_size()) {
        return 0;
    }

    pages = g_new(uint64_t, count);
    for (i = 0; i < count; i++) {
        pages[i] = gpas[i] & ~((uint64_t)page_size - 1);
    }
    qsort(pages, count, sizeof(*pages), reims_vgpu_dirty_cmp_u64);
    n = 0;
    for (i = 0; i < count; i++) {
        if (n == 0 || pages[n - 1] != pages[i]) {
            pages[n++] = pages[i];
        }
    }

    s = g_new0(ReimsVgpuDirtySet, 1);
    s->pages = pages;
    s->page_gen = g_new0(uint64_t, n);
    s->count = n;
    s->page_size = page_size;

    qemu_mutex_lock(&d->lock);
    /*
     * One harvest, and a second only if that harvest has to turn logging on.
     *
     * This used to be an unconditional two, because "the first harvest after
     * this call is the one that may still be turning logging on for these
     * pages, so writes older than it were never recorded". That reason is
     * right, and it is a reason about the *first* set to name a region — not
     * about every set. Guest RAM is one MemoryRegion, so once any surface has
     * been tracked, logging is already on for every later one, no harvest ever
     * enables anything, and the second harvest was waiting for something that
     * had already happened.
     *
     * It is not free to wait. While the set is unarmed its generation reads
     * back 0, `HostOps::guest_write_gen` maps that to "cannot tell", the
     * mapper-ref-texture LOAD elision refuses with `t11_gw_ref_no_stamp`, and
     * every draw onto that surface pays a whole-frame seed read out of guest
     * pages plus a
     * whole-frame staging upload. The window is counted in harvests and
     * harvests are driven by guest doorbells, so on a quiet desktop it lasts as
     * long as the guest stays quiet — which is exactly when a draw arriving
     * into it is a visible hitch (measured at 12-65 ms per draw, against
     * 0.2 ms driven).
     *
     * The second harvest is still taken when it is owed: `reims_vgpu_dirty_harvest`
     * pushes every set still waiting when it had to enable logging, so a set
     * created before the region was logged arms exactly as late as it used to.
     */
    s->arm_at = d->harvests + 1;
    s->seen_seq = d->global_seq;
    d->next_token++;
    token = d->next_token;
    g_hash_table_insert(d->sets, g_memdup2(&token, sizeof(token)), s);
    qemu_mutex_unlock(&d->lock);
    return token;
}

void reims_vgpu_dirty_untrack(ReimsVgpuDirty *d, uint64_t token)
{
    if (!d || token == 0) {
        return;
    }
    qemu_mutex_lock(&d->lock);
    g_hash_table_remove(d->sets, &token);
    qemu_mutex_unlock(&d->lock);
}

static void reims_vgpu_dirty_sync_one(ReimsVgpuDirty *d, uint64_t token);

uint64_t reims_vgpu_dirty_gen(ReimsVgpuDirty *d, uint64_t token)
{
    ReimsVgpuDirtySet *s;
    uint64_t gen = 0;
    bool stale;

    if (!d || token == 0) {
        return 0;
    }
    qemu_mutex_lock(&d->lock);
    s = g_hash_table_lookup(d->sets, &token);
    stale = s && d->ondemand && s->synced_epoch < d->epoch;
    qemu_mutex_unlock(&d->lock);
    if (stale) {
        reims_vgpu_dirty_sync_one(d, token);
    }
    qemu_mutex_lock(&d->lock);
    s = g_hash_table_lookup(d->sets, &token);
    if (s) {
        gen = s->gen;
        s->last_query_epoch = d->epoch;
    }
    d->reads_since_harvest++;
    qemu_mutex_unlock(&d->lock);
    return gen;
}

int64_t reims_vgpu_dirty_written_since(ReimsVgpuDirty *d, uint64_t token,
                                       uint64_t since_gen, uint64_t *out,
                                       size_t max)
{
    ReimsVgpuDirtySet *s;
    int64_t found = 0;
    size_t p;

    if (!d || token == 0 || out == NULL) {
        return -1;
    }
    qemu_mutex_lock(&d->lock);
    s = g_hash_table_lookup(d->sets, &token);
    if (s && d->ondemand && s->synced_epoch < d->epoch) {
        /*
         * A page list read in a later epoch than the set's last sync would
         * miss what the guest wrote since; bring the set up to date first.
         */
        qemu_mutex_unlock(&d->lock);
        reims_vgpu_dirty_sync_one(d, token);
        qemu_mutex_lock(&d->lock);
        s = g_hash_table_lookup(d->sets, &token);
    }
    if (s) {
        s->last_query_epoch = d->epoch;
    }
    /*
     * `since_gen == 0` is a caller that never recorded a readable observation,
     * and `s->gen == 0` is a set still inside its startup window. Neither can
     * be compared against a page stamp, and both mean the same thing to the
     * caller as an unknown token does.
     */
    if (!s || s->gen == 0 || since_gen == 0) {
        qemu_mutex_unlock(&d->lock);
        return -1;
    }
    for (p = 0; p < s->count; p++) {
        if (s->page_gen[p] <= since_gen) {
            continue;
        }
        if ((size_t)found == max) {
            /* Truncation would read as "these pages and no others". */
            qemu_mutex_unlock(&d->lock);
            return -1;
        }
        out[found++] = s->pages[p];
    }
    /*
     * Counted like reims_vgpu_dirty_gen(): a caller asking this question is a
     * consumer of the report, so the next harvest has something to tell it and
     * must not be skipped.
     */
    d->reads_since_harvest++;
    qemu_mutex_unlock(&d->lock);
    return found;
}

typedef struct ReimsVgpuDirtyRamScan {
    ReimsVgpuDirtySlice **buf;
    int *cap;
    int n;
} ReimsVgpuDirtyRamScan;

static bool reims_vgpu_dirty_ram_range(Int128 start, Int128 len,
                                       const MemoryRegion *mr,
                                       hwaddr offset_in_region, void *opaque)
{
    ReimsVgpuDirtyRamScan *scan = opaque;
    ReimsVgpuDirtySlice *sl;

    if (!mr->ram || !int128_nz(len)) {
        return false;
    }
    if (scan->n == *scan->cap) {
        *scan->cap = *scan->cap ? *scan->cap * 2 : 16;
        *scan->buf = g_renew(ReimsVgpuDirtySlice, *scan->buf, *scan->cap);
    }
    sl = &(*scan->buf)[scan->n++];
    sl->logged = false;
    sl->needs_log = false;
    /*
     * Dropping const: the FlatView hands out a const view of a region this
     * harvest goes on to read a dirty bitmap for, turn logging on for, and
     * clear bits in, all of which are non-const operations on a region the
     * caller holds the BQL over.
     */
    sl->mr = (MemoryRegion *)mr;
    sl->gpa = int128_get64(start);
    sl->len = int128_get64(len);
    sl->offset = offset_in_region;
    sl->ram_addr = memory_region_get_ram_addr(sl->mr) + offset_in_region;
    return false;
}

/*
 * The guest-RAM ranges of the system address space. Returns how many were
 * recorded.
 *
 * This asks the FlatView rather than reconstructing it, and the difference is
 * not a simplification. It used to cut a monotone hull of every GPA ever
 * tracked into slices by walking it with address_space_translate(), and that
 * walk could not be made correct: address_space_translate_internal() clamps
 * `plen` to the section *only when the section is RAM* (system/physmem.c). The
 * first non-RAM byte in the hull therefore returned the whole remaining length,
 * and the walk recorded one "not RAM" slice covering every guest page above it.
 *
 * On x86 q35 low RAM ends below the PCI hole and high RAM begins at 4 GiB, so
 * one tracked page landing in low RAM while others sat above the hole put every
 * page in high RAM into that slice. The harvest reads a page it cannot resolve
 * as written — the conservative answer, and the right one for a page that is
 * genuinely gone — so every tracked surface then reported itself permanently
 * overwritten by the guest. The hull never shrank and the clear pass skipped
 * unresolved pages, so nothing could undo it for the life of the VM.
 *
 * Downstream that is a guest-visible latch, not a slow path: every host-side
 * copy of a surface is declared stale, the mapper-ref-texture sampled rung
 * refuses its resident and merges guest pages that never held the composite,
 * and deferred
 * render windows report `deferred_flush_clobber`. It reads as backdrops going
 * transparent and popover geometry breaking, until the guest is rebooted.
 *
 * A range is kept whenever it is `ram`, including device BARs and flash. A
 * tracked page that lands in one is a page this device must answer for rather
 * than skip, and nothing turns logging on for a region until a tracked page
 * resolves into it.
 *
 * Every such range is recorded; the table grows to fit rather than stopping at
 * a bound. A bound here is not a budget that costs latency when it binds — a
 * dropped range makes every page in it read as written for the life of the VM,
 * which is the same latch by a smaller door, and one no counter would name.
 * Growth is amortized and the steady state reallocates nothing, because the
 * table is owned across harvests and the flat view's shape does not churn.
 */
static int reims_vgpu_dirty_ram_slices_into(ReimsVgpuDirtySlice **buf, int *cap)
{
    ReimsVgpuDirtyRamScan scan = { buf, cap, 0 };

    RCU_READ_LOCK_GUARD();
    flatview_for_each_range(address_space_to_flatview(&address_space_memory),
                            reims_vgpu_dirty_ram_range, &scan);
    return scan.n;
}

static int reims_vgpu_dirty_ram_slices(ReimsVgpuDirty *d)
{
    return reims_vgpu_dirty_ram_slices_into(&d->slices, &d->slices_cap);
}

/* Which slice holds `gpa`, or -1. Slices are few and ordered, so a linear
 * scan beats any structure these counts would justify. */
static int reims_vgpu_dirty_slice_of(const ReimsVgpuDirtySlice *slices, int n,
                                     uint64_t gpa)
{
    int i;

    for (i = 0; i < n; i++) {
        if (gpa >= slices[i].gpa && gpa - slices[i].gpa < slices[i].len) {
            return i;
        }
    }
    return -1;
}

/*
 * Grow the page stamps to cover target page frames below `end_pfn`.
 * d->lock held.
 */
static void reims_vgpu_dirty_page_seq_cover(ReimsVgpuDirty *d, uint64_t end_pfn)
{
    uint64_t n;

    if (end_pfn <= d->page_seq_n) {
        return;
    }
    n = MAX(end_pfn, d->page_seq_n * 2);
    d->page_seq = g_renew(uint64_t, d->page_seq, n);
    memset(d->page_seq + d->page_seq_n, 0,
           (n - d->page_seq_n) * sizeof(uint64_t));
    d->page_seq_n = n;
}

/*
 * Read and clear one target page's VGA dirty bit as a single atomic step.
 *
 * Reading the bit and clearing it later — even under the tracker's lock — lost
 * writes: another thread's hypervisor query deposits into the same bitmap
 * without that lock, so a write it moved in between the read and the clear was
 * erased unseen (streaks of stale tile pages in the lab). A bit taken this way
 * that a query sets again afterwards is simply there for the next consumer.
 */
static bool reims_vgpu_dirty_take_bit(ram_addr_t ram)
{
    unsigned long page = ram >> qemu_target_page_bits();
    unsigned long idx = page / DIRTY_MEMORY_BLOCK_SIZE;
    unsigned long off = page % DIRTY_MEMORY_BLOCK_SIZE;
    DirtyMemoryBlocks *blocks;

    RCU_READ_LOCK_GUARD();
    blocks = qatomic_rcu_read(&ram_list.dirty_memory[DIRTY_MEMORY_VGA]);
    return bitmap_test_and_clear_atomic(blocks->blocks[idx], off, 1);
}

/*
 * Move the VGA dirty bit of every target page of one set into the global page
 * stamps, and queue the pages it was set for on `written` for the re-arm pass.
 * A page a second set shares was taken by whichever set consumed it first; the
 * other reads the stamp. d->lock held.
 */
static void reims_vgpu_dirty_consume_set(ReimsVgpuDirty *d,
                                         const ReimsVgpuDirtySet *s,
                                         const ReimsVgpuDirtySlice *slices,
                                         int n, GArray *written)
{
    const int shift = qemu_target_page_bits();
    const uint64_t target = 1ULL << shift;
    size_t p;
    uint64_t off;

    for (p = 0; p < s->count; p++) {
        uint64_t gpa = s->pages[p];
        int si = reims_vgpu_dirty_slice_of(slices, n, gpa);

        if (si < 0 || !slices[si].logged) {
            continue;
        }
        for (off = 0; off < s->page_size; off += target) {
            uint64_t tg = gpa + off;
            uint64_t pfn = tg >> shift;
            ram_addr_t ram = slices[si].ram_addr + (tg - slices[si].gpa);
            ReimsVgpuDirtyWritten rec = { tg, target };

            if (!reims_vgpu_dirty_take_bit(ram)) {
                continue;
            }
            reims_vgpu_dirty_page_seq_cover(d, pfn + 1);
            d->page_seq[pfn] = ++d->global_seq;
            g_array_append_val(written, rec);
        }
    }
}

/*
 * Fold the page stamps into one set's generation and per-page generations: the
 * per-set half of a harvest. `hit` is scratch. Returns whether some page of the
 * set sits in a range this device was not yet recording. d->lock held.
 *
 * `by_harvest` keeps the harvest's arming rule (`arm_at`, counted in
 * harvests). An on-demand evaluation cannot count harvests — there may be none
 * — so it arms a set on its first evaluation with every page in a logged
 * range: logging was on before the set existed, so nothing since its creation
 * went unrecorded. A page in a range not yet logged reads as written, flags the
 * range, and keeps an unarmed set unreadable, as in the harvest.
 */
static bool reims_vgpu_dirty_eval_set(ReimsVgpuDirty *d, ReimsVgpuDirtySet *s,
                                      ReimsVgpuDirtySlice *slices, int n,
                                      GArray *hit, bool by_harvest)
{
    const int shift = qemu_target_page_bits();
    const uint64_t target = 1ULL << shift;
    bool any = false;
    bool unlogged = false;
    size_t p;
    uint64_t off;

    g_array_set_size(hit, 0);
    for (p = 0; p < s->count; p++) {
        uint64_t gpa = s->pages[p];
        int si = reims_vgpu_dirty_slice_of(slices, n, gpa);
        bool written = false;

        /*
         * Not guest RAM this device can read a bitmap for, or RAM whose
         * writes this device was not yet recording when the sync ran.
         * Both mean the host cannot say the page is unwritten, and that
         * reads as written — per page as well as for the set, so a
         * per-page reader is no less conservative than the generation is.
         */
        if (si < 0) {
            any = true;
            g_array_append_val(hit, p);
            continue;
        }
        if (!slices[si].logged) {
            slices[si].needs_log = true;
            unlogged = true;
            any = true;
            g_array_append_val(hit, p);
            continue;
        }
        for (off = 0; off < s->page_size; off += target) {
            uint64_t pfn = (gpa + off) >> shift;

            if (pfn < d->page_seq_n && d->page_seq[pfn] > s->seen_seq) {
                written = true;
                break;
            }
        }
        if (written) {
            any = true;
            g_array_append_val(hit, p);
        }
    }
    if (by_harvest) {
        /*
         * Some page of this set sits in a region whose writes this device
         * was not yet recording when the sync ran, so writes older than it
         * were never recorded and their absence is not evidence. Wait for a
         * harvest that covered every page.
         *
         * This is what lets `reims_vgpu_dirty_track` arm at +1 instead of
         * +2. The +2 was paying for this case on every set; this pays for it
         * on the sets it applies to, which after the first surface of a boot
         * is none. The test is `s->gen == 0` — never armed — because an
         * already armed set must not be sent back through its startup
         * window: its pages read as written above, which is the conservative
         * answer and needs no help.
         */
        if (s->gen == 0 && unlogged) {
            s->arm_at = d->harvests + 1;
        }
        if (d->harvests < s->arm_at) {
            /*
             * Startup window: an absence of reports says nothing about the
             * guest yet, so the generation stays unreadable.
             */
            s->gen = 0;
        } else if (s->gen == 0) {
            s->gen = 1;
        } else if (any) {
            s->gen++;
        }
    } else if (s->gen == 0) {
        /* On demand: armed at the first evaluation with every page logged. */
        if (!unlogged) {
            s->gen = 1;
        }
    } else if (any) {
        s->gen++;
    }
    /*
     * Stamp after the generation, and only once it is readable. A page
     * stamped during the startup window would carry a generation no reader
     * can have recorded, and would then read as written forever.
     */
    if (s->gen != 0) {
        guint h;

        for (h = 0; h < hit->len; h++) {
            s->page_gen[g_array_index(hit, size_t, h)] = s->gen;
        }
    }
    s->seen_seq = d->global_seq;
    return unlogged;
}

/*
 * Re-arm write tracking for the pages a consume pass took a bit for.
 *
 * The bit itself is already clear (reims_vgpu_dirty_take_bit). What is left is
 * what memory_region_reset_dirty() used to do on top of clearing it: let the
 * accelerator re-protect the pages (log_clear, e.g. KVM with manual dirty-log
 * protection) and let TCG drop its not-dirty TLB entries, so the next guest
 * store to them is recorded again. Neither touches the bitmap, so nothing a
 * concurrent sync deposited meanwhile is lost. On WHPX both are no-ops: the
 * query resets the hypervisor's log itself.
 *
 * Only what came back written is re-armed, since re-protecting costs a fault
 * per page; exactly-adjacent pages merge into one call. No tracker lock needed.
 */
static void reims_vgpu_dirty_clear_written(GArray *written,
                                           const ReimsVgpuDirtySlice *slices,
                                           int n)
{
    guint w = 0;

    if (written->len == 0) {
        return;
    }
    g_array_sort(written, reims_vgpu_dirty_cmp_written);
    while (w < written->len) {
        ReimsVgpuDirtyWritten first =
            g_array_index(written, ReimsVgpuDirtyWritten, w);
        int si = reims_vgpu_dirty_slice_of(slices, n, first.gpa);
        uint64_t end = first.gpa + first.page_size;

        while (w + 1 < written->len) {
            ReimsVgpuDirtyWritten next =
                g_array_index(written, ReimsVgpuDirtyWritten, w + 1);

            if (next.gpa + next.page_size <= end) {
                /*
                 * Same page recorded by a second set, or a smaller page
                 * already inside the run.
                 */
                w++;
                continue;
            }
            if (next.gpa != end ||
                reims_vgpu_dirty_slice_of(slices, n, next.gpa) != si) {
                break;
            }
            end = next.gpa + next.page_size;
            w++;
        }
        if (si >= 0) {
            memory_region_clear_dirty_bitmap(
                slices[si].mr, slices[si].offset + (first.gpa - slices[si].gpa),
                end - first.gpa);
            physical_memory_dirty_bits_cleared(
                slices[si].ram_addr + (first.gpa - slices[si].gpa),
                end - first.gpa);
        }
        w++;
    }
}

/*
 * Pages of unwritten gap a sync range may absorb rather than end. Measured on
 * WHPX, a query costs ~4 us a call plus ~26 ns a page walked, so a gap is
 * worth absorbing below ~150 pages and a little past it costs little. At 256
 * the tracked set of a busy desktop (~60k pages) is ~800 ranges over ~360k
 * pages, ~12 ms a harvest, against ~3M pages and ~35 ms for all of RAM; 1024
 * measured no better.
 */
#define REIMS_VGPU_DIRTY_SYNC_GAP_PAGES 256

/*
 * Bring the dirty bitmap up to date for the tracked pages, not all of RAM.
 *
 * The harvest reads the bit of every tracked page and nothing else, so that
 * is all the sync has to cover. A whole-RAM sync walks every page the guest
 * has, tens of milliseconds on a 12 GiB guest, and a vCPU waits through it on
 * every doorbell write that harvests. Only logged slices are synced: pages in
 * the others read as written without consulting a bit.
 *
 * Nothing is lost for any other bitmap consumer. A range sync moves the
 * hypervisor's bits for that range into QEMU's bitmaps for every client; the
 * bits outside it stay with the hypervisor until someone syncs them.
 *
 * A set tracked after the page scan below is synced at the next harvest, the
 * same as one tracked after the whole-RAM sync used to be.
 */
static void reims_vgpu_dirty_sync_tracked(ReimsVgpuDirty *d,
                                          const ReimsVgpuDirtySlice *slices,
                                          int n)
{
    const int shift = qemu_target_page_bits();
    GHashTableIter it;
    gpointer key, val;
    uint64_t bits = 0;
    int i, j;

    for (i = 0; i < n; i++) {
        if (slices[i].logged) {
            bits = MAX(bits, (slices[i].gpa + slices[i].len) >> shift);
        }
    }
    if (bits == 0) {
        return;
    }
    if (d->sync_pages_bits < bits) {
        g_free(d->sync_pages);
        d->sync_pages = bitmap_new(bits);
        d->sync_pages_bits = bits;
    }
    if (!d->sync_ranges) {
        d->sync_ranges = g_array_new(FALSE, FALSE, sizeof(MemoryRegionRange));
    }

    qemu_mutex_lock(&d->lock);
    g_hash_table_iter_init(&it, d->sets);
    while (g_hash_table_iter_next(&it, &key, &val)) {
        const ReimsVgpuDirtySet *s = val;
        size_t p;
        uint64_t off;

        for (p = 0; p < s->count; p++) {
            for (off = 0; off < s->page_size; off += 1ULL << shift) {
                uint64_t pfn = (s->pages[p] + off) >> shift;

                if (pfn < bits) {
                    set_bit(pfn, d->sync_pages);
                }
            }
        }
    }
    qemu_mutex_unlock(&d->lock);

    /*
     * One batch per MemoryRegion: on q35 guest RAM is one region aliased below
     * and above the PCI hole, and a listener that cannot sync sub-ranges syncs
     * the whole region once per batch.
     */
    for (i = 0; i < n; i++) {
        bool seen = false;

        for (j = 0; j < i; j++) {
            seen |= slices[j].logged && slices[j].mr == slices[i].mr;
        }
        if (!slices[i].logged || seen) {
            continue;
        }
        g_array_set_size(d->sync_ranges, 0);
        for (j = i; j < n; j++) {
            const ReimsVgpuDirtySlice *sl = &slices[j];
            uint64_t end = (sl->gpa + sl->len) >> shift;
            uint64_t pfn, first, last, next;

            if (!sl->logged || sl->mr != slices[i].mr) {
                continue;
            }
            pfn = find_next_bit(d->sync_pages, end, sl->gpa >> shift);
            while (pfn < end) {
                MemoryRegionRange r;

                first = last = pfn;
                for (;;) {
                    next = find_next_bit(d->sync_pages, end, last + 1);
                    if (next >= end ||
                        next - last - 1 > REIMS_VGPU_DIRTY_SYNC_GAP_PAGES) {
                        break;
                    }
                    last = next;
                }
                r.start = sl->offset + ((first << shift) - sl->gpa);
                r.len = (last - first + 1) << shift;
                g_array_append_val(d->sync_ranges, r);
                pfn = next;
            }
        }
        memory_region_sync_dirty_ranges(slices[i].mr,
                                        &g_array_index(d->sync_ranges,
                                                       MemoryRegionRange, 0),
                                        d->sync_ranges->len);
    }
    bitmap_zero(d->sync_pages, bits);
}

void reims_vgpu_dirty_harvest(ReimsVgpuDirty *d)
{
    ReimsVgpuDirtySlice *slices;
    g_autoptr(GArray) written = NULL;
    g_autoptr(GArray) hit = NULL;
    GHashTableIter it;
    gpointer key, val;
    int n, i;
    uint64_t epoch;

    if (!d) {
        return;
    }

    qemu_mutex_lock(&d->lock);
    if (g_hash_table_size(d->sets) == 0 || d->reads_since_harvest == 0) {
        qemu_mutex_unlock(&d->lock);
        return;
    }
    /* Every doorbell before this point is covered by the sync below. */
    epoch = d->epoch;
    qemu_mutex_unlock(&d->lock);

    n = reims_vgpu_dirty_ram_slices(d);
    slices = d->slices;
    /*
     * Whether this device was already recording writes to each range when the
     * sync below ran. Answered once per range rather than per page: the page
     * loop runs over every page of every tracked set, and a hash lookup there
     * is the harvest's inner loop.
     *
     * A range that is not logged yet is one whose bits mean nothing, so its
     * pages read as written below and it is queued for logging afterwards.
     * Enabling is deferred past the clear pass because memory_region_set_log()
     * is a MemoryRegion transaction that rebuilds the flat view, which would
     * leave this slice table describing a view that no longer exists.
     */
    qemu_mutex_lock(&d->lock);
    for (i = 0; i < n; i++) {
        slices[i].logged = g_hash_table_contains(d->logged, slices[i].mr);
    }
    qemu_mutex_unlock(&d->lock);

    /* One sync covering every tracked page, then only reads. */
    reims_vgpu_dirty_sync_tracked(d, slices, n);

    written = g_array_new(FALSE, FALSE, sizeof(ReimsVgpuDirtyWritten));
    /*
     * Indices of the current set's written pages. Held across the generation
     * update because the value to stamp them with is the generation this
     * harvest produces, which is not known until every page has been read.
     * Reused between sets so the harvest allocates once, not once per mapping.
     */
    hit = g_array_new(FALSE, FALSE, sizeof(size_t));
    qemu_mutex_lock(&d->lock);
    d->harvests++;
    d->reads_since_harvest = 0;
    /*
     * Consume every tracked page's bit into the page stamps first, then fold
     * the stamps into each set. Two passes, because a bit may be consumed only
     * once and a page may belong to several sets; the stamp is what they share.
     */
    g_hash_table_iter_init(&it, d->sets);
    while (g_hash_table_iter_next(&it, &key, &val)) {
        reims_vgpu_dirty_consume_set(d, val, slices, n, written);
    }
    g_hash_table_iter_init(&it, d->sets);
    while (g_hash_table_iter_next(&it, &key, &val)) {
        ReimsVgpuDirtySet *set = val;

        reims_vgpu_dirty_eval_set(d, set, slices, n, hit, true);
        set->synced_epoch = MAX(set->synced_epoch, epoch);
    }
    qemu_mutex_unlock(&d->lock);

    /*
     * Re-arm only the pages that came back written. Re-arming re-protects, so
     * re-arming the whole tracked window would make every page the guest has
     * mapped refault after every harvest; re-arming what was written costs
     * exactly one refault per page the guest writes again, which is the work
     * the report is worth.
     *
     * Exactly-adjacent pages merge into one call because each call is a
     * hypervisor round trip. Nothing is widened: a merged range covers only
     * pages that were themselves written.
     */
    reims_vgpu_dirty_clear_written(written, slices, n);

    /*
     * Start recording writes to every region a tracked page resolved into and
     * that was not being recorded for the sync above. Last, because this is a
     * MemoryRegion transaction: it rebuilds the flat view, and the slice table
     * every pass above it describes the old one.
     *
     * Nothing is missed by the delay. Every page in such a region already read
     * as written — which is what a region with no recorded history has to say —
     * and no set whose pages were among them armed on this harvest.
     */
    for (i = 0; i < n; i++) {
        bool relock;

        if (!slices[i].needs_log) {
            continue;
        }
        /*
         * The one step of a harvest that needs the BQL: a MemoryRegion
         * transaction. The harvest thread runs everything above without it
         * (RCU and atomic bitmap operations only), so take it here, and only
         * for the regions that ask — after the first surface of a boot, none.
         */
        relock = !bql_locked();
        if (relock) {
            bql_lock();
        }
        /*
         * One MemoryRegion can supply several ranges — on x86 q35 guest RAM is
         * one region aliased below and above the PCI hole — so the insert is
         * what decides, not the flag. g_hash_table_add() reports whether the
         * key was new, which keeps the reference this table owns at one per
         * region however many of its ranges asked.
         */
        qemu_mutex_lock(&d->lock);
        if (g_hash_table_add(d->logged, slices[i].mr)) {
            qemu_mutex_unlock(&d->lock);
            memory_region_ref(slices[i].mr);
            memory_region_set_log(slices[i].mr, true, DIRTY_MEMORY_VGA);
        } else {
            qemu_mutex_unlock(&d->lock);
        }
        if (relock) {
            bql_unlock();
        }
    }
    qemu_mutex_lock(&d->lock);
    d->log_wanted = false;
    qemu_mutex_unlock(&d->lock);
}

/*
 * Bring one set up to date: sync the hypervisor's dirty log for its pages
 * alone, consume their bits into the page stamps, and fold them into its
 * generation. The on-demand half of the tracker; see reims_vgpu_dirty_gen.
 *
 * Runs on the drain without the BQL, which is what makes it possible at all:
 * the ranged sync is RCU and atomic bitmap work, and the one BQL step — turning
 * logging on for a range — is left to the background harvest (`log_wanted`).
 * The set's page list is copied out under the lock and re-looked-up after the
 * sync, so an untrack meanwhile is an answer rather than a use after free.
 */
static void reims_vgpu_dirty_sync_one(ReimsVgpuDirty *d, uint64_t token)
{
    const int shift = qemu_target_page_bits();
    ReimsVgpuDirtySlice *slices = NULL;
    int cap = 0, n, i;
    ReimsVgpuDirtySet *s;
    g_autofree uint64_t *pages = NULL;
    g_autoptr(GArray) ranges = NULL;
    g_autoptr(GArray) written = NULL;
    g_autoptr(GArray) hit = NULL;
    size_t count, p;
    uint64_t page_size, epoch;

    qemu_mutex_lock(&d->lock);
    s = g_hash_table_lookup(d->sets, &token);
    if (!s) {
        qemu_mutex_unlock(&d->lock);
        return;
    }
    epoch = d->epoch;
    /*
     * Up to date already, or being brought up to date for this epoch by the
     * other thread (drain or prefetch): wait for that one rather than repeat
     * its hypervisor calls. The claimant holds no lock while it syncs.
     */
    while (s && s->synced_epoch < epoch && s->claim_epoch >= epoch) {
        qemu_cond_wait(&d->synced_cond, &d->lock);
        s = g_hash_table_lookup(d->sets, &token);
    }
    if (!s || s->synced_epoch >= epoch) {
        qemu_mutex_unlock(&d->lock);
        return;
    }
    s->claim_epoch = epoch;
    count = s->count;
    page_size = s->page_size;
    pages = g_memdup2(s->pages, count * sizeof(uint64_t));
    qemu_mutex_unlock(&d->lock);

    n = reims_vgpu_dirty_ram_slices_into(&slices, &cap);
    qemu_mutex_lock(&d->lock);
    for (i = 0; i < n; i++) {
        slices[i].logged = g_hash_table_contains(d->logged, slices[i].mr);
    }
    qemu_mutex_unlock(&d->lock);

    /* Contiguous runs of this set's pages, per logged region. */
    ranges = g_array_new(FALSE, FALSE, sizeof(MemoryRegionRange));
    for (i = 0; i < n; i++) {
        const ReimsVgpuDirtySlice *sl = &slices[i];
        bool seen = false;
        int j;

        for (j = 0; j < i; j++) {
            seen |= slices[j].logged && slices[j].mr == sl->mr;
        }
        if (!sl->logged || seen) {
            continue;
        }
        g_array_set_size(ranges, 0);
        for (j = i; j < n; j++) {
            const ReimsVgpuDirtySlice *sj = &slices[j];
            MemoryRegionRange r = { 0, 0 };
            bool open = false;

            if (!sj->logged || sj->mr != sl->mr) {
                continue;
            }
            for (p = 0; p < count; p++) {
                uint64_t gpa = pages[p];

                if (gpa < sj->gpa || gpa - sj->gpa >= sj->len) {
                    continue;
                }
                if (open && r.start + r.len + REIMS_VGPU_DIRTY_SYNC_GAP_PAGES *
                    (1ULL << shift) >= sj->offset + (gpa - sj->gpa)) {
                    r.len = sj->offset + (gpa - sj->gpa) + page_size - r.start;
                    continue;
                }
                if (open) {
                    g_array_append_val(ranges, r);
                }
                r.start = sj->offset + (gpa - sj->gpa);
                r.len = page_size;
                open = true;
            }
            if (open) {
                g_array_append_val(ranges, r);
            }
        }
        memory_region_sync_dirty_ranges(sl->mr,
                                        &g_array_index(ranges,
                                                       MemoryRegionRange, 0),
                                        ranges->len);
    }

    written = g_array_new(FALSE, FALSE, sizeof(ReimsVgpuDirtyWritten));
    hit = g_array_new(FALSE, FALSE, sizeof(size_t));
    qemu_mutex_lock(&d->lock);
    s = g_hash_table_lookup(d->sets, &token);
    if (s) {
        reims_vgpu_dirty_consume_set(d, s, slices, n, written);
        if (reims_vgpu_dirty_eval_set(d, s, slices, n, hit, false)) {
            d->log_wanted = true;
        }
        s->synced_epoch = MAX(s->synced_epoch, epoch);
        if (s->claim_epoch == epoch) {
            s->claim_epoch = 0;
        }
    }
    qemu_cond_broadcast(&d->synced_cond);
    qemu_mutex_unlock(&d->lock);
    reims_vgpu_dirty_clear_written(written, slices, n);
    g_free(slices);
}

void reims_vgpu_dirty_set_ondemand(ReimsVgpuDirty *d, bool on)
{
    if (d) {
        const char *op = getenv("REIMS_VGPU_DIRTY_OD_PREFETCH");   /* lab A/B */

        qemu_mutex_lock(&d->lock);
        d->ondemand = on;
        /*
         * On by default (lab: CSS animation ~108 -> ~115 fps, scroll
         * ~88 -> ~100).
         */
        d->od_prefetch = !(op && strcmp(op, "off") == 0);
        qemu_mutex_unlock(&d->lock);
    }
}

/*
 * A doorbell: every guest store before it must be seen by any generation read
 * after it. In on-demand mode that is all a doorbell does — the next read of
 * each set syncs it. Returns whether the background harvest is still owed
 * something (turning logging on), which is the only reason left to ask for one.
 */
bool reims_vgpu_dirty_note_doorbell(ReimsVgpuDirty *d)
{
    bool want;

    if (!d) {
        return false;
    }
    qemu_mutex_lock(&d->lock);
    d->epoch++;
    want = d->log_wanted || g_hash_table_size(d->logged) == 0 || d->od_prefetch;
    qemu_mutex_unlock(&d->lock);
    return want;
}

/*
 * The harvest thread's work: a full harvest, or in on-demand mode with
 * prefetch, a sync of every set read in the last two epochs that is not up to
 * date for this one, so the drain finds them ready. Turning logging on still
 * takes a full harvest.
 */
void reims_vgpu_dirty_background(ReimsVgpuDirty *d)
{
    g_autoptr(GArray) tokens = NULL;
    GHashTableIter it;
    gpointer key, val;
    uint64_t epoch;
    bool full;
    guint t;

    if (!d) {
        return;
    }
    qemu_mutex_lock(&d->lock);
    full = !d->ondemand || !d->od_prefetch || d->log_wanted ||
           g_hash_table_size(d->logged) == 0;
    qemu_mutex_unlock(&d->lock);
    if (full) {
        reims_vgpu_dirty_harvest(d);
        return;
    }
    tokens = g_array_new(FALSE, FALSE, sizeof(uint64_t));
    qemu_mutex_lock(&d->lock);
    epoch = d->epoch;
    g_hash_table_iter_init(&it, d->sets);
    while (g_hash_table_iter_next(&it, &key, &val)) {
        ReimsVgpuDirtySet *set = val;

        if (set->synced_epoch < epoch && set->last_query_epoch + 2 >= epoch) {
            g_array_append_val(tokens, *(uint64_t *)key);
        }
    }
    qemu_mutex_unlock(&d->lock);
    for (t = 0; t < tokens->len; t++) {
        qemu_mutex_lock(&d->lock);
        if (d->epoch != epoch) {
            /* A newer doorbell: the next prefetch covers it from the start. */
            qemu_mutex_unlock(&d->lock);
            break;
        }
        qemu_mutex_unlock(&d->lock);
        reims_vgpu_dirty_sync_one(d, g_array_index(tokens, uint64_t, t));
    }
}
