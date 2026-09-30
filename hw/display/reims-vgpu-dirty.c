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
    uint64_t lab_last_read; /* TEMP diag: harvest count at the last read */
    /*
     * On-demand sync (see reims_vgpu_dirty_gen): the doorbell epoch this set was
     * last brought up to date at, and the global write sequence its generation
     * last accounted for. A page stamped past `seen_seq` was written since.
     */
    uint64_t synced_epoch;
    uint64_t seen_seq;
    uint64_t last_query_epoch;  /* epoch of the last generation or page read */
    uint64_t claim_epoch;       /* epoch a sync of this set is running for, or 0 */
    uint64_t read_order;        /* read counter at this set's first read in its last epoch */
} ReimsVgpuDirtySet;

/* One hypervisor query range of a split job. */
typedef struct ReimsVgpuDirtyChunk {
    MemoryRegion *mr;
    MemoryRegionRange r;
} ReimsVgpuDirtyChunk;

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
     * On-demand mode: the epoch in which each target page frame was last
     * queried from the hypervisor, recorded once that query's bits are in the
     * VGA bitmap. Sets share pages (a framebuffer and the textures that alias
     * it), and a second query of a page in the same epoch can only return
     * stores made after the doorbell, which that epoch does not owe anyone; so
     * a set's sync skips pages already queried in its epoch and takes their
     * bits (or the stamps of whoever took them) like any other. Same indexing
     * and length as `page_seq`.
     */
    uint64_t *page_qepoch;
    /*
     * On-demand mode: doorbells only advance `epoch`; a generation read of a set
     * not synced since the last doorbell syncs that set's pages first. The
     * background harvest is then asked for only to turn logging on
     * (`log_wanted`), the one step that needs the BQL.
     */
    bool ondemand;
    bool log_wanted;
    uint64_t epoch;
    /* TEMP lab A/B (REIMS_VGPU_DIRTY_OD_DELAY_US): hold an on-demand sync until
     * this long after the last doorbell, to test whether the hypervisor's dirty
     * log lags a running vCPU's stores. */
    int64_t doorbell_ns;
    int64_t lab_od_delay_ns;
    /* Lab A/B (REIMS_VGPU_DIRTY_OD_FLUSH=on): flush the accelerator's per-vCPU
     * dirty logs before the first on-demand sync of each epoch. */
    bool od_flush;
    uint64_t flushed_epoch;
    /* Lab A/B (REIMS_VGPU_DIRTY_OD_BATCH=on): the first stale read of an epoch
     * syncs every set read in the previous epoch in one merged range list. */
    bool od_batch;
    uint64_t batched_epoch;
    /* Lab A/B (REIMS_VGPU_DIRTY_OD_PREFETCH=on): at each doorbell the harvest
     * thread syncs the sets read in the last two epochs, in parallel with the
     * drain, which then finds most of them already up to date. */
    bool od_prefetch;
    QemuCond synced_cond;
    /*
     * Prefetch pool (REIMS_VGPU_DIRTY_OD_THREADS, default 7 extra threads):
     * the prefetch's token list is shared with worker threads that take the
     * next token in turn, so an epoch's sets are brought up to date in parallel
     * hypervisor queries and mostly before the drain reaches them.
     */
    QemuThread *od_workers;
    int od_nworkers;
    bool od_stop;
    QemuCond od_work_cond;
    GArray *od_work;          /* uint64_t tokens of the current prefetch */
    guint od_work_next;
    uint64_t od_work_epoch;   /* epoch the list was built for; 0 = none */
    uint64_t read_counter;    /* first reads of a set in an epoch, in order */
    /*
     * The epoch when the drain last read a ring's tail (0 = never, or the lab
     * A/B REIMS_VGPU_DIRTY_WORK_EPOCH=off). The packets it runs were handed over
     * no later than that read, so a generation read for them needs its set
     * synced in this epoch or later — not in every newer one that doorbells
     * arriving meanwhile open, which is what made a lagging drain re-sync its
     * hot sets several times a frame.
     */
    uint64_t work_epoch;
    bool work_epoch_off;
    /*
     * Split queries (lab A/B REIMS_VGPU_DIRTY_OD_SPLIT=on): a sync of a large
     * set hands its ranges out, in pieces, to every idle worker and to any
     * thread waiting for that set's claim, and queries a share itself, so the
     * walk runs in parallel. One job at a time; a sync that finds one running
     * queries alone.
     */
    bool od_split;
    bool od_pbatch;           /* lab A/B REIMS_VGPU_DIRTY_OD_PBATCH=on */
    uint64_t od_hot;          /* epochs a read keeps a set in the prefetch; lab REIMS_VGPU_DIRTY_OD_HOT */
    bool od_splitwait;        /* lab REIMS_VGPU_DIRTY_OD_SPLITWAIT=on */
    bool od_nopticket;        /* lab REIMS_VGPU_DIRTY_OD_PTICKET=off */
    GArray *split_chunks;     /* ReimsVgpuDirtyChunk */
    guint split_next, split_done, split_total;
    bool split_busy;          /* a job is published and its owner not yet done */
    QemuCond split_cond;
    /* Epoch of each channel's latest doorbell (0 = root FIFO). */
    uint64_t chan_epoch[64];
    /*
     * Hypervisor dirty-log queries in flight. A query resets the pages' bits in
     * the hypervisor when it runs and puts them in the VGA bitmap only when it
     * returns; a second query of the same page in between reads it clean, and a
     * consume that follows it would take nothing and call a written page clean
     * (lab: one 4 KiB page stuck stale for ~20 s at 5K, with parallel syncs).
     * So every query holds a ticket from before it starts until its bits are
     * deposited, and a sync consumes only once every ticket issued before its
     * own queries ended has been returned.
     */
    uint64_t q_next;
    uint64_t q_inflight[32];
    int q_ninflight;
    QemuCond q_cond;
    /* TEMP lab diagnostic: on-demand syncs and their cost. */
    uint64_t lab_od_n, lab_od_ns, lab_od_pages, lab_od_skipped;
    uint64_t lab_od_epochs, lab_od_fg_n, lab_od_fg_ns;   /* TEMP lab */
    uint64_t lab_fg_ahead, lab_fg_missed, lab_fg_cold, lab_pb_epoch;  /* TEMP lab */
    uint64_t lab_pb_n, lab_pb_sets, lab_pb_pages, lab_pb_chunks, lab_pb_setpages;  /* TEMP lab */
    int64_t lab_pb_build, lab_pb_query, lab_pb_settle, lab_pb_mark, lab_pb_consume,
            lab_pb_eval, lab_pb_last, lab_pb_lag;
    uint64_t lab_ph_wait, lab_ph_prep, lab_ph_query, lab_ph_settle, lab_ph_consume;  /* TEMP lab */
    int64_t lab_od_last;
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
    d->od_hot = 6;
    qemu_cond_init(&d->synced_cond);
    qemu_cond_init(&d->od_work_cond);
    qemu_cond_init(&d->q_cond);
    qemu_cond_init(&d->split_cond);
    d->split_chunks = g_array_new(FALSE, FALSE, sizeof(ReimsVgpuDirtyChunk));
    d->od_work = g_array_new(FALSE, FALSE, sizeof(uint64_t));
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
    if (d->od_nworkers) {
        int w;

        qemu_mutex_lock(&d->lock);
        d->od_stop = true;
        qemu_cond_broadcast(&d->od_work_cond);
        qemu_mutex_unlock(&d->lock);
        for (w = 0; w < d->od_nworkers; w++) {
            qemu_thread_join(&d->od_workers[w]);
        }
        g_free(d->od_workers);
    }
    g_array_free(d->od_work, TRUE);
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

/* The epoch a generation read must be synced for. d->lock held. */
static uint64_t reims_vgpu_dirty_needed_epoch(ReimsVgpuDirty *d)
{
    return d->work_epoch && !d->work_epoch_off ? MIN(d->work_epoch, d->epoch)
                                               : d->epoch;
}

/*
 * The drain read a ring's tail for work channel `scope`'s doorbells handed
 * over: generation reads for that work need their sets synced for the epoch
 * of that channel's latest doorbell, not for every newer one.
 */
void reims_vgpu_dirty_note_work_scope(ReimsVgpuDirty *d, uint32_t scope)
{
    uint64_t e = 0;

    if (!d) {
        return;
    }
    qemu_mutex_lock(&d->lock);
    if (scope < ARRAY_SIZE(d->chan_epoch)) {
        e = d->chan_epoch[scope];
    }
    d->work_epoch = e ? e : d->epoch;
    qemu_mutex_unlock(&d->lock);
}

uint64_t reims_vgpu_dirty_gen(ReimsVgpuDirty *d, uint64_t token)
{
    ReimsVgpuDirtySet *s;
    uint64_t gen = 0;
    bool stale;
    int64_t fg0;

    if (!d || token == 0) {
        return 0;
    }
    qemu_mutex_lock(&d->lock);
    s = g_hash_table_lookup(d->sets, &token);
    stale = s && d->ondemand && s->synced_epoch < reims_vgpu_dirty_needed_epoch(d);
    if (stale) {   /* TEMP lab */
        if (s->last_query_epoch + d->od_hot < d->epoch) {
            d->lab_fg_cold++;
        } else if (d->lab_pb_epoch >= reims_vgpu_dirty_needed_epoch(d)) {
            d->lab_fg_missed++;
        } else {
            d->lab_fg_ahead++;
        }
    }
    qemu_mutex_unlock(&d->lock);
    fg0 = stale ? get_clock() : 0;
    if (stale) {
        reims_vgpu_dirty_sync_one(d, token);
    }
    qemu_mutex_lock(&d->lock);
    if (stale) {
        d->lab_od_fg_n++;
        d->lab_od_fg_ns += get_clock() - fg0;
    }
    s = g_hash_table_lookup(d->sets, &token);
    if (s) {
        gen = s->gen;
        s->lab_last_read = d->harvests + 1;
        if (s->last_query_epoch != d->epoch) {
            s->read_order = ++d->read_counter;
        }
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
    if (s && d->ondemand && s->synced_epoch < reims_vgpu_dirty_needed_epoch(d)) {
        /* A page list read in a later epoch than the set's last sync would
         * miss what the guest wrote since; bring the set up to date first. */
        int64_t fg0 = get_clock();

        if (s->last_query_epoch + d->od_hot < d->epoch) {   /* TEMP lab */
            d->lab_fg_cold++;
        } else if (d->lab_pb_epoch >= reims_vgpu_dirty_needed_epoch(d)) {
            d->lab_fg_missed++;
        } else {
            d->lab_fg_ahead++;
        }

        qemu_mutex_unlock(&d->lock);
        reims_vgpu_dirty_sync_one(d, token);
        qemu_mutex_lock(&d->lock);
        d->lab_od_fg_n++;
        d->lab_od_fg_ns += get_clock() - fg0;
        s = g_hash_table_lookup(d->sets, &token);
    }
    if (s) {
        if (s->last_query_epoch != d->epoch) {
            s->read_order = ++d->read_counter;
        }
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
/* TEMP diagnostic counters (Windows/WHPX bring-up). */
static uint64_t reims_diag_noslice, reims_diag_unlogged, reims_diag_bit, reims_diag_clean;

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

/* Grow the page stamps to cover target page frames below `end_pfn`. d->lock held. */
static void reims_vgpu_dirty_page_seq_cover(ReimsVgpuDirty *d, uint64_t end_pfn)
{
    uint64_t n;

    if (end_pfn <= d->page_seq_n) {
        return;
    }
    n = MAX(end_pfn, d->page_seq_n * 2);
    d->page_seq = g_renew(uint64_t, d->page_seq, n);
    memset(d->page_seq + d->page_seq_n, 0, (n - d->page_seq_n) * sizeof(uint64_t));
    d->page_qepoch = g_renew(uint64_t, d->page_qepoch, n);
    memset(d->page_qepoch + d->page_seq_n, 0, (n - d->page_seq_n) * sizeof(uint64_t));
    d->page_seq_n = n;
}

/*
 * Move the VGA dirty bit of every target page of one set into the global page
 * stamps, and queue the pages it was set for on `written` for the clear pass.
 * A page already stamped since `pass_seq` — a second set sharing it, in the
 * same pass — is not stamped again. d->lock held.
 */
/*
 * Take the VGA bits of target pages [pfn0, pfn0 + count), which sit at RAM page
 * `page` onward, into the page stamps. A word is read plainly first and only a
 * word with a set bit in range is cleared atomically, so a clean surface costs
 * one load per BITS_PER_LONG pages instead of one locked operation per page.
 *
 * The clear is one atomic step that returns exactly the bits it cleared.
 * Reading bits and clearing them later — even under the tracker's lock — lost
 * writes: another thread's hypervisor query deposits into the same bitmap
 * without that lock, so a write it moved in between was erased unseen (streaks
 * of stale tile pages in the lab). A bit a query sets again after the clear is
 * simply there for the next consumer. d->lock and the RCU read lock held.
 */
static void reims_vgpu_dirty_consume_range(ReimsVgpuDirty *d, DirtyMemoryBlocks *blocks,
                                           uint64_t pfn0, unsigned long page,
                                           unsigned long count)
{
    unsigned long done = 0;

    while (done < count) {
        unsigned long pg = page + done;
        unsigned long idx = pg / DIRTY_MEMORY_BLOCK_SIZE;
        unsigned long off = pg % DIRTY_MEMORY_BLOCK_SIZE;
        unsigned long *w = &blocks->blocks[idx][BIT_WORD(off)];
        unsigned long bit = off % BITS_PER_LONG;
        unsigned long nbits = MIN(BITS_PER_LONG - bit, count - done);
        unsigned long mask = (nbits == BITS_PER_LONG ? ~0UL : ((1UL << nbits) - 1)) << bit;

        if (qatomic_read(w) & mask) {
            unsigned long got = qatomic_fetch_and(w, ~mask) & mask;

            while (got) {
                unsigned long b = ctzl(got);
                uint64_t pfn = pfn0 + done + (b - bit);

                got &= got - 1;
                reims_vgpu_dirty_page_seq_cover(d, pfn + 1);
                d->page_seq[pfn] = ++d->global_seq;
            }
        }
        done += nbits;
    }
}

static void reims_vgpu_dirty_consume_set(ReimsVgpuDirty *d,
                                         const ReimsVgpuDirtySet *s,
                                         const ReimsVgpuDirtySlice *slices,
                                         int n, uint64_t pass_seq,
                                         GArray *written)
{
    const int shift = qemu_target_page_bits();
    DirtyMemoryBlocks *blocks;
    size_t p = 0;

    (void)pass_seq;
    (void)written;
    RCU_READ_LOCK_GUARD();
    blocks = qatomic_rcu_read(&ram_list.dirty_memory[DIRTY_MEMORY_VGA]);
    /* Runs of adjacent pages, each cut at slice ends. */
    while (p < s->count) {
        uint64_t gpa = s->pages[p];
        uint64_t end = gpa + s->page_size;
        int si;

        for (p++; p < s->count && s->pages[p] == end; p++) {
            end += s->page_size;
        }
        while (gpa < end) {
            uint64_t send;

            si = reims_vgpu_dirty_slice_of(slices, n, gpa);
            if (si < 0) {
                gpa += 1ULL << shift;
                continue;
            }
            send = MIN(end, slices[si].gpa + slices[si].len);
            if (slices[si].logged) {
                ram_addr_t ram = slices[si].ram_addr + (gpa - slices[si].gpa);

                reims_vgpu_dirty_consume_range(d, blocks, gpa >> shift, ram >> shift,
                                               (send - gpa) >> shift);
            }
            gpa = send;
        }
    }
}

/*
 * Fold the page stamps into one set's generation and per-page generations: the
 * per-set half of a harvest. `hit` is scratch.
 *
 * `by_harvest` keeps the harvest's arming rule (`arm_at`, counted in
 * harvests). An on-demand evaluation cannot count harvests — there may be none
 * — so it arms a set on its first evaluation with every page in a logged
 * range: logging was on before the set existed, so nothing since its creation
 * went unrecorded. A page in a range not yet logged reads as written, flags the
 * range, and keeps an unarmed set unreadable, as in the harvest. d->lock held.
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

        if (si < 0) {
            reims_diag_noslice++;
            any = true;
            g_array_append_val(hit, p);
            continue;
        }
        if (!slices[si].logged) {
            reims_diag_unlogged++;
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
            reims_diag_bit++;
            any = true;
            g_array_append_val(hit, p);
        } else {
            reims_diag_clean++;
        }
    }
    if (by_harvest) {
        if (s->gen == 0 && unlogged) {
            s->arm_at = d->harvests + 1;
        }
        if (d->harvests < s->arm_at) {
            s->gen = 0;
        } else if (s->gen == 0) {
            s->gen = 1;
        } else if (any) {
            s->gen++;
        }
    } else if (s->gen == 0) {
        if (!unlogged) {
            s->gen = 1;
        }
    } else if (any) {
        s->gen++;
    }
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
 * Clear the VGA bits of the pages a consume pass stamped. Clearing re-protects,
 * so only what came back written is cleared; exactly-adjacent pages merge into
 * one call. No lock needed: RCU and atomic bitmap operations.
 */
static void reims_vgpu_dirty_clear_written(GArray *written,
                                           const ReimsVgpuDirtySlice *slices,
                                           int n)
{
    guint w = 0;

    /* Bits are taken atomically at consumption now; nothing is queued here. */
    if (written->len == 0) {
        return;
    }

    g_array_sort(written, reims_vgpu_dirty_cmp_written);
    while (w < written->len) {
        ReimsVgpuDirtyWritten first = g_array_index(written, ReimsVgpuDirtyWritten, w);
        int si = reims_vgpu_dirty_slice_of(slices, n, first.gpa);
        uint64_t end = first.gpa + first.page_size;

        while (w + 1 < written->len) {
            ReimsVgpuDirtyWritten next =
                g_array_index(written, ReimsVgpuDirtyWritten, w + 1);
            if (next.gpa + next.page_size <= end) {
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
            memory_region_reset_dirty(
                slices[si].mr, slices[si].offset + (first.gpa - slices[si].gpa),
                end - first.gpa, DIRTY_MEMORY_VGA);
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
#define REIMS_VGPU_DIRTY_SYNC_GAP_PAGES_DEFAULT 256
/* Lab A/B: REIMS_VGPU_DIRTY_GAP=<pages> overrides the bridge width. */
static uint64_t reims_vgpu_dirty_gap_pages(void)
{
    static uint64_t gap;

    if (!gap) {
        const char *e = getenv("REIMS_VGPU_DIRTY_GAP");
        long v = e ? atol(e) : 0;

        gap = v > 0 ? (uint64_t)v : REIMS_VGPU_DIRTY_SYNC_GAP_PAGES_DEFAULT;
    }
    return gap;
}
#define REIMS_VGPU_DIRTY_SYNC_GAP_PAGES (reims_vgpu_dirty_gap_pages())

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
/* Take a query ticket before issuing hypervisor queries. d->lock held. */
static uint64_t reims_vgpu_dirty_query_begin(ReimsVgpuDirty *d)
{
    while (d->q_ninflight == (int)ARRAY_SIZE(d->q_inflight)) {
        qemu_cond_wait(&d->q_cond, &d->lock);
    }
    d->q_inflight[d->q_ninflight++] = ++d->q_next;
    return d->q_next;
}

/* Return a ticket whose queries' bits are in the VGA bitmap. d->lock held. */
static void reims_vgpu_dirty_query_end(ReimsVgpuDirty *d, uint64_t ticket)
{
    int i;

    for (i = 0; i < d->q_ninflight; i++) {
        if (d->q_inflight[i] == ticket) {
            d->q_inflight[i] = d->q_inflight[--d->q_ninflight];
            break;
        }
    }
    qemu_cond_broadcast(&d->q_cond);
}

/*
 * Query chunks of the running split job until none are left. d->lock held on
 * entry and on return.
 *
 * Each piece holds its own ticket for just its own query, and the job's owner
 * holds none while the pieces run: a sync that settles meanwhile waits for the
 * few pieces in flight, not for the whole job. The owner settles with no
 * ticket once every piece is done, which waits for every query begun before
 * then — the same guarantee a job-long ticket gave it.
 */
static void reims_vgpu_dirty_split_help(ReimsVgpuDirty *d)
{
    while (d->split_next < d->split_total) {
        ReimsVgpuDirtyChunk c = g_array_index(d->split_chunks, ReimsVgpuDirtyChunk,
                                              d->split_next++);
        uint64_t pt = d->od_nopticket ? 0 : reims_vgpu_dirty_query_begin(d);

        qemu_mutex_unlock(&d->lock);
        memory_region_sync_dirty_ranges(c.mr, &c.r, 1);
        qemu_mutex_lock(&d->lock);
        if (pt) {
            reims_vgpu_dirty_query_end(d, pt);
        }
        if (++d->split_done == d->split_total) {
            qemu_cond_broadcast(&d->split_cond);
        }
    }
}

/*
 * Return a ticket once its queries' bits are in the VGA bitmap, then wait until
 * every query that began before now has done the same. d->lock held.
 */
static void reims_vgpu_dirty_query_settle(ReimsVgpuDirty *d, uint64_t ticket)
{
    uint64_t upto = d->q_next;
    int i;

    reims_vgpu_dirty_query_end(d, ticket);   /* 0 = the caller holds none */
    for (;;) {
        bool pending = false;

        for (i = 0; i < d->q_ninflight; i++) {
            pending |= d->q_inflight[i] <= upto;
        }
        if (!pending) {
            return;
        }
        qemu_cond_wait(&d->q_cond, &d->lock);
    }
}

static void reims_vgpu_dirty_sync_tracked(ReimsVgpuDirty *d,
                                          const ReimsVgpuDirtySlice *slices,
                                          int n)
{
    uint64_t qticket;
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
    qticket = reims_vgpu_dirty_query_begin(d);
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
                                        (MemoryRegionRange *)d->sync_ranges->data,
                                        d->sync_ranges->len);
    }
    qemu_mutex_lock(&d->lock);
    reims_vgpu_dirty_query_settle(d, qticket);
    qemu_mutex_unlock(&d->lock);
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
    /* TEMP diagnostic (Windows/WHPX bring-up): why pages read as written. */
    if ((d->harvests % 256) == 0) {
        GHashTableIter hit_it;
        gpointer hk, hv;
        uint64_t hot1 = 0, hot1_pages = 0, hot120 = 0, hot120_pages = 0, all_pages = 0;

        g_hash_table_iter_init(&hit_it, d->sets);
        while (g_hash_table_iter_next(&hit_it, &hk, &hv)) {
            ReimsVgpuDirtySet *hs = hv;
            uint64_t pg = hs->count * (hs->page_size / 4096);

            all_pages += pg;
            if (hs->lab_last_read + 1 >= d->harvests) {
                hot1++;
                hot1_pages += pg;
            }
            if (hs->lab_last_read + 120 >= d->harvests) {
                hot120++;
                hot120_pages += pg;
            }
        }
        fprintf(stderr, "reims-dirty-hot sets=%u pages=%" PRIu64 " hot1=%" PRIu64
                "/%" PRIu64 "pg hot120=%" PRIu64 "/%" PRIu64 "pg\n",
                g_hash_table_size(d->sets), all_pages, hot1, hot1_pages,
                hot120, hot120_pages);
        fprintf(stderr, "reims-dirty-diag harvests=%" PRIu64 " sets=%u slices=%d"
                " noslice=%" PRIu64 " unlogged=%" PRIu64 " bit=%" PRIu64
                " clean=%" PRIu64 " logged_regions=%u\n",
                (uint64_t)d->harvests, g_hash_table_size(d->sets), n,
                reims_diag_noslice, reims_diag_unlogged, reims_diag_bit,
                reims_diag_clean, g_hash_table_size(d->logged));
        reims_diag_noslice = reims_diag_unlogged = reims_diag_bit =
            reims_diag_clean = 0;
    }
    /*
     * TEMP diagnostic (Windows/WHPX bring-up): how clustered the tracked pages
     * are in guest-physical space. A targeted dirty query costs per page
     * walked plus per call, so the runs (merging gaps up to g pages) and the
     * pages they cover say what a range-limited sync would cost.
     */
    if ((d->harvests % 256) == 0) {
        static const uint64_t gaps[] = { 0, 16, 256, 4096 };
        uint64_t max_pfn = 0, distinct = 0, pfn;
        unsigned long *bm;
        GHashTableIter dit;
        gpointer dk, dv;
        unsigned gi;

        for (i = 0; i < n; i++) {
            max_pfn = MAX(max_pfn, (slices[i].gpa + slices[i].len) >> 12);
        }
        bm = bitmap_new(max_pfn + 1);
        g_hash_table_iter_init(&dit, d->sets);
        while (g_hash_table_iter_next(&dit, &dk, &dv)) {
            ReimsVgpuDirtySet *s = dv;
            size_t p;
            uint64_t off;

            for (p = 0; p < s->count; p++) {
                for (off = 0; off < s->page_size; off += 4096) {
                    pfn = (s->pages[p] + off) >> 12;
                    if (pfn <= max_pfn && !test_and_set_bit(pfn, bm)) {
                        distinct++;
                    }
                }
            }
        }
        fprintf(stderr, "reims-dirty-cluster harvests=%" PRIu64
                " distinct_pages=%" PRIu64 " max_pfn=%" PRIu64,
                (uint64_t)d->harvests, distinct, max_pfn);
        for (gi = 0; gi < ARRAY_SIZE(gaps); gi++) {
            uint64_t runs = 0, covered = 0, run_start = 0, last = 0;
            bool open = false;

            for (pfn = find_first_bit(bm, max_pfn + 1); pfn <= max_pfn;
                 pfn = find_next_bit(bm, max_pfn + 1, pfn + 1)) {
                if (open && pfn - last - 1 <= gaps[gi]) {
                    last = pfn;
                    continue;
                }
                if (open) {
                    covered += last - run_start + 1;
                }
                runs++;
                run_start = last = pfn;
                open = true;
            }
            if (open) {
                covered += last - run_start + 1;
            }
            fprintf(stderr, " g%" PRIu64 "=%" PRIu64 "runs/%" PRIu64 "pg",
                    gaps[gi], runs, covered);
        }
        fprintf(stderr, "\n");
        g_free(bm);
    }
    /*
     * Consume every tracked page's bit into the page stamps first, then fold
     * the stamps into each set. Two passes, because a bit may be consumed only
     * once and a page may belong to several sets; the stamp is what they share.
     */
    {
        uint64_t pass_seq = d->global_seq;

        g_hash_table_iter_init(&it, d->sets);
        while (g_hash_table_iter_next(&it, &key, &val)) {
            reims_vgpu_dirty_consume_set(d, val, slices, n, pass_seq, written);
        }
        g_hash_table_iter_init(&it, d->sets);
        while (g_hash_table_iter_next(&it, &key, &val)) {
            ReimsVgpuDirtySet *set = val;

            reims_vgpu_dirty_eval_set(d, set, slices, n, hit, true);
            set->synced_epoch = MAX(set->synced_epoch, epoch);
        }
    }
    qemu_mutex_unlock(&d->lock);

    /*
     * Clear only the pages that came back written. Clearing re-protects, so
     * clearing the whole tracked window would make every page the guest has
     * mapped refault after every harvest; clearing what was written costs
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
static int reims_vgpu_dirty_cmp_u64_asc(gconstpointer a, gconstpointer b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;

    return x < y ? -1 : x > y;
}

/*
 * The first stale read of an epoch: sync, in one merged range list, every set
 * read in the previous epoch (and the one asked for), and fold them all. A
 * drain pass reads mostly the sets the previous pass read, and their pages sit
 * near each other, so one list costs far fewer hypervisor calls than a list
 * per set. Returns whether the asked-for set was brought up to date here.
 */
static bool reims_vgpu_dirty_sync_batch(ReimsVgpuDirty *d, uint64_t token)
{
    uint64_t qticket;
    const int shift = qemu_target_page_bits();
    const uint64_t target = 1ULL << shift;
    ReimsVgpuDirtySlice *slices = NULL;
    int cap = 0, n, i;
    g_autoptr(GArray) tokens = g_array_new(FALSE, FALSE, sizeof(uint64_t));
    g_autoptr(GArray) pages = g_array_new(FALSE, FALSE, sizeof(uint64_t));
    g_autoptr(GArray) ranges = NULL;
    g_autoptr(GArray) written = NULL;
    g_autoptr(GArray) hit = NULL;
    GHashTableIter it;
    gpointer key, val;
    uint64_t epoch;
    bool covered = false;
    guint t, pi;
    int64_t t0 = get_clock();

    qemu_mutex_lock(&d->lock);
    epoch = d->epoch;
    if (d->batched_epoch >= epoch) {
        qemu_mutex_unlock(&d->lock);
        return false;
    }
    d->batched_epoch = epoch;
    g_hash_table_iter_init(&it, d->sets);
    while (g_hash_table_iter_next(&it, &key, &val)) {
        ReimsVgpuDirtySet *set = val;
        uint64_t tk = *(uint64_t *)key;
        size_t p;
        uint64_t off;

        if (set->synced_epoch >= epoch ||
            (tk != token && set->last_query_epoch + 1 < epoch)) {
            continue;
        }
        g_array_append_val(tokens, tk);
        covered |= tk == token;
        for (p = 0; p < set->count; p++) {
            for (off = 0; off < set->page_size; off += target) {
                uint64_t g = set->pages[p] + off;

                g_array_append_val(pages, g);
            }
        }
    }
    qemu_mutex_unlock(&d->lock);
    if (tokens->len == 0) {
        return false;
    }
    g_array_sort(pages, reims_vgpu_dirty_cmp_u64_asc);

    n = reims_vgpu_dirty_ram_slices_into(&slices, &cap);
    qemu_mutex_lock(&d->lock);
    for (i = 0; i < n; i++) {
        slices[i].logged = g_hash_table_contains(d->logged, slices[i].mr);
    }
    qticket = reims_vgpu_dirty_query_begin(d);
    qemu_mutex_unlock(&d->lock);

    ranges = g_array_new(FALSE, FALSE, sizeof(MemoryRegionRange));
    for (i = 0; i < n; i++) {
        bool seen = false;
        int j;

        for (j = 0; j < i; j++) {
            seen |= slices[j].logged && slices[j].mr == slices[i].mr;
        }
        if (!slices[i].logged || seen) {
            continue;
        }
        g_array_set_size(ranges, 0);
        for (j = i; j < n; j++) {
            const ReimsVgpuDirtySlice *sj = &slices[j];
            MemoryRegionRange r = { 0, 0 };
            bool open = false;

            if (!sj->logged || sj->mr != slices[i].mr) {
                continue;
            }
            for (pi = 0; pi < pages->len; pi++) {
                uint64_t gpa = g_array_index(pages, uint64_t, pi);
                uint64_t off;

                if (gpa < sj->gpa || gpa - sj->gpa >= sj->len) {
                    continue;
                }
                off = sj->offset + (gpa - sj->gpa);
                if (open && off < r.start + r.len) {
                    continue;   /* the same page, from a second set */
                }
                if (open && r.start + r.len +
                    REIMS_VGPU_DIRTY_SYNC_GAP_PAGES * target >= off) {
                    r.len = off + target - r.start;
                    continue;
                }
                if (open) {
                    g_array_append_val(ranges, r);
                }
                r.start = off;
                r.len = target;
                open = true;
            }
            if (open) {
                g_array_append_val(ranges, r);
            }
        }
        memory_region_sync_dirty_ranges(slices[i].mr,
                                        (MemoryRegionRange *)ranges->data,
                                        ranges->len);
    }

    written = g_array_new(FALSE, FALSE, sizeof(ReimsVgpuDirtyWritten));
    hit = g_array_new(FALSE, FALSE, sizeof(size_t));
    qemu_mutex_lock(&d->lock);
    reims_vgpu_dirty_query_settle(d, qticket);
    {
        uint64_t pass_seq = d->global_seq;

        for (t = 0; t < tokens->len; t++) {
            ReimsVgpuDirtySet *set = g_hash_table_lookup(d->sets,
                                        &g_array_index(tokens, uint64_t, t));
            if (set) {
                reims_vgpu_dirty_consume_set(d, set, slices, n, pass_seq,
                                             written);
            }
        }
    }
    for (t = 0; t < tokens->len; t++) {
        ReimsVgpuDirtySet *set = g_hash_table_lookup(d->sets,
                                    &g_array_index(tokens, uint64_t, t));
        if (set) {
            if (reims_vgpu_dirty_eval_set(d, set, slices, n, hit, false)) {
                d->log_wanted = true;
            }
            set->synced_epoch = MAX(set->synced_epoch, epoch);
        }
    }
    d->lab_od_n += tokens->len;
    d->lab_od_ns += get_clock() - t0;
    d->lab_od_pages += pages->len;
    qemu_mutex_unlock(&d->lock);
    reims_vgpu_dirty_clear_written(written, slices, n);
    g_free(slices);
    return covered;
}

static void reims_vgpu_dirty_sync_one(ReimsVgpuDirty *d, uint64_t token)
{
    const int shift = qemu_target_page_bits();
    ReimsVgpuDirtySlice *slices = NULL;
    int cap = 0, n, i;
    ReimsVgpuDirtySet *s;
    g_autofree uint64_t *pages = NULL;
    g_autoptr(GArray) ranges = NULL;
    g_autoptr(GArray) walked = NULL;
    g_autoptr(GArray) written = NULL;
    g_autoptr(GArray) hit = NULL;
    size_t count, p;
    uint64_t page_size, epoch, qticket, walked_pages = 0;
    int64_t ph = 0, ph_prep_end = 0, ph_query_end = 0, ph_settle_end = 0;
    g_autoptr(GArray) chunks = g_array_new(FALSE, FALSE, sizeof(ReimsVgpuDirtyChunk));
    int64_t t0;

    if (d->od_batch && reims_vgpu_dirty_sync_batch(d, token)) {
        return;
    }
    t0 = get_clock();
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
    ph = get_clock();
    while (s && s->synced_epoch < epoch && s->claim_epoch >= epoch) {
        /* Waiting for the claimant: query its split chunks meanwhile. */
        if (d->split_next < d->split_total) {
            reims_vgpu_dirty_split_help(d);
        } else {
            qemu_cond_wait(&d->synced_cond, &d->lock);
        }
        s = g_hash_table_lookup(d->sets, &token);
    }
    d->lab_ph_wait += get_clock() - ph;
    if (!s || s->synced_epoch >= epoch) {
        qemu_mutex_unlock(&d->lock);
        return;
    }
    ph = get_clock();
    s->claim_epoch = epoch;
    page_size = s->page_size;
    pages = g_new(uint64_t, s->count);
    count = 0;
    for (p = 0; p < s->count; p++) {
        uint64_t pfn = s->pages[p] >> shift;

        if (pfn < d->page_seq_n && d->page_qepoch[pfn] >= epoch) {
            d->lab_od_skipped++;
            continue;
        }
        pages[count++] = s->pages[p];
    }
    {
        int64_t wait = d->doorbell_ns + d->lab_od_delay_ns - get_clock();

        bool flush = d->od_flush && d->flushed_epoch < epoch &&
                     memory_dirty_log_flush_hook;

        if (flush) {
            d->flushed_epoch = epoch;
        }
        qemu_mutex_unlock(&d->lock);
        if (d->lab_od_delay_ns > 0 && wait > 0) {
            g_usleep(wait / 1000);
        }
        if (flush) {
            memory_dirty_log_flush_hook();
        }
    }

    n = reims_vgpu_dirty_ram_slices_into(&slices, &cap);
    qemu_mutex_lock(&d->lock);
    for (i = 0; i < n; i++) {
        slices[i].logged = g_hash_table_contains(d->logged, slices[i].mr);
    }
    qemu_mutex_unlock(&d->lock);

    /* Contiguous runs of this set's pages, per logged region. */
    ranges = g_array_new(FALSE, FALSE, sizeof(MemoryRegionRange));
    walked = g_array_new(FALSE, FALSE, sizeof(MemoryRegionRange));  /* GPA spans */
    qemu_mutex_lock(&d->lock);
    qticket = reims_vgpu_dirty_query_begin(d);
    qemu_mutex_unlock(&d->lock);
    ph_prep_end = get_clock();
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
                    MemoryRegionRange w = { sj->gpa + (r.start - sj->offset), r.len };

                    g_array_append_val(ranges, r);
                    g_array_append_val(walked, w);
                }
                r.start = sj->offset + (gpa - sj->gpa);
                r.len = page_size;
                open = true;
            }
            if (open) {
                MemoryRegionRange w = { sj->gpa + (r.start - sj->offset), r.len };

                g_array_append_val(ranges, r);
                g_array_append_val(walked, w);
            }
        }
        {
            guint k;
            uint64_t walk = 0;

            for (k = 0; k < ranges->len; k++) {
                ReimsVgpuDirtyChunk c = {
                    sl->mr, g_array_index(ranges, MemoryRegionRange, k)
                };

                walk += c.r.len >> shift;
                g_array_append_val(chunks, c);
            }
            walked_pages += walk;
        }
    }
    /*
     * A large foreground sync splits its queries across the prefetch workers;
     * anything else (a worker's own sync, a small set, a job already running)
     * queries alone, one call per region as before.
     */
    qemu_mutex_lock(&d->lock);
    if (d->od_split && d->od_nworkers > 0 && walked_pages >= 4096 &&
        !d->split_busy) {
        guint k;

        /* The pieces carry their own tickets; this one has queried nothing. */
        if (!d->od_nopticket) {
            reims_vgpu_dirty_query_end(d, qticket);
            qticket = 0;
        }
        /* Pieces of at most 1024 pages, so a set of a few long runs still
         * spreads over every worker. */
        g_array_set_size(d->split_chunks, 0);
        for (k = 0; k < chunks->len; k++) {
            ReimsVgpuDirtyChunk c = g_array_index(chunks, ReimsVgpuDirtyChunk, k);
            const uint64_t piece = 1024ULL << shift;

            while (c.r.len > piece) {
                ReimsVgpuDirtyChunk head = { c.mr, { c.r.start, piece } };

                g_array_append_val(d->split_chunks, head);
                c.r.start += piece;
                c.r.len -= piece;
            }
            g_array_append_val(d->split_chunks, c);
        }
        d->split_next = d->split_done = 0;
        d->split_total = d->split_chunks->len;
        d->split_busy = true;
        qemu_cond_broadcast(&d->od_work_cond);
        qemu_cond_broadcast(&d->synced_cond);   /* claim waiters help too */
        reims_vgpu_dirty_split_help(d);
        while (d->split_done < d->split_total) {
            qemu_cond_wait(&d->split_cond, &d->lock);
        }
        d->split_total = d->split_done = d->split_next = 0;
        d->split_busy = false;
        qemu_cond_broadcast(&d->split_cond);   /* a batch may be waiting to split */
        qemu_mutex_unlock(&d->lock);
    } else {
        guint k = 0;

        qemu_mutex_unlock(&d->lock);
        while (k < chunks->len) {
            MemoryRegion *mr = g_array_index(chunks, ReimsVgpuDirtyChunk, k).mr;
            guint e = k;

            g_array_set_size(ranges, 0);
            while (e < chunks->len &&
                   g_array_index(chunks, ReimsVgpuDirtyChunk, e).mr == mr) {
                g_array_append_val(ranges, g_array_index(chunks, ReimsVgpuDirtyChunk, e).r);
                e++;
            }
            memory_region_sync_dirty_ranges(mr, (MemoryRegionRange *)ranges->data,
                                            ranges->len);
            k = e;
        }
    }

    written = g_array_new(FALSE, FALSE, sizeof(ReimsVgpuDirtyWritten));
    hit = g_array_new(FALSE, FALSE, sizeof(size_t));
    qemu_mutex_lock(&d->lock);
    ph_query_end = get_clock();
    reims_vgpu_dirty_query_settle(d, qticket);
    ph_settle_end = get_clock();
    d->lab_ph_prep += ph_prep_end - ph;
    d->lab_ph_query += ph_query_end - ph_prep_end;
    d->lab_ph_settle += ph_settle_end - ph_query_end;
    /*
     * Every page the queries walked, the gaps they bridged included: those gaps
     * are mostly other sets' pages (surfaces interleave in guest RAM), whose
     * bits are now in the VGA bitmap too. Only logged slices were queried, so a
     * page outside them stays unmarked (the consume reads it as written).
     */
    for (p = 0; p < walked->len; p++) {
        const MemoryRegionRange *w = &g_array_index(walked, MemoryRegionRange, p);
        uint64_t pfn = w->start >> shift, end = (w->start + w->len) >> shift;

        reims_vgpu_dirty_page_seq_cover(d, end);
        for (; pfn < end; pfn++) {
            d->page_qepoch[pfn] = MAX(d->page_qepoch[pfn], epoch);
        }
    }
    s = g_hash_table_lookup(d->sets, &token);
    if (s) {
        reims_vgpu_dirty_consume_set(d, s, slices, n, d->global_seq, written);
        if (reims_vgpu_dirty_eval_set(d, s, slices, n, hit, false)) {
            d->log_wanted = true;
        }
        s->synced_epoch = MAX(s->synced_epoch, epoch);
        if (s->claim_epoch == epoch) {
            s->claim_epoch = 0;
        }
    }
    qemu_cond_broadcast(&d->synced_cond);
    {
        int64_t now = get_clock();

        d->lab_ph_consume += now - ph_settle_end;
        d->lab_od_n++;
        d->lab_od_ns += now - t0;
        d->lab_od_pages += count;
        if (now - d->lab_od_last >= NANOSECONDS_PER_SECOND) {
            fprintf(stderr, "reims-dirty-ondemand epoch_ms=%" PRId64 " syncs=%"
                    PRIu64 " us=%" PRIu64 " pages=%" PRIu64 " skipped=%" PRIu64
                    " epochs=%" PRIu64 " fg=%" PRIu64 " fg_us=%" PRIu64
                    " wait_us=%" PRIu64 " prep_us=%" PRIu64 " query_us=%" PRIu64
                    " settle_us=%" PRIu64 " consume_us=%" PRIu64
                    " fg_ahead=%" PRIu64 " fg_missed=%" PRIu64 " fg_cold=%" PRIu64
                    "\n",
                    g_get_real_time() / 1000, d->lab_od_n, d->lab_od_ns / 1000,
                    d->lab_od_pages, d->lab_od_skipped, d->lab_od_epochs,
                    d->lab_od_fg_n, d->lab_od_fg_ns / 1000, d->lab_ph_wait / 1000,
                    d->lab_ph_prep / 1000, d->lab_ph_query / 1000,
                    d->lab_ph_settle / 1000, d->lab_ph_consume / 1000,
                    d->lab_fg_ahead, d->lab_fg_missed, d->lab_fg_cold);
            d->lab_fg_ahead = d->lab_fg_missed = d->lab_fg_cold = 0;
            d->lab_ph_wait = d->lab_ph_prep = d->lab_ph_query = 0;
            d->lab_ph_settle = d->lab_ph_consume = 0;
            d->lab_od_n = d->lab_od_ns = d->lab_od_pages = d->lab_od_skipped = 0;
            d->lab_od_epochs = d->lab_od_fg_n = d->lab_od_fg_ns = 0;
            d->lab_od_last = now;
        }
    }
    qemu_mutex_unlock(&d->lock);
    reims_vgpu_dirty_clear_written(written, slices, n);
    g_free(slices);
}

/*
 * Take the prefetch's tokens in turn until the list is used up or a newer
 * doorbell makes it stale. Called by the harvest thread and by each worker.
 */
static void reims_vgpu_dirty_prefetch_drain(ReimsVgpuDirty *d, uint64_t epoch)
{
    for (;;) {
        uint64_t token;

        qemu_mutex_lock(&d->lock);
        if (d->od_stop || d->od_work_epoch != epoch || d->epoch != epoch ||
            d->od_work_next >= d->od_work->len) {
            qemu_mutex_unlock(&d->lock);
            return;
        }
        token = g_array_index(d->od_work, uint64_t, d->od_work_next++);
        qemu_mutex_unlock(&d->lock);
        reims_vgpu_dirty_sync_one(d, token);
    }
}

static void *reims_vgpu_dirty_od_worker(void *opaque)
{
    ReimsVgpuDirty *d = opaque;
    uint64_t done = 0;

    rcu_register_thread();   /* sync_one walks the FlatView under RCU */
    qemu_mutex_lock(&d->lock);
    while (!d->od_stop) {
        uint64_t epoch = d->od_work_epoch;

        if (d->split_next < d->split_total) {
            reims_vgpu_dirty_split_help(d);
            continue;
        }
        if (epoch == done || epoch == 0) {
            qemu_cond_wait(&d->od_work_cond, &d->lock);
            continue;
        }
        done = epoch;
        qemu_mutex_unlock(&d->lock);
        reims_vgpu_dirty_prefetch_drain(d, epoch);
        qemu_mutex_lock(&d->lock);
    }
    qemu_mutex_unlock(&d->lock);
    rcu_unregister_thread();
    return NULL;
}

void reims_vgpu_dirty_set_ondemand(ReimsVgpuDirty *d, bool on)
{
    if (d) {
        const char *dl = getenv("REIMS_VGPU_DIRTY_OD_DELAY_US");   /* lab A/B */
        const char *fl = getenv("REIMS_VGPU_DIRTY_OD_FLUSH");      /* lab A/B */
        const char *ob = getenv("REIMS_VGPU_DIRTY_OD_BATCH");      /* lab A/B */
        const char *op = getenv("REIMS_VGPU_DIRTY_OD_PREFETCH");   /* lab A/B */

        qemu_mutex_lock(&d->lock);
        d->ondemand = on;
        d->od_flush = fl && strcmp(fl, "on") == 0;
        d->od_batch = ob && strcmp(ob, "on") == 0;
        /* On by default (lab: CSS animation ~108 -> ~115, scroll ~88 -> ~100). */
        d->od_prefetch = !(op && strcmp(op, "off") == 0);
        d->lab_od_delay_ns = dl ? (int64_t)atoi(dl) * 1000 : 0;
        /*
         * Defaults measured on the 5120x2160 Dell (2026-09-29): a set read once a
         * frame spans ~3.5 doorbells, so the prefetch keeps sets read in the last
         * 6; one batched prefetch per doorbell; large syncs split across the
         * pool; a batch helps a running split instead of walking alone. Each has
         * a lab switch that turns it back off.
         */
        d->od_hot = getenv("REIMS_VGPU_DIRTY_OD_HOT") ?                /* lab A/B */
                    MAX(1, atoi(getenv("REIMS_VGPU_DIRTY_OD_HOT"))) : 6;
        d->od_splitwait = !(getenv("REIMS_VGPU_DIRTY_OD_SPLITWAIT") && /* lab A/B */
                            strcmp(getenv("REIMS_VGPU_DIRTY_OD_SPLITWAIT"), "off") == 0);
        d->od_nopticket = getenv("REIMS_VGPU_DIRTY_OD_PTICKET") &&    /* lab A/B */
                          strcmp(getenv("REIMS_VGPU_DIRTY_OD_PTICKET"), "off") == 0;
        d->od_pbatch = !(getenv("REIMS_VGPU_DIRTY_OD_PBATCH") &&      /* lab A/B */
                         strcmp(getenv("REIMS_VGPU_DIRTY_OD_PBATCH"), "off") == 0);
        d->od_split = !(getenv("REIMS_VGPU_DIRTY_OD_SPLIT") &&        /* lab A/B */
                        strcmp(getenv("REIMS_VGPU_DIRTY_OD_SPLIT"), "off") == 0);
        d->work_epoch_off = getenv("REIMS_VGPU_DIRTY_WORK_EPOCH") &&   /* lab A/B */
                            strcmp(getenv("REIMS_VGPU_DIRTY_WORK_EPOCH"), "off") == 0;
        qemu_mutex_unlock(&d->lock);
        if (on && d->od_prefetch && d->od_nworkers == 0) {
            const char *nt = getenv("REIMS_VGPU_DIRTY_OD_THREADS");   /* lab A/B */
            int w, want = nt ? atoi(nt) : 7;

            want = MAX(0, MIN(want, 8));
            d->od_workers = g_new0(QemuThread, MAX(want, 1));
            for (w = 0; w < want; w++) {
                qemu_thread_create(&d->od_workers[w], "reims-vgpu-od",
                                   reims_vgpu_dirty_od_worker, d,
                                   QEMU_THREAD_JOINABLE);
            }
            d->od_nworkers = want;
        }
    }
}

/*
 * A doorbell: every guest store before it must be seen by any generation read
 * after it. In on-demand mode that is all a doorbell does — the next read of
 * each set syncs it. Returns whether the background harvest is still owed
 * something (turning logging on), which is the only reason left to ask for one.
 */
bool reims_vgpu_dirty_note_doorbell(ReimsVgpuDirty *d, int channel)
{
    bool want;
    guint c;

    if (!d) {
        return false;
    }
    qemu_mutex_lock(&d->lock);
    d->epoch++;
    if (channel >= 0 && channel < (int)ARRAY_SIZE(d->chan_epoch)) {
        d->chan_epoch[channel] = d->epoch;
    } else {
        /* Not a doorbell the device names: binds every channel. */
        for (c = 0; c < ARRAY_SIZE(d->chan_epoch); c++) {
            d->chan_epoch[c] = d->epoch;
        }
    }
    d->lab_od_epochs++;
    d->doorbell_ns = get_clock();
    want = d->log_wanted || g_hash_table_size(d->logged) == 0 || d->od_prefetch;
    qemu_mutex_unlock(&d->lock);
    return want;
}

/*
 * The harvest thread's work in on-demand mode: sync every set read in the last
 * two epochs that is not up to date for this one, most recently read first, so
 * the drain finds them ready. Turning logging on still takes a full harvest.
 */
typedef struct ReimsVgpuDirtyOrdered {
    uint64_t order;
    uint64_t token;
} ReimsVgpuDirtyOrdered;

static gint reims_vgpu_dirty_cmp_ordered(gconstpointer a, gconstpointer b)
{
    uint64_t x = ((const ReimsVgpuDirtyOrdered *)a)->order;
    uint64_t y = ((const ReimsVgpuDirtyOrdered *)b)->order;

    return x < y ? -1 : x > y;
}

static gint reims_vgpu_dirty_cmp_span(gconstpointer a, gconstpointer b)
{
    uint64_t x = ((const MemoryRegionRange *)a)->start;
    uint64_t y = ((const MemoryRegionRange *)b)->start;

    return x < y ? -1 : x > y;
}

/*
 * Prefetch batch: bring every hot set up to date for `epoch` under one query
 * ticket. The sets are claimed first, so a drain that needs one waits for the
 * batch (and queries its pieces meanwhile) instead of syncing it alone; their
 * pages are walked once as a union, pages another sync walked this epoch are
 * skipped, and the walk is split across the pool. One ticket in flight instead
 * of one per worker is what the ticket settle waits on.
 */
static void reims_vgpu_dirty_prefetch_batch(ReimsVgpuDirty *d, GArray *tokens,
                                            uint64_t epoch)
{
    const int shift = qemu_target_page_bits();
    const uint64_t target = 1ULL << shift;
    ReimsVgpuDirtySlice *slices = NULL;
    int cap = 0, n, i;
    g_autoptr(GArray) claimed = g_array_new(FALSE, FALSE, sizeof(uint64_t));
    g_autoptr(GArray) pages = g_array_new(FALSE, FALSE, sizeof(MemoryRegionRange));
    g_autoptr(GArray) chunks = g_array_new(FALSE, FALSE, sizeof(ReimsVgpuDirtyChunk));
    g_autoptr(GArray) walked = g_array_new(FALSE, FALSE, sizeof(MemoryRegionRange));
    g_autoptr(GArray) written = NULL;
    g_autoptr(GArray) hit = NULL;
    uint64_t qticket, npages = 0;
    guint t, pi;
    int64_t pb0 = get_clock(), pb1, pb2, pb3, pb4, pb5, pb6, pb7;  /* TEMP lab */
    uint64_t setpages = 0;

    qemu_mutex_lock(&d->lock);
    if (d->epoch != epoch) {
        qemu_mutex_unlock(&d->lock);
        return;
    }
    d->lab_pb_lag += pb0 - d->doorbell_ns;
    d->lab_pb_epoch = epoch;
    /*
     * The union of the claimed sets' pages (GPA). Pages another sync walked
     * this epoch are walked again here: a second query of a page is harmless,
     * and filtering page by page cost more than the query it saved.
     */
    for (t = 0; t < tokens->len; t++) {
        uint64_t tk = g_array_index(tokens, uint64_t, t);
        ReimsVgpuDirtySet *set = g_hash_table_lookup(d->sets, &tk);
        size_t p;

        if (!set || set->synced_epoch >= epoch || set->claim_epoch >= epoch) {
            continue;
        }
        set->claim_epoch = epoch;
        g_array_append_val(claimed, tk);
        setpages += set->count * (set->page_size / target);
        /* The set's runs of adjacent pages; the sort below merges the sets. */
        for (p = 0; p < set->count;) {
            MemoryRegionRange run = { set->pages[p], set->page_size };

            for (p++; p < set->count && set->pages[p] == run.start + run.len; p++) {
                run.len += set->page_size;
            }
            g_array_append_val(pages, run);
        }
    }
    qemu_mutex_unlock(&d->lock);
    g_array_sort(pages, reims_vgpu_dirty_cmp_span);
    if (claimed->len == 0) {
        return;
    }

    n = reims_vgpu_dirty_ram_slices_into(&slices, &cap);
    qemu_mutex_lock(&d->lock);
    for (i = 0; i < n; i++) {
        slices[i].logged = g_hash_table_contains(d->logged, slices[i].mr);
    }
    qemu_mutex_unlock(&d->lock);

    /*
     * The union of the runs, cut at slice ends, merged across gaps below the
     * sync gap within one region; a page two sets share walks once.
     */
    for (pi = 0; pi < pages->len; pi++) {
        const MemoryRegionRange *run = &g_array_index(pages, MemoryRegionRange, pi);
        uint64_t gpa = run->start, end = run->start + run->len;

        while (gpa < end) {
            uint64_t send, off;
            ReimsVgpuDirtyChunk *last;

            i = reims_vgpu_dirty_slice_of(slices, n, gpa);
            if (i < 0) {
                gpa += target;
                continue;
            }
            send = MIN(end, slices[i].gpa + slices[i].len);
            if (!slices[i].logged) {
                gpa = send;
                continue;
            }
            off = slices[i].offset + (gpa - slices[i].gpa);
            last = chunks->len ? &g_array_index(chunks, ReimsVgpuDirtyChunk,
                                                chunks->len - 1) : NULL;
            if (last && last->mr == slices[i].mr && off >= last->r.start &&
                last->r.start + last->r.len +
                REIMS_VGPU_DIRTY_SYNC_GAP_PAGES * target >= off) {
                uint64_t nend = MAX(last->r.start + last->r.len, off + (send - gpa));

                npages += (nend - (last->r.start + last->r.len)) >> shift;
                last->r.len = nend - last->r.start;
            } else {
                ReimsVgpuDirtyChunk c = { slices[i].mr, { off, send - gpa } };

                npages += (send - gpa) >> shift;
                g_array_append_val(chunks, c);
            }
            gpa = send;
        }
    }
    /*
     * Every page the queries walk, in GPA: each chunk mapped back through the
     * logged slices of its region, the gaps it bridges included.
     */
    for (pi = 0; pi < chunks->len; pi++) {
        const ReimsVgpuDirtyChunk *c = &g_array_index(chunks, ReimsVgpuDirtyChunk, pi);

        for (i = 0; i < n; i++) {
            uint64_t lo, hi;

            if (slices[i].mr != c->mr || !slices[i].logged) {
                continue;
            }
            lo = MAX(c->r.start, slices[i].offset);
            hi = MIN(c->r.start + c->r.len, slices[i].offset + slices[i].len);
            if (lo < hi) {
                MemoryRegionRange w = { slices[i].gpa + (lo - slices[i].offset),
                                        hi - lo };

                g_array_append_val(walked, w);
            }
        }
    }

    pb1 = get_clock();
    qemu_mutex_lock(&d->lock);
    qticket = 0;   /* the split pieces carry their own tickets */
    while (d->od_splitwait && d->od_nworkers > 0 && chunks->len && d->split_busy) {
        reims_vgpu_dirty_split_help(d);
        if (d->split_busy) {
            qemu_cond_wait(&d->split_cond, &d->lock);
        }
    }
    if (d->od_nworkers > 0 && chunks->len && !d->split_busy) {
        if (d->od_nopticket) {
            qticket = reims_vgpu_dirty_query_begin(d);
        }
        guint k;

        g_array_set_size(d->split_chunks, 0);
        for (k = 0; k < chunks->len; k++) {
            ReimsVgpuDirtyChunk c = g_array_index(chunks, ReimsVgpuDirtyChunk, k);
            const uint64_t piece = 1024ULL << shift;

            while (c.r.len > piece) {
                ReimsVgpuDirtyChunk head = { c.mr, { c.r.start, piece } };

                g_array_append_val(d->split_chunks, head);
                c.r.start += piece;
                c.r.len -= piece;
            }
            g_array_append_val(d->split_chunks, c);
        }
        d->split_next = d->split_done = 0;
        d->split_total = d->split_chunks->len;
        d->split_busy = true;
        qemu_cond_broadcast(&d->od_work_cond);
        qemu_cond_broadcast(&d->synced_cond);
        reims_vgpu_dirty_split_help(d);
        while (d->split_done < d->split_total) {
            qemu_cond_wait(&d->split_cond, &d->lock);
        }
        d->split_total = d->split_done = d->split_next = 0;
        d->split_busy = false;
        qemu_cond_broadcast(&d->split_cond);   /* a batch may be waiting to split */
        qemu_mutex_unlock(&d->lock);
    } else {
        guint k;

        qticket = reims_vgpu_dirty_query_begin(d);
        qemu_mutex_unlock(&d->lock);
        for (k = 0; k < chunks->len; k++) {
            ReimsVgpuDirtyChunk c = g_array_index(chunks, ReimsVgpuDirtyChunk, k);

            memory_region_sync_dirty_ranges(c.mr, &c.r, 1);
        }
    }

    written = g_array_new(FALSE, FALSE, sizeof(ReimsVgpuDirtyWritten));
    hit = g_array_new(FALSE, FALSE, sizeof(size_t));
    pb2 = get_clock();
    qemu_mutex_lock(&d->lock);
    reims_vgpu_dirty_query_settle(d, qticket);
    pb3 = get_clock();
    /*
     * Every page the queries walked, the gaps they bridged included, as in
     * reims_vgpu_dirty_sync_one: their bits are in the VGA bitmap now.
     */
    for (pi = 0; pi < walked->len; pi++) {
        const MemoryRegionRange *w = &g_array_index(walked, MemoryRegionRange, pi);
        uint64_t pfn = w->start >> shift, end = (w->start + w->len) >> shift;

        reims_vgpu_dirty_page_seq_cover(d, end);
        for (; pfn < end; pfn++) {
            d->page_qepoch[pfn] = MAX(d->page_qepoch[pfn], epoch);
        }
    }
    pb4 = get_clock();
    for (t = 0; t < claimed->len; t++) {
        ReimsVgpuDirtySet *set = g_hash_table_lookup(d->sets,
                                    &g_array_index(claimed, uint64_t, t));
        if (set) {
            reims_vgpu_dirty_consume_set(d, set, slices, n, d->global_seq, written);
        }
    }
    pb5 = get_clock();
    for (t = 0; t < claimed->len; t++) {
        ReimsVgpuDirtySet *set = g_hash_table_lookup(d->sets,
                                    &g_array_index(claimed, uint64_t, t));
        if (set) {
            if (reims_vgpu_dirty_eval_set(d, set, slices, n, hit, false)) {
                d->log_wanted = true;
            }
            set->synced_epoch = MAX(set->synced_epoch, epoch);
            if (set->claim_epoch == epoch) {
                set->claim_epoch = 0;
            }
        }
    }
    d->lab_od_n += claimed->len;
    d->lab_od_pages += npages;
    pb6 = get_clock();
    d->lab_pb_n++;
    d->lab_pb_sets += claimed->len;
    d->lab_pb_pages += npages;
    d->lab_pb_setpages += setpages;
    d->lab_pb_chunks += chunks->len;
    d->lab_pb_build += pb1 - pb0;
    d->lab_pb_query += pb2 - pb1;
    d->lab_pb_settle += pb3 - pb2;
    d->lab_pb_mark += pb4 - pb3;
    d->lab_pb_consume += pb5 - pb4;
    d->lab_pb_eval += pb6 - pb5;
    qemu_cond_broadcast(&d->synced_cond);
    qemu_mutex_unlock(&d->lock);
    reims_vgpu_dirty_clear_written(written, slices, n);
    g_free(slices);
    pb7 = get_clock();
    qemu_mutex_lock(&d->lock);
    if (pb7 - d->lab_pb_last >= NANOSECONDS_PER_SECOND) {
        fprintf(stderr, "reims-pbatch epoch_ms=%" PRId64 " n=%" PRIu64 " sets=%" PRIu64
                " setpages=%" PRIu64 " pages=%" PRIu64 " chunks=%" PRIu64
                " lag_us=%" PRId64 " build_us=%" PRId64 " query_us=%" PRId64
                " settle_us=%" PRId64 " mark_us=%" PRId64 " consume_us=%" PRId64
                " eval_us=%" PRId64 "\n",
                g_get_real_time() / 1000, d->lab_pb_n, d->lab_pb_sets,
                d->lab_pb_setpages, d->lab_pb_pages, d->lab_pb_chunks,
                d->lab_pb_lag / 1000, d->lab_pb_build / 1000, d->lab_pb_query / 1000,
                d->lab_pb_settle / 1000, d->lab_pb_mark / 1000,
                d->lab_pb_consume / 1000, d->lab_pb_eval / 1000);
        d->lab_pb_n = d->lab_pb_sets = d->lab_pb_pages = d->lab_pb_chunks = 0;
        d->lab_pb_setpages = 0;
        d->lab_pb_build = d->lab_pb_query = d->lab_pb_settle = d->lab_pb_mark = 0;
        d->lab_pb_consume = d->lab_pb_eval = d->lab_pb_lag = 0;
        d->lab_pb_last = pb7;
    }
    qemu_mutex_unlock(&d->lock);
}

void reims_vgpu_dirty_background(ReimsVgpuDirty *d)
{
    g_autoptr(GArray) tokens = NULL;
    g_autoptr(GArray) order = NULL;
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
    order = g_array_new(FALSE, FALSE, sizeof(ReimsVgpuDirtyOrdered));
    qemu_mutex_lock(&d->lock);
    epoch = d->epoch;
    g_hash_table_iter_init(&it, d->sets);
    while (g_hash_table_iter_next(&it, &key, &val)) {
        ReimsVgpuDirtySet *set = val;

        if (set->synced_epoch < epoch && set->last_query_epoch + d->od_hot >= epoch) {
            ReimsVgpuDirtyOrdered o = { set->read_order, *(uint64_t *)key };

            g_array_append_val(order, o);
        }
    }
    /*
     * In the order the drain first read them last time: it reads a frame's sets
     * in much the same order every frame, so a prefetch in that order stays
     * ahead of it instead of syncing, in hash order, sets it needs last.
     */
    g_array_sort(order, reims_vgpu_dirty_cmp_ordered);
    for (t = 0; t < order->len; t++) {
        g_array_append_val(tokens, g_array_index(order, ReimsVgpuDirtyOrdered, t).token);
    }
    if (d->od_pbatch) {
        qemu_mutex_unlock(&d->lock);
        reims_vgpu_dirty_prefetch_batch(d, tokens, epoch);
        return;
    }
    /* Publish the list; the workers and this thread take tokens in turn. A
     * newer doorbell makes it stale and the next prefetch starts over. */
    g_array_set_size(d->od_work, 0);
    g_array_append_vals(d->od_work, tokens->data, tokens->len);
    d->od_work_next = 0;
    d->od_work_epoch = epoch;
    qemu_cond_broadcast(&d->od_work_cond);
    qemu_mutex_unlock(&d->lock);
    (void)t;
    reims_vgpu_dirty_prefetch_drain(d, epoch);
}
