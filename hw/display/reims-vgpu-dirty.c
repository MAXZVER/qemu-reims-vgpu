/*
 * reims-vgpu: guest-write tracking over the hypervisor dirty bitmap.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/main-loop.h"
#include "qemu/thread.h"
#include "qemu/timer.h"
#include "qemu/rcu.h"
#include "qemu/bitmap.h"
#include "exec/target_page.h"
#include "system/address-spaces.h"
#include "system/memory.h"
#include "system/physmem.h"
#include "system/ram_addr.h"
#include "system/ramlist.h"
#include "system/cpus.h"
#include "system/whpx-wp.h"

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
    struct rcu_head rcu;    /* first: call_rcu needs offset 0 (lab GEN_LOCKFREE frees through it) */
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
    uint64_t pb_claim;          /* epoch a two-phase batch pre-claimed it for, or 0 */
    uint64_t read_order;        /* read counter at this set's first read in its last epoch */
    uint64_t lab_created_epoch; /* TEMP lab (FGTRACE): d->epoch at track */
    uint64_t lab_needed_seen;   /* TEMP lab (FGTRACE): needed epoch of the last read */
    uint64_t lab_nreads;        /* TEMP lab (FGTRACE): reads of this set */
    uint32_t lab_pref;          /* TEMP lab (FGTRACE): a batch synced it, no read has used that yet */
    /*
     * Epoch of the prefetch batch that holds this set's claim, or 0. A batch
     * sets it with `claim_epoch` and clears both when it publishes the set; a
     * sync of the drain's own sets only `claim_epoch`. Lab STEAL tells a drain
     * that finds a claim apart: a batch's can be bypassed, a thread's cannot.
     */
    uint64_t bclaim;
    /*
     * Lab GEN_MEMO: the generation this set answered for an epoch a read had to
     * be synced for, packed as (needed << 32) | generation, both below 2^32;
     * 0 = none. One word so that concurrent readers see a consistent pair and
     * install it with one compare-and-swap.
     */
    uint64_t memo;
    /*
     * Lab GEN_LOCKFREE: a seqlock over what an evaluation rewrites (`gen`,
     * `page_gen`). Odd while one is writing (under `lock`); a lock-free reader
     * of the page list takes the same value before and after and retries on a
     * difference, so it sees the generation and the stamps of one evaluation.
     */
    uint32_t eval_seq;
    /*
     * Lab DIRTY_PERIOD: the needed epoch of this set's last read, and the gaps
     * (in epochs, saturating at 255) between its last four distinct needed
     * epochs, newest in the low byte; 0 = no history. Written by readers with
     * plain atomic stores: a lost update only makes the hint older.
     */
    uint64_t rd_epoch;
    uint32_t rd_gaps;
    /*
     * Lab DIRTY_WP: this set's token (the protect queue and the fault hook name
     * sets by it), whether its pages are write-protected now, how many syncs
     * in a row found it clean, and how often a protection was lost (each loss
     * doubles the clean run the next protection needs).
     */
    uint64_t token;
    uint8_t wp_state;       /* WP_NONE / WP_PROTECTING / WP_ON / WP_DROPPING; under `lock` */
    size_t wp_next;         /* protecting: next page to protect; dropping: next page to look at */
    uint32_t wp_streak;
    uint32_t wp_lost;
    uint32_t lab_fg_n;      /* TEMP lab: foreground syncs since the last fgtop line */
} ReimsVgpuDirtySet;

/*
 * Lab GEN_LOCKFREE: a read-mostly mirror of the token -> set map that readers
 * walk without any lock. Open addressing, the token is the key, 0 = empty and
 * TOMB = deleted; a slot's value is stored (release) before its key, so a reader
 * that sees the key sees the set. Writers (track, untrack) hold `lock`. A table
 * that has to grow is replaced whole and retired through RCU; a removed set is
 * freed through RCU too, so a reader inside an RCU read-side section can still
 * be holding the pointer it found.
 */
typedef struct ReimsVgpuDirtyTab {
    struct rcu_head rcu;    /* first: call_rcu needs offset 0 */
    uint64_t mask;
    uint32_t shift;         /* 64 - log2(capacity) */
    uint64_t *keys;
    struct ReimsVgpuDirtySet **vals;
} ReimsVgpuDirtyTab;
#define REIMS_VGPU_DIRTY_TOMB UINT64_MAX

/* One hypervisor query range of a split job. */
typedef struct ReimsVgpuDirtyChunk {
    MemoryRegion *mr;
    MemoryRegionRange r;
} ReimsVgpuDirtyChunk;

/*
 * TEMP lab (REIMS_VGPU_LAB_FGTRACE=<file>): one 72-byte record per traced event,
 * appended to <file> about once a second. Kinds:
 *   0 fast-path hit, first read of the set in its needed epoch
 *   2 foreground sync from a generation read
 *   3 foreground sync from a written-pages read
 *   4 lagged generation read
 *   5 the drain's work scope moved (needed = work epoch)
 *   6 prefetch batch (needed = epoch, pages = walked pages, aux = sets)
 *   7 doorbell (needed = the epoch it opened)
 *   8 track (pages, epoch)   9 untrack (aux = lifetime in epochs, total_ns = reads)
 *  10 written-pages read that needed no sync
 */
typedef struct ReimsVgpuLabRec {
    uint64_t t_ns;
    uint64_t token;
    uint64_t needed;
    uint64_t epoch;
    uint64_t last_q;
    int64_t since_db_ns;
    uint32_t pages;
    uint8_t kind;
    uint8_t cls;      /* sync kinds: 0 cold, 1 missed, 2 ahead, 3 never read before */
    uint8_t dirty;    /* the sync moved the generation */
    uint8_t valid;
    uint32_t total_ns;
    uint32_t wait_ns;
    uint32_t query_ns;
    uint32_t aux;
} ReimsVgpuLabRec;
#define REIMS_VGPU_LAB_TR_CAP (1u << 20)

struct ReimsVgpuDirty {
    QemuMutex lock;
    GHashTable *sets;       /* token -> ReimsVgpuDirtySet* */
    /*
     * Guards `sets` membership for the generation fast path, which must not
     * queue behind `lock`: a prefetch batch holds `lock` through its mark,
     * consume and eval passes (~0.6 ms, ~220 batches a second at 5K), and every
     * generation read of an already-synced set used to wait that out on the
     * drain thread. Membership changes (track/untrack) take `lock` and then
     * this; the fast path takes only this. Other threads only look sets up
     * under `lock`, and concurrent lookups are read-only.
     */
    QemuMutex sets_lock;
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
    /* Generation reads may skip `lock` for synced sets (lab A/B
     * REIMS_VGPU_DIRTY_GEN_FAST=off restores the locked read). */
    bool gen_fast;
    uint64_t lab_gen_fast;    /* TEMP lab: fast-path answers since the last report */
    uint64_t lab_gen_lagged;  /* TEMP lab: lagged answers since the last report */
    /*
     * The prefetch batch's page union: one bit per target page frame, set for
     * every page of every claimed set and cleared again as the runs are read
     * back out, so it is all-zero between batches. Scanning it yields the
     * union's maximal runs in GPA order, which the batch used to get by
     * sorting every set's runs with g_array_sort — a comparator call per
     * element, a fifth of the harvest thread at 5K. Owned by the batch, whose
     * only caller is the device's harvest thread; bits are set under `lock`
     * with the claims and read back without it.
     */
    /*
     * Two-phase prefetch (lab A/B REIMS_VGPU_DIRTY_PB_SPLIT=on): claim every
     * hot set, then sync and publish the first third in read order before the
     * rest. A drain that reaches a set claimed by a running batch waits for the
     * whole batch; the sets it reads first are the ones worth publishing early.
     */
    bool pb_split;
    /*
     * Doorbell coalescing (lab A/B REIMS_VGPU_DIRTY_COALESCE=on). A doorbell
     * opens a new epoch so that a sync begun before it cannot vouch for a
     * store made before it. If no sync has claimed a set or begun a query
     * since the epoch opened, every sync still to come begins after this
     * doorbell too, so the open epoch already carries the guarantee and a new
     * one would only invalidate every set a second time. `epoch_queried` is
     * set under `lock` at every claim and every query ticket.
     */
    bool coalesce;
    bool epoch_queried;
    uint64_t lab_db_coalesced;  /* TEMP lab */
    unsigned long *pb_bits;
    uint64_t pb_nbits;
    uint64_t pb_lo, pb_hi;    /* frames that may hold a set bit: [pb_lo, pb_hi) */
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
    /*
     * Lab DIRTY_RSETTLE: what each in-flight ticket queries, sorted by (region,
     * offset), or NULL for "not known" (every ticket until its owner says, and
     * always for the full-harvest paths). A settle then waits only for tickets
     * whose ranges meet its own: a query that touches none of a sync's pages
     * cannot hold one of their bits. Parallel to `q_inflight`.
     */
    const struct ReimsVgpuDirtyChunk *q_fp[32];
    guint q_fpn[32];
    uint32_t rsettle_mode;
    uint64_t lab_rs_waits, lab_rs_skipped;
    /* TEMP lab diagnostic: on-demand syncs and their cost. */
    uint64_t lab_od_n, lab_od_ns, lab_od_pages, lab_od_skipped;
    uint64_t lab_od_epochs, lab_od_fg_n, lab_od_fg_ns;   /* TEMP lab */
    uint64_t lab_fg_ahead, lab_fg_missed, lab_fg_cold, lab_pb_epoch;  /* TEMP lab */
    uint64_t lab_pb_n, lab_pb_sets, lab_pb_pages, lab_pb_chunks, lab_pb_setpages;  /* TEMP lab */
    int64_t lab_pb_build, lab_pb_query, lab_pb_settle, lab_pb_mark, lab_pb_consume,
            lab_pb_eval, lab_pb_last, lab_pb_lag;
    uint64_t lab_ph_wait, lab_ph_prep, lab_ph_query, lab_ph_settle, lab_ph_consume;  /* TEMP lab */
    int64_t lab_od_last;
    /* TEMP lab (FGTRACE): the trace buffer and the cost counters beside it. */
    ReimsVgpuLabRec *lab_tr;
    uint64_t lab_tr_n, lab_tr_flushed;
    FILE *lab_tr_f;
    int64_t lab_tr_flush_at, lab_t0, lab_epoch_ns[4096];
    uint64_t lab_first_reads, lab_dup_reads, lab_fast_n, lab_fast_ns;
    uint64_t lab_pref_used, lab_pref_wasted, lab_pref_wasted_pages;
    uint64_t lab_whv_ns, lab_whv_calls, lab_fg_walked, lab_fg_split;
    uint64_t lab_ws_n, lab_ws_sync_n, lab_ws_ns, lab_ntrack, lab_nuntrack;
    uint64_t lab_lagged_n;
    /*
     * TEMP lab gates, each 0 = off (the original behaviour), 1 = on, 2 = verify:
     *   REIMS_VGPU_LAB_DIRTY_STEAL  a foreground sync does not wait for the
     *                               prefetch batch that claimed its set; it
     *                               queries the set itself. verify does that and
     *                               then waits the batch out as the original did,
     *                               answering with the later generation, and
     *                               counts how often the batch moved it.
     *   REIMS_VGPU_LAB_GEN_MEMO     a set answers one generation per needed
     *                               epoch. verify answers as the original did and
     *                               counts how often the memo would have differed.
     */
    uint32_t steal_mode, gen_memo_mode;
    /*
     * REIMS_VGPU_LAB_EPOCH_NAMED=on: a gfx register write that names no doorbell
     * (gfx_write_doorbell_channel() answers none: it schedules no FIFO drain,
     * so it hands the device no packet) does not open an epoch. 3/4 of the
     * epochs of a 5K window drag are such writes (the display-interrupt enable
     * register, ~250 a second), each invalidating every set's sync.
     */
    uint32_t epoch_named;
    uint64_t lab_unnamed_skipped;
    /*
     * REIMS_VGPU_LAB_DIRTY_HOLE=census: count, per channel, the named doorbells
     * (db), the packets the drain consumed (pk), the rings it left empty after
     * consuming (dr) and the rings it found empty on arrival (em). Packets far
     * outnumber doorbells, so a packet cannot be matched to the doorbell that
     * handed it over, and a read that needs one has only the epoch to go by.
     *
     * REIMS_VGPU_LAB_DIRTY_HOLE=close also acts on it: a run that starts with no
     * doorbell pending consumed packets published before their doorbell was
     * noted, so a sync that began after the last epoch opened may have begun
     * before the stores those packets depend on. The drain's own read opens a
     * fresh epoch (the packet has been read by then, so any sync that begins
     * after it is exact in the strict sense) and its reads need that epoch.
     */
    uint32_t hole_mode;           /* 0 off, 1 census, 2 close, 3 tail */
    /*
     * REIMS_VGPU_LAB_DIRTY_FRONTIER=on (effective with DIRTY_HOLE=tail): the
     * epoch a read needs is the newest epoch that covers a packet the drain has
     * read so far, not the newest epoch of any channel. A packet's guest stores
     * are ordered before its own publication (covered by its channel's doorbell
     * epoch or by the epoch the drain opened for an uncovered tail), and any
     * store another channel's packet made it depend on (an event it waits for)
     * reached the device through a packet the drain read first, whose cover the
     * frontier already holds. A doorbell for packets not read yet (the next
     * frame of another channel) therefore no longer makes every set stale for
     * the packets being run. Root-ring reads keep the newest epoch (their tail
     * is the doorbell write itself; conservative). `cover_pending` is the cover
     * note_ring_tail computed for the read that note_work_scope(All) follows.
     */
    uint32_t frontier_mode;
    uint64_t cover_pending;
    bool cover_valid;
    /* REIMS_VGPU_LAB_DIRTY_TAILWAKE=on: an epoch the drain opens wakes the prefetch. */
    uint32_t tailwake_mode;
    /*
     * REIMS_VGPU_LAB_DIRTY_WP (on by default, =off disables): a set of at least `wp_min_pages` target pages
     * whose last `wp_streak_need` syncs all found it clean is write-protected
     * in the hypervisor (accel/whpx: unmapped and mapped again read-only, still
     * tracked) and gets `synced_epoch` = REIMS_VGPU_DIRTY_WP_EPOCH: every read
     * finds it synced and the prefetch skips it, so it costs no query at all.
     * The guest's first store into any of its pages exits to
     * reims_vgpu_dirty_wp_fault on that vCPU *before* the store happens; the
     * set is made writable again there (claimed while its mapping changes,
     * `synced_epoch` 0) and the vCPU re-runs the store against the tracked
     * mapping, so the next sync finds it by query exactly as without WP.
     * Protecting happens in a main-loop BH with every vCPU paused: a fresh
     * query of the set, then the read-only remap, with no store possible in
     * between (a remap may drop the bits a range had collected). The 5K
     * surfaces this is for are ~10 800 pages, re-queried ~once a frame for
     * ~0.3 ms each and never written.
     */
    uint32_t wp_mode;
    bool wp_active;               /* DIRTY_WP=toggle:<file>: the file exists (else always) */
    bool wp_drop_all;             /* the BH drops every protection (toggled off) */
    char *wp_toggle;
    uint32_t wp_streak_need;
    uint64_t wp_min_pages;
    QemuMutex wp_mutex;           /* remaps and `wp_list`; taken before `lock` */
    GArray *wp_queue;             /* tokens for the protect BH (under `lock`) */
    GArray *wp_list;              /* tokens of sets in any WP state but NONE (under `wp_mutex`) */
    GArray *wp_orphans;           /* read-only runs no set owns any more (under `wp_mutex`) */
    uint64_t wp_cur;              /* the set being protected, or 0 (under `wp_mutex`) */
    ReimsVgpuDirtySlice *wp_slices;
    int wp_slices_cap;
    uint32_t wp_chunk;            /* runs per step (DIRTY_WP_CHUNK, 24) */
    uint32_t wp_pace_ms;          /* between steps (DIRTY_WP_PACE_MS, 4) */
    uint32_t wp_duty;             /* max % of time the guest is paused (DIRTY_WP_DUTY, 10) */
    /*
     * DIRTY_WP_NOPAUSE=on: protect without pausing the vCPUs. The remap runs under a query ticket and
     * every remapped page is marked written in the VGA bitmap before the ticket ends: whatever a store
     * before the remap left in the hypervisor bitmap (the remap may drop it) is covered by the marks,
     * and a sync that settles after the ticket began waits for them. The set's pages are re-read once.
     */
    bool wp_nopause;
    uint64_t wp_delay_ms;         /* the next step's wait after a chunk's pause */
    QEMUTimer *wp_timer;
    QEMUBH *wp_bh;
    unsigned long *wp_bits;       /* pfns of protected sets (they never overlap); under `wp_mutex` */
    uint64_t wp_bits_n;
    uint64_t lab_wp_protects, lab_wp_faults, lab_wp_refused, lab_wp_chunks, lab_wp_remaps;
    uint64_t lab_wp_lost, lab_wp_dropped, lab_wp_remap_ns, lab_wp_pause_ns, lab_wp_pause_max_ns;
    /*
     * REIMS_VGPU_LAB_DIRTY_PERIOD=on: a set of at least `period_big` pages joins
     * a prefetch batch only when the batch's epoch is at least its shortest
     * recent read gap past its last read. A set read once a frame spans ~3
     * doorbell epochs, and the batch of every doorbell synced it: two of three
     * syncs of the full-screen sets were never read (5K light scene trace,
     * 2026-10-01: 7 sets of 2-11k pages, 98 % of batch-synced first reads
     * unused), and their hypervisor queries are what the drain's own syncs queue
     * behind. Only which sets a batch takes changes; a read that finds its set
     * stale still syncs it, as before.
     */
    uint32_t period_mode;
    uint64_t period_big;
    /*
     * REIMS_VGPU_LAB_DIRTY_BIGSPLIT=on: sets of at least `period_big` pages are
     * not part of the prefetch batch. The batch publishes every set it holds
     * only once the whole union has been queried, consumed and evaluated (~1.9
     * ms after its doorbell at 5K, median), and the drain reaches a full-screen
     * set ~1.5 ms after it (median; a quarter before 0.6 ms), so it synced most
     * of them itself, beside the batch querying the same pages. Here each big set
     * goes to a worker as its own sync (read order, no split, so they run side by
     * side), published when that sync ends; a drain that reaches one a worker
     * holds waits for that one sync. The small sets keep the batch.
     */
    uint32_t bigsplit_mode;
    uint64_t lab_bigsplit_sets;
    /*
     * REIMS_VGPU_LAB_DIRTY_BIGPHASE=on: the batch walks its sets in read order
     * as phases instead of one union: every set of at least `period_big` pages
     * is a phase of its own (split across the pool like any batch, published as
     * soon as its pieces are in), and the small sets between two big ones form
     * one phase. A big set's phase claims it without `bclaim`, so a drain that
     * reaches it mid-phase waits for (and helps with) that one set rather than
     * querying the same pages beside it; a set a later phase has not reached yet
     * is unclaimed, and a drain that needs it first syncs it itself, which that
     * phase then skips. The single union published ~1.9 ms after its doorbell
     * (5K light scene, median) while the drain reached the full-screen sets
     * after ~1.5 ms (a quarter of them before 0.6 ms).
     */
    uint32_t bigphase_mode;
    uint64_t bigphase_big;   /* pages from which a set is a phase of its own (DIRTY_BIGPHASE_BIG, 2048) */
    uint64_t lab_bigphase_big, lab_bigphase_groups;
    uint64_t lab_frontier_child;
    uint64_t lab_period_skipped, lab_period_taken;
    uint64_t lab_hole_closed;
    /*
     * DIRTY_HOLE=tail. Per channel, a child-ring tail every packet up to which
     * an epoch no newer than the current one already covers: either a tail a
     * doorbell read before it opened (or coalesced into) its epoch, or a tail
     * the drain read before it opened one itself. See
     * reims_vgpu_dirty_note_ring_tail. Under `lock`.
     */
    uint32_t ring_tail[64];
    bool ring_tail_valid[64];
    uint64_t lab_db[64], lab_pk[64], lab_dr[64], lab_em[64], lab_unnamed;
    /*
     * A run is the drain's visit to a ring from its first packet to the moment it
     * leaves the ring empty. `lab_dbp` counts named doorbells since the last run
     * ended; a run that starts with none pending (`lab_hole`) consumed a batch whose
     * doorbell had not been noted yet. An upper bound: a doorbell for a batch
     * published after a run's last tail read is also dropped when that run ends.
     */
    uint32_t lab_run[64];
    uint64_t lab_dbp[64], lab_hole[64];
    uint64_t lab_steals, lab_steal_same, lab_steal_late;
    uint64_t lab_memo_hit, lab_memo_fresh, lab_memo_diff, lab_memo_bad;
    /*
     * REIMS_VGPU_LAB_GEN_LOCKFREE: 0 = a generation read of a synced set takes
     * `sets_lock` (the original), 1 = it walks `tab` under RCU and takes nothing,
     * 2 = verify: the original, with the lock-free read beside it, compared.
     */
    uint32_t lf_mode;
    ReimsVgpuDirtyTab *tab;   /* RCU pointer */
    uint64_t tab_used, tab_live;   /* under lock: slots taken (tombstones included), live */
    uint64_t lab_lf_hit, lab_lf_hitdiff, lab_lf_gendiff, lab_lf_race, lab_gen_fast_s;
    uint64_t lab_lfw_hit, lab_lfw_miss, lab_lfw_diff, lab_lfw_race;
};

static void reims_vgpu_lab_rec(ReimsVgpuDirty *d, const ReimsVgpuLabRec *r)
{
    uint64_t i;

    if (!d->lab_tr) {
        return;
    }
    i = qatomic_fetch_inc(&d->lab_tr_n);
    if (i >= REIMS_VGPU_LAB_TR_CAP) {
        qatomic_set(&d->lab_tr_n, REIMS_VGPU_LAB_TR_CAP);
        return;
    }
    d->lab_tr[i] = *r;
    qatomic_store_release(&d->lab_tr[i].valid, 1);
}

/* Write what is complete so far. Not thread safe: callers hold d->lock. */
static void reims_vgpu_lab_flush(ReimsVgpuDirty *d, bool force)
{
    int64_t now = get_clock();
    uint64_t n, i;

    if (!d->lab_tr || !d->lab_tr_f) {
        return;
    }
    if (!force && now - d->lab_tr_flush_at < NANOSECONDS_PER_SECOND) {
        return;
    }
    d->lab_tr_flush_at = now;
    n = MIN(qatomic_read(&d->lab_tr_n), (uint64_t)REIMS_VGPU_LAB_TR_CAP);
    for (i = d->lab_tr_flushed; i < n; i++) {
        if (!qatomic_load_acquire(&d->lab_tr[i].valid)) {
            break;
        }
    }
    if (i > d->lab_tr_flushed) {
        fwrite(&d->lab_tr[d->lab_tr_flushed], sizeof(ReimsVgpuLabRec),
               i - d->lab_tr_flushed, d->lab_tr_f);
        fflush(d->lab_tr_f);
        d->lab_tr_flushed = i;
    }
}

/* Nanoseconds since the doorbell that opened `epoch`, or -1. */
static int64_t reims_vgpu_lab_since_db(ReimsVgpuDirty *d, uint64_t epoch)
{
    int64_t t = epoch ? qatomic_read(&d->lab_epoch_ns[epoch & 4095]) : 0;

    return t ? get_clock() - t : -1;
}

/* A TEMP lab gate: unset or "off" = 0, "on" = 1, "verify" = 2. */
static uint32_t reims_vgpu_dirty_lab_mode(const char *name)
{
    const char *e = getenv(name);

    if (!e) {
        return 0;
    }
    if (strcmp(e, "on") == 0) {
        return 1;
    }
    if (strcmp(e, "verify") == 0) {
        return 2;
    }
    return 0;
}

static void reims_vgpu_dirty_set_free(gpointer p)
{
    ReimsVgpuDirtySet *s = p;

    g_free(s->pages);
    g_free(s->page_gen);
    g_free(s);
}

static void reims_vgpu_dirty_set_free_rcu(ReimsVgpuDirtySet *s)
{
    reims_vgpu_dirty_set_free(s);
}

static void reims_vgpu_dirty_tab_free(ReimsVgpuDirtyTab *t)
{
    g_free(t->keys);
    g_free(t->vals);
    g_free(t);
}

static inline uint64_t reims_vgpu_dirty_tab_slot(const ReimsVgpuDirtyTab *t, uint64_t token)
{
    return (token * 0x9E3779B97F4A7C15ULL) >> t->shift;
}

/* The set `token` names, or NULL. The caller is inside an RCU read-side section. */
static ReimsVgpuDirtySet *reims_vgpu_dirty_tab_lookup(ReimsVgpuDirty *d, uint64_t token)
{
    ReimsVgpuDirtyTab *t = qatomic_rcu_read(&d->tab);
    uint64_t i, n;

    if (!t || token == 0 || token == REIMS_VGPU_DIRTY_TOMB) {
        return NULL;
    }
    i = reims_vgpu_dirty_tab_slot(t, token);
    for (n = 0; n <= t->mask; n++, i = (i + 1) & t->mask) {
        uint64_t k = qatomic_load_acquire(&t->keys[i]);

        if (k == token) {
            ReimsVgpuDirtySet *found = qatomic_load_acquire(&t->vals[i]);

            /*
             * A slot whose set was removed can be given to another token between
             * the two loads (the writer stores the value, then the key): the key
             * is read again, and a slot that no longer holds this token is a miss.
             */
            return qatomic_load_acquire(&t->keys[i]) == token ? found : NULL;
        }
        if (k == 0) {
            return NULL;
        }
    }
    return NULL;
}

static void reims_vgpu_dirty_tab_put(ReimsVgpuDirtyTab *t, uint64_t token,
                                     ReimsVgpuDirtySet *s, bool *fresh)
{
    uint64_t i = reims_vgpu_dirty_tab_slot(t, token);

    while (t->keys[i] != 0 && t->keys[i] != REIMS_VGPU_DIRTY_TOMB) {
        i = (i + 1) & t->mask;
    }
    *fresh = t->keys[i] == 0;
    qatomic_store_release(&t->vals[i], s);
    qatomic_store_release(&t->keys[i], token);
}

/* Publish a new set. d->lock held. */
static void reims_vgpu_dirty_tab_add(ReimsVgpuDirty *d, uint64_t token, ReimsVgpuDirtySet *s)
{
    ReimsVgpuDirtyTab *t = d->tab;
    bool fresh;

    if (!t || (d->tab_used + 1) * 2 > t->mask + 1) {
        uint64_t cap = MAX(64, pow2ceil((d->tab_live + 1) * 4));
        ReimsVgpuDirtyTab *nt = g_new0(ReimsVgpuDirtyTab, 1);
        uint64_t i;

        nt->mask = cap - 1;
        nt->shift = 64 - ctz64(cap);
        nt->keys = g_new0(uint64_t, cap);
        nt->vals = g_new0(ReimsVgpuDirtySet *, cap);
        d->tab_used = 0;
        for (i = 0; t && i <= t->mask; i++) {
            if (t->keys[i] != 0 && t->keys[i] != REIMS_VGPU_DIRTY_TOMB) {
                reims_vgpu_dirty_tab_put(nt, t->keys[i], t->vals[i], &fresh);
                d->tab_used++;
            }
        }
        qatomic_rcu_set(&d->tab, nt);
        if (t) {
            call_rcu(t, reims_vgpu_dirty_tab_free, rcu);
        }
        t = nt;
    }
    reims_vgpu_dirty_tab_put(t, token, s, &fresh);
    d->tab_used += fresh;
    d->tab_live++;
}

/* Unpublish a set; the caller frees it through RCU. d->lock held. */
static void reims_vgpu_dirty_tab_del(ReimsVgpuDirty *d, uint64_t token)
{
    ReimsVgpuDirtyTab *t = d->tab;
    uint64_t i, n;

    if (!t) {
        return;
    }
    i = reims_vgpu_dirty_tab_slot(t, token);
    for (n = 0; n <= t->mask; n++, i = (i + 1) & t->mask) {
        if (t->keys[i] == token) {
            qatomic_store_release(&t->keys[i], REIMS_VGPU_DIRTY_TOMB);
            d->tab_live--;
            return;
        }
        if (t->keys[i] == 0) {
            return;
        }
    }
}

ReimsVgpuDirty *reims_vgpu_dirty_new(void)
{
    ReimsVgpuDirty *d = g_new0(ReimsVgpuDirty, 1);

    qemu_mutex_init(&d->lock);
    qemu_mutex_init(&d->sets_lock);
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
    if (d->lab_tr) {   /* TEMP lab (FGTRACE) */
        qemu_mutex_lock(&d->lock);
        reims_vgpu_lab_flush(d, true);
        qemu_mutex_unlock(&d->lock);
        fclose(d->lab_tr_f);
        g_free(d->lab_tr);
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
    if (d->tab) {   /* TEMP lab GEN_LOCKFREE: every reader thread is gone by now */
        reims_vgpu_dirty_tab_free(d->tab);
    }
    g_free(d->slices);
    g_free(d->sync_pages);
    if (d->sync_ranges) {
        g_array_free(d->sync_ranges, TRUE);
    }
    g_free(d->pb_bits);
    qemu_mutex_destroy(&d->sets_lock);
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
    s->lab_created_epoch = d->epoch;   /* TEMP lab (FGTRACE) */
    d->next_token++;
    token = d->next_token;
    s->token = token;   /* lab DIRTY_WP */
    d->lab_ntrack++;
    if (d->lab_tr) {
        ReimsVgpuLabRec r = { .t_ns = get_clock() - d->lab_t0, .token = token,
                              .needed = d->epoch, .epoch = d->epoch,
                              .pages = (uint32_t)(n * (page_size / 4096)), .kind = 8 };

        reims_vgpu_lab_rec(d, &r);
    }
    qemu_mutex_lock(&d->sets_lock);
    g_hash_table_insert(d->sets, g_memdup2(&token, sizeof(token)), s);
    qemu_mutex_unlock(&d->sets_lock);
    if (d->lf_mode) {   /* TEMP lab GEN_LOCKFREE: publish to the lock-free readers last */
        reims_vgpu_dirty_tab_add(d, token, s);
    }
    qemu_mutex_unlock(&d->lock);
    return token;
}

/* Lab DIRTY_WP ---------------------------------------------------------- */
/* A protected set reads as synced for every epoch (far above any real one). */
#define REIMS_VGPU_DIRTY_WP_EPOCH (UINT64_MAX >> 2)

static ReimsVgpuDirty *reims_vgpu_dirty_wp_dev;
#ifdef REIMS_DIRTY_STRESS   /* scripts/dirty-stress: its own main loop runs the BH */
QEMUBH *reims_stress_bh_new(QEMUBHFunc *cb, void *opaque);
void reims_stress_bh_schedule(QEMUBH *bh);
#define REIMS_WP_BH_NEW(cb, op) reims_stress_bh_new(cb, op)
#define REIMS_WP_BH_SCHEDULE(bh) reims_stress_bh_schedule(bh)
#else
#define REIMS_WP_BH_NEW(cb, op) qemu_bh_new(cb, op)
#define REIMS_WP_BH_SCHEDULE(bh) qemu_bh_schedule(bh)
#endif
static void reims_vgpu_dirty_sync_one(ReimsVgpuDirty *d, uint64_t token, bool fg,
                                      int64_t *lab_wait, int64_t *lab_query);

static int reims_vgpu_dirty_ram_slices_into(ReimsVgpuDirtySlice **buf, int *cap);
static int reims_vgpu_dirty_slice_of(const ReimsVgpuDirtySlice *slices, int n, uint64_t gpa);
static uint64_t reims_vgpu_dirty_query_begin(ReimsVgpuDirty *d);
static void reims_vgpu_dirty_query_footprint(ReimsVgpuDirty *d, uint64_t ticket,
                                             const ReimsVgpuDirtyChunk *chunks, guint n);
static void reims_vgpu_dirty_query_end(ReimsVgpuDirty *d, uint64_t ticket);

enum { WP_NONE, WP_PROTECTING, WP_ON, WP_DROPPING };

typedef struct ReimsVgpuDirtyRun {
    uint64_t gpa, size;
} ReimsVgpuDirtyRun;

#ifdef REIMS_DIRTY_STRESS
#define REIMS_WP_PACE(d) REIMS_WP_BH_SCHEDULE((d)->wp_bh)
#else
#define REIMS_WP_PACE(d) \
    timer_mod((d)->wp_timer, qemu_clock_get_ms(QEMU_CLOCK_REALTIME) + \
              MAX((uint64_t)(d)->wp_pace_ms, (d)->wp_delay_ms))
#endif

static inline bool reims_vgpu_dirty_wp_ro(ReimsVgpuDirty *d, uint64_t gpa)
{
    uint64_t pfn = gpa >> qemu_target_page_bits();

    return pfn < d->wp_bits_n && test_bit(pfn, d->wp_bits);
}

/* One remap of [gpa, gpa+size), with `wp_bits` kept equal to what is read-only. `wp_mutex` held. */
static int reims_vgpu_dirty_wp_remap(ReimsVgpuDirty *d, uint64_t gpa, uint64_t size, bool writable)
{
    const uint64_t step = 1ULL << qemu_target_page_bits();
    int64_t t0 = get_clock();
    uint64_t a;
    int rc;

    if (!writable) {
        /* Marked first: a store exit on it from here on is ours. */
        for (a = gpa; a < gpa + size; a += step) {
            set_bit_atomic(a >> qemu_target_page_bits(), d->wp_bits);
        }
    }
    rc = whpx_wp_remap(gpa, size, writable);
    if (writable || rc) {
        for (a = gpa; a < gpa + size; a += step) {
            clear_bit_atomic(a >> qemu_target_page_bits(), d->wp_bits);
        }
    }
    d->lab_wp_remap_ns += get_clock() - t0;
    d->lab_wp_remaps++;
    return rc;
}

/*
 * Make the read-only target pages in [gpa, end) writable, in maximal runs of pages that are still
 * read-only: a page in between that a fault already gave back may hold bits a remap would drop.
 * Counts each remap against `*budget`. `wp_mutex` held.
 */
static void reims_vgpu_dirty_wp_release(ReimsVgpuDirty *d, uint64_t gpa, uint64_t end, int *budget)
{
    const uint64_t pg = 1ULL << qemu_target_page_bits();

    while (gpa < end) {
        uint64_t a;

        if (!reims_vgpu_dirty_wp_ro(d, gpa)) {
            gpa += pg;
            continue;
        }
        for (a = gpa + pg; a < end && reims_vgpu_dirty_wp_ro(d, a); a += pg) {
        }
        reims_vgpu_dirty_wp_remap(d, gpa, a - gpa, true);
        (*budget)--;
        gpa = a;
    }
}

/* The run of `s` that starts at page index `p`: returns the index after it, `*end` its end. */
static size_t reims_vgpu_dirty_wp_run(const ReimsVgpuDirtySet *s, size_t p, uint64_t *end)
{
    size_t q = p + 1;

    *end = s->pages[p] + s->page_size;
    while (q < s->count && s->pages[q] == *end) {
        *end += s->page_size;
        q++;
    }
    return q;
}

static bool reims_vgpu_dirty_wp_has(const ReimsVgpuDirtySet *s, uint64_t gpa)
{
    size_t lo = 0, hi = s->count;

    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;

        if (s->pages[mid] + s->page_size <= gpa) {
            lo = mid + 1;
        } else if (s->pages[mid] > gpa) {
            hi = mid;
        } else {
            return true;
        }
    }
    return false;
}

static bool reims_vgpu_dirty_wp_any_ro(ReimsVgpuDirty *d, const ReimsVgpuDirtySet *s,
                                       size_t p, size_t q)
{
    for (; p < q; p++) {
        uint64_t a;

        for (a = s->pages[p]; a < s->pages[p] + s->page_size; a += 1ULL << qemu_target_page_bits()) {
            if (reims_vgpu_dirty_wp_ro(d, a)) {
                return true;
            }
        }
    }
    return false;
}

/* The read-only runs of `s` go on the orphan list (the step makes them writable). `wp_mutex` held. */
static void reims_vgpu_dirty_wp_orphan(ReimsVgpuDirty *d, const ReimsVgpuDirtySet *s)
{
    size_t p = 0;

    while (p < s->count) {
        uint64_t end;
        size_t q = reims_vgpu_dirty_wp_run(s, p, &end);

        if (reims_vgpu_dirty_wp_any_ro(d, s, p, q)) {
            ReimsVgpuDirtyRun r = { .gpa = s->pages[p], .size = end - s->pages[p] };

            g_array_append_val(d->wp_orphans, r);
        }
        p = q;
    }
}

/*
 * vCPU thread, a store exit at `gpa` (whpx_handle_mmio). The set holding the page stops being
 * protected (reads stop trusting it before anything is writable); only the page is made writable
 * here, the step function does the rest of the set without a pause.
 */
static bool reims_vgpu_dirty_wp_fault(uint64_t gpa)
{
    ReimsVgpuDirty *d = reims_vgpu_dirty_wp_dev;
    guint i;

    if (!d || !qatomic_read(&d->wp_bits) || !reims_vgpu_dirty_wp_ro(d, gpa)) {
        return false;
    }
    qemu_mutex_lock(&d->wp_mutex);
    if (reims_vgpu_dirty_wp_ro(d, gpa)) {
        for (i = 0; i < d->wp_list->len; i++) {
            uint64_t token = g_array_index(d->wp_list, uint64_t, i);
            ReimsVgpuDirtySet *s;
            bool hit;

            qemu_mutex_lock(&d->lock);
            s = g_hash_table_lookup(d->sets, &token);
            hit = s && (s->wp_state == WP_PROTECTING || s->wp_state == WP_ON) &&
                  reims_vgpu_dirty_wp_has(s, gpa);
            if (hit) {
                if (s->wp_state == WP_ON) {
                    qatomic_store_release(&s->synced_epoch, 0);
                }
                s->wp_state = WP_DROPPING;
                s->wp_next = 0;
                s->wp_streak = 0;
                s->wp_lost++;
                d->lab_wp_lost++;
                if (d->wp_cur == token) {
                    d->wp_cur = 0;
                }
            }
            qemu_mutex_unlock(&d->lock);
            if (hit) {
                break;
            }
        }
        reims_vgpu_dirty_wp_remap(d, gpa & ~(uint64_t)4095, 4096, true);
    }
    d->lab_wp_faults++;
    qemu_mutex_unlock(&d->wp_mutex);
    REIMS_WP_BH_SCHEDULE(d->wp_bh);
    return true;
}

/* QEMU is about to remap writable RAM, or WP was toggled off: every protection ends. */
static void reims_vgpu_dirty_wp_invalidate(void)
{
    ReimsVgpuDirty *d = reims_vgpu_dirty_wp_dev;
    guint i;

    if (!d) {
        return;
    }
    qemu_mutex_lock(&d->wp_mutex);
    for (i = 0; i < d->wp_list->len; i++) {
        uint64_t token = g_array_index(d->wp_list, uint64_t, i);
        ReimsVgpuDirtySet *s;

        qemu_mutex_lock(&d->lock);
        s = g_hash_table_lookup(d->sets, &token);
        if (s) {
            if (s->wp_state == WP_ON) {
                qatomic_store_release(&s->synced_epoch, 0);
            }
            s->wp_state = WP_NONE;
            s->wp_streak = 0;
        }
        qemu_mutex_unlock(&d->lock);
        if (s && d->wp_bits) {
            reims_vgpu_dirty_wp_orphan(d, s);
        }
    }
    g_array_set_size(d->wp_list, 0);
    d->wp_cur = 0;
    qemu_mutex_unlock(&d->wp_mutex);
    REIMS_WP_BH_SCHEDULE(d->wp_bh);
}

/* Take the next eligible queued set as the one being protected. `wp_mutex` and `lock` held. */
static bool reims_vgpu_dirty_wp_pick(ReimsVgpuDirty *d)
{
    while (!d->wp_cur && d->wp_queue->len) {
        uint64_t token = g_array_index(d->wp_queue, uint64_t, d->wp_queue->len - 1);
        ReimsVgpuDirtySet *c = g_hash_table_lookup(d->sets, &token);

        g_array_set_size(d->wp_queue, d->wp_queue->len - 1);
        if (c && c->wp_state == WP_NONE && c->gen != 0 && c->count &&
            ((c->pages[c->count - 1] + c->page_size - 1) >> qemu_target_page_bits()) < d->wp_bits_n &&
            !reims_vgpu_dirty_wp_any_ro(d, c, 0, c->count)) {
            c->wp_state = WP_PROTECTING;
            c->wp_next = 0;
            d->wp_cur = token;
            g_array_append_val(d->wp_list, token);
        }
    }
    return d->wp_cur != 0;
}

/*
 * Every page of `s` is read-only. A sync for a fresh epoch consumes what the chunk queries put in
 * the VGA bitmap (its query begins after theirs); after it nothing can change, so the set reads
 * as synced for good. `wp_mutex` held, `lock` not.
 */
static void reims_vgpu_dirty_wp_finish(ReimsVgpuDirty *d, uint64_t token, ReimsVgpuDirtySet *s)
{
    uint64_t fin_epoch;

    qemu_mutex_lock(&d->lock);
    qatomic_set(&d->epoch, d->epoch + 1);
    d->epoch_queried = false;
    fin_epoch = d->epoch;
    qemu_mutex_unlock(&d->lock);
    reims_vgpu_dirty_sync_one(d, token, true, NULL, NULL);
    qemu_mutex_lock(&d->lock);
    if (s->wp_state == WP_PROTECTING && s->claim_epoch == 0 && s->synced_epoch >= fin_epoch) {
        s->wp_state = WP_ON;
        qatomic_store_release(&s->synced_epoch, REIMS_VGPU_DIRTY_WP_EPOCH);
        if (d->wp_cur == token) {
            d->wp_cur = 0;
        }
        d->lab_wp_protects++;
    }
    qemu_mutex_unlock(&d->lock);
}

/*
 * DIRTY_WP_NOPAUSE: mark the target pages of [gpa, end) written in the VGA bitmap, as a query that
 * found them written would have. A page outside every RAM slice reads as written without a bit.
 */
static void reims_vgpu_dirty_wp_mark(const ReimsVgpuDirtySlice *slices, int n, uint64_t gpa,
                                     uint64_t end)
{
    const int shift = qemu_target_page_bits();
    DirtyMemoryBlocks *blocks;

    RCU_READ_LOCK_GUARD();
    blocks = qatomic_rcu_read(&ram_list.dirty_memory[DIRTY_MEMORY_VGA]);
    for (; gpa < end; gpa += 1ULL << shift) {
        int si = reims_vgpu_dirty_slice_of(slices, n, gpa);
        unsigned long pg;

        if (si < 0) {
            continue;
        }
        pg = (slices[si].ram_addr + (gpa - slices[si].gpa)) >> shift;
        set_bit_atomic(pg % DIRTY_MEMORY_BLOCK_SIZE, blocks->blocks[pg / DIRTY_MEMORY_BLOCK_SIZE]);
    }
}

/*
 * Protect the next chunk of runs of set `token` (WP_PROTECTING) with every vCPU paused. Called
 * with neither lock: the pause comes first, because a vCPU inside the fault hook waits for
 * `wp_mutex` and could not stop while this held it.
 */
static int64_t reims_vgpu_dirty_wp_protect_chunk(ReimsVgpuDirty *d, uint64_t token, int budget)
{
    ReimsVgpuDirtySet *s;
    size_t p0, p, p_end;
    int64_t t0;
    int n, runs, rc = 0;
#ifndef REIMS_DIRTY_STRESS
    bool relock = false;
#endif

    n = reims_vgpu_dirty_ram_slices_into(&d->wp_slices, &d->wp_slices_cap);
    t0 = get_clock();
    if (d->wp_nopause) {
#ifndef REIMS_DIRTY_STRESS
        /* Nothing here needs the BQL; a vCPU's MMIO exit would wait through the remaps for it. */
        relock = bql_locked();
        if (relock) {
            bql_unlock();
        }
#endif
    } else {
        pause_all_vcpus();
    }
    qemu_mutex_lock(&d->wp_mutex);
  next_set:
    /* Every store before the pause, into the VGA bitmap; the set's next sync consumes it. */
    qemu_mutex_lock(&d->lock);
    s = g_hash_table_lookup(d->sets, &token);
    if (budget <= 0 || !s || s->wp_state != WP_PROTECTING || d->wp_cur != token) {
        qemu_mutex_unlock(&d->lock);
        goto done;
    }
    p0 = s->wp_next;
    if (d->wp_nopause) {
        /* The ticket is taken before the first remap and returned after the last mark. */
        uint64_t pt = reims_vgpu_dirty_query_begin(d);

        qemu_mutex_unlock(&d->lock);
        for (p = p0, runs = 0; p < s->count && runs < budget && !rc; runs++) {
            uint64_t end;
            size_t q = reims_vgpu_dirty_wp_run(s, p, &end);

            rc = reims_vgpu_dirty_wp_remap(d, s->pages[p], end - s->pages[p], false);
            reims_vgpu_dirty_wp_mark(d->wp_slices, n, s->pages[p], end);
            p = q;
        }
        p_end = p;
        qemu_mutex_lock(&d->lock);
        reims_vgpu_dirty_query_end(d, pt);
        qemu_mutex_unlock(&d->lock);
        goto remapped;
    }
    for (p = p0, runs = 0; p < s->count && runs < budget; runs++) {
        uint64_t end;
        size_t q = reims_vgpu_dirty_wp_run(s, p, &end);
        int si = reims_vgpu_dirty_slice_of(d->wp_slices, n, s->pages[p]);

        if (si >= 0) {
            ReimsVgpuDirtyChunk c = { .mr = d->wp_slices[si].mr };
            uint64_t pt = reims_vgpu_dirty_query_begin(d);

            c.r.start = d->wp_slices[si].offset + (s->pages[p] - d->wp_slices[si].gpa);
            c.r.len = end - s->pages[p];
            reims_vgpu_dirty_query_footprint(d, pt, &c, 1);
            qemu_mutex_unlock(&d->lock);
            memory_region_sync_dirty_ranges(c.mr, &c.r, 1);
            qemu_mutex_lock(&d->lock);
            reims_vgpu_dirty_query_end(d, pt);
        }
        p = q;
    }
    qemu_mutex_unlock(&d->lock);
    p_end = p;
    /* Then read-only: no store can land between that query and the remap. */
    for (p = p0; p < p_end && !rc;) {
        uint64_t end;
        size_t q = reims_vgpu_dirty_wp_run(s, p, &end);

        rc = reims_vgpu_dirty_wp_remap(d, s->pages[p], end - s->pages[p], false);
        p = q;
    }
  remapped:
    budget -= runs;
    d->lab_wp_chunks++;
    qemu_mutex_lock(&d->lock);
    if (rc) {
        s->wp_state = WP_DROPPING;
        s->wp_next = 0;
        d->wp_cur = 0;
        d->wp_mode = 0;
        d->lab_wp_refused++;
        fprintf(stderr, "reims-dirty-wp: read-only remap refused (rc=%d), DIRTY_WP off\n", rc);
        qemu_mutex_unlock(&d->lock);
        goto done;
    }
    s->wp_next = p_end;
    qemu_mutex_unlock(&d->lock);
    if (s->wp_next >= s->count) {
        /*
         * Done inside this pause, and the next set taken in it too: small sets are a run or a
         * few each, and one pause per set would cost far more than their queries. The next
         * one is chosen only now that this one is wholly read-only, so the read-only check
         * keeps them disjoint.
         */
        reims_vgpu_dirty_wp_finish(d, token, s);
        qemu_mutex_lock(&d->lock);
        if (budget > 0 && reims_vgpu_dirty_wp_pick(d)) {
            token = d->wp_cur;
            qemu_mutex_unlock(&d->lock);
            goto next_set;
        }
        qemu_mutex_unlock(&d->lock);
    }
  done:
    if (d->wp_nopause) {
        /* Not a pause: the step's time is reported in the same fields, nothing paces by it. */
        t0 = get_clock() - t0;
        d->lab_wp_pause_ns += t0;
        d->lab_wp_pause_max_ns = MAX(d->lab_wp_pause_max_ns, (uint64_t)t0);
        qemu_mutex_unlock(&d->wp_mutex);
#ifndef REIMS_DIRTY_STRESS
        if (relock) {
            bql_lock();
        }
#endif
        return 0;
    }
    resume_all_vcpus();
    t0 = get_clock() - t0;
    d->lab_wp_pause_ns += t0;
    d->lab_wp_pause_max_ns = MAX(d->lab_wp_pause_max_ns, (uint64_t)t0);
    qemu_mutex_unlock(&d->wp_mutex);
    return t0;
}

/*
 * Main loop (BH or pacing timer), BQL held. One step: up to `wp_chunk` runs made writable again
 * (orphans, then dropping sets), then one chunk of the set being protected, then reschedule while
 * anything is left.
 */
static void reims_vgpu_dirty_wp_bh(void *opaque)
{
    ReimsVgpuDirty *d = opaque;
    int budget = d->wp_chunk;
    bool more = false;
    guint i;

    d->wp_delay_ms = 0;
    if (qatomic_xchg(&d->wp_drop_all, false)) {
        reims_vgpu_dirty_wp_invalidate();
    }
    if (!d->wp_bits) {
        /*
         * Every pfn below 64 GiB of guest physical address space (2 MiB). Not
         * `page_seq_n`: that grows with the pages seen so far, and the BH first
         * runs early in boot (an invalidate from QEMU's own remaps schedules it),
         * when it covered almost nothing and no set would ever have fitted.
         */
        d->wp_bits_n = 1ULL << (36 - qemu_target_page_bits());
        qatomic_set(&d->wp_bits, bitmap_new(d->wp_bits_n));
    }
    qemu_mutex_lock(&d->wp_mutex);
    while (budget > 0 && d->wp_orphans->len) {
        ReimsVgpuDirtyRun r = g_array_index(d->wp_orphans, ReimsVgpuDirtyRun, d->wp_orphans->len - 1);

        g_array_set_size(d->wp_orphans, d->wp_orphans->len - 1);
        reims_vgpu_dirty_wp_release(d, r.gpa, r.gpa + r.size, &budget);
    }
    for (i = 0; i < d->wp_list->len && budget > 0;) {
        uint64_t token = g_array_index(d->wp_list, uint64_t, i);
        ReimsVgpuDirtySet *s;

        qemu_mutex_lock(&d->lock);
        s = g_hash_table_lookup(d->sets, &token);
        qemu_mutex_unlock(&d->lock);
        if (!s) {
            g_array_remove_index_fast(d->wp_list, i);
            continue;
        }
        if (s->wp_state != WP_DROPPING) {
            i++;
            continue;
        }
        while (budget > 0 && s->wp_next < s->count) {
            uint64_t end;
            size_t q = reims_vgpu_dirty_wp_run(s, s->wp_next, &end);

            reims_vgpu_dirty_wp_release(d, s->pages[s->wp_next], end, &budget);
            s->wp_next = q;
        }
        if (s->wp_next >= s->count) {
            qemu_mutex_lock(&d->lock);
            s->wp_state = WP_NONE;
            qemu_mutex_unlock(&d->lock);
            g_array_remove_index_fast(d->wp_list, i);
            d->lab_wp_dropped++;
            continue;
        }
        i++;
    }
    for (i = 0; i < d->wp_list->len && !more; i++) {   /* `more` = something is being given back */
        uint64_t token = g_array_index(d->wp_list, uint64_t, i);
        ReimsVgpuDirtySet *s;

        qemu_mutex_lock(&d->lock);
        s = g_hash_table_lookup(d->sets, &token);
        more = s && s->wp_state == WP_DROPPING;
        qemu_mutex_unlock(&d->lock);
    }
    more = more || d->wp_orphans->len;
    if (!more && budget > 0 && d->wp_mode && qatomic_read(&d->wp_active)) {
        ReimsVgpuDirtySet *s = NULL;

        qemu_mutex_lock(&d->lock);
        reims_vgpu_dirty_wp_pick(d);
        if (d->wp_cur) {
            s = g_hash_table_lookup(d->sets, &d->wp_cur);
            if (!s || s->wp_state != WP_PROTECTING) {
                d->wp_cur = 0;
                s = NULL;
            }
        }
        qemu_mutex_unlock(&d->lock);
        if (s && s->wp_next < s->count) {
            uint64_t token = d->wp_cur;

            qemu_mutex_unlock(&d->wp_mutex);
            int64_t paused = reims_vgpu_dirty_wp_protect_chunk(d, token, budget);

            /* Keep the guest paused at most `wp_duty` percent of the time. */
            d->wp_delay_ms = (uint64_t)(paused / 1000000) * (100 - d->wp_duty) / d->wp_duty;
            qemu_mutex_lock(&d->wp_mutex);
        } else if (s) {
            reims_vgpu_dirty_wp_finish(d, d->wp_cur, s);
        }
        qemu_mutex_lock(&d->lock);
        more = d->wp_cur || d->wp_queue->len;
        qemu_mutex_unlock(&d->lock);
    }
    qemu_mutex_unlock(&d->wp_mutex);
    if (more) {
        REIMS_WP_PACE(d);
    }
}

void reims_vgpu_dirty_untrack(ReimsVgpuDirty *d, uint64_t token)
{
    if (!d || token == 0) {
        return;
    }
    if (d->wp_list) {   /* lab DIRTY_WP: its read-only runs outlive it, the step makes them writable */
        guint i;

        qemu_mutex_lock(&d->wp_mutex);
        for (i = 0; i < d->wp_list->len; i++) {
            if (g_array_index(d->wp_list, uint64_t, i) == token) {
                ReimsVgpuDirtySet *ws;

                qemu_mutex_lock(&d->lock);
                ws = g_hash_table_lookup(d->sets, &token);
                if (ws) {
                    ws->wp_state = WP_NONE;
                }
                qemu_mutex_unlock(&d->lock);
                if (ws) {
                    reims_vgpu_dirty_wp_orphan(d, ws);
                }
                if (d->wp_cur == token) {
                    d->wp_cur = 0;
                }
                g_array_remove_index_fast(d->wp_list, i);
                REIMS_WP_BH_SCHEDULE(d->wp_bh);
                break;
            }
        }
        qemu_mutex_unlock(&d->wp_mutex);
    }
    qemu_mutex_lock(&d->lock);
    d->lab_nuntrack++;
    if (d->lab_tr) {   /* TEMP lab (FGTRACE) */
        ReimsVgpuDirtySet *ls = g_hash_table_lookup(d->sets, &token);

        if (ls) {
            ReimsVgpuLabRec r = { .t_ns = get_clock() - d->lab_t0, .token = token,
                                  .epoch = d->epoch, .last_q = ls->last_query_epoch,
                                  .pages = (uint32_t)(ls->count * (ls->page_size / 4096)),
                                  .kind = 9, .total_ns = (uint32_t)ls->lab_nreads,
                                  .aux = (uint32_t)(d->epoch - ls->lab_created_epoch) };

            reims_vgpu_lab_rec(d, &r);
        }
    }
    qemu_mutex_lock(&d->sets_lock);
    if (d->lf_mode) {
        /* TEMP lab GEN_LOCKFREE: a lock-free reader may hold the set, so it is
         * unpublished first and freed once every reader that could have found it
         * has left its read-side section. */
        gpointer okey, oval;

        if (g_hash_table_steal_extended(d->sets, &token, &okey, &oval)) {
            reims_vgpu_dirty_tab_del(d, token);
            g_free(okey);
            call_rcu((ReimsVgpuDirtySet *)oval, reims_vgpu_dirty_set_free_rcu, rcu);
        }
    } else {
        g_hash_table_remove(d->sets, &token);
    }
    qemu_mutex_unlock(&d->sets_lock);
    qemu_mutex_unlock(&d->lock);
}


/* The epoch a generation read must be synced for. d->lock held. */
static uint64_t reims_vgpu_dirty_needed_epoch(ReimsVgpuDirty *d)
{
    return d->work_epoch && !d->work_epoch_off ? MIN(d->work_epoch, d->epoch)
                                               : d->epoch;
}

/*
 * Lab DIRTY_HOLE=close: the drain read a batch whose doorbell has not been noted.
 * Open an epoch now, after the read, and make the reads that follow need it.
 */
static void reims_vgpu_dirty_open_epoch_from_drain(ReimsVgpuDirty *d)
{
    guint c;

    qemu_mutex_lock(&d->lock);
    qatomic_set(&d->epoch, d->epoch + 1);
    d->epoch_queried = false;
    for (c = 0; c < ARRAY_SIZE(d->chan_epoch); c++) {
        qatomic_set(&d->chan_epoch[c], d->epoch);
    }
    d->lab_od_epochs++;
    d->doorbell_ns = get_clock();
    if (d->lab_tr) {
        qatomic_set(&d->lab_epoch_ns[d->epoch & 4095], d->doorbell_ns);
    }
    qatomic_set(&d->work_epoch, d->epoch);
    d->lab_hole_closed++;
    qemu_mutex_unlock(&d->lock);
}

/*
 * DIRTY_HOLE=tail, the shim's half: a doorbell for child channel `channel`
 * read `tail` from the guest's ring registers *before* it called
 * reims_vgpu_dirty_note_doorbell. Every packet up to that tail was published
 * before the tail was read, so before the doorbell's epoch opened (or before
 * any sync of the epoch it coalesced into began): that epoch covers them.
 * `valid` false is a doorbell that could not read the tail; the drain's next
 * read then opens its own epoch.
 */
void reims_vgpu_dirty_note_child_tail(ReimsVgpuDirty *d, int channel, uint32_t tail,
                                      bool valid)
{
    if (!d || d->hole_mode != 3 || channel < 0 ||
        channel >= (int)ARRAY_SIZE(d->ring_tail)) {
        return;
    }
    qemu_mutex_lock(&d->lock);
    d->ring_tail[channel] = tail;
    d->ring_tail_valid[channel] = valid;
    qemu_mutex_unlock(&d->lock);
}

/*
 * DIRTY_HOLE=tail, the drain's half: it has just read `tail` from child ring
 * `channel`'s registers in guest memory, and will run the packets up to it.
 *
 * The guest publishes a packet by storing the tail after the packet and every
 * buffer it names, so a packet the drain sees was complete before this read.
 * What the drain does not know is whether an epoch it can need was opened
 * after that: the doorbell for these packets may not have been noted yet (the
 * tail is read from memory, not from the doorbell), or may have been noted for
 * an earlier, shorter tail while more packets were appended behind it. In
 * both cases a sync begun before the packets were published could vouch for
 * them, which is the hole.
 *
 * So: if this tail is the one an epoch is already known to cover (a doorbell's
 * snapshot, or this drain's own earlier read), nothing changes. Otherwise open
 * an epoch now, after the read, so that every sync the following reads can
 * accept began after these packets were published, and remember the tail.
 * Called before note_work_scope, which then makes the reads need that epoch.
 * Exact without any assumption about how many doorbells a batch rings.
 */
bool reims_vgpu_dirty_note_ring_tail(ReimsVgpuDirty *d, uint32_t channel, uint32_t tail)
{
    bool opened = false;

    if (!d || d->hole_mode != 3 || channel >= ARRAY_SIZE(d->ring_tail)) {
        return false;
    }
    qemu_mutex_lock(&d->lock);
    if (!d->ring_tail_valid[channel] || d->ring_tail[channel] != tail) {
        opened = true;
        qatomic_set(&d->epoch, d->epoch + 1);
        d->epoch_queried = false;
        qatomic_set(&d->chan_epoch[channel], d->epoch);
        d->lab_od_epochs++;
        d->doorbell_ns = get_clock();
        if (d->lab_tr) {
            qatomic_set(&d->lab_epoch_ns[d->epoch & 4095], d->doorbell_ns);
        }
        d->ring_tail[channel] = tail;
        d->ring_tail_valid[channel] = true;
        d->lab_hole_closed++;
    }
    if (d->frontier_mode) {
        /*
         * The epoch covering every packet up to `tail`: the doorbell epoch whose
         * snapshot this tail is (chan_epoch only grows, so a newer doorbell of
         * this channel only asks for more), or the one just opened above.
         */
        qatomic_set(&d->cover_pending, qatomic_read(&d->chan_epoch[channel]));
        qatomic_set(&d->cover_valid, true);
    }
    qemu_mutex_unlock(&d->lock);
    return opened && d->tailwake_mode && (qatomic_read(&d->log_wanted) || d->od_prefetch);
}

/*
 * Lock-free: the drain is the only writer of `work_epoch`, and the epochs it
 * reads are only ever stored whole. Taking `lock` here queued the drain behind
 * a prefetch batch's locked passes before every submission.
 */
void reims_vgpu_dirty_note_work_scope(ReimsVgpuDirty *d, uint32_t scope)
{
    uint64_t e = 0;

    if (!d) {
        return;
    }
    if (scope != 0xffffffffu && scope >= ARRAY_SIZE(d->chan_epoch)) {
        /* TEMP lab (DIRTY_HOLE): a census notification from the drain (high bits
         * 8 = packet read, 4 = ring drained, 2 = ring found empty), not a scope:
         * it never changes what a read is owed. */
        uint32_t ch = scope & 0xffff;

        if (d->hole_mode && ch < ARRAY_SIZE(d->lab_pk)) {
            switch (scope >> 28) {
            case 0x8:
                qatomic_inc(&d->lab_pk[ch]);
                if (!qatomic_read(&d->lab_run[ch])) {
                    qatomic_set(&d->lab_run[ch], 1);
                    if (qatomic_xchg(&d->lab_dbp[ch], 0) == 0) {
                        qatomic_inc(&d->lab_hole[ch]);
                        if (d->hole_mode == 2) {
                            reims_vgpu_dirty_open_epoch_from_drain(d);
                        }
                    }
                }
                break;
            case 0x4:
                qatomic_inc(&d->lab_dr[ch]);
                qatomic_set(&d->lab_run[ch], 0);
                qatomic_set(&d->lab_dbp[ch], 0);
                break;
            case 0x2:
                qatomic_inc(&d->lab_em[ch]);
                /* nothing was waiting: any doorbell noted since belongs to packets already read */
                qatomic_set(&d->lab_dbp[ch], 0);
                break;
            default:
                break;
            }
        }
        return;
    }
    if (scope < ARRAY_SIZE(d->chan_epoch)) {
        e = qatomic_read(&d->chan_epoch[scope]);
    }
    if (scope == 0xffffffffu && d->frontier_mode) {
        uint64_t cover;
        bool child;

        /* The drain thread's own words (note_ring_tail runs on it too): no lock. */
        child = qatomic_read(&d->cover_valid);
        cover = qatomic_read(&d->cover_pending);
        qatomic_set(&d->cover_valid, false);
        /* A child ring's cover; a root read (no tail noted) needs the newest. */
        e = child ? cover : qatomic_read(&d->epoch);
        e = MAX(e, qatomic_read(&d->work_epoch));   /* the frontier never recedes */
        if (child) {
            d->lab_frontier_child++;
        }
    }
    e = e ? e : qatomic_read(&d->epoch);
    if (d->lab_tr && qatomic_read(&d->work_epoch) != e) {   /* TEMP lab (FGTRACE) */
        ReimsVgpuLabRec r = { .t_ns = get_clock() - d->lab_t0, .needed = e,
                              .epoch = qatomic_read(&d->epoch),
                              .since_db_ns = reims_vgpu_lab_since_db(d, e), .kind = 5 };

        reims_vgpu_lab_rec(d, &r);
    }
    qatomic_set(&d->work_epoch, e);
}

/*
 * Lab GEN_MEMO: what a set answers for a read that needs `needed`. `live` is
 * the generation as it stands. The caller has established that the set's
 * `synced_epoch` is >= `needed` (the memo key must be an epoch the set vouches
 * for). Lock-free: the memo is one word per set, installed by compare-and-swap.
 *
 * Why one answer per (set, needed epoch) is exact. The answer for `needed` is a
 * generation read from a set whose sync began after the doorbell that opened an
 * epoch >= `needed` (the caller has just checked `synced_epoch >= needed`, and
 * `needed` is the epoch every store ordered before the packet being run is
 * ordered before), so it reflects every guest store the packet can depend on.
 * `needed` changes only when the drain announces a new work scope, so every read
 * under the same `needed` is owed exactly that and nothing more. A newer sync
 * that moves `live` while the drain is still inside the epoch has seen a store
 * made after the doorbell: one the packet does not depend on, which the original
 * path happened to report to whichever read came after the sync and not to the
 * ones before it, handing one draw two answers for one set. A later `needed`
 * takes `live`, which is never older than a memoized value (a generation only
 * grows), so no read is ever told less than it was before.
 *
 * 0 ("cannot tell") is never memoized as an answer a later read must repeat: the
 * set can arm within the epoch, and "cannot tell" is the conservative reading
 * either way. Under concurrent readers the first to install an answer for a
 * `needed` wins and the rest adopt it; an install never replaces a newer epoch's.
 */
static uint64_t reims_vgpu_dirty_memo_answer(ReimsVgpuDirty *d, ReimsVgpuDirtySet *s,
                                             uint64_t needed, uint64_t live)
{
    uint64_t m, nm;

    if (!d->gen_memo_mode || live == 0 || (live >> 32) || (needed >> 32)) {
        return live;   /* "cannot tell" is never memoized; nor is what does not fit */
    }
    nm = (needed << 32) | live;
    m = qatomic_read(&s->memo);
    for (;;) {
        uint64_t mn = m >> 32, old;

        if (m != 0 && mn == needed) {
            uint64_t memo = (uint32_t)m;

            qatomic_inc(&d->lab_memo_hit);
            if (memo != live) {
                qatomic_inc(&d->lab_memo_diff);
                if (memo > live) {
                    qatomic_inc(&d->lab_memo_bad);   /* a generation never shrinks */
                }
            }
            return d->gen_memo_mode == 1 ? memo : live;
        }
        if (m != 0 && mn > needed) {
            return live;   /* a later epoch's answer is installed: this read is a straggler */
        }
        old = qatomic_cmpxchg(&s->memo, m, nm);
        if (old == m) {
            qatomic_inc(&d->lab_memo_fresh);
            return live;
        }
        m = old;   /* somebody installed one meanwhile: adopt it, or yield to a later one */
    }
}

/*
 * What a generation read of a set already synced for `needed` records, shared by
 * the locked and the lock-free paths. Atomics only: it runs under `sets_lock` in
 * one and under nothing in the other. The read-order bookkeeping only steers the
 * prefetch; a race over it costs a worse prefetch order, never an answer.
 */
/* Lab DIRTY_PERIOD: remember the needed epoch of a read and its gap. */
static inline void reims_vgpu_dirty_note_read_epoch(ReimsVgpuDirty *d,
                                                    ReimsVgpuDirtySet *s,
                                                    uint64_t needed)
{
    uint64_t last;

    if (!d->period_mode || needed == 0) {
        return;
    }
    last = qatomic_read(&s->rd_epoch);
    if (needed <= last) {
        return;
    }
    qatomic_set(&s->rd_epoch, needed);
    if (last) {
        uint32_t gap = (uint32_t)MIN(needed - last, 255);

        qatomic_set(&s->rd_gaps, (qatomic_read(&s->rd_gaps) << 8) | gap);
    }
}

/*
 * Lab DIRTY_PERIOD: whether a batch for `epoch` should take `s`. A set with no
 * gap history is taken (as every hot set was); otherwise only from its shortest
 * recent gap past its last read on. d->lock held.
 */
static bool reims_vgpu_dirty_period_due(const ReimsVgpuDirtySet *s, uint64_t epoch)
{
    uint32_t gaps = qatomic_read(&s->rd_gaps), floor = 255, g;
    uint64_t last = qatomic_read(&s->rd_epoch);
    int k;

    if (gaps == 0 || last == 0) {
        return true;
    }
    for (k = 0; k < 4; k++) {
        g = (gaps >> (8 * k)) & 0xff;
        if (g) {
            floor = MIN(floor, g);
        }
    }
    return epoch >= last + floor;
}

static void reims_vgpu_dirty_gen_hit(ReimsVgpuDirty *d, ReimsVgpuDirtySet *s,
                                     uint64_t token, uint64_t needed, uint64_t epoch)
{
    reims_vgpu_dirty_note_read_epoch(d, s, needed);
    uint64_t hv = qatomic_read(&d->harvests) + 1;

    if (qatomic_read(&s->lab_last_read) != hv) {
        qatomic_set(&s->lab_last_read, hv);
    }
    if (d->lab_tr) {   /* TEMP lab (FGTRACE) */
        qatomic_set(&s->lab_nreads, qatomic_read(&s->lab_nreads) + 1);
        if (qatomic_read(&s->lab_needed_seen) != needed) {
            ReimsVgpuLabRec r = { .t_ns = get_clock() - d->lab_t0, .token = token,
                                  .needed = needed, .epoch = epoch,
                                  .last_q = qatomic_read(&s->last_query_epoch),
                                  .since_db_ns = reims_vgpu_lab_since_db(d, needed),
                                  .pages = (uint32_t)(s->count * (s->page_size / 4096)),
                                  .kind = 0, .cls = qatomic_read(&s->lab_pref) != 0,
                                  .aux = (uint32_t)qatomic_read(&s->lab_nreads) };

            qatomic_set(&s->lab_needed_seen, needed);
            qatomic_inc(&d->lab_first_reads);
            reims_vgpu_lab_rec(d, &r);
        } else {
            qatomic_inc(&d->lab_dup_reads);
        }
        if (qatomic_read(&s->lab_pref)) {
            qatomic_set(&s->lab_pref, 0);
            qatomic_inc(&d->lab_pref_used);
        }
    }
    if (qatomic_read(&s->last_query_epoch) != epoch) {
        qatomic_set(&s->read_order, qatomic_fetch_inc(&d->read_counter) + 1);
        qatomic_set(&s->last_query_epoch, epoch);
    }
}

/*
 * The lock-free read of a synced set: look the token up in the RCU mirror and
 * answer from the set's own words. Decides and reads only, so the verify arm can
 * run it beside the locked path. false when the set is absent or stale.
 *
 * Why it answers what the locked path would. The decision is the same test on
 * the same words: `synced_epoch >= needed`, loaded with acquire, and every writer
 * of `gen` stores it before the release store of `synced_epoch` that vouches for
 * it, so a set seen synced carries a generation at least that fresh. The lookup
 * differs only in how the set is found: a set is published to the mirror after
 * it is fully built and before its token is returned, and unpublished before it
 * is freed through RCU, so a read either finds a live set or finds none, as it
 * would under the lock.
 */
static bool reims_vgpu_dirty_gen_peek_lf(ReimsVgpuDirty *d, uint64_t token,
                                         uint64_t needed, ReimsVgpuDirtySet **sp,
                                         uint64_t *live)
{
    ReimsVgpuDirtySet *s = reims_vgpu_dirty_tab_lookup(d, token);

    if (!s || qatomic_load_acquire(&s->synced_epoch) < needed) {
        return false;
    }
    *sp = s;
    *live = qatomic_read(&s->gen);
    return true;
}

/*
 * The generation of a set already synced for the epoch a read needs, without
 * taking `lock`; false when the set is absent or stale, for the locked path.
 *
 * `synced_epoch` is published with release order after the generation it
 * covers, so a set seen synced here carries a generation at least that fresh.
 * `work_epoch` is only ever an epoch that has been reached, so reading it and
 * `epoch` apart cannot ask for more than the locked path would.
 */
static bool reims_vgpu_dirty_gen_synced_locked(ReimsVgpuDirty *d, uint64_t token,
                                               uint64_t epoch, uint64_t needed,
                                               uint64_t *gen, uint64_t *live_out)
{
    ReimsVgpuDirtySet *s;
    bool hit = false;

    qemu_mutex_lock(&d->sets_lock);
    s = g_hash_table_lookup(d->sets, &token);
    if (s && qatomic_load_acquire(&s->synced_epoch) >= needed) {
        *live_out = qatomic_read(&s->gen);
        *gen = reims_vgpu_dirty_memo_answer(d, s, needed, *live_out);
        reims_vgpu_dirty_gen_hit(d, s, token, needed, epoch);
        hit = true;
    }
    qemu_mutex_unlock(&d->sets_lock);
    if (hit) {
        qatomic_inc(&d->reads_since_harvest);
        qatomic_inc(&d->lab_gen_fast);
    }
    return hit;
}

static bool reims_vgpu_dirty_gen_synced(ReimsVgpuDirty *d, uint64_t token,
                                        uint64_t *gen)
{
    uint64_t epoch, work, needed, live = 0, lf_live = 0;
    ReimsVgpuDirtySet *s = NULL;
    bool hit, lf_hit = false;

    if (!qatomic_read(&d->ondemand)) {
        return false;
    }
    epoch = qatomic_read(&d->epoch);
    work = qatomic_read(&d->work_epoch);
    needed = work && !d->work_epoch_off ? MIN(work, epoch) : epoch;
    if (d->lf_mode == 1) {
        RCU_READ_LOCK_GUARD();
        if (!reims_vgpu_dirty_gen_peek_lf(d, token, needed, &s, &live)) {
            return false;
        }
        *gen = reims_vgpu_dirty_memo_answer(d, s, needed, live);
        reims_vgpu_dirty_gen_hit(d, s, token, needed, epoch);
        /* One shared flag, written only when it has been cleared; the counter is
         * sampled by token so that concurrent readers do not queue on one line. */
        if (qatomic_read(&d->reads_since_harvest) == 0) {
            qatomic_set(&d->reads_since_harvest, 1);
        }
        if ((token & 7) == 0) {
            qatomic_inc(&d->lab_gen_fast_s);
        }
        return true;
    }
    if (d->lf_mode == 2) {
        RCU_READ_LOCK_GUARD();
        lf_hit = reims_vgpu_dirty_gen_peek_lf(d, token, needed, &s, &lf_live);
    }
    hit = reims_vgpu_dirty_gen_synced_locked(d, token, epoch, needed, gen, &live);
    if (d->lf_mode == 2) {
        /* Lab verify: the lock-free read against the locked one, taken one after
         * the other. A set a sync completes in between moves between them, so a
         * difference is read again before it counts. */
        if (lf_hit != hit) {
            RCU_READ_LOCK_GUARD();
            bool again = reims_vgpu_dirty_gen_peek_lf(d, token, needed, &s, &lf_live);

            if (again != hit) {
                qatomic_inc(&d->lab_lf_hitdiff);
            } else {
                qatomic_inc(&d->lab_lf_race);
            }
        } else if (hit) {
            qatomic_inc(&d->lab_lf_hit);
            if (lf_live != live) {
                RCU_READ_LOCK_GUARD();
                uint64_t cur = 0;
                ReimsVgpuDirtySet *s2;

                /* A generation only grows: a sync that landed between the two reads
                 * shows as lf_live < live <= what the set holds now. */
                if (reims_vgpu_dirty_gen_peek_lf(d, token, needed, &s2, &cur) &&
                    cur >= live && live > lf_live) {
                    qatomic_inc(&d->lab_lf_race);
                } else {
                    qatomic_inc(&d->lab_lf_gendiff);
                }
            }
        }
    }
    return hit;
}

/*
 * TEMP lab (REIMS_VGPU_LAB_STATED_VOUCH): a token with this bit set asks for the
 * generation the set was last synced to, without bringing it up to date for
 * the current epoch. The caller holds a second witness (the guest's own stated
 * writes) and accepts that a write the guest did not state is seen one read
 * late. The read still keeps the set hot, so the prefetch keeps syncing it.
 */
#define REIMS_VGPU_DIRTY_TOKEN_LAGGED (1ULL << 63)

static uint64_t reims_vgpu_dirty_gen_lagged(ReimsVgpuDirty *d, uint64_t token)
{
    ReimsVgpuDirtySet *s;
    uint64_t gen = 0, epoch = qatomic_read(&d->epoch);

    if (d->lf_mode == 1) {   /* TEMP lab GEN_LOCKFREE: the same read, found through the mirror */
        RCU_READ_LOCK_GUARD();
        s = reims_vgpu_dirty_tab_lookup(d, token);
        if (s) {
            uint64_t hv = qatomic_read(&d->harvests) + 1;

            gen = qatomic_read(&s->gen);
            if (qatomic_read(&s->lab_last_read) != hv) {
                qatomic_set(&s->lab_last_read, hv);
            }
            if (qatomic_read(&s->last_query_epoch) != epoch) {
                qatomic_set(&s->read_order, qatomic_fetch_inc(&d->read_counter) + 1);
                qatomic_set(&s->last_query_epoch, epoch);
            }
        }
        if (qatomic_read(&d->reads_since_harvest) == 0) {
            qatomic_set(&d->reads_since_harvest, 1);
        }
        return gen;
    }
    qemu_mutex_lock(&d->sets_lock);
    s = g_hash_table_lookup(d->sets, &token);
    if (s) {
        gen = qatomic_read(&s->gen);
        qatomic_set(&s->lab_last_read, qatomic_read(&d->harvests) + 1);
        if (qatomic_read(&s->last_query_epoch) != epoch) {
            qatomic_set(&s->read_order, qatomic_fetch_inc(&d->read_counter) + 1);
            qatomic_set(&s->last_query_epoch, epoch);
        }
    }
    qemu_mutex_unlock(&d->sets_lock);
    qatomic_inc(&d->reads_since_harvest);
    qatomic_inc(&d->lab_gen_lagged);
    return gen;
}

uint64_t reims_vgpu_dirty_gen(ReimsVgpuDirty *d, uint64_t token)
{
    ReimsVgpuDirtySet *s;
    uint64_t gen = 0;
    bool stale, lagged = token & REIMS_VGPU_DIRTY_TOKEN_LAGGED;
    int64_t fg0;
    /* TEMP lab (FGTRACE) */
    int64_t lt0 = 0, lw = 0, lq = 0;
    uint64_t l_needed = 0, l_last_q = 0, l_gen0 = 0, l_epoch = 0, l_created = 0;
    uint32_t l_pages = 0;
    uint8_t l_cls = 0;
    uint64_t needed_pre = 0;   /* TEMP lab GEN_MEMO: the epoch this read needed when it began */

    token &= ~REIMS_VGPU_DIRTY_TOKEN_LAGGED;
    if (!d || token == 0) {
        return 0;
    }
    if (d->lab_tr) {
        lt0 = get_clock();
    }
    if (d->gen_fast && reims_vgpu_dirty_gen_synced(d, token, &gen)) {
        if (lt0) {
            qatomic_add(&d->lab_fast_ns, get_clock() - lt0);
            qatomic_inc(&d->lab_fast_n);
        }
        return gen;
    }
    if (lagged && qatomic_read(&d->ondemand)) {
        return reims_vgpu_dirty_gen_lagged(d, token);
    }
    qemu_mutex_lock(&d->lock);
    s = g_hash_table_lookup(d->sets, &token);
    stale = s && d->ondemand && s->synced_epoch < reims_vgpu_dirty_needed_epoch(d);
    needed_pre = reims_vgpu_dirty_needed_epoch(d);
    if (stale) {   /* TEMP lab */
        s->lab_fg_n++;
        if (s->last_query_epoch + d->od_hot < d->epoch) {
            d->lab_fg_cold++;
            l_cls = 0;
        } else if (d->lab_pb_epoch >= reims_vgpu_dirty_needed_epoch(d)) {
            d->lab_fg_missed++;
            l_cls = 1;
        } else {
            d->lab_fg_ahead++;
            l_cls = 2;
        }
        if (d->lab_tr) {
            l_needed = reims_vgpu_dirty_needed_epoch(d);
            l_last_q = s->last_query_epoch;
            l_gen0 = s->gen;
            l_epoch = d->epoch;
            l_created = s->lab_created_epoch;
            l_pages = (uint32_t)(s->count * (s->page_size / 4096));
            if (s->lab_nreads == 0) {
                l_cls = 3;
            }
        }
    }
    qemu_mutex_unlock(&d->lock);
    fg0 = stale ? get_clock() : 0;
    if (stale) {
        reims_vgpu_dirty_sync_one(d, token, true, &lw, &lq);
    }
    qemu_mutex_lock(&d->lock);
    if (stale) {
        d->lab_od_fg_n++;
        d->lab_od_fg_ns += get_clock() - fg0;
    }
    s = g_hash_table_lookup(d->sets, &token);
    if (s) {
        gen = s->gen;
        if (d->gen_memo_mode &&
            qatomic_load_acquire(&s->synced_epoch) >= needed_pre) {   /* TEMP lab GEN_MEMO */
            /*
             * The epoch this read needed when it began, not the one the drain
             * announces by now: a sync only vouches for the epochs that had been
             * reached when it started, and a memo installed under a later key
             * would hand the next read of that epoch an answer older than it is
             * owed. (The stress harness caught exactly that.) And only if the set
             * does vouch for it.
             */
            gen = reims_vgpu_dirty_memo_answer(d, s, needed_pre, gen);
        }
        s->lab_last_read = d->harvests + 1;
        reims_vgpu_dirty_note_read_epoch(d, s, needed_pre);
        if (d->lab_tr) {   /* TEMP lab (FGTRACE) */
            s->lab_nreads++;
            s->lab_pref = 0;
            if (stale) {
                ReimsVgpuLabRec r = { .t_ns = get_clock() - d->lab_t0, .token = token,
                                      .needed = l_needed, .epoch = l_epoch,
                                      .last_q = l_last_q,
                                      .since_db_ns = -1, .pages = l_pages, .kind = 2,
                                      .cls = l_cls, .dirty = gen != l_gen0,
                                      .total_ns = (uint32_t)(get_clock() - fg0),
                                      .wait_ns = (uint32_t)lw, .query_ns = (uint32_t)lq,
                                      .aux = (uint32_t)(l_epoch - l_created) };
                int64_t db = l_needed ? d->lab_epoch_ns[l_needed & 4095] : 0;

                r.since_db_ns = db ? fg0 - db : -1;   /* when the read began, not ended */
                s->lab_needed_seen = l_needed;
                reims_vgpu_lab_rec(d, &r);
            }
        }
        if (s->last_query_epoch != d->epoch) {
            qatomic_set(&s->read_order, qatomic_fetch_inc(&d->read_counter) + 1);
        }
        qatomic_set(&s->last_query_epoch, d->epoch);
    }
    qatomic_inc(&d->reads_since_harvest);
    qemu_mutex_unlock(&d->lock);
    return gen;
}

/*
 * Lab GEN_LOCKFREE: the page list of a set that is already synced for the epoch
 * the read needs, without `lock`. `*ok` false = not answerable here (the set is
 * absent or stale, or an evaluation kept rewriting it), take the locked path.
 *
 * It returns what the locked path returns for such a set: the pages whose stamp
 * is newer than `since_gen`, -1 for "assume everything" (a set still unarmed, a
 * `since_gen` of 0, a list longer than `max`). The one difference is how the
 * stamps are read: an evaluation rewrites `gen` and the stamps under `lock`
 * inside an odd `eval_seq`, and this reads the page list between two equal even
 * values of it, so it sees one evaluation's generation and stamps whole.
 */
static int64_t reims_vgpu_dirty_written_since_lf(ReimsVgpuDirty *d, uint64_t token,
                                                 uint64_t since_gen, uint64_t *out,
                                                 size_t max, bool *ok)
{
    ReimsVgpuDirtySet *s;
    uint64_t epoch, work, needed;
    int tries;

    *ok = false;
    epoch = qatomic_read(&d->epoch);
    work = qatomic_read(&d->work_epoch);
    needed = work && !d->work_epoch_off ? MIN(work, epoch) : epoch;
    RCU_READ_LOCK_GUARD();
    s = reims_vgpu_dirty_tab_lookup(d, token);
    if (!s || qatomic_load_acquire(&s->synced_epoch) < needed) {
        return 0;
    }
    for (tries = 0; tries < 8; tries++) {
        uint32_t seq1 = qatomic_load_acquire(&s->eval_seq), seq2;
        int64_t found = 0;
        bool over = false;
        uint64_t gen;
        size_t p;

        if (seq1 & 1) {
            cpu_relax();
            continue;
        }
        gen = qatomic_read(&s->gen);
        if (gen != 0 && since_gen != 0) {
            for (p = 0; p < s->count; p++) {
                if (qatomic_read(&s->page_gen[p]) <= since_gen) {
                    continue;
                }
                if ((size_t)found == max) {
                    over = true;   /* truncation would read as "these pages and no others" */
                    break;
                }
                out[found++] = s->pages[p];
            }
        }
        smp_rmb();
        seq2 = qatomic_load_acquire(&s->eval_seq);
        if (seq1 != seq2) {
            continue;
        }
        if (qatomic_read(&s->last_query_epoch) != epoch) {
            qatomic_set(&s->read_order, qatomic_fetch_inc(&d->read_counter) + 1);
            qatomic_set(&s->last_query_epoch, epoch);
        }
        if (qatomic_read(&d->reads_since_harvest) == 0) {
            qatomic_set(&d->reads_since_harvest, 1);
        }
        *ok = true;
        return (gen == 0 || since_gen == 0 || over) ? -1 : found;
    }
    return 0;
}

int64_t reims_vgpu_dirty_written_since(ReimsVgpuDirty *d, uint64_t token,
                                       uint64_t since_gen, uint64_t *out,
                                       size_t max)
{
    ReimsVgpuDirtySet *s;
    int64_t found = 0;
    size_t p;
    /* TEMP lab GEN_LOCKFREE */
    int64_t lf_res = 0;
    bool lf_ok = false;
    uint64_t *lf_buf = NULL;

    int64_t ws0 = 0, lw = 0, lq = 0;   /* TEMP lab (FGTRACE) */
    bool l_synced = false;
    uint64_t l_needed = 0, l_last_q = 0, l_gen0 = 0, l_epoch = 0, l_created = 0;
    uint32_t l_pages = 0;
    uint8_t l_cls = 0;

    if (!d || token == 0 || out == NULL) {
        return -1;
    }
    if (d->lab_tr) {
        ws0 = get_clock();
    }
    if (d->lf_mode == 1 && d->ondemand) {
        lf_res = reims_vgpu_dirty_written_since_lf(d, token, since_gen, out, max, &lf_ok);
        if (lf_ok) {
            qatomic_inc(&d->lab_lfw_hit);
            return lf_res;
        }
        qatomic_inc(&d->lab_lfw_miss);
    } else if (d->lf_mode == 2 && d->ondemand) {
        lf_buf = g_new(uint64_t, max ? max : 1);
        lf_res = reims_vgpu_dirty_written_since_lf(d, token, since_gen, lf_buf, max, &lf_ok);
    }
    qemu_mutex_lock(&d->lock);
    s = g_hash_table_lookup(d->sets, &token);
    if (s && d->ondemand && s->synced_epoch < reims_vgpu_dirty_needed_epoch(d)) {
        /* A page list read in a later epoch than the set's last sync would
         * miss what the guest wrote since; bring the set up to date first. */
        int64_t fg0 = get_clock();

        s->lab_fg_n++;   /* TEMP lab */
        if (s->last_query_epoch + d->od_hot < d->epoch) {   /* TEMP lab */
            d->lab_fg_cold++;
        } else if (d->lab_pb_epoch >= reims_vgpu_dirty_needed_epoch(d)) {
            d->lab_fg_missed++;
            l_cls = 1;
        } else {
            d->lab_fg_ahead++;
            l_cls = 2;
        }
        if (d->lab_tr) {
            l_synced = true;
            l_needed = reims_vgpu_dirty_needed_epoch(d);
            l_last_q = s->last_query_epoch;
            l_gen0 = s->gen;
            l_epoch = d->epoch;
            l_created = s->lab_created_epoch;
            l_pages = (uint32_t)(s->count * (s->page_size / 4096));
            if (s->lab_nreads == 0) {
                l_cls = 3;
            }
        }

        qemu_mutex_unlock(&d->lock);
        reims_vgpu_dirty_sync_one(d, token, true, &lw, &lq);
        qemu_mutex_lock(&d->lock);
        d->lab_od_fg_n++;
        d->lab_od_fg_ns += get_clock() - fg0;
        s = g_hash_table_lookup(d->sets, &token);
        if (s && l_synced) {   /* TEMP lab (FGTRACE) */
            ReimsVgpuLabRec r = { .t_ns = get_clock() - d->lab_t0, .token = token,
                                  .needed = l_needed, .epoch = l_epoch,
                                  .last_q = l_last_q, .since_db_ns = -1,
                                  .pages = l_pages, .kind = 3, .cls = l_cls,
                                  .dirty = s->gen != l_gen0,
                                  .total_ns = (uint32_t)(get_clock() - fg0),
                                  .wait_ns = (uint32_t)lw, .query_ns = (uint32_t)lq,
                                  .aux = (uint32_t)(l_epoch - l_created) };
            int64_t db = l_needed ? d->lab_epoch_ns[l_needed & 4095] : 0;

            r.since_db_ns = db ? fg0 - db : -1;
            reims_vgpu_lab_rec(d, &r);
        }
    } else if (s && d->lab_tr) {   /* TEMP lab (FGTRACE): no sync was needed */
        ReimsVgpuLabRec r = { .t_ns = get_clock() - d->lab_t0, .token = token,
                              .needed = reims_vgpu_dirty_needed_epoch(d), .epoch = d->epoch,
                              .last_q = s->last_query_epoch, .since_db_ns = -1,
                              .pages = (uint32_t)(s->count * (s->page_size / 4096)),
                              .kind = 10 };

        reims_vgpu_lab_rec(d, &r);
    }
    if (s) {
        if (d->lab_tr) {
            s->lab_nreads++;
            s->lab_pref = 0;
        }
        if (s->last_query_epoch != d->epoch) {
            qatomic_set(&s->read_order, qatomic_fetch_inc(&d->read_counter) + 1);
        }
        qatomic_set(&s->last_query_epoch, d->epoch);
    }
    /*
     * `since_gen == 0` is a caller that never recorded a readable observation,
     * and `s->gen == 0` is a set still inside its startup window. Neither can
     * be compared against a page stamp, and both mean the same thing to the
     * caller as an unknown token does.
     */
    if (ws0) {   /* TEMP lab (FGTRACE) */
        d->lab_ws_n++;
        d->lab_ws_sync_n += l_synced;
        d->lab_ws_ns += get_clock() - ws0;
    }
    if (!s || s->gen == 0 || since_gen == 0) {
        qemu_mutex_unlock(&d->lock);
        if (lf_buf) {   /* TEMP lab verify */
            if (lf_ok && lf_res != -1) {
                qatomic_inc(&d->lab_lfw_diff);
            }
            g_free(lf_buf);
        }
        return -1;
    }
    for (p = 0; p < s->count; p++) {
        if (s->page_gen[p] <= since_gen) {
            continue;
        }
        if ((size_t)found == max) {
            /* Truncation would read as "these pages and no others". */
            qemu_mutex_unlock(&d->lock);
            if (lf_buf) {   /* TEMP lab verify */
                if (lf_ok && lf_res != -1) {
                    qatomic_inc(&d->lab_lfw_diff);
                }
                g_free(lf_buf);
            }
            return -1;
        }
        out[found++] = s->pages[p];
    }
    /*
     * Counted like reims_vgpu_dirty_gen(): a caller asking this question is a
     * consumer of the report, so the next harvest has something to tell it and
     * must not be skipped.
     */
    qatomic_inc(&d->reads_since_harvest);
    qemu_mutex_unlock(&d->lock);
    if (lf_buf) {
        /* TEMP lab verify: the lock-free list against the locked one. An evaluation
         * between the two reads moves one of them, so a difference is read again
         * (lock-free) before it counts. */
        bool same = lf_ok && lf_res == found &&
                    memcmp(lf_buf, out, found * sizeof(uint64_t)) == 0;

        if (lf_ok && !same) {
            bool ok2;
            int64_t r2 = reims_vgpu_dirty_written_since_lf(d, token, since_gen, lf_buf, max, &ok2);

            if (ok2 && r2 == found && memcmp(lf_buf, out, found * sizeof(uint64_t)) == 0) {
                qatomic_inc(&d->lab_lfw_race);
            } else {
                qatomic_inc(&d->lab_lfw_diff);
            }
        } else if (lf_ok) {
            qatomic_inc(&d->lab_lfw_hit);
        }
        g_free(lf_buf);
    }
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
    /* Stored atomically: the generation fast path reads it without `lock`. */
    qatomic_set(&s->eval_seq, s->eval_seq + 1);   /* odd: rewriting gen and the stamps */
    smp_wmb();
    if (by_harvest) {
        if (s->gen == 0 && unlogged) {
            s->arm_at = d->harvests + 1;
        }
        if (d->harvests < s->arm_at) {
            qatomic_set(&s->gen, 0);
        } else if (s->gen == 0) {
            qatomic_set(&s->gen, 1);
        } else if (any) {
            qatomic_set(&s->gen, s->gen + 1);
        }
    } else if (s->gen == 0) {
        if (!unlogged) {
            qatomic_set(&s->gen, 1);
        }
    } else if (any) {
        qatomic_set(&s->gen, s->gen + 1);
    }
    if (s->gen != 0) {
        guint h;

        for (h = 0; h < hit->len; h++) {
            qatomic_set(&s->page_gen[g_array_index(hit, size_t, h)], s->gen);
        }
    }
    smp_wmb();
    qatomic_set(&s->eval_seq, s->eval_seq + 1);   /* even again */
    s->seen_seq = d->global_seq;
    if (d->wp_mode && d->wp_active && !by_harvest && s->wp_state == WP_NONE && s->gen != 0 &&
        s->count * (s->page_size >> shift) >= d->wp_min_pages) {   /* lab DIRTY_WP */
        s->wp_streak = any ? 0 : s->wp_streak + 1;
        if (s->wp_streak == (d->wp_streak_need << MIN(s->wp_lost, 6))) {
            g_array_append_val(d->wp_queue, s->token);
            REIMS_WP_BH_SCHEDULE(d->wp_bh);
        }
    }
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
    d->epoch_queried = true;
    while (d->q_ninflight == (int)ARRAY_SIZE(d->q_inflight)) {
        qemu_cond_wait(&d->q_cond, &d->lock);
    }
    d->q_fp[d->q_ninflight] = NULL;
    d->q_fpn[d->q_ninflight] = 0;
    d->q_inflight[d->q_ninflight++] = ++d->q_next;
    return d->q_next;
}

/*
 * Lab DIRTY_RSETTLE: name what `ticket` queries (`chunks` sorted by region and
 * offset, alive until the ticket ends). Before the queries are issued, so a
 * settle never sees a ticket's footprint narrower than its queries. d->lock held.
 */
static void reims_vgpu_dirty_query_footprint(ReimsVgpuDirty *d, uint64_t ticket,
                                             const ReimsVgpuDirtyChunk *chunks, guint n)
{
    int i;

    if (!d->rsettle_mode || ticket == 0) {
        return;
    }
    for (i = 0; i < d->q_ninflight; i++) {
        if (d->q_inflight[i] == ticket) {
            d->q_fp[i] = chunks;
            d->q_fpn[i] = n;
            return;
        }
    }
}

static int reims_vgpu_dirty_cmp_chunk(gconstpointer a, gconstpointer b)
{
    const ReimsVgpuDirtyChunk *x = a, *y = b;

    if (x->mr != y->mr) {
        return (uintptr_t)x->mr < (uintptr_t)y->mr ? -1 : 1;
    }
    return x->r.start < y->r.start ? -1 : x->r.start > y->r.start;
}

/*
 * Whether two footprints share a byte. Both sorted by (region, offset); NULL
 * is "not known" and meets everything. An interval that ends before the other
 * list's current one starts meets nothing later in that list either.
 */
static bool reims_vgpu_dirty_fp_meet(const ReimsVgpuDirtyChunk *a, guint na,
                                     const ReimsVgpuDirtyChunk *b, guint nb)
{
    guint i = 0, j = 0;

    if (!a || !b) {
        return true;
    }
    while (i < na && j < nb) {
        if (a[i].mr != b[j].mr) {
            if ((uintptr_t)a[i].mr < (uintptr_t)b[j].mr) {
                i++;
            } else {
                j++;
            }
        } else if (a[i].r.start + a[i].r.len <= b[j].r.start) {
            i++;
        } else if (b[j].r.start + b[j].r.len <= a[i].r.start) {
            j++;
        } else {
            return true;
        }
    }
    return false;
}

/* Return a ticket whose queries' bits are in the VGA bitmap. d->lock held. */
static void reims_vgpu_dirty_query_end(ReimsVgpuDirty *d, uint64_t ticket)
{
    int i;

    for (i = 0; i < d->q_ninflight; i++) {
        if (d->q_inflight[i] == ticket) {
            --d->q_ninflight;
            d->q_inflight[i] = d->q_inflight[d->q_ninflight];
            d->q_fp[i] = d->q_fp[d->q_ninflight];
            d->q_fpn[i] = d->q_fpn[d->q_ninflight];
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

        reims_vgpu_dirty_query_footprint(d, pt, &c, 1);
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
static void reims_vgpu_dirty_query_settle_fp(ReimsVgpuDirty *d, uint64_t ticket,
                                             const ReimsVgpuDirtyChunk *mine,
                                             guint nmine)
{
    uint64_t upto = d->q_next;
    bool waited = false, skipped = false;
    int i;

    reims_vgpu_dirty_query_end(d, ticket);   /* 0 = the caller holds none */
    if (!d->rsettle_mode) {
        mine = NULL;
    }
    for (;;) {
        bool pending = false;

        for (i = 0; i < d->q_ninflight; i++) {
            if (d->q_inflight[i] > upto) {
                continue;
            }
            if (reims_vgpu_dirty_fp_meet(d->q_fp[i], d->q_fpn[i], mine, nmine)) {
                pending = true;
            } else {
                skipped = true;
            }
        }
        if (!pending) {
            break;
        }
        waited = true;
        qemu_cond_wait(&d->q_cond, &d->lock);
    }
    if (d->rsettle_mode) {   /* TEMP lab */
        d->lab_rs_waits += waited;
        d->lab_rs_skipped += skipped;
    }
}

static void reims_vgpu_dirty_query_settle(ReimsVgpuDirty *d, uint64_t ticket)
{
    reims_vgpu_dirty_query_settle_fp(d, ticket, NULL, 0);
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
    qatomic_set(&d->reads_since_harvest, 0);
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
            qatomic_store_release(&set->synced_epoch, MAX(set->synced_epoch, epoch));
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
            qatomic_store_release(&set->synced_epoch, MAX(set->synced_epoch, epoch));
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

static void reims_vgpu_dirty_sync_one(ReimsVgpuDirty *d, uint64_t token, bool fg,
                                      int64_t *lab_wait, int64_t *lab_query)
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
    bool stole = false;   /* TEMP lab STEAL */
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
        /*
         * Lab STEAL: the claimant is a prefetch batch, which publishes every set
         * it holds only after the whole union has been queried, consumed and
         * evaluated (~1 ms), and this set needs one query. Query it here. The
         * claim's only job was to spare a second thread the same hypervisor
         * calls; skipping it repeats them, and the ticket protocol (every query
         * holds a ticket from before it starts until its bits are in the VGA
         * bitmap, and a sync consumes only after every earlier ticket is back)
         * is what makes two overlapping queries of one page safe: whichever
         * took the hypervisor's bit, the settle below waits for it to be
         * deposited before this sync consumes. A set claimed by another
         * *thread's* sync (`bclaim` 0) is still waited for.
         */
        if (d->steal_mode && fg && s->bclaim >= epoch) {
            stole = true;
            d->lab_steals++;
            break;
        }
        /* Waiting for the claimant: query its split chunks meanwhile. */
        if (d->split_next < d->split_total) {
            reims_vgpu_dirty_split_help(d);
        } else {
            qemu_cond_wait(&d->synced_cond, &d->lock);
        }
        s = g_hash_table_lookup(d->sets, &token);
    }
    d->lab_ph_wait += get_clock() - ph;
    if (lab_wait) {   /* TEMP lab (FGTRACE) */
        *lab_wait = get_clock() - ph;
    }
    if (!s || s->synced_epoch >= epoch) {
        qemu_mutex_unlock(&d->lock);
        return;
    }
    ph = get_clock();
    s->claim_epoch = epoch;
    d->epoch_queried = true;
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
    /*
     * Lab DIRTY_RSETTLE: the ticket is taken below, with its footprint, once
     * the ranges are known: a ticket held through the walk that builds them
     * has no footprint yet, and every settle meanwhile has to wait for it.
     */
    qticket = d->rsettle_mode ? 0 : reims_vgpu_dirty_query_begin(d);
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
    if (d->rsettle_mode) {
        g_array_sort(chunks, reims_vgpu_dirty_cmp_chunk);
    }
    qemu_mutex_lock(&d->lock);
    if (d->rsettle_mode) {
        qticket = reims_vgpu_dirty_query_begin(d);
    }
    reims_vgpu_dirty_query_footprint(d, qticket, (const ReimsVgpuDirtyChunk *)chunks->data,
                                     chunks->len);
    if (d->od_split && d->od_nworkers > 0 && walked_pages >= 4096 &&
        !d->split_busy && (fg || !d->bigsplit_mode)) {
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
        d->lab_fg_split++;   /* TEMP lab (FGTRACE) */
        qemu_mutex_unlock(&d->lock);
    } else {
        guint k = 0;
        int64_t w0 = d->lab_tr ? get_clock() : 0;   /* TEMP lab (FGTRACE) */

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
        if (w0) {
            qatomic_add(&d->lab_whv_ns, get_clock() - w0);
            qatomic_add(&d->lab_whv_calls, chunks->len);
            qatomic_add(&d->lab_fg_walked, walked_pages);
        }
    }

    written = g_array_new(FALSE, FALSE, sizeof(ReimsVgpuDirtyWritten));
    hit = g_array_new(FALSE, FALSE, sizeof(size_t));
    qemu_mutex_lock(&d->lock);
    ph_query_end = get_clock();
    if (lab_query) {   /* TEMP lab (FGTRACE) */
        *lab_query = ph_query_end - ph_prep_end;
    }
    reims_vgpu_dirty_query_settle_fp(d, qticket, (const ReimsVgpuDirtyChunk *)chunks->data,
                                     chunks->len);
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
        qatomic_store_release(&s->synced_epoch, MAX(s->synced_epoch, epoch));
        if (s->claim_epoch == epoch) {
            s->claim_epoch = 0;
        }
    }
    qemu_cond_broadcast(&d->synced_cond);
    if (stole && d->steal_mode == 2) {
        /* Lab STEAL verify: answer as the original did, after the batch. */
        uint64_t g1 = s ? s->gen : 0;

        while ((s = g_hash_table_lookup(d->sets, &token)) && s->bclaim >= epoch) {
            qemu_cond_wait(&d->synced_cond, &d->lock);
        }
        if (s) {
            if (s->gen != g1) {
                d->lab_steal_late++;
            } else {
                d->lab_steal_same++;
            }
        }
    }
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
                    " gen_fast=%" PRIu64 " db_coalesced=%" PRIu64 " gen_lagged=%" PRIu64
                    " period=%u period_skipped=%" PRIu64 " period_taken=%" PRIu64
                    " bigsplit=%u bigsplit_sets=%" PRIu64 " rsettle=%u rs_waits=%" PRIu64
                    " rs_skipped=%" PRIu64 " bigphase=%u bp_big=%" PRIu64
                    " bp_groups=%" PRIu64 " frontier=%u fr_child=%" PRIu64 "\n",
                    g_get_real_time() / 1000, d->lab_od_n, d->lab_od_ns / 1000,
                    d->lab_od_pages, d->lab_od_skipped, d->lab_od_epochs,
                    d->lab_od_fg_n, d->lab_od_fg_ns / 1000, d->lab_ph_wait / 1000,
                    d->lab_ph_prep / 1000, d->lab_ph_query / 1000,
                    d->lab_ph_settle / 1000, d->lab_ph_consume / 1000,
                    d->lab_fg_ahead, d->lab_fg_missed, d->lab_fg_cold,
                    qatomic_xchg(&d->lab_gen_fast, 0), d->lab_db_coalesced,
                    qatomic_xchg(&d->lab_gen_lagged, 0), d->period_mode,
                    d->lab_period_skipped, d->lab_period_taken,
                    d->bigsplit_mode, d->lab_bigsplit_sets, d->rsettle_mode,
                    d->lab_rs_waits, d->lab_rs_skipped, d->bigphase_mode,
                    d->lab_bigphase_big, d->lab_bigphase_groups, d->frontier_mode,
                    d->lab_frontier_child);
            d->lab_frontier_child = 0;
            if (d->wp_toggle) {   /* lab DIRTY_WP=toggle: follow the flag file */
                bool want = g_file_test(d->wp_toggle, G_FILE_TEST_EXISTS);

                if (want != qatomic_read(&d->wp_active)) {
                    qatomic_set(&d->wp_active, want);
                    if (!want) {
                        qatomic_set(&d->wp_drop_all, true);
                        REIMS_WP_BH_SCHEDULE(d->wp_bh);
                    }
                }
            }
            if (d->wp_mode || d->lab_wp_refused) {   /* lab DIRTY_WP */
                fprintf(stderr, "reims-dirty-wp active=%u listed=%u protecting=%u protects=%" PRIu64
                        " chunks=%" PRIu64 " faults=%" PRIu64 " lost=%" PRIu64 " dropped=%" PRIu64
                        " orphans=%u remaps=%" PRIu64 " refused=%" PRIu64 " remap_us=%" PRId64
                        " pause_us=%" PRId64 " pause_max_us=%" PRId64 "\n",
                        d->wp_active, d->wp_list ? d->wp_list->len : 0, d->wp_cur != 0,
                        d->lab_wp_protects, d->lab_wp_chunks, d->lab_wp_faults, d->lab_wp_lost,
                        d->lab_wp_dropped, d->wp_orphans ? d->wp_orphans->len : 0, d->lab_wp_remaps,
                        d->lab_wp_refused, (int64_t)d->lab_wp_remap_ns / 1000,
                        (int64_t)d->lab_wp_pause_ns / 1000, (int64_t)d->lab_wp_pause_max_ns / 1000);
                d->lab_wp_protects = d->lab_wp_chunks = d->lab_wp_faults = d->lab_wp_lost = 0;
                d->lab_wp_dropped = d->lab_wp_remaps = d->lab_wp_remap_ns = 0;
                d->lab_wp_pause_ns = d->lab_wp_pause_max_ns = 0;
            }
            {   /* TEMP lab: the sets the foreground syncs most, and why WP leaves them */
                GHashTableIter it;
                gpointer k, v;
                ReimsVgpuDirtySet *top[5] = { 0 };
                GString *line = g_string_new("reims-dirty-fgtop");
                int ti;

                g_hash_table_iter_init(&it, d->sets);
                while (g_hash_table_iter_next(&it, &k, &v)) {
                    ReimsVgpuDirtySet *ts = v;

                    for (ti = 0; ti < 5; ti++) {
                        if (!top[ti] || ts->lab_fg_n > top[ti]->lab_fg_n) {
                            memmove(&top[ti + 1], &top[ti], (4 - ti) * sizeof(top[0]));
                            top[ti] = ts;
                            break;
                        }
                    }
                }
                for (ti = 0; ti < 5 && top[ti] && top[ti]->lab_fg_n; ti++) {
                    g_string_append_printf(line, " [n=%u pages=%" PRIu64 " wp=%u streak=%u lost=%u gen=%" PRIu64 "]",
                                           top[ti]->lab_fg_n,
                                           (uint64_t)(top[ti]->count * (top[ti]->page_size >> qemu_target_page_bits())),
                                           top[ti]->wp_state, top[ti]->wp_streak, top[ti]->wp_lost,
                                           top[ti]->gen);
                }
                fprintf(stderr, "%s\n", line->str);
                g_string_free(line, TRUE);
                g_hash_table_iter_init(&it, d->sets);
                while (g_hash_table_iter_next(&it, &k, &v)) {
                    ((ReimsVgpuDirtySet *)v)->lab_fg_n = 0;
                }
            }
            d->lab_rs_waits = d->lab_rs_skipped = 0;
            d->lab_bigphase_big = d->lab_bigphase_groups = 0;
            d->lab_period_skipped = d->lab_period_taken = 0;
            d->lab_bigsplit_sets = 0;
            d->lab_db_coalesced = 0;
            d->lab_fg_ahead = d->lab_fg_missed = d->lab_fg_cold = 0;
            d->lab_ph_wait = d->lab_ph_prep = d->lab_ph_query = 0;
            d->lab_ph_settle = d->lab_ph_consume = 0;
            d->lab_od_n = d->lab_od_ns = d->lab_od_pages = d->lab_od_skipped = 0;
            d->lab_od_epochs = d->lab_od_fg_n = d->lab_od_fg_ns = 0;
            d->lab_od_last = now;
            if (d->steal_mode || d->gen_memo_mode || d->epoch_named ||
                d->lf_mode) {   /* TEMP lab gates */
                fprintf(stderr, "reims-dirty-lab epoch_ms=%" PRId64 " steal=%u memo=%u"
                        " steals=%" PRIu64 " steal_same=%" PRIu64 " steal_late=%" PRIu64
                        " memo_hit=%" PRIu64 " memo_fresh=%" PRIu64
                        " memo_diff=%" PRIu64 " memo_bad=%" PRIu64 " named=%u unnamed_skipped=%" PRIu64 " lf=%u lf_hit=%" PRIu64 " lf_hitdiff=%" PRIu64
                        " lf_gendiff=%" PRIu64 " lf_race=%" PRIu64 " lf_fast=%" PRIu64 " lfw_hit=%" PRIu64
                        " lfw_miss=%" PRIu64 " lfw_diff=%" PRIu64 " lfw_race=%" PRIu64 "\n",
                        g_get_real_time() / 1000, d->steal_mode, d->gen_memo_mode,
                        d->lab_steals, d->lab_steal_same, d->lab_steal_late,
                        qatomic_xchg(&d->lab_memo_hit, 0),
                        qatomic_xchg(&d->lab_memo_fresh, 0),
                        qatomic_xchg(&d->lab_memo_diff, 0),
                        qatomic_xchg(&d->lab_memo_bad, 0), d->epoch_named,
                        qatomic_xchg(&d->lab_unnamed_skipped, 0), d->lf_mode,
                        qatomic_xchg(&d->lab_lf_hit, 0), qatomic_xchg(&d->lab_lf_hitdiff, 0),
                        qatomic_xchg(&d->lab_lf_gendiff, 0), qatomic_xchg(&d->lab_lf_race, 0),
                        qatomic_xchg(&d->lab_gen_fast_s, 0) * 8,
                        qatomic_xchg(&d->lab_lfw_hit, 0), qatomic_xchg(&d->lab_lfw_miss, 0),
                        qatomic_xchg(&d->lab_lfw_diff, 0), qatomic_xchg(&d->lab_lfw_race, 0));
                d->lab_steals = d->lab_steal_same = d->lab_steal_late = 0;
            }
            if (d->hole_mode) {   /* TEMP lab (DIRTY_HOLE) */
                unsigned ch;

                fprintf(stderr, "reims-dirty-hole epoch_ms=%" PRId64 " closed=%" PRIu64
                        " unnamed=%" PRIu64, g_get_real_time() / 1000,
                        d->lab_hole_closed, qatomic_xchg(&d->lab_unnamed, 0));
                d->lab_hole_closed = 0;
                for (ch = 0; ch < 8; ch++) {
                    fprintf(stderr, " db%u=%" PRIu64 " pk%u=%" PRIu64 " dr%u=%" PRIu64
                            " em%u=%" PRIu64 " hl%u=%" PRIu64, ch, qatomic_xchg(&d->lab_db[ch], 0),
                            ch, qatomic_xchg(&d->lab_pk[ch], 0),
                            ch, qatomic_xchg(&d->lab_dr[ch], 0),
                            ch, qatomic_xchg(&d->lab_em[ch], 0),
                            ch, qatomic_xchg(&d->lab_hole[ch], 0));
                }
                fputc(10, stderr);
            }
            if (d->lab_tr) {   /* TEMP lab (FGTRACE) */
                GHashTableIter lit;
                gpointer lk, lv;
                uint64_t setpg = 0;

                g_hash_table_iter_init(&lit, d->sets);
                while (g_hash_table_iter_next(&lit, &lk, &lv)) {
                    const ReimsVgpuDirtySet *ls = lv;

                    setpg += ls->count * (ls->page_size / 4096);
                }
                fprintf(stderr, "reims-fgdiag epoch_ms=%" PRId64 " first=%" PRIu64
                        " dup=%" PRIu64 " fast_n=%" PRIu64 " fast_us=%" PRIu64
                        " pref_used=%" PRIu64 " pref_wasted=%" PRIu64
                        " wasted_pg=%" PRIu64 " whv_us=%" PRIu64 " whv_calls=%" PRIu64
                        " fg_walked=%" PRIu64 " fg_split=%" PRIu64 " ws_n=%" PRIu64
                        " ws_sync=%" PRIu64 " ws_us=%" PRIu64 " ntrack=%" PRIu64
                        " nuntrack=%" PRIu64 " nsets=%u setpg=%" PRIu64 "\n",
                        g_get_real_time() / 1000,
                        qatomic_xchg(&d->lab_first_reads, 0),
                        qatomic_xchg(&d->lab_dup_reads, 0),
                        qatomic_xchg(&d->lab_fast_n, 0),
                        qatomic_xchg(&d->lab_fast_ns, 0) / 1000,
                        qatomic_xchg(&d->lab_pref_used, 0),
                        d->lab_pref_wasted, d->lab_pref_wasted_pages,
                        qatomic_xchg(&d->lab_whv_ns, 0) / 1000,
                        qatomic_xchg(&d->lab_whv_calls, 0),
                        qatomic_xchg(&d->lab_fg_walked, 0), d->lab_fg_split,
                        d->lab_ws_n, d->lab_ws_sync_n, d->lab_ws_ns / 1000,
                        d->lab_ntrack, d->lab_nuntrack,
                        g_hash_table_size(d->sets), setpg);
                d->lab_pref_wasted = d->lab_pref_wasted_pages = d->lab_fg_split = 0;
                d->lab_ws_n = d->lab_ws_sync_n = d->lab_ws_ns = 0;
                d->lab_ntrack = d->lab_nuntrack = 0;
                reims_vgpu_lab_flush(d, false);
            }
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
        reims_vgpu_dirty_sync_one(d, token, false, NULL, NULL);
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
        const char *ft = getenv("REIMS_VGPU_LAB_FGTRACE");         /* lab: trace file */

        if (ft && *ft && strcmp(ft, "off") != 0 && !d->lab_tr) {
            d->lab_tr_f = fopen(ft, "wb");
            if (d->lab_tr_f) {
                d->lab_tr = g_new0(ReimsVgpuLabRec, REIMS_VGPU_LAB_TR_CAP);
                d->lab_t0 = get_clock();
            }
        }
        qemu_mutex_lock(&d->lock);
        qatomic_set(&d->ondemand, on);
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
        d->coalesce = getenv("REIMS_VGPU_DIRTY_COALESCE") &&           /* lab A/B */
                      strcmp(getenv("REIMS_VGPU_DIRTY_COALESCE"), "on") == 0;
        d->pb_split = getenv("REIMS_VGPU_DIRTY_PB_SPLIT") &&           /* lab A/B */
                      strcmp(getenv("REIMS_VGPU_DIRTY_PB_SPLIT"), "on") == 0;
        d->gen_fast = !(getenv("REIMS_VGPU_DIRTY_GEN_FAST") &&        /* lab A/B */
                        strcmp(getenv("REIMS_VGPU_DIRTY_GEN_FAST"), "off") == 0);
        d->od_split = !(getenv("REIMS_VGPU_DIRTY_OD_SPLIT") &&        /* lab A/B */
                        strcmp(getenv("REIMS_VGPU_DIRTY_OD_SPLIT"), "off") == 0);
        d->work_epoch_off = getenv("REIMS_VGPU_DIRTY_WORK_EPOCH") &&   /* lab A/B */
                            strcmp(getenv("REIMS_VGPU_DIRTY_WORK_EPOCH"), "off") == 0;
        /*
         * On by default (2026-10-01): STEAL and GEN_LOCKFREE; `=off` restores the
         * earlier path. EPOCH_NAMED and DIRTY_HOLE stay opt-in: with them the 5K
         * light scene fell from 116.7 to 91-95 Hz (fzl0/fzl1/fzl2), the drain's
         * hole epochs turning every target currency check into a foreground sync
         * of a full-screen set; without them it ran 118.4 (fzl3). GEN_MEMO stays
         * opt-in (exact, no measurable effect).
         */
        d->steal_mode = getenv("REIMS_VGPU_LAB_DIRTY_STEAL")
                        ? reims_vgpu_dirty_lab_mode("REIMS_VGPU_LAB_DIRTY_STEAL") : 1;
        d->gen_memo_mode = reims_vgpu_dirty_lab_mode("REIMS_VGPU_LAB_GEN_MEMO");
        d->epoch_named = reims_vgpu_dirty_lab_mode("REIMS_VGPU_LAB_EPOCH_NAMED") == 1;
        d->lf_mode = getenv("REIMS_VGPU_LAB_GEN_LOCKFREE")
                     ? reims_vgpu_dirty_lab_mode("REIMS_VGPU_LAB_GEN_LOCKFREE") : 1;
        d->period_mode = getenv("REIMS_VGPU_LAB_DIRTY_PERIOD") &&     /* lab A/B */
                         strcmp(getenv("REIMS_VGPU_LAB_DIRTY_PERIOD"), "on") == 0;
        d->rsettle_mode = getenv("REIMS_VGPU_LAB_DIRTY_RSETTLE") &&     /* lab A/B */
                          strcmp(getenv("REIMS_VGPU_LAB_DIRTY_RSETTLE"), "on") == 0;
        /* On by default (2026-10-01, 5K light: drain busy 915/982 -> 786/801 ms/s). */
        d->bigphase_mode = !(getenv("REIMS_VGPU_LAB_DIRTY_BIGPHASE") &&   /* lab A/B */
                             strcmp(getenv("REIMS_VGPU_LAB_DIRTY_BIGPHASE"), "off") == 0);
        d->bigsplit_mode = getenv("REIMS_VGPU_LAB_DIRTY_BIGSPLIT") &&   /* lab A/B */
                           strcmp(getenv("REIMS_VGPU_LAB_DIRTY_BIGSPLIT"), "on") == 0;
        d->bigphase_big = getenv("REIMS_VGPU_LAB_DIRTY_BIGPHASE_BIG") ?
                          MAX(1, atoi(getenv("REIMS_VGPU_LAB_DIRTY_BIGPHASE_BIG"))) : 2048;
        d->period_big = getenv("REIMS_VGPU_LAB_DIRTY_PERIOD_BIG") ?
                        MAX(1, atoi(getenv("REIMS_VGPU_LAB_DIRTY_PERIOD_BIG"))) : 2048;
        d->frontier_mode = getenv("REIMS_VGPU_LAB_DIRTY_FRONTIER") &&    /* lab A/B */
                           strcmp(getenv("REIMS_VGPU_LAB_DIRTY_FRONTIER"), "on") == 0;
        d->tailwake_mode = getenv("REIMS_VGPU_LAB_DIRTY_TAILWAKE") &&    /* lab A/B */
                           strcmp(getenv("REIMS_VGPU_LAB_DIRTY_TAILWAKE"), "on") == 0;
        d->hole_mode = !getenv("REIMS_VGPU_LAB_DIRTY_HOLE") ? 0 :
                       strcmp(getenv("REIMS_VGPU_LAB_DIRTY_HOLE"), "census") == 0 ? 1 :
                       strcmp(getenv("REIMS_VGPU_LAB_DIRTY_HOLE"), "close") == 0 ? 2 :
                       strcmp(getenv("REIMS_VGPU_LAB_DIRTY_HOLE"), "tail") == 0 ? 3 : 0;
        if (d->hole_mode != 3) {
            d->frontier_mode = 0;   /* the cover of a read is only known under tail */
        }
        if (!d->wp_list) {   /* lab DIRTY_WP */
            const char *w = getenv("REIMS_VGPU_LAB_DIRTY_WP");

            /*
             * On by default (5K, VM B idle: drag 88.4/88.3 Hz against 63.2, light
             * drain busy 457 against ~780 ms/s, verification tour clean);
             * REIMS_VGPU_LAB_DIRTY_WP=off is the query-every-sync path.
             */
            d->wp_mode = !(w && strcmp(w, "off") == 0);
            d->wp_active = true;
            if (w && strncmp(w, "toggle:", 7) == 0) {
                d->wp_toggle = g_strdup(w + 7);
                d->wp_active = g_file_test(d->wp_toggle, G_FILE_TEST_EXISTS);
            }
            d->wp_streak_need = getenv("REIMS_VGPU_LAB_DIRTY_WP_STREAK") ?
                                MAX(1, atoi(getenv("REIMS_VGPU_LAB_DIRTY_WP_STREAK"))) : 16;
            d->wp_min_pages = getenv("REIMS_VGPU_LAB_DIRTY_WP_MIN") ?
                              MAX(1, atoi(getenv("REIMS_VGPU_LAB_DIRTY_WP_MIN"))) : 2048;
            d->wp_chunk = getenv("REIMS_VGPU_LAB_DIRTY_WP_CHUNK") ?
                          MAX(1, atoi(getenv("REIMS_VGPU_LAB_DIRTY_WP_CHUNK"))) : 24;
            d->wp_duty = getenv("REIMS_VGPU_LAB_DIRTY_WP_DUTY") ?
                         MIN(100, MAX(1, atoi(getenv("REIMS_VGPU_LAB_DIRTY_WP_DUTY")))) : 10;
            d->wp_pace_ms = getenv("REIMS_VGPU_LAB_DIRTY_WP_PACE_MS") ?
                            MAX(1, atoi(getenv("REIMS_VGPU_LAB_DIRTY_WP_PACE_MS"))) : 4;
            d->wp_nopause = getenv("REIMS_VGPU_LAB_DIRTY_WP_NOPAUSE") &&
                            strcmp(getenv("REIMS_VGPU_LAB_DIRTY_WP_NOPAUSE"), "on") == 0;
            qemu_mutex_init(&d->wp_mutex);
            d->wp_queue = g_array_new(FALSE, FALSE, sizeof(uint64_t));
            d->wp_list = g_array_new(FALSE, FALSE, sizeof(uint64_t));
            d->wp_orphans = g_array_new(FALSE, FALSE, sizeof(ReimsVgpuDirtyRun));
            if (d->wp_mode) {
#ifndef REIMS_DIRTY_STRESS
                d->wp_timer = timer_new_ms(QEMU_CLOCK_REALTIME, reims_vgpu_dirty_wp_bh, d);
#endif
                d->wp_bh = REIMS_WP_BH_NEW(reims_vgpu_dirty_wp_bh, d);
                reims_vgpu_dirty_wp_dev = d;
                whpx_wp_invalidate_hook = reims_vgpu_dirty_wp_invalidate;
                whpx_wp_fault_hook = reims_vgpu_dirty_wp_fault;
            }
        }
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
    if (d->hole_mode) {   /* TEMP lab (DIRTY_HOLE) */
        if (channel >= 0 && channel < (int)ARRAY_SIZE(d->lab_db)) {
            qatomic_inc(&d->lab_db[channel]);
            qatomic_inc(&d->lab_dbp[channel]);
        } else {
            qatomic_inc(&d->lab_unnamed);
        }
    }
    if (channel < 0 && d->epoch_named && d->ondemand) {
        /* TEMP lab EPOCH_NAMED: no doorbell, no epoch, and nothing to prefetch. */
        qatomic_inc(&d->lab_unnamed_skipped);
        return qatomic_read(&d->log_wanted);
    }
    qemu_mutex_lock(&d->lock);
    if (d->coalesce && d->ondemand && d->epoch != 0 && !d->epoch_queried) {
        d->lab_db_coalesced++;
    } else {
        qatomic_set(&d->epoch, d->epoch + 1);
        d->epoch_queried = false;
        if (d->lab_tr) {   /* TEMP lab (FGTRACE) */
            int64_t now = get_clock();
            ReimsVgpuLabRec r = { .t_ns = now - d->lab_t0, .needed = d->epoch,
                                  .epoch = d->epoch, .kind = 7, .aux = (uint32_t)channel };

            qatomic_set(&d->lab_epoch_ns[d->epoch & 4095], now);
            reims_vgpu_lab_rec(d, &r);
        }
    }
    if (channel >= 0 && channel < (int)ARRAY_SIZE(d->chan_epoch)) {
        qatomic_set(&d->chan_epoch[channel], d->epoch);
    } else {
        /* Not a doorbell the device names: binds every channel. */
        for (c = 0; c < ARRAY_SIZE(d->chan_epoch); c++) {
            qatomic_set(&d->chan_epoch[c], d->epoch);
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

/*
 * Prefetch batch: bring every hot set up to date for `epoch` under one query
 * ticket. The sets are claimed first, so a drain that needs one waits for the
 * batch (and queries its pieces meanwhile) instead of syncing it alone; their
 * pages are walked once as a union, pages another sync walked this epoch are
 * skipped, and the walk is split across the pool. One ticket in flight instead
 * of one per worker is what the ticket settle waits on.
 */
/* Add one set's target page frames to the batch union. d->lock held. */
static void reims_vgpu_dirty_pb_mark(ReimsVgpuDirty *d, const ReimsVgpuDirtySet *s,
                                     int shift)
{
    const uint64_t per = s->page_size >> shift;
    uint64_t lo, hi;
    size_t p;

    if (s->count == 0) {
        return;
    }
    lo = s->pages[0] >> shift;
    hi = (s->pages[s->count - 1] + s->page_size) >> shift;
    if (hi > d->pb_nbits) {
        uint64_t nbits = MAX(hi, d->pb_nbits * 2);

        d->pb_bits = bitmap_zero_extend(d->pb_bits, d->pb_nbits, nbits);
        d->pb_nbits = nbits;
    }
    for (p = 0; p < s->count; p++) {
        uint64_t pfn = s->pages[p] >> shift;

        if (per == 1) {
            set_bit(pfn, d->pb_bits);
        } else {
            bitmap_set(d->pb_bits, pfn, per);
        }
    }
    if (d->pb_hi == 0) {
        d->pb_lo = lo;
    }
    d->pb_lo = MIN(d->pb_lo, lo);
    d->pb_hi = MAX(d->pb_hi, hi);
}

/*
 * Read the batch union back as maximal runs in GPA order, clearing it. Pages
 * two sets share appear once, where the sorted per-set runs held them twice.
 */
static void reims_vgpu_dirty_pb_runs(ReimsVgpuDirty *d, int shift, GArray *runs)
{
    unsigned long bit, end;

    if (d->pb_hi == 0) {
        return;
    }
    bit = find_next_bit(d->pb_bits, d->pb_hi, d->pb_lo);
    while (bit < d->pb_hi) {
        MemoryRegionRange run;

        end = find_next_zero_bit(d->pb_bits, d->pb_hi, bit);
        run.start = (uint64_t)bit << shift;
        run.len = (uint64_t)(end - bit) << shift;
        g_array_append_val(runs, run);
        bitmap_clear(d->pb_bits, bit, end - bit);
        bit = find_next_bit(d->pb_bits, d->pb_hi, end);
    }
    d->pb_lo = d->pb_hi = 0;
}

/*
 * One prefetch pass over `tokens`. `preclaimed` is the second phase of a split
 * batch: its sets were claimed for `epoch` up front, so it runs even if a newer
 * doorbell has arrived (a claim nobody completes would park a waiting drain),
 * and it takes the sets it pre-claimed as its own.
 */
static void reims_vgpu_dirty_prefetch_phase(ReimsVgpuDirty *d, GArray *tokens,
                                            uint64_t epoch, bool preclaimed,
                                            bool no_steal)
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
    if (!preclaimed && d->epoch != epoch) {
        qemu_mutex_unlock(&d->lock);
        return;
    }
    if (!preclaimed) {
        d->lab_pb_lag += pb0 - d->doorbell_ns;
        d->lab_pb_epoch = epoch;
    }
    /*
     * The union of the claimed sets' pages (GPA). Pages another sync walked
     * this epoch are walked again here: a second query of a page is harmless,
     * and filtering page by page cost more than the query it saved.
     */
    for (t = 0; t < tokens->len; t++) {
        uint64_t tk = g_array_index(tokens, uint64_t, t);
        ReimsVgpuDirtySet *set = g_hash_table_lookup(d->sets, &tk);

        if (!set || set->synced_epoch >= epoch) {
            continue;
        }
        if (set->claim_epoch >= epoch && !(preclaimed && set->pb_claim == epoch)) {
            continue;
        }
        set->claim_epoch = epoch;
        set->bclaim = no_steal ? 0 : epoch;   /* lab BIGPHASE: a big set's own phase is waited for */
        set->pb_claim = 0;
        d->epoch_queried = true;
        g_array_append_val(claimed, tk);
        setpages += set->count * (set->page_size / target);
        reims_vgpu_dirty_pb_mark(d, set, shift);
    }
    qemu_mutex_unlock(&d->lock);
    reims_vgpu_dirty_pb_runs(d, shift, pages);
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
            {
                /* Set pages this batch queries: the ones it may vouch for. */
                MemoryRegionRange w = { gpa, send - gpa };

                g_array_append_val(walked, w);
            }
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

    pb1 = get_clock();
    if (d->rsettle_mode) {
        g_array_sort(chunks, reims_vgpu_dirty_cmp_chunk);
    }
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
        reims_vgpu_dirty_query_footprint(d, qticket,
                                         (const ReimsVgpuDirtyChunk *)chunks->data,
                                         chunks->len);
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
    reims_vgpu_dirty_query_settle_fp(d, qticket, (const ReimsVgpuDirtyChunk *)chunks->data,
                                     chunks->len);
    pb3 = get_clock();
    /*
     * Stamp the set pages as queried for this epoch. The gap pages between
     * them were queried too, but no set of this batch names them, and leaving
     * them unstamped only lets a later sync of some other set query them
     * again — the harmless direction. Stamping them was three quarters of
     * this loop, run under `lock`.
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
            qatomic_store_release(&set->synced_epoch, MAX(set->synced_epoch, epoch));
            if (set->claim_epoch == epoch) {
                set->claim_epoch = 0;
            }
            if (set->bclaim == epoch) {
                set->bclaim = 0;
            }
            if (d->lab_tr) {   /* TEMP lab (FGTRACE): was the last prefetch ever used? */
                if (qatomic_read(&set->lab_pref)) {
                    d->lab_pref_wasted++;
                    d->lab_pref_wasted_pages += set->count * (set->page_size / 4096);
                }
                qatomic_set(&set->lab_pref, 1);
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
    if (d->lab_tr) {   /* TEMP lab (FGTRACE) */
        int64_t db = d->lab_epoch_ns[epoch & 4095];
        ReimsVgpuLabRec r = { .t_ns = pb6 - d->lab_t0, .needed = epoch, .epoch = d->epoch,
                              .since_db_ns = db ? pb0 - db : -1, .pages = (uint32_t)npages,
                              .kind = 6, .cls = preclaimed,
                              .total_ns = (uint32_t)(pb6 - pb0),
                              .wait_ns = (uint32_t)(pb1 - pb0),
                              .query_ns = (uint32_t)(pb2 - pb1),
                              .aux = claimed->len };

        reims_vgpu_lab_rec(d, &r);
    }
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

static void reims_vgpu_dirty_prefetch_batch(ReimsVgpuDirty *d, GArray *tokens,
                                            uint64_t epoch)
{
    g_autoptr(GArray) head = NULL;
    g_autoptr(GArray) tail = NULL;
    guint first, t;

    first = d->pb_split ? MAX(4u, tokens->len / 3) : tokens->len;
    if (first >= tokens->len) {
        reims_vgpu_dirty_prefetch_phase(d, tokens, epoch, false, false);
        return;
    }
    head = g_array_new(FALSE, FALSE, sizeof(uint64_t));
    tail = g_array_new(FALSE, FALSE, sizeof(uint64_t));
    g_array_append_vals(head, tokens->data, first);
    qemu_mutex_lock(&d->lock);
    if (d->epoch != epoch) {
        qemu_mutex_unlock(&d->lock);
        return;
    }
    for (t = first; t < tokens->len; t++) {
        uint64_t tk = g_array_index(tokens, uint64_t, t);
        ReimsVgpuDirtySet *set = g_hash_table_lookup(d->sets, &tk);

        if (!set || set->synced_epoch >= epoch || set->claim_epoch >= epoch) {
            continue;
        }
        set->claim_epoch = epoch;
        set->bclaim = epoch;
        set->pb_claim = epoch;
        d->epoch_queried = true;
        g_array_append_val(tail, tk);
    }
    qemu_mutex_unlock(&d->lock);
    reims_vgpu_dirty_prefetch_phase(d, head, epoch, false, false);
    if (tail->len) {
        reims_vgpu_dirty_prefetch_phase(d, tail, epoch, true, false);
    }
}

/* Lab BIGPHASE: the batch as read-ordered phases (see `bigphase_mode`). */
static void reims_vgpu_dirty_prefetch_phased(ReimsVgpuDirty *d, GArray *tokens,
                                             uint64_t epoch)
{
    g_autoptr(GArray) group = g_array_new(FALSE, FALSE, sizeof(uint64_t));
    g_autoptr(GArray) one = g_array_new(FALSE, FALSE, sizeof(uint64_t));
    g_autofree bool *big = g_new0(bool, MAX(tokens->len, 1));
    guint t;

    qemu_mutex_lock(&d->lock);
    for (t = 0; t < tokens->len; t++) {
        uint64_t tk = g_array_index(tokens, uint64_t, t);
        ReimsVgpuDirtySet *set = g_hash_table_lookup(d->sets, &tk);

        big[t] = set && set->count * (set->page_size >> qemu_target_page_bits()) >=
                        d->bigphase_big;
    }
    qemu_mutex_unlock(&d->lock);
    for (t = 0; t < tokens->len; t++) {
        uint64_t tk = g_array_index(tokens, uint64_t, t);

        if (qatomic_read(&d->epoch) != epoch) {
            return;   /* a newer doorbell's prefetch takes over */
        }
        if (!big[t]) {
            g_array_append_val(group, tk);
            continue;
        }
        if (group->len) {
            reims_vgpu_dirty_prefetch_phase(d, group, epoch, false, false);
            g_array_set_size(group, 0);
            d->lab_bigphase_groups++;
        }
        g_array_set_size(one, 0);
        g_array_append_val(one, tk);
        reims_vgpu_dirty_prefetch_phase(d, one, epoch, false, true);
        d->lab_bigphase_big++;
    }
    if (group->len && qatomic_read(&d->epoch) == epoch) {
        reims_vgpu_dirty_prefetch_phase(d, group, epoch, false, false);
        d->lab_bigphase_groups++;
    }
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
    reims_vgpu_lab_flush(d, false);   /* TEMP lab (FGTRACE) */
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

            if (d->period_mode &&
                set->count * (set->page_size >> qemu_target_page_bits()) >= d->period_big) {
                if (!reims_vgpu_dirty_period_due(set, epoch)) {
                    d->lab_period_skipped++;
                    continue;
                }
                d->lab_period_taken++;
            }
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
    if (d->od_pbatch && d->bigsplit_mode && d->od_nworkers > 0) {
        g_autoptr(GArray) small = g_array_new(FALSE, FALSE, sizeof(uint64_t));
        guint nbig = 0;

        g_array_set_size(d->od_work, 0);
        for (t = 0; t < tokens->len; t++) {
            uint64_t tk = g_array_index(tokens, uint64_t, t);
            ReimsVgpuDirtySet *set = g_hash_table_lookup(d->sets, &tk);

            if (set && set->count * (set->page_size >> qemu_target_page_bits()) >=
                d->period_big) {
                g_array_append_val(d->od_work, tk);
                nbig++;
            } else {
                g_array_append_val(small, tk);
            }
        }
        if (nbig) {
            d->od_work_next = 0;
            d->od_work_epoch = epoch;
            d->lab_bigsplit_sets += nbig;
            qemu_cond_broadcast(&d->od_work_cond);
        }
        qemu_mutex_unlock(&d->lock);
        if (small->len) {
            reims_vgpu_dirty_prefetch_batch(d, small, epoch);
        }
        return;
    }
    if (d->od_pbatch && d->bigphase_mode) {
        qemu_mutex_unlock(&d->lock);
        reims_vgpu_dirty_prefetch_phased(d, tokens, epoch);
        return;
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
