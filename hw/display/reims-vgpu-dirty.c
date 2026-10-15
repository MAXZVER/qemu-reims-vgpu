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
#include <immintrin.h>   /* TEMP lab: the AVX2 page hash */

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

/* TEMP lab DIRTY_PSTAMP: a batch piece's set pages waiting to be stamped (see `pstamp`). */
typedef struct ReimsVgpuDirtyPstamp {
    uint64_t upto;        /* every ticket up to this one must have ended */
    uint64_t gpa, end;    /* the piece's GPA span ... */
    guint wa, wb;         /* ... and the batch's walked runs [wa, wb) that meet it */
    uint64_t epoch;
} ReimsVgpuDirtyPstamp;

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
     * Epoch a thread's own sync of this set (reims_vgpu_dirty_sync_one) is
     * running for, or 0; set with `claim_epoch` and cleared by that sync when
     * it publishes. A batch's claim leaves it alone. It is what tells a
     * foreground read that finds the set claimed by a batch whether a thread
     * already bypassed that claim (lab STEAL) for an epoch the read can use:
     * then the read waits for that thread (and helps its split pieces) instead
     * of querying the same pages a third time. Without it every concurrent
     * foreground read of a batch-claimed set stole and queried on its own.
     */
    uint64_t tclaim;
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
     * TEMP lab DIRTY_CHAN_PREFETCH: channels whose work read this set within `od_hot` epochs of
     * `rd_chans_epoch` (bit 63: a read under the All scope, which any doorbell makes stale).
     */
    uint64_t rd_chans;
    uint64_t rd_chans_epoch;
    /*
     * Lab DIRTY_WP: this set's token (the protect queue and the fault hook name
     * sets by it), whether its pages are write-protected now, how many syncs
     * in a row found it clean, and how often a protection was lost (each loss
     * doubles the clean run the next protection needs).
     */
    uint64_t token;
    uint8_t wp_state;       /* WP_NONE / WP_PROTECTING / WP_ON / WP_DROPPING / WP_HOLED; `lock` */
    size_t wp_next;         /* protecting: next page to protect; dropping: next page to look at */
    uint32_t wp_streak;
    uint32_t wp_lost;
    uint32_t wp_holes;      /* TEMP lab DIRTY_WP_HOLE: pages faults opened since the set was last whole */
    /*
     * Lab DIRTY_WP_CALM: when the set was tracked and how many runs it spans (a protection costs a
     * remap per run, and as many again when it is given back), and whether its token is on
     * `wp_list` — written under `lock` as well as `wp_mutex`, so an untrack that reads it under
     * `lock` knows whether the WP step may still be holding the set.
     */
    int64_t born_ns;
    uint32_t nruns;
    bool wp_listed;
    bool wp_counted;        /* TEMP lab DIRTY_WP_OVERLAP: its pages are counted in `wp_refs`; `wp_mutex` */
    uint32_t lab_fg_n;      /* TEMP lab: foreground syncs since the last fgtop line */
    /*
     * Lab DIRTY_HASH: a set scattered over many runs of guest RAM is brought up to date by
     * hashing each page's content and comparing with the hash its last sync took, instead of
     * hypervisor queries: a query costs a call per run plus every page of the gaps it bridges, a
     * hash costs the set's own bytes. fgt-glass (n166): 98k of 133k foreground syncs were
     * 129-256-page sets at ~736 us each (2.9 us per page), 2.6 % of them written; a hash of 1 MiB
     * is ~60-100 us and needs no ticket and no settle. NULL: a set the hypervisor answers for.
     *
     * Hashes see content, not stores, so a store of the bytes a page already held is not a write
     * here; nothing downstream can tell the two apart, since what a reader re-gathers would be the
     * same bytes. `hash` is touched only by the sync holding `hash_busy` (under `lock`), with
     * `hash_base` set once it holds a baseline.
     */
    uint64_t *hash;
    bool hash_base;
    bool hash_busy;
    /*
     * TEMP lab DIRTY_VERIFY (live, default on): a set the hypervisor answers for keeps its pages'
     * content hashes too, so a page the bitmap calls written whose bytes did not move is not a
     * write - the contract `hash` already gives the sets it is kept for. p080: a 5K framebuffer the
     * device renders only in its resident (its guest pages never painted) read "all 10800 pages
     * written" when the guest stored back the bytes those pages already held, and the reader laid
     * the unpainted pages over the resident - the boot wallpaper stripes of p79c/p80a. Baseline at
     * track, refreshed for the pages an unarmed evaluation names, re-taken by rebase for the
     * device's own stores. Read and written under `lock` once the set is published.
     */
    uint64_t *vhash;
    bool vhash_base;
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
 * TEMP lab DIRTY_FGSLOTS: one foreground split job. Its owner publishes the
 * pieces, helpers take them in turn (`next`), and the owner resets the job once
 * every piece is in (`done == total`); a piece taken holds the job until it is
 * counted done, so a helper's `done` pointer stays valid. Under `lock`.
 */
typedef struct ReimsVgpuDirtySplitJob {
    GArray *chunks;           /* ReimsVgpuDirtyChunk */
    guint next, done, total;
    bool busy;                /* published and its owner not yet done */
} ReimsVgpuDirtySplitJob;
#define REIMS_VGPU_DIRTY_FGSLOTS_MAX 8
#define REIMS_VGPU_DIRTY_HJOBS 4   /* TEMP lab: concurrent split hash jobs */

/*
 * The target page frames' query epochs (see `qep`), published through RCU so
 * that a sync's prep can read them without `lock`. Written under `lock` only,
 * with atomic stores; growing copies into a new array, publishes it and
 * retires the old one through RCU.
 */
typedef struct ReimsVgpuDirtyQep {
    struct rcu_head rcu;      /* first: call_rcu needs offset 0 */
    uint64_t n;
    uint64_t e[];
} ReimsVgpuDirtyQep;

/* TEMP lab DIRTY_TIGHT: tracked sets per target page frame (see `trk`), saturating. */
typedef struct ReimsVgpuDirtyTrk {
    struct rcu_head rcu;      /* first: call_rcu needs offset 0 */
    uint64_t n;
    uint16_t c[];
} ReimsVgpuDirtyTrk;

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
     * and length as `page_seq` (NULL until the first stamp).
     *
     * RCU (an RCU pointer; written under `lock`): a sync's prep reads it
     * without `lock` (lab DIRTY_PREPLOCK). A reader holding an array that a
     * growth has since replaced sees older epochs than the current ones, which
     * only makes it query a page it could have skipped.
     */
    ReimsVgpuDirtyQep *qep;
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
    /*
     * Lab ARM_FLUSH (on unless REIMS_VGPU_LAB_ARM_FLUSH=off): flush the per-vCPU
     * dirty logs before the sync that arms a new set, and query all its pages.
     * The guest zero-fills a surface's pages just before handing them over; a
     * store still in a running vCPU's log is invisible to the arming query and
     * surfaced one generation later as a write after the device's first Store,
     * which overlaid whole 16 KiB granules of zeros onto WebKit tiles.
     */
    bool arm_flush;
    bool pb_arm_flush;   /* TEMP lab (REIMS_VGPU_LAB_PB_ARM_FLUSH=off): the batch's half */
    uint64_t lab_arm_flushes;
    uint64_t lab_pb_arm_flushes;   /* TEMP lab: arming flushes of the batch (qatomic) */
    int64_t lab_arm_flush_ns;
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
    /*
     * TEMP lab SYNC_AHEAD (reims_vgpu_dirty_sync_ahead): tokens the device named as about to be
     * read — a pass's later chunks — for the workers to sync now, ahead of the prefetch list.
     * Under `lock`; reset to empty when its last token is taken.
     */
    GArray *ahead;
    guint ahead_next;
    uint64_t lab_ahead_req, lab_ahead_q, lab_ahead_done, lab_ahead_hit, lab_ahead_busy,
             lab_ahead_full;
    uint64_t read_counter;    /* first reads of a set in an epoch, in order */
    /* Generation reads may skip `lock` for synced sets (lab A/B
     * REIMS_VGPU_DIRTY_GEN_FAST=off restores the locked read). */
    bool gen_fast;
    /*
     * TEMP (lab, diagnostic only) REIMS_VGPU_LAB_DIRTY_NOSYNC=on: every generation read takes the lagged
     * answer (what the prefetch workers last synced) instead of a foreground sync. Not correct — a store's
     * witness may miss guest writes — it measures how much of a frame the foreground syncs cost.
     */
    bool lab_nosync;
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
    /*
     * TEMP lab REIMS_VGPU_LAB_DIRTY_CHAN_PREFETCH=on: with reads scoped to their packet's channel
     * (the Rust side's PACKET_SCOPE), a doorbell of channel c makes stale only the sets c's work
     * reads, so a prefetch batch takes only those (and every set an All-scope read touched). Under
     * the All scope every read is bit 63 and the batch is what it was.
     */
    bool chan_prefetch;
    /*
     * Lab REIMS_VGPU_LAB_DIRTY_HASH=on (see `hash` in ReimsVgpuDirtySet): which sets are hashed is
     * decided at track by a cost model — the hypervisor's (calls x 5 us + walked pages x 26 ns,
     * the gap bridge included) against `hash_ns` per own page — for sets of at most `hash_max`
     * target pages (REIMS_VGPU_LAB_DIRTY_HASH_NS, _MAX; _ALL=on hashes every set up to the max,
     * for the stress harness). A hashed set never enters a prefetch batch, a harvest or a merged
     * sync: its generation moves only on what its own hashes show.
     */
    bool hash_mode;
    bool hash_all;
    uint32_t hash_ns;
    uint64_t hash_max;
    uint64_t lab_hash_sets, lab_hash_syncs, lab_hash_pages, lab_hash_ns, lab_hash_dirty;
    uint64_t lab_hash_waits, lab_hash_noram, lab_hash_fg, lab_hash_fg_ns, lab_hash_pf;
    /* reims_vgpu_dirty_rebase: calls, pages re-baselined, calls that found no hashed baseline */
    uint64_t lab_rebase_calls, lab_rebase_pages, lab_rebase_unhashed;
    /* TEMP lab DIRTY_VERIFY: sets given a baseline, its cost, and the pages it judged. */
    uint64_t lab_vhash_sets, lab_vhash_base_ns, lab_vhash_same, lab_vhash_moved, lab_vhash_rebase;
    int64_t lab_hash_last;
    uint32_t work_chan;             /* channel of the drain's work scope; 63 = All */
    uint64_t db_chans;              /* channels rung since the last batch took them; bit 63 = unnamed */
    uint64_t lab_cpf_taken, lab_cpf_skipped;
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
    bool hot_new;             /* lab REIMS_VGPU_LAB_DIRTY_HOT_NEW: new sets start hot */
    bool od_splitwait;        /* lab REIMS_VGPU_DIRTY_OD_SPLITWAIT=on */
    bool od_nopticket;        /* lab REIMS_VGPU_DIRTY_OD_PTICKET=off */
    GArray *split_chunks;     /* ReimsVgpuDirtyChunk */
    guint split_next, split_done, split_total;
    /*
     * TEMP lab REIMS_VGPU_LAB_DIRTY_PSTAMP=on: a prefetch batch's pieces stamp the batch's set pages
     * they walked as queried for the batch's epoch as soon as every ticket begun before the piece
     * ended — not all at once after the whole union. A reader that bypasses the batch mid-way
     * (STEAL) then queries only what no finished piece covered, and settles behind nobody when
     * that is nothing (SETTLE_SKIP). The stamp keeps the settle's invariant: written only once
     * every query that could have taken one of those pages' bits before the piece has deposited.
     * `pstamps` is a FIFO (append order = `upto` order); `query_end` pops what became safe.
     */
    bool pstamp;
    bool split_pstamp;                      /* the running split job is a batch's */
    const struct ReimsVgpuDirtySlice *split_slices;
    int split_nslices;
    const GArray *split_walked;             /* MemoryRegionRange, GPA order */
    uint64_t split_epoch;
    GArray *pstamps;                        /* ReimsVgpuDirtyPstamp */
    guint pstamps_head;
    uint64_t lab_pstamps, lab_pstamp_pages;
    bool split_busy;          /* a job is published and its owner not yet done */
    QemuCond split_cond;
    /*
     * Lab DIRTY_HASH: one set's hash cut into page chunks that the prefetch workers and any waiter
     * take in turn (`next`), the owner (the sync holding the set's `hash_busy`) taking its share and
     * waiting for `done == total` on `split_cond`. Fields are written by the owner before `busy` and
     * not again until every chunk is done; a chunk writes only its own pages' `hash` and `moved`.
     * Under `lock`. TEMP lab: REIMS_VGPU_DIRTY_HJOBS slots (was one), so a second large set's hash
     * is split across the workers too instead of running alone on its reader.
     */
    struct {
        ReimsVgpuDirtySet *s;
        const ReimsVgpuDirtySlice *slices;
        int n;
        uint8_t *moved;       /* per page: 1 = its content moved, or it is outside guest RAM */
        bool base;
        size_t chunk;
        guint next, done, total;
        bool busy;
        /*
         * TEMP lab REBASE_SPLIT: non-NULL for a rebase job - the set's page indices `idx[0..nidx)` are hashed into
         * `out` (the set's `hash` or `vhash`), with no comparison and no `moved`.
         */
        const size_t *idx;
        size_t nidx;
        uint64_t *out;
    } hjob[REIMS_VGPU_DIRTY_HJOBS];
    uint64_t lab_hash_split, lab_hash_split_helped;
    uint64_t lab_rebase_split;   /* TEMP lab REBASE_SPLIT: rebases hashed on the workers */
    /*
     * TEMP lab DIRTY_FGSLOT (on unless =off): a large foreground sync gets a
     * split job of its own, which every helper serves before the batch's.
     * With one job a prefetch batch held it through most of a transition, and
     * a 5K set the drain needed then queried alone (~0.6 ms instead of ~0.1).
     *
     * TEMP lab DIRTY_FGSLOTS=N (1..8, default 4): N such jobs, so foreground
     * syncs on several threads at once (the drain, and the device's own worker
     * threads through reims_vgpu_dirty_gen) each split instead of all but one
     * querying alone. A sync takes a free job; helpers take one piece at a time
     * from the open jobs in turn (`fjob_rr`), and the batch's only when none is
     * open; an owner takes only its own job's pieces. `fgslots` = 0 is
     * DIRTY_FGSLOT=off. `fjob_open` has bit k set while job k has pieces left.
     */
    uint32_t fgslots;
    ReimsVgpuDirtySplitJob fjob[REIMS_VGPU_DIRTY_FGSLOTS_MAX];
    uint32_t fjob_open;
    uint32_t fjob_rr;
    uint32_t fjob_busy;       /* jobs published and not yet reset */
    uint64_t lab_fg_split_slot, lab_fg_split_shared, lab_fg_big_alone;
    uint64_t lab_fg_slots_max, lab_fg_slots_multi, lab_fg_noslot;   /* TEMP lab (FGSLOTS) */
    /*
     * TEMP lab: `lock` in reims_vgpu_dirty_sync_one, per sync that got past the
     * claim (lk_n): time spent acquiring it (lk_wait), held for the prep from
     * the claim to the walk (lk_prep, of which lk_list builds the page list),
     * for the ticket and the split publication (lk_split), and after the settle
     * for the stamps, consume and evaluation (lk_post).
     */
    uint64_t lab_hold_n;
    int64_t lab_hold_prep, lab_hold_list, lab_hold_split, lab_hold_post, lab_lock_wait;
    uint64_t lab_holdb_n;     /* the same for sets of >= 4096 target pages (lkb_*) */
    int64_t lab_holdb_prep, lab_holdb_list, lab_holdb_post;
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
    /*
     * TEMP lab DIRTY_SETTLE_SKIP (on; =off restores): a sync with no page left to query takes no
     * ticket and settles behind nobody. Every page it skips was stamped for this epoch or later
     * by a query that began after this epoch's doorbell, and every stamp is written after that
     * query's own settle, so whatever bits any query took for those pages before then are in the
     * VGA bitmap already; the consume pass reads them there. A read-only page (RO_SKIP) is not
     * such a page while a protect run's ticket is open (its marks may not have landed yet), so a
     * list with one skips only while `wp_tickets_open` is 0: every read-only page then had its
     * protect's ticket end, after the marks of its run, before this sync decided.
     */
    uint32_t settle_skip;
    uint32_t wp_tickets_open;   /* protect-run query tickets held now; under `lock` */
    /*
     * TEMP lab REIMS_VGPU_LAB_DIRTY_TICKET_BATCH=<ranges> (0 = off): a sync that queries alone takes
     * a ticket per that many ranges instead of one across all of them. Every settle that began
     * meanwhile waited for the whole sequence (a scattered 5K set is hundreds of calls, ~0.5-2 ms);
     * now it waits for the batch in flight. The sync's own settle (no ticket) still waits for every
     * query begun before it, as a split job's owner does.
     */
    uint32_t ticket_batch;
    uint64_t lab_tb_syncs, lab_tb_batches;
    /*
     * TEMP lab REIMS_VGPU_LAB_DIRTY_FG_SPLIT_MIN=<pages> (default 4096): the walk past which a
     * foreground sync splits its queries across the helpers. FGTRACE (n166): the scroll scene's
     * costliest syncs are 504-page sets walking ~1-3k pages in ~60 calls, ~0.5 ms of query alone.
     */
    uint64_t fg_split_min;
    uint64_t lab_settle_skips, lab_settle_skip_cand;
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
     * TEMP lab DIRTY_FGJOIN (on unless =off): a read waits for a sync already
     * running for the epoch it needs (not only for the newest epoch), and a
     * foreground read bypasses a batch's claim only if no thread already did
     * for an epoch it can use (`tclaim`). Off: the earlier rule, where only a
     * claim for the newest epoch was waited for and every foreground read of a
     * batch-claimed set queried it on its own. `lab_dup_inflight` counts syncs
     * begun while a thread's sync of the same set for an epoch the read could
     * use was running (0 by construction when on), `lab_ov_older` those begun
     * beside a thread's sync for an older epoch, which the read cannot use.
     */
    uint32_t fgjoin_mode;
    uint64_t lab_dup_inflight, lab_ov_older;
    /*
     * Lab DIRTY_PREPLOCK (on unless =off): a sync's prep holds `lock` only to
     * claim the set. The page list (the set's pages less those already queried
     * this epoch) was ~97 % of the prep's hold (stress, sets of 4-8k pages:
     * ~9 us of ~9.5) and is built after it, under RCU: the set's `pages` are
     * immutable and a set is freed through RCU, and `qep` is RCU-published. A
     * stale epoch read there only queries a page again (harmless); an epoch
     * that does let it skip a page was stamped under `lock` after that query's
     * bits were in the VGA bitmap, and this sync consumes under `lock` later.
     * The query ticket is taken where the queries are decided (as DIRTY_RSETTLE
     * did), still before any of them, so the lock is taken three times, not
     * four. Off: the list under `lock` and the ticket before the walk.
     */
    uint32_t prep_unlocked;
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
    uint32_t wp_chunk;            /* runs per step (DIRTY_WP_CHUNK, 4) */
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
    int64_t wp_next_ok_ns;        /* no protect pause before this (duty gate) */
    QEMUTimer *wp_timer;
    QEMUBH *wp_bh;
    /*
     * TEMP lab REIMS_VGPU_LAB_DIRTY_WP_THREAD=on: the protect step runs on a thread of its own, not in
     * the main loop, and always without pausing the vCPUs (DIRTY_WP_NOPAUSE). The main-loop BH and
     * the pacing timer only wake it. With the vCPUs paused a 2048-page set took seconds to protect
     * at the 10 % pause duty (one set at a time, 4 runs a pause, ~0.3 ms a run) and a page-scattered
     * hot set was never reached (n149: worker reads of clean mapper sets 685 ms/s); NOPAUSE in the
     * main loop instead held the loop that delivers the device's interrupts (n80: 36.7 s a boot).
     * `wp_thread_duty` caps the share of time the thread spends in a step (the remaps share the
     * hypervisor with every dirty query).
     */
    bool wp_thread_mode;
    /*
     * The last step found read-only runs still to give back (a dropping set, orphans): the thread
     * then takes the next step at once instead of resting by the duty, because every vCPU store
     * into one of those pages is an exit and a remap of its own until the step gets there.
     */
    bool wp_giving_back;
    bool wp_run_ticket;   /* REIMS_VGPU_LAB_DIRTY_WP_RUN_TICKET (on unless =off): see protect_chunk */
    uint32_t wp_thread_duty;
    QemuThread wp_thread;
    QemuEvent wp_ev;
    unsigned long *wp_bits;       /* pfns of protected sets (they never overlap); under `wp_mutex` */
    /*
     * TEMP lab DIRTY_WP_HOLE: pfns a fault opened in a set that stays protected. Still that set's:
     * no other set may be protected over one (a release of either would then leave the other's page
     * writable while it reads as protected), so a pick refuses a set holding one, as it refuses a
     * set holding a read-only page. Under `wp_mutex`.
     */
    unsigned long *wp_hole_bits;
    uint64_t wp_bits_n;
    /*
     * TEMP lab DIRTY_WP_OVERLAP: per pfn, how many protected (or being-protected) sets hold the page;
     * a release leaves a page with a non-zero count read-only. Allocated with `wp_bits` when on.
     */
    uint16_t *wp_refs;
    bool wp_overlap;
    /*
     * TEMP lab REIMS_VGPU_LAB_DIRTY_WP_BRIDGE=on (b451): a protect remaps a set's runs together with the gaps between
     * them when every gap page is already read-only or held only by hashed sets. b439/b450 canvas 5K: the clean
     * surfaces (256 and 57 pages, live the whole run) sit in runs of ~4 pages interleaved with each other; a remap
     * costs ~0.4 ms a call whatever its size, so a protect took ~60 remaps and the thread at its 30 % duty protected
     * ~10 sets a second while ~3000 foreground hashes a second (15-28 % of the drain) went to sets never written.
     * A hashed set's generation moves only on its own hashes, so a remap that drops a page's dirty bits loses it
     * nothing; a store into a bridged page faults and gets the page back like an orphan's. `wp_hc` / `wp_lc` count,
     * per pfn, the hashed and the logged (query-synced) sets that hold the page (under `lock`, read racily by the
     * step: a logged set that comes in between is covered by the marks of the bridged pages). `wp_bridged` keeps the
     * gap runs made read-only for an invalidate; an untrack orphans the read-only runs of an unlisted set.
     * The switch also queues a set its syncs never found written (gen 1) after `wp_fast_streak` clean syncs (the
     * age gate unchanged), and protects such never-written sets from 16 pages: hashed ones (not 64) and queried ones (not
     * `wp_min_pages`, 2048 - the canvas surfaces turned out to be queried sets of a few runs, never protected).
     */
    bool wp_bridge;
    uint32_t wp_fast_streak;
    uint16_t *wp_hc, *wp_lc;
    GArray *wp_bridged;
    GArray *wp_bridge_marks;
    uint64_t lab_wp_br_spans, lab_wp_br_runs, lab_wp_br_ro, lab_wp_br_hashed;
    uint64_t lab_wp_br_stop_log, lab_wp_br_stop_free, lab_wp_br_stop_far, lab_wp_br_orphaned;
    /*
     * TEMP lab DIRTY_WP_FAIR (on; `=off` restores): a store fault never waits for a whole WP step. b452: at a UI
     * transition one step held `wp_mutex` up to 72 ms (32 remaps of ~1 ms each, between them a plain unlock and lock
     * that lets nobody in), MMIO exits averaged 0.5-1 ms against 3 us, and the guest's audio stopped 30-150 ms. Under
     * it the protect loop, the merged give-back and the orphan release step aside after each remap while a fault
     * waits, and a fault itself gives back at most REIMS_WP_FAIR_FAULT_SPANS spans.
     */
    unsigned wp_fair;   /* bits: 1 give-back stops, 2 step yields, 4 fault budget, 8 protect yields */
    /*
     * TEMP lab DIRTY_WP_ORPHAN_CALM (on, =off restores; b457): orphans (read-only runs of sets the guest freed) are not given back
     * while sets keep being freed (a UI transition) - at most `wp_orphan_hold_ms` - and protects wait with them; a
     * guest store into one still faults it back. b454 F0: at Safari's full-screen exit the WP thread remapped
     * 2000-2400 runs (50-106k pages, ~950 ms) a second, outside any duty, in exactly the seconds every guest exit
     * cost 250-300 us and the audio stopped. b447's 3 ms rest between give-back steps never lowered that rate.
     */
    bool wp_orphan_calm;
    uint64_t wp_orphan_near, lab_wp_near_runs;   /* TEMP lab DIRTY_WP_ORPHAN_NEAR_KB (bytes; 0 = off) */
    int64_t wp_orphaned_ns, wp_orphan_hold_since;
    uint32_t wp_orphan_quiet_ms, wp_orphan_hold_ms;
    /* TEMP lab (b454): remaps by cause - protect, fault, orphan give-back, dropping give-back, other - per second. */
    uint64_t lab_wp_cause_n[5], lab_wp_cause_pages[5], lab_wp_held;
    int64_t lab_wp_cause_last;
    uint64_t lab_wp_fwait_ns, lab_wp_fwait_max_ns, lab_wp_fhold_ns, lab_wp_fhold_max_ns, lab_wp_fslow;
    uint64_t lab_wp_protects, lab_wp_faults, lab_wp_refused, lab_wp_chunks, lab_wp_remaps;
    /*
     * TEMP lab (n148): why the protect step passed over a queued set — gone or no longer WP_NONE,
     * unarmed, past the WP bitmap, or sharing a page that is already read-only — and how often a
     * set was queued again because its streak went on past the one moment it used to be queued at.
     */
    uint64_t lab_wp_rej_state, lab_wp_rej_gen, lab_wp_rej_bits, lab_wp_rej_ro, lab_wp_requeued;
    /* TEMP lab (n148): target pages a sync left out of its query because they are read-only. */
    uint64_t lab_ro_skipped;
    /* TEMP lab (PREP_SYNC): merged syncs the drain asked for ahead of a pass's planned binds. */
    uint64_t lab_prep_n, lab_prep_sets, lab_prep_pages, lab_prep_ns, lab_prep_asked;
    uint64_t lab_prep_query_ns, lab_prep_settle_ns, lab_prep_fold_ns, lab_prep_lock_ns;
    /*
     * TEMP lab REIMS_VGPU_LAB_DIRTY_WP_RETRY (on unless =off): a clean set past its streak is queued
     * again every `wp_streak_need` clean syncs, not only when the streak equals the need.
     * REIMS_VGPU_LAB_DIRTY_RO_SKIP (on unless =off): a sync does not query read-only pages.
     */
    bool wp_retry, ro_skip;
    /*
     * TEMP lab REIMS_VGPU_LAB_DIRTY_WP_HOLE=on (WP thread with RO_SKIP): a store fault opens its page
     * only, not the whole set (WP_HOLED). Giving a 5K surface back cost a remap per run (~2 000,
     * ~0.6 s) and protecting it again as much, after a streak doubled by every loss — so one
     * stray store cost the whole protection for seconds.
     */
    bool wp_hole;
    /*
     * More holes than this and the set is given back whole, as before: a guest rewriting a surface
     * it had left alone would otherwise take one exit and one remap per page (~0.4 ms each).
     */
    uint32_t wp_hole_max;
    uint64_t lab_wp_holed, lab_wp_heal_queued, lab_wp_heals, lab_wp_heal_skipped, lab_wp_hole_drops;
    uint64_t lab_wp_lost, lab_wp_dropped, lab_wp_remap_ns, lab_wp_pause_ns, lab_wp_pause_max_ns;
    /*
     * Lab DIRTY_WP_CALM (on unless =off). Full-screen transitions create 5K surfaces scattered over
     * ~2 000 runs that live a few seconds; protecting one cost a remap per run (~0.35 ms) and giving
     * it back as much again (n185 fsz0: 30.8k remaps, 11.9 s of them, in 80 s of scenario), while
     * the step held `wp_mutex` across 32 runs and vCPU store faults and the drain's untracks waited
     * for it (43 % of the drain's map/unmap handling). Calm: a set is queued only once it has lived
     * longer than its protection costs; the step lets a waiting fault in between the runs it gives
     * back; and an untrack of a set still on `wp_list` hands the set to the step (`wp_dead`) instead
     * of waiting for `wp_mutex`. (A fault still opens only its page: see the fault hook.)
     */
    bool wp_calm;
    bool wp_layout;   /* TEMP lab (b357) REIMS_VGPU_LAB_DIRTY_WP_LAYOUT=1: big sets' layout at protect */
    /*
     * Lab DIRTY_HASH_DUE (on by default; =off restores the old window): the background hashes a
     * hashed set when its next read is due — 4 doorbells from its shortest recent read gap past its last read — instead of at every
     * doorbell for the 4 doorbells right after any read. A set read once a frame (~5 doorbells
     * apart) was hashed up to 4 times for nothing and then, outside the window by its next read,
     * hashed again on the drain; n185: 210-224 s of hashing a boot (~1 core, ~9 GB/s), 20 s of it
     * on the drain. Without a gap history the 4-doorbell window stays.
     */
    bool hash_due;
    uint64_t lab_hash_due_skipped;
    uint32_t wp_age_ms, wp_age_run_us;   /* the age floor and its share per run (DIRTY_WP_AGE_MS) */
    GPtrArray *wp_dead;           /* untracked sets still on `wp_list`, for the step (under `lock`) */
    int wp_waiters;               /* store faults waiting for `wp_mutex` (atomic) */
    uint64_t lab_wp_young, lab_wp_dead, lab_wp_yields;
    /*
     * TEMP lab REIMS_VGPU_LAB_DIRTY_WP_MERGE (on by default, =off restores the one-run give-back and the
     * one-page fault): read-only pages go back to writable in merged spans
     * (gaps of at most `wp_merge_gap` pages, DIRTY_WP_MERGE_GAP, 512), and a store fault on a dropped
     * set's or an orphan's page gives the whole of it back at once. A remap costs ~200 us whatever its
     * size and a 5K surface is ~2 000 runs: p005 fc0a had 2 500 remaps a second through the
     * transitions, and one lost set took 7 387 faults (each an exit plus a one-page remap, ~1.5 s of
     * vCPU time) before the step had given it back. See reims_vgpu_dirty_wp_release_merged().
     */
    bool wp_merge;
    uint32_t wp_merge_gap;
    ReimsVgpuDirtySlice *wp_merge_slices;   /* under `wp_mutex` */
    int wp_merge_slices_cap;
    GArray *wp_merge_runs;                  /* scratch, under `wp_mutex` */
    uint64_t lab_wp_merge_spans, lab_wp_merge_runs, lab_wp_merge_pages, lab_wp_merge_fault;
    /* TEMP lab (b357): remaps by size - 1, 2-16, 17-256, >256 pages - count and time. */
    uint64_t lab_wp_rm_n[4], lab_wp_rm_ns[4];
    /*
     * TEMP lab REIMS_VGPU_LAB_UNTRACK_POPULATE=on: the queried (not hashed) sets this tracker stops
     * tracking have their writable pages' second-level entries filled in for write on threads of
     * their own (`up_threads`, UNTRACK_POPULATE_THREADS, 2), before the guest reuses the pages. A
     * query leaves its pages write-protected for dirty logging, and on a host without PML (AMD) the
     * guest's first store into each is a hypervisor exit: WindowServer zeroes the pages of the
     * surfaces a full-screen transition frees and allocates again at ~2 us a page there - 51 % of
     * its main thread's CPU in transitions (hq051 wp1: __bzero 4285 of 9667 samples, 2.08M pages
     * wired in 5.2 s over 24 transitions), and on AMD the guest, not the device, paces the
     * transition. A populate for write marks its pages dirty, so a set that reads them later finds
     * them so and no write is lost; read-only (DIRTY_WP) pages are left to the WP step.
     */
    bool up_on;
    bool up_ready;                /* the populate threads exist (on, or a live file that may turn it on) */
    int up_threads;
    QemuMutex up_mutex;
    QemuCond up_cond;
    GArray *up_queue;               /* ReimsVgpuDirtyRun, under `up_mutex` */
    uint64_t lab_up_sets, lab_up_runs, lab_up_pages, lab_up_calls, lab_up_fail, lab_up_ns, lab_up_dropped;
    int64_t lab_up_last_ns;
    /*
     * TEMP lab REIMS_VGPU_LAB_DIRTY_TIGHT=on (live through REIMS_VGPU_LAB_LIVE_FILE): a sync bridges a gap
     * between two runs of the pages it queries only when every page of the gap belongs to a tracked set.
     * A hypervisor query write-protects every page it walks again, the bridged gaps included, and the
     * gaps of the 256-page bridge are mostly not surfaces: the guest's free pages, heaps, other
     * processes' memory (a busy desktop's ~60k tracked pages were walked as ~360k). The guest's next
     * store to each faults into the hypervisor - and the free pages next to a surface just released are
     * the ones the kernel hands out next: at 5K WindowServer zeroes ~1.6M fresh pages a phase at
     * ~3 GB/s (hq062/hq072, AMD: 46-65k queries a second over 15.7M pages). `trk` counts the tracked
     * sets holding each target page frame (written under `lock` by track and untrack, read lock-free
     * under RCU; grown by copy, like `qep`), kept whenever the switch may be on.
     */
    struct ReimsVgpuDirtyTrk *trk;
    bool trk_on;                  /* `trk` is kept (the switch set, or a live file that may set it) */
    bool tight;                   /* the switch's environment value */
    uint64_t tight_slack;         /* REIMS_VGPU_LAB_DIRTY_TIGHT_SLACK: untracked pages a bridge may still cross (0) */
    bool host_intel;              /* an Intel host (drain-paced): picks the hash defaults, see set_ondemand */
    uint64_t lab_tight_kept, lab_tight_cut, lab_tight_cut_pages;
    /* TEMP lab: foreground syncs of large sets (>= 4096 pages) and their time, by the drain's work channel (7 = All). */
    uint64_t lab_fgbig_n[8], lab_fgbig_ns[8];
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
    uint64_t lab_steals, lab_steal_same, lab_steal_late, lab_steal_joined;
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

/*
 * TEMP lab REIMS_VGPU_LAB_LIVE_FILE (crates/reims-vgpu/src/runtime/lab_live.rs): the live value of lab switch `name`
 * (without the REIMS_VGPU_LAB_ prefix) in the host-written file - 1 for `on`, 0 for any other value, -1 when the
 * variable is unset or the file is missing or does not name it. So an A/B runs as phases inside one boot instead of a
 * boot an arm. Re-read at most every 200 ms. Any thread.
 */
static int reims_vgpu_lab_live_get(const char *name, long *num)
{
    static GMutex lock;
    static int state;            /* 0 not looked, 1 no file variable, 2 file variable set */
    static char *path;
    static gint64 checked_us;
    static char *keys[32];
    static int vals[32];
    static long nums[32];        /* the value as a number (strtol), for a numeric switch */
    static int n;
    gint64 now;
    int i, r = -1;

    g_mutex_lock(&lock);
    if (state == 0) {
        const char *e = getenv("REIMS_VGPU_LAB_LIVE_FILE");

        state = e && *e ? 2 : 1;
        path = state == 2 ? g_strdup(e) : NULL;
    }
    if (state == 2) {
        now = g_get_monotonic_time();
        if (checked_us == 0 || now - checked_us >= 200000) {
            gchar *text = NULL;

            checked_us = now;
            for (i = 0; i < n; i++) {
                g_free(keys[i]);
            }
            n = 0;
            if (g_file_get_contents(path, &text, NULL, NULL)) {
                gchar **lines = g_strsplit(text, "\n", -1);

                for (i = 0; lines[i] && n < (int)ARRAY_SIZE(keys); i++) {
                    gchar *ln = g_strstrip(lines[i]);
                    gchar *eq = strchr(ln, '=');

                    if (!eq || eq == ln) {
                        continue;
                    }
                    *eq = '\0';
                    if (g_str_has_prefix(ln, "REIMS_VGPU_LAB_")) {
                        ln += strlen("REIMS_VGPU_LAB_");
                    }
                    keys[n] = g_strdup(g_strstrip(ln));
                    vals[n] = strcmp(g_strstrip(eq + 1), "on") == 0;
                    nums[n] = strtol(g_strstrip(eq + 1), NULL, 10);
                    n++;
                }
                g_strfreev(lines);
                g_free(text);
            }
        }
        for (i = 0; i < n; i++) {
            if (strcmp(keys[i], name) == 0) {
                r = vals[i];
                if (num) {
                    *num = nums[i];
                }
                break;
            }
        }
    }
    g_mutex_unlock(&lock);
    return r;
}

static int reims_vgpu_lab_live(const char *name)
{
    return reims_vgpu_lab_live_get(name, NULL);
}

/* The live value of numeric lab switch `name`, or -1 when the file does not name it. */
static long reims_vgpu_lab_live_num(const char *name)
{
    long v = -1;

    return reims_vgpu_lab_live_get(name, &v) < 0 ? -1 : v;
}

/*
 * TEMP lab live DIRTY_GAP=<pages>: the sync gap bridge (REIMS_VGPU_DIRTY_GAP) from the live file, 0 = not
 * set. Refreshed once a second by the on-demand stats (a lookup takes the live file's lock, and the bridge
 * is read for every run of every sync). hq078: the hypervisor's price is the call, not the page walked -
 * DIRTY_TIGHT cut the pages walked 17x and tripled the calls, and the guest got slower (Safari 39 -> 27
 * frames a transition, WindowServer's __bzero 1754 -> 4365 samples).
 */
static uint64_t reims_vgpu_dirty_gap_live;

/*
 * TEMP lab live DIRTY_CALL_NS=<ns>: what the hash/query choice of a new set (reims_vgpu_dirty_hash_pick) prices
 * a hypervisor call at, 0 = not set (5000, the host's own time for one). hq078 says the guest pays far more a
 * call than the host does, and a hashed set makes no call at all.
 */
static uint64_t reims_vgpu_dirty_call_ns_live;
/* TEMP lab live DIRTY_HASH_MAX=<pages>: REIMS_VGPU_LAB_DIRTY_HASH_MAX from the live file, 0 = not set. */
static uint64_t reims_vgpu_dirty_hash_max_live;
/*
 * TEMP lab live DIRTY_WP_STREAK=<syncs> and DIRTY_WP_AGE_MS=<ms> (0 = not set): the clean syncs in a row and the age
 * a set needs before the WP step protects it. The thread mode's 128 syncs and 1 s + 2 ms a run were sized for
 * transition surfaces that die young; the 5K sets the drain syncs most are long-lived and never written (hq079
 * fgtop: 10.8k pages, streak 112, gen 1, not protected - every sync of one is hundreds of hypervisor calls).
 */
static uint64_t reims_vgpu_dirty_wp_streak_live, reims_vgpu_dirty_wp_age_live;
/*
 * TEMP lab live DIRTY_WP_THREAD_DUTY=<percent> (0 = not set): the WP thread's duty, REIMS_VGPU_LAB_DIRTY_WP_THREAD_DUTY
 * (30) otherwise. b348 (Intel, full-screen Maps): the thread ran at its 30 % cap the whole time (~700 run remaps a
 * second, ~0.4 ms each) and finished 0-2 protects a second, while the drain's most-synced sets were 5K surfaces
 * (10 800 pages, gen 1, clean streak 165-377) left unprotected - each sync of one a run of hypervisor queries.
 */
static uint64_t reims_vgpu_dirty_wp_thread_duty_live;
/*
 * TEMP lab live DIRTY_HASH_BG_MAX=<pages> (0 = no bound): a hashed set larger than this is not hashed by
 * the background prefetch at every doorbell, only when a read needs it (split across the workers) or
 * a pass names it ahead. hq080: hashing the 5K sets (10.8k pages) in the background as well tied up
 * ~4 GB/s of hashing and put ch5's serve at 15 ms.
 */
static uint64_t reims_vgpu_dirty_hash_bg_max_live;
/*
 * TEMP lab: DIRTY_HASH_BG_MAX when the live file does not name it - 4096 by default (hq085), set at init from
 * REIMS_VGPU_LAB_DIRTY_HASH_BG_MAX (0 = no bound).
 */
static uint64_t reims_vgpu_dirty_hash_bg_max_default = 4096;
/*
 * TEMP lab live DIRTY_HASH_BG_ONCE=on: a hashed set past DIRTY_HASH_BG_MAX is still hashed by the background,
 * but once a predicted read - in its DIRTY_HASH_DUE window, while nothing has synced it since the window
 * opened - so its reader finds it synced instead of hashing 44 MB on the drain (hq082: fg sync time
 * 89 -> 281 ms a second with the large sets hashed only when read).
 */
static int reims_vgpu_dirty_hash_bg_once_live;
/*
 * TEMP lab live DIRTY_WP_HASHED=on: a hashed set gets the WP step too - its clean hash syncs count as the
 * clean streak a queried set's syncs give it, under the same need and age. hq082: with the 5K sets hashed, the
 * readers hashed the same 4-5 of them (10.8k pages, written 0-5 times a boot) 40-82 times a second each; a
 * protected set reads as synced with no hash and no hypervisor call.
 */
static int reims_vgpu_dirty_wp_hashed_live;
/*
 * TEMP lab live DIRTY_WP_HASHED_STREAK=<syncs>: the clean hash syncs a hashed set needs before WP_HASHED
 * protects it, apart from DIRTY_WP_STREAK (the queried sets'). hq082 (AMD): the readers re-hash 5K sets the guest
 * leaves alone 40-82 times a second; hq084/hq085 (AMD): WP_HASHED at the queried sets' streak (16) protected sets
 * the guest wrote again soon, and the remaps cost the guest more than the hashes. (p009, Intel: the drain's
 * foreground syncs are 23-29 ms a second with the 5K sets hashed - little left to win there.) 0 = DIRTY_WP_STREAK's
 * need.
 */
static uint64_t reims_vgpu_dirty_wp_hashed_streak_live;
/*
 * TEMP lab live DIRTY_WP_HASHED_MIN=<pages>: the smallest hashed set WP_HASHED protects, apart from DIRTY_WP_MIN (the
 * queried sets', 2048). b099 (Intel, Weather): the hashed sets the drain re-hashes are mostly 150-2110 pages - under the
 * queried sets' floor, so WP_HASHED protected none - and 6.2k hash syncs a window over 2.7M pages found 2 dirty.
 * 0 = DIRTY_WP_MIN's floor.
 */
static uint64_t reims_vgpu_dirty_wp_hashed_min_live;
/*
 * DIRTY_WP_HASHED and its floor when the live file does not name them: on, 64 pages, on an Intel host (b099: the
 * hashed sets there are the small and mid ones, hash_max 4096, and protecting them took the drain's foreground hashes
 * from ~44 to ~3 ms a second - Weather 11.3 -> 9.2 us a draw, reveal-desktop 15.1 -> 11.9, Safari 27 -> 22, Finder
 * 18.3 -> 16.5; no artifacts); off elsewhere (hq084/hq085, AMD: with the 5K sets hashed the remaps cost the guest
 * more than the hashes). REIMS_VGPU_LAB_DIRTY_WP_HASHED=on|off and _WP_HASHED_MIN=<pages> override at boot.
 */
static int reims_vgpu_dirty_wp_hashed_default;
static uint64_t reims_vgpu_dirty_wp_hashed_min_default;
/*
 * TEMP lab live DIRTY_VERIFY=on|off (see `ReimsVgpuDirtySet::vhash`): -1 while the live file does not name it, and
 * then REIMS_VGPU_LAB_DIRTY_VERIFY=off at boot is the only way to turn it off.
 */
static int reims_vgpu_dirty_verify_live = -1;

static bool reims_vgpu_dirty_verify_on(void)
{
    static int env = -1;
    int v = qatomic_read(&reims_vgpu_dirty_verify_live);

    if (v >= 0) {
        return v == 1;
    }
    if (qatomic_read(&env) < 0) {
        const char *e = getenv("REIMS_VGPU_LAB_DIRTY_VERIFY");

        qatomic_set(&env, !(e && strcmp(e, "off") == 0));
    }
    return qatomic_read(&env) == 1;
}

static uint64_t reims_vgpu_dirty_call_ns(void)
{
    uint64_t v = qatomic_read(&reims_vgpu_dirty_call_ns_live);

    return v ? v : 5000;
}

static void reims_vgpu_dirty_gap_refresh(void)
{
    long v = reims_vgpu_lab_live_num("DIRTY_GAP");

    qatomic_set(&reims_vgpu_dirty_gap_live, v > 0 ? (uint64_t)v : 0);
    v = reims_vgpu_lab_live_num("DIRTY_CALL_NS");
    qatomic_set(&reims_vgpu_dirty_call_ns_live, v > 0 ? (uint64_t)v : 0);
    v = reims_vgpu_lab_live_num("DIRTY_HASH_MAX");
    qatomic_set(&reims_vgpu_dirty_hash_max_live, v > 0 ? (uint64_t)v : 0);
    v = reims_vgpu_lab_live_num("DIRTY_WP_STREAK");
    qatomic_set(&reims_vgpu_dirty_wp_streak_live, v > 0 ? (uint64_t)v : 0);
    v = reims_vgpu_lab_live_num("DIRTY_WP_AGE_MS");
    qatomic_set(&reims_vgpu_dirty_wp_age_live, v > 0 ? (uint64_t)v : 0);
    v = reims_vgpu_lab_live_num("DIRTY_WP_THREAD_DUTY");
    qatomic_set(&reims_vgpu_dirty_wp_thread_duty_live, v > 0 ? (uint64_t)MIN(v, 100) : 0);
    v = reims_vgpu_lab_live_num("DIRTY_HASH_BG_MAX");
    qatomic_set(&reims_vgpu_dirty_hash_bg_max_live, v < 0 ? reims_vgpu_dirty_hash_bg_max_default : (uint64_t)v);
    qatomic_set(&reims_vgpu_dirty_hash_bg_once_live, reims_vgpu_lab_live("DIRTY_HASH_BG_ONCE") == 1);
    {
        int on = reims_vgpu_lab_live("DIRTY_WP_HASHED");

        qatomic_set(&reims_vgpu_dirty_wp_hashed_live, on < 0 ? reims_vgpu_dirty_wp_hashed_default : on == 1);
    }
    v = reims_vgpu_lab_live_num("DIRTY_WP_HASHED_STREAK");
    qatomic_set(&reims_vgpu_dirty_wp_hashed_streak_live, v > 0 ? (uint64_t)v : 0);
    v = reims_vgpu_lab_live_num("DIRTY_WP_HASHED_MIN");
    qatomic_set(&reims_vgpu_dirty_wp_hashed_min_live, v > 0 ? (uint64_t)v : reims_vgpu_dirty_wp_hashed_min_default);
    qatomic_set(&reims_vgpu_dirty_verify_live, reims_vgpu_lab_live("DIRTY_VERIFY"));
}

/* TEMP lab DIRTY_TIGHT: whether a sync bridges only gaps of tracked pages (one look a sync). */
static bool reims_vgpu_dirty_tight_on(ReimsVgpuDirty *d)
{
    int v;

    if (!d->trk_on) {
        return false;
    }
    v = reims_vgpu_lab_live("DIRTY_TIGHT");
    return v < 0 ? d->tight : v == 1;
}

static void reims_vgpu_dirty_trk_free(ReimsVgpuDirtyTrk *t)
{
    g_free(t);
}

/* TEMP lab DIRTY_WP_BRIDGE: count `s`'s target pages in (1) or out (-1) of `wp_hc` / `wp_lc`. d->lock held. */
static void reims_vgpu_dirty_wp_hold_add(ReimsVgpuDirty *d, const ReimsVgpuDirtySet *s, int delta)
{
    const int shift = qemu_target_page_bits();
    uint16_t *c = s->hash ? d->wp_hc : d->wp_lc;
    uint64_t per, n = 1ULL << (36 - shift);
    size_t i;

    if (!c) {
        return;
    }
    per = s->page_size >> shift;
    for (i = 0; i < s->count; i++) {
        uint64_t pfn = s->pages[i] >> shift, k;

        for (k = 0; k < per && pfn + k < n; k++) {
            uint16_t v = c[pfn + k];

            if (delta > 0 ? v < UINT16_MAX : v > 0) {
                qatomic_set(&c[pfn + k], (uint16_t)(v + delta));
            }
        }
    }
}

/*
 * TEMP lab DIRTY_TIGHT: count `s`'s target page frames in (`delta` 1, at track) or out (-1, at
 * untrack) of `trk`. d->lock held.
 */
static void reims_vgpu_dirty_trk_add(ReimsVgpuDirty *d, const ReimsVgpuDirtySet *s, int delta)
{
    const int shift = qemu_target_page_bits();
    ReimsVgpuDirtyTrk *t = d->trk;
    uint64_t per, end_pfn;
    size_t i;

    if (!d->trk_on || s->count == 0) {
        return;
    }
    per = s->page_size >> shift;
    end_pfn = (s->pages[s->count - 1] >> shift) + per;
    if (delta > 0 && (!t || end_pfn > t->n)) {
        uint64_t n = MAX(end_pfn, t ? t->n * 2 : end_pfn);
        ReimsVgpuDirtyTrk *nt = g_malloc0(sizeof(*nt) + n * sizeof(uint16_t));

        nt->n = n;
        if (t) {
            memcpy(nt->c, t->c, t->n * sizeof(uint16_t));
        }
        qatomic_rcu_set(&d->trk, nt);
        if (t) {
            call_rcu(t, reims_vgpu_dirty_trk_free, rcu);
        }
        t = nt;
    }
    if (!t) {
        return;
    }
    for (i = 0; i < s->count; i++) {
        uint64_t pfn = s->pages[i] >> shift, k;

        for (k = 0; k < per && pfn + k < t->n; k++) {
            uint16_t v = t->c[pfn + k];

            if (delta > 0 ? v < UINT16_MAX : v > 0) {
                qatomic_set(&t->c[pfn + k], (uint16_t)(v + delta));
            }
        }
    }
}

/*
 * TEMP lab DIRTY_TIGHT: whether a sync may bridge target page frames [from, to) between two runs it
 * queries - always without the switch; with it, only when a tracked set holds every one of them but
 * at most `tight_slack` (a scattered set's runs, each its own call, can cost more than the walk). Any
 * thread; a count read while a set is being tracked or untracked only moves where a range ends.
 */
static bool reims_vgpu_dirty_bridge_ok(ReimsVgpuDirty *d, bool tight, uint64_t from, uint64_t to)
{
    const ReimsVgpuDirtyTrk *t;
    bool ok = true;
    uint64_t p;

    if (!tight || from >= to) {
        return true;
    }
    WITH_RCU_READ_LOCK_GUARD() {
        t = qatomic_rcu_read(&d->trk);
        if (!t || to > t->n) {
            ok = to - from <= d->tight_slack;
        } else {
            uint64_t foreign = 0;

            for (p = from; p < to; p++) {
                if (!qatomic_read(&t->c[p]) && ++foreign > d->tight_slack) {
                    ok = false;
                    break;
                }
            }
        }
    }
    if (ok) {
        qatomic_inc(&d->lab_tight_kept);
    } else {
        qatomic_inc(&d->lab_tight_cut);
        qatomic_add(&d->lab_tight_cut_pages, to - from);
    }
    return ok;
}

static void reims_vgpu_dirty_set_free(gpointer p)
{
    ReimsVgpuDirtySet *s = p;

    g_free(s->pages);
    g_free(s->page_gen);
    g_free(s->hash);
    g_free(s->vhash);
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
    /*
     * 24 (was 6): a set read once a frame spans ~5.5 doorbell epochs at 5K
     * drag, so 6 left 97 % of the foreground syncs cold — chiefly the GVA
     * store witness of every layer Store (gva_store_witness::note_store,
     * ~2300 hypervisor queries/s on the drain at ~23 us each). At 24 the
     * prefetch workers sync them ahead: fg 2250 -> 660/s, drag 99.1/99.0 ->
     * 101.8/102.0 Hz (ftc*, 2026-10-02). REIMS_VGPU_DIRTY_OD_HOT overrides.
     */
    d->od_hot = 24;
    qemu_cond_init(&d->synced_cond);
    qemu_cond_init(&d->od_work_cond);
    qemu_cond_init(&d->q_cond);
    qemu_cond_init(&d->split_cond);
    d->split_chunks = g_array_new(FALSE, FALSE, sizeof(ReimsVgpuDirtyChunk));
    d->fg_split_min = 4096;   /* until set_ondemand reads the lab switch */
    d->pstamps = g_array_new(FALSE, FALSE, sizeof(ReimsVgpuDirtyPstamp));
    for (int k = 0; k < REIMS_VGPU_DIRTY_FGSLOTS_MAX; k++) {
        d->fjob[k].chunks = g_array_new(FALSE, FALSE, sizeof(ReimsVgpuDirtyChunk));
    }
    d->od_work = g_array_new(FALSE, FALSE, sizeof(uint64_t));
    d->ahead = g_array_new(FALSE, FALSE, sizeof(uint64_t));   /* lab SYNC_AHEAD */
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
    g_array_free(d->ahead, TRUE);   /* lab SYNC_AHEAD */
    g_array_free(d->split_chunks, TRUE);
    g_array_free(d->pstamps, TRUE);
    for (int k = 0; k < REIMS_VGPU_DIRTY_FGSLOTS_MAX; k++) {
        g_array_free(d->fjob[k].chunks, TRUE);
    }
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
    g_free(d->page_seq);
    g_free(d->qep);   /* every reader thread is gone by now */
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

/* Lab DIRTY_HASH ------------------------------------------------------- */
static uint64_t reims_vgpu_dirty_gap_pages(void);
#define REIMS_VGPU_DIRTY_HASH_HOT 4   /* epochs a hashed set stays in the background prefetch */

/*
 * Whether a set of these (sorted, distinct) pages is cheaper to hash than to query: the
 * hypervisor's price is a call per bridged run (runs closer than the gap bridge share one call
 * and walk the pages between) plus every page walked; a hash's is the set's own bytes.
 */
static bool reims_vgpu_dirty_hash_pick(ReimsVgpuDirty *d, const uint64_t *pages, size_t n,
                                       size_t page_size)
{
    const int shift = qemu_target_page_bits();
    const uint64_t bridge = reims_vgpu_dirty_gap_pages() << shift;
    uint64_t own = n * (page_size >> shift), walked = 0, calls = 0, start = 0, end = 0;
    size_t i;

    if (own > (qatomic_read(&reims_vgpu_dirty_hash_max_live) ?: d->hash_max)) {   /* TEMP lab live DIRTY_HASH_MAX */
        return false;
    }
    if (d->hash_all) {
        return true;
    }
    for (i = 0; i < n; i++) {
        if (calls && pages[i] <= end + bridge) {
            end = pages[i] + page_size;
            continue;
        }
        if (calls) {
            walked += (end - start) >> shift;
        }
        calls++;
        start = pages[i];
        end = pages[i] + page_size;
    }
    walked += (end - start) >> shift;
    return own * d->hash_ns < calls * reims_vgpu_dirty_call_ns() + walked * 26;
}

/*
 * TEMP lab: the same question a lane-multiply hash answers, at memory speed - XXH3's accumulate step
 * (each 64-bit word: its halves swapped, plus the product of the low and high halves of word ^ key)
 * over two 256-bit accumulators, then a scalar avalanche. 19-21 GB/s a thread on the i7-14700K
 * against 12 for the scalar lanes (out-of-cache pages); every single-bit change of a 4 KiB page moves
 * it. `len` a multiple of 64. One kernel for the process's life: hashes are only compared with hashes
 * the same kernel took (REIMS_VGPU_LAB_DIRTY_HASH_AVX2=off keeps the scalar one).
 */
__attribute__((target("avx2")))
static uint64_t reims_vgpu_dirty_hash_bytes_avx2(const uint8_t *p, size_t len)
{
    const __m256i k0 = _mm256_set_epi64x(0x1cad21f72c81017cLL, (long long)0xdb979083e96dd4deULL,
                                         0x1f67b3b7a4a44072LL, 0x78e5c0cc4ee679cbLL);
    const __m256i k1 = _mm256_set_epi64x(0x2172ffcc7dd05a82LL, (long long)0x8e2443f7744608b8ULL,
                                         0x4c263a81e69035e0LL, (long long)0xcb00c391bb52283cULL);
    __m256i a0 = _mm256_set1_epi64x((long long)0x9E3779B185EBCA87ULL);
    __m256i a1 = _mm256_set1_epi64x((long long)0xC2B2AE3D27D4EB4FULL);
    uint64_t l[8], h = len * 0x9E3779B185EBCA87ULL;
    size_t i;
    int k;

    for (i = 0; i < len; i += 64) {
        __m256i d0 = _mm256_loadu_si256((const __m256i *)(p + i));
        __m256i d1 = _mm256_loadu_si256((const __m256i *)(p + i + 32));
        __m256i x0 = _mm256_xor_si256(d0, k0), x1 = _mm256_xor_si256(d1, k1);
        __m256i m0 = _mm256_mul_epu32(x0, _mm256_srli_epi64(x0, 32));
        __m256i m1 = _mm256_mul_epu32(x1, _mm256_srli_epi64(x1, 32));

        a0 = _mm256_add_epi64(a0, _mm256_add_epi64(_mm256_shuffle_epi32(d0, 0x4E), m0));
        a1 = _mm256_add_epi64(a1, _mm256_add_epi64(_mm256_shuffle_epi32(d1, 0x4E), m1));
    }
    _mm256_storeu_si256((__m256i *)l, a0);
    _mm256_storeu_si256((__m256i *)(l + 4), a1);
    for (k = 0; k < 8; k++) {
        h ^= l[k] * 0xC2B2AE3D27D4EB4FULL;
        h = rol64(h, 27) * 0x9E3779B185EBCA87ULL + 0x165667B19E3779F9ULL;
    }
    h ^= h >> 33;
    h *= 0xC2B2AE3D27D4EB4FULL;
    h ^= h >> 29;
    h *= 0x165667B19E3779F9ULL;
    h ^= h >> 32;
    return h;
}

/* TEMP lab: whether this process hashes with the AVX2 kernel (decided once). */
static bool reims_vgpu_dirty_hash_avx2(void)
{
    static int v = -1;

    if (v < 0) {
        const char *e = getenv("REIMS_VGPU_LAB_DIRTY_HASH_AVX2");

        __builtin_cpu_init();
        v = __builtin_cpu_supports("avx2") && !(e && strcmp(e, "off") == 0);
        fprintf(stderr, "reims-dirty-hash-kernel avx2=%d\n", v);
    }
    return v;
}

/* A 64-bit hash of `len` bytes (a multiple of 32): four independent multiply-rotate lanes. */
static uint64_t reims_vgpu_dirty_hash_bytes_scalar(const uint8_t *p, size_t len)
{
    const uint64_t k1 = 0x9E3779B185EBCA87ULL, k2 = 0xC2B2AE3D27D4EB4FULL;
    uint64_t a = k1 + k2, b = k2, c = 0, e = -k1, h;
    size_t i;

    for (i = 0; i < len; i += 32) {
        uint64_t w[4];

        memcpy(w, p + i, sizeof(w));
        a = rol64(a + w[0] * k2, 31) * k1;
        b = rol64(b + w[1] * k2, 31) * k1;
        c = rol64(c + w[2] * k2, 31) * k1;
        e = rol64(e + w[3] * k2, 31) * k1;
    }
    h = rol64(a, 1) + rol64(b, 7) + rol64(c, 12) + rol64(e, 18) + len;
    h ^= h >> 33;
    h *= k2;
    h ^= h >> 29;
    h *= k1;
    h ^= h >> 32;
    return h;
}

/* A page's content hash: the AVX2 kernel where the host has it (TEMP lab), else the scalar lanes. */
static uint64_t reims_vgpu_dirty_hash_bytes(const uint8_t *p, size_t len)
{
    if (len % 64 == 0 && reims_vgpu_dirty_hash_avx2()) {
        return reims_vgpu_dirty_hash_bytes_avx2(p, len);
    }
    return reims_vgpu_dirty_hash_bytes_scalar(p, len);
}

static int reims_vgpu_dirty_ram_slices_into(ReimsVgpuDirtySlice **buf, int *cap);
static int reims_vgpu_dirty_slice_of(const ReimsVgpuDirtySlice *slices, int n, uint64_t gpa);

/*
 * TEMP lab DIRTY_VERIFY: the content hash of page `gpa`, or false when it is not inside one host mapping of guest RAM
 * (such a page reads as written by every evaluation anyway). RCU read lock held.
 */
static bool reims_vgpu_dirty_page_hash(const ReimsVgpuDirtySlice *slices, int n, uint64_t gpa,
                                       uint64_t page_size, uint64_t *h)
{
    int si = reims_vgpu_dirty_slice_of(slices, n, gpa);

    if (si < 0 || gpa + page_size > slices[si].gpa + slices[si].len) {
        return false;
    }
    *h = reims_vgpu_dirty_hash_bytes((const uint8_t *)memory_region_get_ram_ptr(slices[si].mr) +
                                     slices[si].offset + (gpa - slices[si].gpa), page_size);
    return true;
}

/*
 * TEMP lab DIRTY_VERIFY: hash every page of a set not yet published (no lock needed for it) into its baseline. The
 * evaluation that arms the set takes the hashes again for the pages it names, so a store between here and the
 * arming is baseline too, as it is for a hashed set's first sync.
 */
static void reims_vgpu_dirty_verify_base(ReimsVgpuDirty *d, ReimsVgpuDirtySet *s)
{
    ReimsVgpuDirtySlice *slices = NULL;
    int64_t t0 = get_clock();
    int cap = 0, n;
    size_t p;

    s->vhash = g_new0(uint64_t, MAX(s->count, 1));
    n = reims_vgpu_dirty_ram_slices_into(&slices, &cap);
    rcu_read_lock();
    for (p = 0; p < s->count; p++) {
        reims_vgpu_dirty_page_hash(slices, n, s->pages[p], s->page_size, &s->vhash[p]);
    }
    rcu_read_unlock();
    g_free(slices);
    s->vhash_base = true;
    qatomic_inc(&d->lab_vhash_sets);
    qatomic_add(&d->lab_vhash_base_ns, (uint64_t)(get_clock() - t0));
}

/*
 * TEMP lab DIRTY_VERIFY: of the pages an armed evaluation found stamped, keep those whose bytes moved since the
 * baseline (and take their new hashes) and those it cannot hash - outside guest RAM, or in a range not yet logged,
 * which read as written whatever their bytes. Returns whether any is left. `lock` held.
 */
static bool reims_vgpu_dirty_verify_filter(ReimsVgpuDirty *d, ReimsVgpuDirtySet *s,
                                           const ReimsVgpuDirtySlice *slices, int n, GArray *hit)
{
    guint h, keep = 0;

    rcu_read_lock();
    for (h = 0; h < hit->len; h++) {
        size_t p = g_array_index(hit, size_t, h);
        int si = reims_vgpu_dirty_slice_of(slices, n, s->pages[p]);
        uint64_t v;

        if (si >= 0 && slices[si].logged &&
            reims_vgpu_dirty_page_hash(slices, n, s->pages[p], s->page_size, &v)) {
            if (v == s->vhash[p]) {
                d->lab_vhash_same++;
                continue;
            }
            s->vhash[p] = v;
            d->lab_vhash_moved++;
        }
        g_array_index(hit, size_t, keep++) = p;
    }
    rcu_read_unlock();
    g_array_set_size(hit, keep);
    return keep > 0;
}

/* TEMP lab DIRTY_VERIFY: an unarmed set's baseline follows the pages its evaluations name. `lock` held. */
static void reims_vgpu_dirty_verify_refresh(ReimsVgpuDirtySet *s, const ReimsVgpuDirtySlice *slices, int n,
                                            const GArray *hit)
{
    guint h;

    rcu_read_lock();
    for (h = 0; h < hit->len; h++) {
        size_t p = g_array_index(hit, size_t, h);

        reims_vgpu_dirty_page_hash(slices, n, s->pages[p], s->page_size, &s->vhash[p]);
    }
    rcu_read_unlock();
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
    if (qatomic_read(&d->hash_mode) && reims_vgpu_dirty_hash_pick(d, pages, n, page_size)) {
        s->hash = g_new0(uint64_t, n);   /* lab DIRTY_HASH */
    }
    if (!s->hash && reims_vgpu_dirty_verify_on()) {
        reims_vgpu_dirty_verify_base(d, s);   /* TEMP lab DIRTY_VERIFY */
    }

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
    s->born_ns = get_clock();          /* lab DIRTY_WP_CALM */
    s->nruns = n ? 1 : 0;
    for (i = 1; i < n; i++) {
        s->nruns += pages[i] != pages[i - 1] + page_size;
    }
    /*
     * TEMP lab REIMS_VGPU_LAB_DIRTY_HOT_NEW=on: a new set counts as read this
     * epoch, so the next prefetch arms it on a worker instead of the drain's
     * first read doing it in the foreground. Full-screen transitions create
     * 5K surfaces every frame; ~55 % of foreground syncs were cold sets.
     */
    if (d->hot_new) {
        s->last_query_epoch = d->epoch;
    }
    d->next_token++;
    token = d->next_token;
    s->token = token;   /* lab DIRTY_WP */
    d->lab_ntrack++;
    d->lab_hash_sets += s->hash != NULL;   /* lab DIRTY_HASH */
    if (d->lab_tr) {
        ReimsVgpuLabRec r = { .t_ns = get_clock() - d->lab_t0, .token = token,
                              .needed = d->epoch, .epoch = d->epoch,
                              .pages = (uint32_t)(n * (page_size / 4096)), .kind = 8 };

        reims_vgpu_lab_rec(d, &r);
    }
    reims_vgpu_dirty_trk_add(d, s, 1);   /* TEMP lab DIRTY_TIGHT */
    reims_vgpu_dirty_wp_hold_add(d, s, 1);   /* TEMP lab DIRTY_WP_BRIDGE */
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
                                      uint64_t needed, int64_t *lab_wait,
                                      int64_t *lab_query);

/* reims_vgpu_dirty_ram_slices_into / reims_vgpu_dirty_slice_of: declared above reims_vgpu_dirty_track. */
static uint64_t reims_vgpu_dirty_query_begin(ReimsVgpuDirty *d);
static void reims_vgpu_dirty_query_footprint(ReimsVgpuDirty *d, uint64_t ticket,
                                             const ReimsVgpuDirtyChunk *chunks, guint n);
static void reims_vgpu_dirty_query_end(ReimsVgpuDirty *d, uint64_t ticket);

/*
 * WP_HOLED (TEMP lab DIRTY_WP_HOLE): a protected set a store faulted on. Only the faulted pages were
 * made writable; the rest stay read-only and keep being skipped by every query (RO_SKIP), so the
 * set syncs again, but a sync queries only its holes. A heal re-protects just the runs with a hole
 * once the set has read clean for a streak again, and the set is WP_ON once more.
 */
enum { WP_NONE, WP_PROTECTING, WP_ON, WP_DROPPING, WP_HOLED };

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

/*
 * TEMP lab DIRTY_WP_OVERLAP (REIMS_VGPU_LAB_DIRTY_WP_OVERLAP=on, not with DIRTY_WP_HOLE): sets may be
 * protected over pages another protected set already holds read-only. b103/b104: the 5K surfaces and
 * the long-lived queried sets were refused for read-only pages (`rej_ro`, 500-3000 a boot once the
 * hashed windows over the same memory were protected). Each page counts the protected sets holding
 * it; a store fault drops every set holding the page (the fault loop already walks them all), and a
 * release - a dropped set's runs, an orphan, a merged span - leaves a page some other set still holds
 * read-only. `wp_mutex` held.
 */
static inline bool reims_vgpu_dirty_wp_kept(ReimsVgpuDirty *d, uint64_t gpa)
{
    uint64_t pfn = gpa >> qemu_target_page_bits();

    return d->wp_refs && pfn < d->wp_bits_n && d->wp_refs[pfn] != 0;
}

/* TEMP lab DIRTY_WP_OVERLAP: whether any page of [gpa, end) is held by a protected set. */
static bool reims_vgpu_dirty_wp_run_kept(ReimsVgpuDirty *d, uint64_t gpa, uint64_t end)
{
    for (; gpa < end; gpa += 1ULL << qemu_target_page_bits()) {
        if (reims_vgpu_dirty_wp_kept(d, gpa)) {
            return true;
        }
    }
    return false;
}

/* TEMP lab DIRTY_WP_OVERLAP: count `s` in (`up`) or out of `wp_refs`, once each way. `wp_mutex` held. */
static void reims_vgpu_dirty_wp_count(ReimsVgpuDirty *d, ReimsVgpuDirtySet *s, bool up)
{
    const uint64_t pg = 1ULL << qemu_target_page_bits();
    size_t p;

    if (!d->wp_refs || s->wp_counted == up) {
        return;
    }
    s->wp_counted = up;
    for (p = 0; p < s->count; p++) {
        uint64_t a;

        for (a = s->pages[p]; a < s->pages[p] + s->page_size; a += pg) {
            uint64_t pfn = a >> qemu_target_page_bits();

            if (pfn >= d->wp_bits_n) {
                continue;
            }
            if (up) {
                d->wp_refs[pfn] += d->wp_refs[pfn] != UINT16_MAX;
            } else {
                d->wp_refs[pfn] -= d->wp_refs[pfn] != 0;
            }
        }
    }
}

/* One remap of [gpa, gpa+size), with `wp_bits` kept equal to what is read-only. `wp_mutex` held. */
/*
 * TEMP lab (b461 freeze watch): `wp_mutex` with its holder recorded. b461-B2 froze a guest 3 s after the 5K mode
 * switch - no vCPU exit, doorbell or WP line for 8 min while ~3 vCPUs spun in the guest - the shape of a TLB shootdown
 * waiting for a vCPU parked in a store fault. A waiter past 1 s says where the holder took the lock, once a second.
 */
static int reims_wp_holder_line, reims_wp_holder_tid;
static int64_t reims_wp_holder_ns;
/* What the holder is doing now: 0 nothing marked, 1 query ticket, 2 hypervisor remap, 3 marks. */
static int reims_wp_stage;

static void reims_wp_lock(ReimsVgpuDirty *d, int line)
{
    if (qemu_mutex_trylock(&d->wp_mutex) != 0) {   /* 0: taken */
        int64_t t0 = get_clock(), next = t0 + NANOSECONDS_PER_SECOND;

        for (;;) {
            int64_t now;

            g_usleep(20);
            if (qemu_mutex_trylock(&d->wp_mutex) == 0) {
                break;
            }
            now = get_clock();
            if (now >= next) {
                fprintf(stderr, "reims-dirty-wp-stuck: line %d thread %d waited %" PRId64 " ms; held from line %d by "
                        "thread %d for %" PRId64 " ms, waiters %d\n", line, qemu_get_thread_id(), (now - t0) / SCALE_MS,
                        qatomic_read(&reims_wp_holder_line), qatomic_read(&reims_wp_holder_tid),
                        (now - qatomic_read(&reims_wp_holder_ns)) / SCALE_MS, qatomic_read(&d->wp_waiters));
                next = now + NANOSECONDS_PER_SECOND;
            }
        }
    }
    qatomic_set(&reims_wp_holder_line, line);
    qatomic_set(&reims_wp_holder_tid, qemu_get_thread_id());
    qatomic_set(&reims_wp_holder_ns, get_clock());
}

static int reims_vgpu_dirty_cmp_run(gconstpointer a, gconstpointer b);

/* TEMP lab (b454): what the next remaps on this thread are for (lab_wp_cause_n). */
enum { WP_CAUSE_PROTECT, WP_CAUSE_FAULT, WP_CAUSE_ORPHAN, WP_CAUSE_DROP, WP_CAUSE_OTHER };
static __thread int reims_wp_cause = WP_CAUSE_OTHER;

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
    qatomic_set(&reims_wp_stage, 2);   /* TEMP lab (b461 freeze watch) */
    rc = whpx_wp_remap(gpa, size, writable);
    qatomic_set(&reims_wp_stage, 0);
    if (writable || rc) {
        for (a = gpa; a < gpa + size; a += step) {
            clear_bit_atomic(a >> qemu_target_page_bits(), d->wp_bits);
        }
    }
    {
        int64_t ns = get_clock() - t0;
        uint64_t np = size >> qemu_target_page_bits();
        int b = np <= 1 ? 0 : np <= 16 ? 1 : np <= 256 ? 2 : 3;

        d->lab_wp_remap_ns += ns;
        d->lab_wp_rm_n[b]++;   /* TEMP lab (b357) */
        d->lab_wp_rm_ns[b] += ns;
        d->lab_wp_cause_n[reims_wp_cause]++;   /* TEMP lab (b454) */
        d->lab_wp_cause_pages[reims_wp_cause] += np;
        if (t0 - d->lab_wp_cause_last >= NANOSECONDS_PER_SECOND) {
            if (d->lab_wp_cause_last) {
                fprintf(stderr, "reims-dirty-wp-cause epoch_ms=%" PRId64 " protect=%" PRIu64 "/%" PRIu64
                        " fault=%" PRIu64 "/%" PRIu64 " orphan=%" PRIu64 "/%" PRIu64 " drop=%" PRIu64 "/%" PRIu64
                        " other=%" PRIu64 "/%" PRIu64 " held=%" PRIu64 " orphans_left=%u\n",
                        g_get_real_time() / 1000,
                        d->lab_wp_cause_n[0], d->lab_wp_cause_pages[0], d->lab_wp_cause_n[1], d->lab_wp_cause_pages[1],
                        d->lab_wp_cause_n[2], d->lab_wp_cause_pages[2], d->lab_wp_cause_n[3], d->lab_wp_cause_pages[3],
                        d->lab_wp_cause_n[4], d->lab_wp_cause_pages[4], d->lab_wp_held,
                        d->wp_orphans ? d->wp_orphans->len : 0);
            }
            memset(d->lab_wp_cause_n, 0, sizeof(d->lab_wp_cause_n));
            memset(d->lab_wp_cause_pages, 0, sizeof(d->lab_wp_cause_pages));
            d->lab_wp_held = 0;
            d->lab_wp_cause_last = t0;
        }
    }
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

        if (!reims_vgpu_dirty_wp_ro(d, gpa) || reims_vgpu_dirty_wp_kept(d, gpa)) {
            gpa += pg;
            continue;
        }
        for (a = gpa + pg; a < end && reims_vgpu_dirty_wp_ro(d, a) && !reims_vgpu_dirty_wp_kept(d, a);
             a += pg) {
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

/* TEMP lab DIRTY_WP_HOLE: whether any page of `s` is read-only or another protected set's hole. */
static bool reims_vgpu_dirty_wp_any_held(ReimsVgpuDirty *d, const ReimsVgpuDirtySet *s)
{
    size_t p;

    if (reims_vgpu_dirty_wp_any_ro(d, s, 0, s->count)) {
        return true;
    }
    if (!d->wp_hole_bits) {
        return false;
    }
    for (p = 0; p < s->count; p++) {
        uint64_t pfn = s->pages[p] >> qemu_target_page_bits();

        if (pfn < d->wp_bits_n && test_bit(pfn, d->wp_hole_bits)) {
            return true;
        }
    }
    return false;
}

/* TEMP lab DIRTY_WP_HOLE: `s` is no longer protected; its holes are nobody's. */
static void reims_vgpu_dirty_wp_holes_clear(ReimsVgpuDirty *d, const ReimsVgpuDirtySet *s)
{
    size_t p;

    if (!d->wp_hole_bits) {
        return;
    }
    for (p = 0; p < s->count; p++) {
        uint64_t pfn = s->pages[p] >> qemu_target_page_bits();

        if (pfn < d->wp_bits_n) {
            clear_bit_atomic(pfn, d->wp_hole_bits);
        }
    }
}

/* TEMP lab DIRTY_WP_HOLE: whether every target page of [gpa, end) is read-only. */
static bool reims_vgpu_dirty_wp_all_ro(ReimsVgpuDirty *d, uint64_t gpa, uint64_t end)
{
    for (; gpa < end; gpa += 1ULL << qemu_target_page_bits()) {
        if (!reims_vgpu_dirty_wp_ro(d, gpa)) {
            return false;
        }
    }
    return true;
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
            qatomic_set(&d->wp_orphaned_ns, get_clock());   /* TEMP lab DIRTY_WP_ORPHAN_CALM */
        }
        p = q;
    }
}

/* Lab DIRTY_WP_CALM: how long a set must have lived before protecting it is worth its remaps. */
static int64_t reims_vgpu_dirty_wp_age_need(const ReimsVgpuDirty *d, const ReimsVgpuDirtySet *s)
{
    uint64_t live = qatomic_read(&reims_vgpu_dirty_wp_age_live);   /* TEMP lab live DIRTY_WP_AGE_MS */

    if (live) {
        return (int64_t)live * SCALE_MS;
    }
    /*
     * DIRTY_WP_BRIDGE keeps this age, per-run part included: b451 capped it at 250 ms for never-written sets, and
     * the WP thread then took Maps' scattered 5K surfaces (~2300 runs) at once - 600-900 remaps a second at its
     * full duty, 0-1 protects, Maps 41.8 -> 32.6 fps.
     */
    return (int64_t)d->wp_age_ms * SCALE_MS + (int64_t)d->wp_age_run_us * s->nruns * SCALE_US;
}

/*
 * TEMP lab DIRTY_WP_BRIDGE: the clean streak a set needs before it is queued - `need`, or `wp_fast_streak` for a set
 * no sync ever found written (b439 canvas: the costly sets lived the whole 20 s run, written once at most, and
 * 88 % of the foreground cost came after their 16th clean sync, 52 % after the 128th).
 */
static uint32_t reims_vgpu_dirty_wp_need(const ReimsVgpuDirty *d, const ReimsVgpuDirtySet *s, uint32_t need)
{
    if (d->wp_bridge && s->gen <= 1 && !s->wp_lost) {
        return MIN(need, d->wp_fast_streak);
    }
    return need;
}

/*
 * Lab DIRTY_WP_CALM: untracked sets the drain left on `wp_list` — their read-only runs become
 * orphans, then the sets are freed. `wp_mutex` held, `lock` not.
 */
static void reims_vgpu_dirty_wp_bury(ReimsVgpuDirty *d)
{
    GPtrArray *dead;
    guint k, i;

    qemu_mutex_lock(&d->lock);
    if (!d->wp_dead->len) {
        qemu_mutex_unlock(&d->lock);
        return;
    }
    dead = d->wp_dead;
    d->wp_dead = g_ptr_array_new();
    qemu_mutex_unlock(&d->lock);
    for (k = 0; k < dead->len; k++) {
        ReimsVgpuDirtySet *ds = g_ptr_array_index(dead, k);

        for (i = 0; i < d->wp_list->len; i++) {
            if (g_array_index(d->wp_list, uint64_t, i) == ds->token) {
                g_array_remove_index_fast(d->wp_list, i);
                break;
            }
        }
        if (d->wp_cur == ds->token) {
            d->wp_cur = 0;
        }
        reims_vgpu_dirty_wp_count(d, ds, false);   /* TEMP lab DIRTY_WP_OVERLAP: before its runs go back */
        reims_vgpu_dirty_wp_orphan(d, ds);
        reims_vgpu_dirty_wp_holes_clear(d, ds);   /* TEMP lab DIRTY_WP_HOLE */
        call_rcu(ds, reims_vgpu_dirty_set_free_rcu, rcu);
    }
    g_ptr_array_free(dead, TRUE);
}

/*
 * Lab DIRTY_WP_CALM: give `wp_mutex` up for a store fault waiting on it (the lock lets its holder
 * take it straight back, so a plain unlock and lock lets nobody in). `wp_mutex` held.
 */
static void reims_vgpu_dirty_wp_yield(ReimsVgpuDirty *d)
{
    int spins;

    if (!qatomic_read(&d->wp_waiters)) {
        return;
    }
    d->lab_wp_yields++;
    qemu_mutex_unlock(&d->wp_mutex);
    for (spins = 0; qatomic_read(&d->wp_waiters) && spins < 200; spins++) {
        g_usleep(10);
    }
    reims_wp_lock(d, __LINE__);
}

static size_t reims_vgpu_dirty_wp_release_merged(ReimsVgpuDirty *d, const ReimsVgpuDirtyRun *runs,
                                                 size_t nr, int *budget);
static void reims_vgpu_dirty_wp_set_runs(ReimsVgpuDirty *d, const ReimsVgpuDirtySet *s, size_t p);
static void reims_vgpu_dirty_wp_release_orphans(ReimsVgpuDirty *d, int *budget);

/* TEMP lab DIRTY_WP_MERGE, its live value when a live file names it. */
static bool reims_vgpu_dirty_wp_merge_on(ReimsVgpuDirty *d)
{
    int v = reims_vgpu_lab_live("DIRTY_WP_MERGE");

    return v < 0 ? d->wp_merge : v;
}

/* TEMP lab DIRTY_WP_MERGE: spans one store fault gives back before the step takes over. */
#define REIMS_WP_MERGE_FAULT_SPANS 64
#define REIMS_WP_FAIR_FAULT_SPANS 8   /* TEMP lab DIRTY_WP_FAIR: ~1 ms a remap at a transition */

/* TEMP lab DIRTY_WP_FAIR: this thread runs the WP step (a give-back here may stop for a waiting fault). */
static __thread bool reims_wp_in_step;

/*
 * vCPU thread, a store exit at `gpa` (whpx_handle_mmio). The set holding the page stops being
 * protected (reads stop trusting it before anything is writable); only the page is made writable
 * here, the step function does the rest of the set without a pause. Under DIRTY_WP_MERGE the fault
 * gives the dropped set (or, for a page no set holds, the orphans) back itself, merged.
 */
static bool reims_vgpu_dirty_wp_fault(uint64_t gpa)
{
    ReimsVgpuDirty *d = reims_vgpu_dirty_wp_dev;
    ReimsVgpuDirtySet *dropped = NULL;   /* TEMP lab DIRTY_WP_MERGE */
    int64_t t_wait, t_held;
    guint i;

    if (!d || !qatomic_read(&d->wp_bits) || !reims_vgpu_dirty_wp_ro(d, gpa)) {
        return false;
    }
    t_wait = get_clock();
    reims_wp_cause = WP_CAUSE_FAULT;   /* TEMP lab (b454) */
    qatomic_inc(&d->wp_waiters);
    reims_wp_lock(d, __LINE__);
    qatomic_dec(&d->wp_waiters);
    t_held = get_clock();
    if (reims_vgpu_dirty_wp_ro(d, gpa)) {
        for (i = 0; i < d->wp_list->len; i++) {
            uint64_t token = g_array_index(d->wp_list, uint64_t, i);
            ReimsVgpuDirtySet *s;
            bool hit;

            qemu_mutex_lock(&d->lock);
            s = g_hash_table_lookup(d->sets, &token);
            hit = s && (s->wp_state == WP_PROTECTING || s->wp_state == WP_ON ||
                        s->wp_state == WP_HOLED) &&
                  reims_vgpu_dirty_wp_has(s, gpa);
            if (hit && d->wp_hole && s->wp_holes >= d->wp_hole_max) {
                d->lab_wp_hole_drops++;   /* TEMP lab DIRTY_WP_HOLE: too many; give it all back */
            }
            if (hit && d->wp_hole && s->wp_holes < d->wp_hole_max) {
                /*
                 * TEMP lab DIRTY_WP_HOLE: reads stop trusting the set before the page is writable
                 * (the remap below), and sync again; their queries take the holes only. A protect
                 * under way stops at its next run; the runs it did stay read-only.
                 */
                if (s->wp_state == WP_ON) {
                    qatomic_store_release(&s->synced_epoch, 0);
                }
                if (s->wp_state != WP_HOLED) {
                    d->lab_wp_holed++;
                }
                s->wp_holes++;
                s->wp_state = WP_HOLED;
                if (d->wp_hole_bits && (gpa >> qemu_target_page_bits()) < d->wp_bits_n) {
                    set_bit_atomic(gpa >> qemu_target_page_bits(), d->wp_hole_bits);
                }
                s->wp_streak = 0;
                s->wp_lost++;
                d->lab_wp_lost++;
                if (d->wp_cur == token) {
                    d->wp_cur = 0;
                }
            } else if (hit) {
                if (s->wp_state == WP_ON) {
                    qatomic_store_release(&s->synced_epoch, 0);
                }
                s->wp_state = WP_DROPPING;
                reims_vgpu_dirty_wp_count(d, s, false);   /* TEMP lab DIRTY_WP_OVERLAP */
                s->wp_next = 0;
                s->wp_streak = 0;
                s->wp_lost++;
                d->lab_wp_lost++;
                if (d->wp_cur == token) {
                    d->wp_cur = 0;
                }
                dropped = s;   /* alive while `wp_mutex` is held: only the step frees listed sets */
            }
            qemu_mutex_unlock(&d->lock);
            if (hit && !d->wp_hole && !d->wp_refs) {
                /*
                 * Protected sets never share a page without holes - unless DIRTY_WP_OVERLAP lets them,
                 * and then every set holding the page has to stop trusting it before it is writable.
                 */
                break;
            }
        }
        /*
         * Only the page, also under DIRTY_WP_CALM: a whole run made writable here would open pages a
         * concurrent sync is leaving out of its query as read-only (RO_SKIP) while the guest is
         * storing into that very region — the stress gate's wp_hole rows lost writes that way.
         * DIRTY_WP_MERGE gives more back only under a query ticket with marks, which closes that.
         */
        if (reims_vgpu_dirty_wp_merge_on(d) && !d->wp_hole) {
            int budget = (d->wp_fair & 4) ? REIMS_WP_FAIR_FAULT_SPANS : REIMS_WP_MERGE_FAULT_SPANS;

            d->lab_wp_merge_fault++;
            for (i = 0; i < d->wp_list->len && !dropped; i++) {
                /* A set an earlier fault began dropping, with runs still read-only. */
                uint64_t token = g_array_index(d->wp_list, uint64_t, i);
                ReimsVgpuDirtySet *s;

                qemu_mutex_lock(&d->lock);
                s = g_hash_table_lookup(d->sets, &token);
                if (s && s->wp_state == WP_DROPPING && reims_vgpu_dirty_wp_has(s, gpa)) {
                    dropped = s;
                }
                qemu_mutex_unlock(&d->lock);
            }
            if (dropped) {
                reims_vgpu_dirty_wp_set_runs(d, dropped, 0);
                reims_vgpu_dirty_wp_release_merged(d, &g_array_index(d->wp_merge_runs,
                                                                     ReimsVgpuDirtyRun, 0),
                                                   d->wp_merge_runs->len, &budget);
            } else {
                if (d->wp_calm) {
                    reims_vgpu_dirty_wp_bury(d);   /* a set untracked since the last step */
                }
                if (d->wp_orphan_calm) {
                    /*
                     * TEMP lab DIRTY_WP_ORPHAN_CALM: the orphan run holding the page, not the first runs of the
                     * list - with the orphans held, a guest filling a freed surface again would otherwise fault
                     * page by page.
                     */
                    guint k;

                    for (k = 0; k < d->wp_orphans->len; k++) {
                        ReimsVgpuDirtyRun r = g_array_index(d->wp_orphans, ReimsVgpuDirtyRun, k);

                        if (gpa >= r.gpa && gpa < r.gpa + r.size) {
                            g_array_remove_index_fast(d->wp_orphans, k);
                            if (d->wp_orphan_near) {
                                /*
                                 * TEMP lab DIRTY_WP_ORPHAN_NEAR_KB: the orphan runs within that distance go with
                                 * it, merged - a freed surface's pages are scattered runs of ~4 pages, and the
                                 * guest filling it again faulted once a run (b457 C1: 585-922 faults a report).
                                 */
                                uint64_t lo = r.gpa > d->wp_orphan_near ? r.gpa - d->wp_orphan_near : 0;
                                uint64_t hi = r.gpa + r.size + d->wp_orphan_near;
                                GArray *nearby = g_array_new(FALSE, FALSE, sizeof(ReimsVgpuDirtyRun));
                                size_t used;

                                g_array_append_val(nearby, r);
                                for (guint j = d->wp_orphans->len; j-- > 0;) {
                                    ReimsVgpuDirtyRun o = g_array_index(d->wp_orphans, ReimsVgpuDirtyRun, j);

                                    if (o.gpa < hi && o.gpa + o.size > lo) {
                                        g_array_append_val(nearby, o);
                                        g_array_remove_index_fast(d->wp_orphans, j);
                                    }
                                }
                                g_array_sort(nearby, reims_vgpu_dirty_cmp_run);
                                used = reims_vgpu_dirty_wp_release_merged(d, &g_array_index(nearby, ReimsVgpuDirtyRun, 0),
                                                                          nearby->len, &budget);
                                if (used < nearby->len) {   /* the budget ran out: the rest stay orphans */
                                    g_array_append_vals(d->wp_orphans, &g_array_index(nearby, ReimsVgpuDirtyRun, used),
                                                        nearby->len - used);
                                }
                                d->lab_wp_near_runs += nearby->len;
                                g_array_free(nearby, TRUE);
                            } else {
                                reims_vgpu_dirty_wp_release_merged(d, &r, 1, &budget);
                            }
                            break;
                        }
                    }
                } else {
                    reims_vgpu_dirty_wp_release_orphans(d, &budget);
                }
            }
        }
        if (reims_vgpu_dirty_wp_ro(d, gpa)) {
            reims_vgpu_dirty_wp_remap(d, gpa & ~(uint64_t)4095, 4096, true);
        }
    }
    d->lab_wp_faults++;
    {   /* TEMP lab DIRTY_WP_FAIR: how long the faulting vCPU stood still, waiting and inside */
        int64_t now = get_clock();
        uint64_t w = t_held - t_wait, h = now - t_held;

        d->lab_wp_fwait_ns += w;
        d->lab_wp_fwait_max_ns = MAX(d->lab_wp_fwait_max_ns, w);
        d->lab_wp_fhold_ns += h;
        d->lab_wp_fhold_max_ns = MAX(d->lab_wp_fhold_max_ns, h);
        d->lab_wp_fslow += w + h >= 4 * SCALE_MS;
    }
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
    reims_wp_lock(d, __LINE__);
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
            reims_vgpu_dirty_wp_count(d, s, false);   /* TEMP lab DIRTY_WP_OVERLAP */
        }
        qemu_mutex_unlock(&d->lock);
        if (s && d->wp_bits) {
            reims_vgpu_dirty_wp_orphan(d, s);
            reims_vgpu_dirty_wp_holes_clear(d, s);   /* TEMP lab DIRTY_WP_HOLE */
        }
        if (s) {
            qemu_mutex_lock(&d->lock);
            s->wp_listed = false;   /* lab DIRTY_WP_CALM: an untrack may free it from here on */
            qemu_mutex_unlock(&d->lock);
        }
    }
    g_array_set_size(d->wp_list, 0);
    d->wp_cur = 0;
    if (d->wp_bridged && d->wp_bridged->len) {
        /* TEMP lab DIRTY_WP_BRIDGE: the bridged gaps belong to no listed set; give them back too. */
        g_array_append_vals(d->wp_orphans, d->wp_bridged->data, d->wp_bridged->len);
        g_array_set_size(d->wp_bridged, 0);
    }
    qemu_mutex_unlock(&d->wp_mutex);
    REIMS_WP_BH_SCHEDULE(d->wp_bh);
}

/*
 * TEMP lab DIRTY_WP_THREAD (pick by heat): the queued set worth most to protect — the longest clean
 * streak times its pages, i.e. the queries protecting it saves — rather than the last one queued.
 * One pass drops what the step cannot take (gone, not WP_NONE, unarmed, past the bitmap) and the
 * duplicates WP_RETRY leaves; only the winner pays the read-only walk, and a set that shares a
 * read-only page goes back for a later pass. n151: the LIFO step spent its whole duty on
 * short-lived sets while the hottest ones (clean for 1 400-3 600 syncs) waited.
 */
/*
 * TEMP lab (b357): the layout of a big set taken for protection - its runs, the gaps between them by
 * size, how many gap pages other sets track or hold read-only, and how many remaps a protect would
 * take if gaps up to 16 / 64 / 256 pages were bridged (all of them, or only untracked writable ones).
 * `lock` and `wp_mutex` held.
 */
static void reims_vgpu_dirty_wp_layout(ReimsVgpuDirty *d, const ReimsVgpuDirtySet *s)
{
    static int64_t sec_ns;
    static unsigned in_sec;
    const int shift = qemu_target_page_bits();
    const uint64_t per = s->page_size >> shift;
    uint64_t gb[5] = { 0 }, gp[5] = { 0 }, gt[5] = { 0 }, gr[5] = { 0 };
    uint64_t c16 = 1, c64 = 1, c256 = 1, u16 = 1, u64 = 1, u256 = 1, prev_end, span;
    int64_t now = get_clock();
    size_t p;

    if (!d->wp_layout || !d->wp_bits || (uint64_t)s->count * per < 1024) {
        return;
    }
    if (now - sec_ns > NANOSECONDS_PER_SECOND) {
        sec_ns = now;
        in_sec = 0;
    }
    if (++in_sec > 20) {
        return;
    }
    WITH_RCU_READ_LOCK_GUARD() {
        const ReimsVgpuDirtyTrk *t = qatomic_rcu_read(&d->trk);

        prev_end = (s->pages[0] >> shift) + per;
        for (p = 1; p < s->count; p++) {
            uint64_t pfn = s->pages[p] >> shift, gap, q, tn = 0, rn = 0;
            int b;
            bool clean;

            if (pfn <= prev_end) {
                prev_end = MAX(prev_end, pfn + per);
                continue;
            }
            gap = pfn - prev_end;
            b = gap <= 16 ? 0 : gap <= 64 ? 1 : gap <= 256 ? 2 : gap <= 1024 ? 3 : 4;
            if (gap <= 1024) {
                for (q = prev_end; q < pfn; q++) {
                    tn += t && q < t->n && qatomic_read(&t->c[q]) != 0;
                    rn += q < d->wp_bits_n && test_bit(q, d->wp_bits);
                }
            }
            gb[b]++;
            gp[b] += gap;
            gt[b] += tn;
            gr[b] += rn;
            clean = tn == 0 && rn == 0;
            c16 += gap > 16;
            c64 += gap > 64;
            c256 += gap > 256;
            u16 += !(clean && gap <= 16);
            u64 += !(clean && gap <= 64);
            u256 += !(clean && gap <= 256);
            prev_end = pfn + per;
        }
    }
    span = prev_end - (s->pages[0] >> shift);
    fprintf(stderr, "reims-dirty-wp-layout pages=%" PRIu64 " runs=%u span=%" PRIu64 " streak=%u"
            " gaps=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64
            " gpages=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64
            " gtrk=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64
            " gro=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64
            " calls=%" PRIu64 "/%" PRIu64 "/%" PRIu64 " ucalls=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "\n",
            (uint64_t)s->count * per, s->nruns, span, s->wp_streak,
            gb[0], gb[1], gb[2], gb[3], gb[4], gp[0], gp[1], gp[2], gp[3],
            gt[0], gt[1], gt[2], gt[3], gr[0], gr[1], gr[2], gr[3], c16, c64, c256, u16, u64, u256);
}

static bool reims_vgpu_dirty_wp_pick_hot(ReimsVgpuDirty *d)
{
    while (!d->wp_cur && d->wp_queue->len) {
        guint i, keep = 0, best = G_MAXUINT;
        uint64_t best_score = 0;
        ReimsVgpuDirtySet *c;

        for (i = 0; i < d->wp_queue->len; i++) {
            uint64_t token = g_array_index(d->wp_queue, uint64_t, i);
            guint j;
            bool dup = false;

            c = g_hash_table_lookup(d->sets, &token);
            if (!c || !c->count ||
                (c->wp_state != WP_NONE && !(d->wp_hole && c->wp_state == WP_HOLED))) {
                d->lab_wp_rej_state++;
                continue;
            }
            if (c->gen == 0) {
                d->lab_wp_rej_gen++;
                continue;
            }
            if (((c->pages[c->count - 1] + c->page_size - 1) >> qemu_target_page_bits()) >=
                d->wp_bits_n) {
                d->lab_wp_rej_bits++;
                continue;
            }
            for (j = 0; j < keep; j++) {
                if (g_array_index(d->wp_queue, uint64_t, j) == token) {
                    dup = true;
                    break;
                }
            }
            if (dup) {
                continue;
            }
            g_array_index(d->wp_queue, uint64_t, keep) = token;
            {
                uint64_t score = (uint64_t)c->wp_streak * c->count *
                                 (c->page_size >> qemu_target_page_bits());

                if (best == G_MAXUINT || score > best_score) {
                    best = keep;
                    best_score = score;
                }
            }
            keep++;
        }
        g_array_set_size(d->wp_queue, keep);
        if (best == G_MAXUINT) {
            break;
        }
        {
            uint64_t token = g_array_index(d->wp_queue, uint64_t, best);

            g_array_remove_index_fast(d->wp_queue, best);
            c = g_hash_table_lookup(d->sets, &token);
            if (c->wp_state == WP_HOLED) {
                /* TEMP lab DIRTY_WP_HOLE: a heal; the set is on the list already. */
                c->wp_state = WP_PROTECTING;
                c->wp_next = 0;
                d->wp_cur = token;
                d->lab_wp_heals++;
                continue;
            }
            if (!d->wp_refs && (d->wp_hole ? reims_vgpu_dirty_wp_any_held(d, c)
                                            : reims_vgpu_dirty_wp_any_ro(d, c, 0, c->count))) {
                d->lab_wp_rej_ro++;
                continue;   /* WP_RETRY queues it again later */
            }
            c->wp_state = WP_PROTECTING;
            c->wp_next = 0;
            c->wp_listed = true;
            reims_vgpu_dirty_wp_count(d, c, true);   /* TEMP lab DIRTY_WP_OVERLAP */
            d->wp_cur = token;
            g_array_append_val(d->wp_list, token);
            reims_vgpu_dirty_wp_layout(d, c);   /* TEMP lab (b357) */
        }
    }
    return d->wp_cur != 0;
}

/* Take the next eligible queued set as the one being protected. `wp_mutex` and `lock` held. */
static bool reims_vgpu_dirty_wp_pick(ReimsVgpuDirty *d)
{
    if (d->wp_thread_mode) {
        return reims_vgpu_dirty_wp_pick_hot(d);
    }
    while (!d->wp_cur && d->wp_queue->len) {
        uint64_t token = g_array_index(d->wp_queue, uint64_t, d->wp_queue->len - 1);
        ReimsVgpuDirtySet *c = g_hash_table_lookup(d->sets, &token);

        g_array_set_size(d->wp_queue, d->wp_queue->len - 1);
        if (c && c->count && d->wp_hole && c->wp_state == WP_HOLED) {
            /* TEMP lab DIRTY_WP_HOLE: a heal; the set is on the list already. */
            c->wp_state = WP_PROTECTING;
            c->wp_next = 0;
            d->wp_cur = token;
            d->lab_wp_heals++;
        } else if (!c || c->wp_state != WP_NONE || !c->count) {
            d->lab_wp_rej_state++;
        } else if (c->gen == 0) {
            d->lab_wp_rej_gen++;
        } else if (((c->pages[c->count - 1] + c->page_size - 1) >> qemu_target_page_bits()) >=
                   d->wp_bits_n) {
            d->lab_wp_rej_bits++;
        } else if (!d->wp_refs && (d->wp_hole ? reims_vgpu_dirty_wp_any_held(d, c)
                                               : reims_vgpu_dirty_wp_any_ro(d, c, 0, c->count))) {
            d->lab_wp_rej_ro++;
        } else {
            c->wp_state = WP_PROTECTING;
            c->wp_next = 0;
            c->wp_listed = true;
            reims_vgpu_dirty_wp_count(d, c, true);   /* TEMP lab DIRTY_WP_OVERLAP */
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
    reims_vgpu_dirty_sync_one(d, token, true, 0, NULL, NULL);   /* 0: an epoch from now on */
    qemu_mutex_lock(&d->lock);
    if (s->wp_state == WP_PROTECTING && s->claim_epoch == 0 && s->synced_epoch >= fin_epoch) {
        s->wp_state = WP_ON;
        s->wp_holes = 0;
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
 * TEMP lab DIRTY_WP_MERGE: make the read-only target pages of `runs` (ascending gpa) writable in as
 * few remaps as possible, at most `*budget` remaps (each counted against it). Returns how many runs
 * were consumed.
 *
 * A span starts at a run's first read-only page and takes the next run in while the gap up to it is
 * at most `wp_merge_gap` pages, holds no read-only page (another set's protection stays whole) and
 * the span stays inside one RAM slice (one remap is one memory region). The gap pages and the run
 * pages a fault already gave back are writable and tracked; the remap drops their dirty bits as the
 * protect's RW->RO remap does, so each span is remapped under a query ticket and marked written in
 * the VGA bitmap after the remap, exactly as protect_chunk does under DIRTY_WP_NOPAUSE: a sync that
 * settles after the ticket began waits for the marks, and a mark covers every store before the
 * remap. That is also why a fault may give a whole set back here, where the plain one-page remap of
 * the fault hook could not (the stress gate lost writes to RO_SKIP that way).
 *
 * `wp_mutex` held, `lock` not.
 */
static size_t reims_vgpu_dirty_wp_release_merged(ReimsVgpuDirty *d, const ReimsVgpuDirtyRun *runs,
                                                 size_t nr, int *budget)
{
    const uint64_t pg = 1ULL << qemu_target_page_bits();
    const uint64_t gap_max = (uint64_t)d->wp_merge_gap << qemu_target_page_bits();
    int n = reims_vgpu_dirty_ram_slices_into(&d->wp_merge_slices, &d->wp_merge_slices_cap);
    size_t i = 0;

    while (i < nr && *budget > 0) {
        uint64_t start = runs[i].gpa, end = runs[i].gpa + runs[i].size, slice_end, a, pt;
        size_t i0 = i, k;
        int si;

        if ((d->wp_fair & 1) && reims_wp_in_step && i && qatomic_read(&d->wp_waiters)) {
            /*
             * TEMP lab DIRTY_WP_FAIR: a fault waits; stop here (the caller consumes what was done and
             * steps aside). Not a yield in here: `runs` points into a list the fault may change.
             */
            break;
        }

        if (d->wp_refs && reims_vgpu_dirty_wp_run_kept(d, start, end)) {
            /* TEMP lab DIRTY_WP_OVERLAP: another set holds part of it; page by page. */
            reims_vgpu_dirty_wp_release(d, start, end, budget);
            i++;
            continue;
        }
        while (start < end && !reims_vgpu_dirty_wp_ro(d, start)) {
            start += pg;
        }
        si = start < end ? reims_vgpu_dirty_slice_of(d->wp_merge_slices, n, start) : -1;
        i++;
        if (si < 0) {
            continue;   /* nothing read-only in the run (or not RAM: the plain release skips it too) */
        }
        slice_end = d->wp_merge_slices[si].gpa + d->wp_merge_slices[si].len;
        end = MIN(end, slice_end);
        while (i < nr) {
            uint64_t g = runs[i].gpa, g_end = runs[i].gpa + runs[i].size;

            if (g_end > slice_end || (g > end && g - end > gap_max)) {
                break;
            }
            if (d->wp_refs && reims_vgpu_dirty_wp_run_kept(d, g, g_end)) {
                break;   /* TEMP lab DIRTY_WP_OVERLAP: that run goes page by page */
            }
            for (a = end; a < g && !reims_vgpu_dirty_wp_ro(d, a); a += pg) {
            }
            if (a < g) {
                break;   /* a read-only page of another set lies in the gap */
            }
            end = MAX(end, g_end);
            i++;
            d->lab_wp_merge_runs++;
        }
        qemu_mutex_lock(&d->lock);
        pt = reims_vgpu_dirty_query_begin(d);
        d->wp_tickets_open++;   /* TEMP lab DIRTY_SETTLE_SKIP */
        qemu_mutex_unlock(&d->lock);
        reims_vgpu_dirty_wp_remap(d, start, end - start, true);
        for (k = i0; k < i; k++) {
            /* The runs only: a gap page stays RW->RW, which keeps its dirty bits (whpover.c). */
            uint64_t ms = MAX(runs[k].gpa, start), me = MIN(runs[k].gpa + runs[k].size, end);

            if (ms < me) {
                reims_vgpu_dirty_wp_mark(d->wp_merge_slices, n, ms, me);
                d->lab_wp_merge_pages += (me - ms) / pg;
            }
        }
        qemu_mutex_lock(&d->lock);
        reims_vgpu_dirty_query_end(d, pt);
        d->wp_tickets_open--;   /* TEMP lab DIRTY_SETTLE_SKIP */
        qemu_mutex_unlock(&d->lock);
        d->lab_wp_merge_spans++;
        d->lab_wp_merge_runs++;
        (*budget)--;
    }
    return i;
}

/*
 * TEMP lab DIRTY_WP_BRIDGE: the run of `s` at page index `p` ends at `*end`, its next run at index `q`. Takes the
 * following runs into one remap while each gap up to the next run is at most `wp_merge_gap` pages, stays inside
 * the RAM slice, and holds only pages that are read-only already or held by hashed sets alone (none by a logged
 * set, none by no set). `wp_bridge_marks` gets what the protect marks written after the remap: the runs and the gap
 * pages that were writable (a read-only one cannot have been stored to, and marking it would read as a write);
 * `wp_bridged` the gaps made read-only. Returns the index after the last run taken. `wp_mutex` held, `lock` not.
 */
static size_t reims_vgpu_dirty_wp_bridge(ReimsVgpuDirty *d, const ReimsVgpuDirtySet *s, size_t p, size_t q,
                                         uint64_t *end, int n)
{
    const int shift = qemu_target_page_bits();
    const uint64_t pg = 1ULL << shift;
    const uint64_t gap_max = (uint64_t)d->wp_merge_gap << shift;
    ReimsVgpuDirtyRun r = { .gpa = s->pages[p], .size = *end - s->pages[p] };
    uint64_t slice_end;
    size_t taken = 0;
    int si;

    g_array_set_size(d->wp_bridge_marks, 0);
    g_array_append_val(d->wp_bridge_marks, r);
    si = reims_vgpu_dirty_slice_of(d->wp_slices, n, s->pages[p]);
    if (si < 0) {
        return q;
    }
    slice_end = d->wp_slices[si].gpa + d->wp_slices[si].len;
    while (q < s->count) {
        uint64_t g = s->pages[q], e2, a, nro = 0, nh = 0;
        size_t q2;
        bool ok = true;

        if (g - *end > gap_max) {
            d->lab_wp_br_stop_far++;
            break;
        }
        q2 = reims_vgpu_dirty_wp_run(s, q, &e2);
        if (e2 > slice_end) {
            d->lab_wp_br_stop_far++;
            break;
        }
        for (a = *end; a < g; a += pg) {
            uint64_t pfn = a >> shift;

            if (test_bit(pfn, d->wp_bits)) {
                nro++;
            } else if (qatomic_read(&d->wp_lc[pfn])) {
                d->lab_wp_br_stop_log++;
                ok = false;
                break;
            } else if (!qatomic_read(&d->wp_hc[pfn])) {
                d->lab_wp_br_stop_free++;
                ok = false;
                break;
            } else {
                nh++;
            }
        }
        if (!ok) {
            break;
        }
        for (a = *end; a < g;) {
            uint64_t from;

            if (test_bit(a >> shift, d->wp_bits)) {
                a += pg;
                continue;
            }
            for (from = a; a < g && !test_bit(a >> shift, d->wp_bits); a += pg) {
            }
            r.gpa = from;
            r.size = a - from;
            g_array_append_val(d->wp_bridge_marks, r);
            g_array_append_val(d->wp_bridged, r);
        }
        r.gpa = g;
        r.size = e2 - g;
        g_array_append_val(d->wp_bridge_marks, r);
        d->lab_wp_br_runs++;
        d->lab_wp_br_ro += nro;
        d->lab_wp_br_hashed += nh;
        *end = e2;
        q = q2;
        taken++;
    }
    d->lab_wp_br_spans += taken != 0;
    return q;
}

/*
 * TEMP lab DIRTY_WP_BRIDGE: drop the bridged gaps no page of which is read-only any more, once the list has grown.
 * `wp_mutex` held.
 */
static void reims_vgpu_dirty_wp_bridged_prune(ReimsVgpuDirty *d)
{
    guint i, keep = 0;

    if (d->wp_bridged->len < 65536) {
        return;
    }
    for (i = 0; i < d->wp_bridged->len; i++) {
        ReimsVgpuDirtyRun r = g_array_index(d->wp_bridged, ReimsVgpuDirtyRun, i);
        uint64_t a;

        for (a = r.gpa; a < r.gpa + r.size; a += 1ULL << qemu_target_page_bits()) {
            if (reims_vgpu_dirty_wp_ro(d, a)) {
                g_array_index(d->wp_bridged, ReimsVgpuDirtyRun, keep++) = r;
                break;
            }
        }
    }
    g_array_set_size(d->wp_bridged, keep);
    if (keep >= 32768) {
        /* Still long: the older half goes back writable (orphans), so the next prune is far off again. */
        g_array_append_vals(d->wp_orphans, d->wp_bridged->data, keep / 2);
        g_array_remove_range(d->wp_bridged, 0, keep / 2);
    }
}

/* TEMP lab DIRTY_WP_MERGE: the runs of `s` from page index `p` on, into `wp_merge_runs`. */
static void reims_vgpu_dirty_wp_set_runs(ReimsVgpuDirty *d, const ReimsVgpuDirtySet *s, size_t p)
{
    g_array_set_size(d->wp_merge_runs, 0);
    while (p < s->count) {
        uint64_t end;
        size_t q = reims_vgpu_dirty_wp_run(s, p, &end);
        ReimsVgpuDirtyRun r = { .gpa = s->pages[p], .size = end - s->pages[p] };

        g_array_append_val(d->wp_merge_runs, r);
        p = q;
    }
}

static int reims_vgpu_dirty_cmp_run(gconstpointer a, gconstpointer b)
{
    const ReimsVgpuDirtyRun *x = a, *y = b;

    return x->gpa < y->gpa ? -1 : x->gpa > y->gpa;
}

/*
 * TEMP lab DIRTY_WP_MERGE: give the orphan runs back, merged, at most `*budget` remaps. The list is
 * sorted first (it is appended set by set). `wp_mutex` held, `lock` not.
 */
static void reims_vgpu_dirty_wp_release_orphans(ReimsVgpuDirty *d, int *budget)
{
    size_t used;

    if (!d->wp_orphans->len) {
        return;
    }
    g_array_sort(d->wp_orphans, reims_vgpu_dirty_cmp_run);
    used = reims_vgpu_dirty_wp_release_merged(d, &g_array_index(d->wp_orphans, ReimsVgpuDirtyRun, 0),
                                              d->wp_orphans->len, budget);
    g_array_remove_range(d->wp_orphans, 0, used);
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
    reims_wp_cause = WP_CAUSE_PROTECT;   /* TEMP lab (b454) */
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
    reims_wp_lock(d, __LINE__);
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

        d->wp_tickets_open++;   /* TEMP lab DIRTY_SETTLE_SKIP */

        bool gone = false;

        qemu_mutex_unlock(&d->lock);
        for (p = p0, runs = 0; p < s->count && runs < budget && !rc; runs++) {
            uint64_t end;
            size_t q = reims_vgpu_dirty_wp_run(s, p, &end);

            if (d->wp_hole) {
                /*
                 * TEMP lab DIRTY_WP_HOLE: a run with no writable page needs no remap and no mark
                 * (a heal re-protects only what faults opened; a first protect finds none).
                 */
                while (reims_vgpu_dirty_wp_all_ro(d, s->pages[p], end)) {
                    d->lab_wp_heal_skipped++;
                    p = q;
                    if (p >= s->count) {
                        break;
                    }
                    q = reims_vgpu_dirty_wp_run(s, p, &end);
                }
                if (p >= s->count) {
                    break;
                }
            }
            if (d->wp_bridge && d->wp_refs && !d->wp_hole) {
                guint k;

                /* TEMP lab DIRTY_WP_BRIDGE: the next runs too, over gaps that are read-only or hashed-only. */
                q = reims_vgpu_dirty_wp_bridge(d, s, p, q, &end, n);
                rc = reims_vgpu_dirty_wp_remap(d, s->pages[p], end - s->pages[p], false);
                for (k = 0; k < d->wp_bridge_marks->len; k++) {
                    ReimsVgpuDirtyRun *mr = &g_array_index(d->wp_bridge_marks, ReimsVgpuDirtyRun, k);

                    reims_vgpu_dirty_wp_mark(d->wp_slices, n, mr->gpa, mr->gpa + mr->size);
                }
                reims_vgpu_dirty_wp_bridged_prune(d);
            } else {
                rc = reims_vgpu_dirty_wp_remap(d, s->pages[p], end - s->pages[p], false);
                reims_vgpu_dirty_wp_mark(d->wp_slices, n, s->pages[p], end);
            }
            if (d->wp_hole_bits && !rc) {   /* TEMP lab DIRTY_WP_HOLE */
                uint64_t a;

                for (a = s->pages[p]; a < end; a += 1ULL << qemu_target_page_bits()) {
                    if ((a >> qemu_target_page_bits()) < d->wp_bits_n) {
                        clear_bit_atomic(a >> qemu_target_page_bits(), d->wp_hole_bits);
                    }
                }
            }
            p = q;
            if (d->wp_thread_mode && d->wp_run_ticket) {
                /*
                 * TEMP lab DIRTY_WP_THREAD: a ticket per run, not per chunk. The ticket makes a
                 * sync that settles after it began wait for the marks; held across a whole chunk
                 * (32 runs, ~13 ms) it held every foreground sync's settle that long. The marks of
                 * a run cover the stores before that run's remap, so its own ticket is enough.
                 */
                qemu_mutex_lock(&d->lock);
                reims_vgpu_dirty_query_end(d, pt);
                pt = (!rc && p < s->count && runs + 1 < budget) ? reims_vgpu_dirty_query_begin(d) : 0;
                if (!pt) {
                    d->wp_tickets_open--;   /* TEMP lab DIRTY_SETTLE_SKIP */
                }
                qemu_mutex_unlock(&d->lock);
            }
            if (d->wp_thread_mode && !rc && p < s->count && runs + 1 < budget) {
                /*
                 * TEMP lab DIRTY_WP_THREAD: `wp_mutex` is given up between runs, so a vCPU whose
                 * store faulted on a read-only page waits for one remap (~0.4 ms), not a whole
                 * chunk (32 runs, ~13 ms). The set is looked up again: the fault may have started
                 * dropping it (its read-only runs are then given back from the first), or an
                 * untrack freed it (its runs are on the orphan list).
                 */
                bool ok;

                if (d->wp_fair & 8) {
                    /* TEMP lab DIRTY_WP_FAIR: a plain unlock and lock lets nobody in (b452: 72 ms). */
                    reims_vgpu_dirty_wp_yield(d);
                } else {
                    qemu_mutex_unlock(&d->wp_mutex);
                    reims_wp_lock(d, __LINE__);
                }
                qemu_mutex_lock(&d->lock);
                s = g_hash_table_lookup(d->sets, &token);
                ok = s && s->wp_state == WP_PROTECTING && d->wp_cur == token;
                qemu_mutex_unlock(&d->lock);
                if (!ok) {
                    gone = true;
                    runs++;
                    break;
                }
            }
        }
        p_end = p;
        if (pt) {
            qemu_mutex_lock(&d->lock);
            reims_vgpu_dirty_query_end(d, pt);
            d->wp_tickets_open--;   /* TEMP lab DIRTY_SETTLE_SKIP */
            qemu_mutex_unlock(&d->lock);
        }
        if (gone) {
            budget -= runs;
            d->lab_wp_chunks++;
            goto done;
        }
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
        reims_vgpu_dirty_wp_count(d, s, false);   /* TEMP lab DIRTY_WP_OVERLAP */
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
 * One step: up to `wp_chunk` runs made writable again (orphans, then dropping sets), then one chunk
 * of the set being protected. Returns whether anything is left. Main loop (BH or pacing timer, BQL
 * held), or the WP thread (DIRTY_WP_THREAD, no BQL, never pausing).
 */
static bool reims_vgpu_dirty_wp_step(ReimsVgpuDirty *d)
{
    int budget = d->wp_chunk;
    bool more = false, held;
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
        if (d->wp_overlap && !d->wp_hole) {
            /* Before the bitmap is published: a pick that sees `wp_bits` sees these too. */
            d->wp_refs = g_new0(uint16_t, d->wp_bits_n);   /* TEMP lab DIRTY_WP_OVERLAP */
        }
        qatomic_set(&d->wp_bits, bitmap_new(d->wp_bits_n));
        if (d->wp_hole) {
            d->wp_hole_bits = bitmap_new(d->wp_bits_n);   /* TEMP lab DIRTY_WP_HOLE */
        }
    }
    reims_wp_lock(d, __LINE__);
    if (d->wp_calm) {
        reims_vgpu_dirty_wp_bury(d);   /* lab DIRTY_WP_CALM */
    }
    reims_wp_in_step = true;   /* TEMP lab DIRTY_WP_FAIR */
    reims_wp_cause = WP_CAUSE_ORPHAN;   /* TEMP lab (b454) */
    held = false;
    if (d->wp_orphan_calm && d->wp_orphans->len) {
        /* TEMP lab DIRTY_WP_ORPHAN_CALM: not while sets are still being freed, up to the hold limit. */
        int64_t now = get_clock();

        if (!d->wp_orphan_hold_since) {
            d->wp_orphan_hold_since = now;
        }
        held = now - qatomic_read(&d->wp_orphaned_ns) < (int64_t)d->wp_orphan_quiet_ms * SCALE_MS &&
               now - d->wp_orphan_hold_since < (int64_t)d->wp_orphan_hold_ms * SCALE_MS;
        if (held) {
            d->lab_wp_held++;
        } else {
            d->wp_orphan_hold_since = 0;
        }
    }
    if (reims_vgpu_dirty_wp_merge_on(d) && !held) {
        reims_vgpu_dirty_wp_release_orphans(d, &budget);   /* TEMP lab DIRTY_WP_MERGE */
        if (d->wp_fair & 2) {
            reims_vgpu_dirty_wp_yield(d);   /* the give-back stopped early for a waiting fault */
        }
    }
    while (!held && budget > 0 && d->wp_orphans->len) {
        ReimsVgpuDirtyRun r = g_array_index(d->wp_orphans, ReimsVgpuDirtyRun, d->wp_orphans->len - 1);

        g_array_set_size(d->wp_orphans, d->wp_orphans->len - 1);
        reims_vgpu_dirty_wp_release(d, r.gpa, r.gpa + r.size, &budget);
        if (d->wp_calm) {
            reims_vgpu_dirty_wp_yield(d);   /* a fault waits for one run, not 32 */
        }
    }
    reims_wp_cause = WP_CAUSE_DROP;   /* TEMP lab (b454) */
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
        if (reims_vgpu_dirty_wp_merge_on(d) && s->wp_next < s->count) {
            /* TEMP lab DIRTY_WP_MERGE: merged spans, no yield (a span or a few a set). */
            size_t used;

            reims_vgpu_dirty_wp_set_runs(d, s, s->wp_next);
            used = reims_vgpu_dirty_wp_release_merged(d, &g_array_index(d->wp_merge_runs,
                                                                        ReimsVgpuDirtyRun, 0),
                                                      d->wp_merge_runs->len, &budget);
            while (used-- && s->wp_next < s->count) {
                uint64_t end;

                s->wp_next = reims_vgpu_dirty_wp_run(s, s->wp_next, &end);
            }
        }
        while (budget > 0 && s->wp_next < s->count) {
            uint64_t end;
            size_t q = reims_vgpu_dirty_wp_run(s, s->wp_next, &end);

            reims_vgpu_dirty_wp_release(d, s->pages[s->wp_next], end, &budget);
            s->wp_next = q;
            if (d->wp_calm && qatomic_read(&d->wp_waiters)) {
                /*
                 * Lab DIRTY_WP_CALM: let the fault in; the list may be emptied meanwhile (an
                 * invalidate) and the set then freed by an untrack, so both are looked up again.
                 */
                reims_vgpu_dirty_wp_yield(d);
                if (i >= d->wp_list->len || g_array_index(d->wp_list, uint64_t, i) != token) {
                    s = NULL;
                    break;
                }
                qemu_mutex_lock(&d->lock);
                s = g_hash_table_lookup(d->sets, &token);
                qemu_mutex_unlock(&d->lock);
                if (!s || s->wp_state != WP_DROPPING) {
                    break;
                }
            }
        }
        if (!s) {
            i = 0;   /* the list changed under the yield: look again from the start */
            continue;
        }
        if (s->wp_state != WP_DROPPING) {
            i++;
            continue;
        }
        if (s->wp_next >= s->count) {
            qemu_mutex_lock(&d->lock);
            s->wp_state = WP_NONE;
            s->wp_holes = 0;
            qemu_mutex_unlock(&d->lock);
            reims_vgpu_dirty_wp_holes_clear(d, s);   /* TEMP lab DIRTY_WP_HOLE */
            qemu_mutex_lock(&d->lock);
            s->wp_listed = false;   /* lab DIRTY_WP_CALM: not touched again below */
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
    d->wp_giving_back = more || (d->wp_orphans->len && !held);
    if (held && !more) {
        /* TEMP lab DIRTY_WP_ORPHAN_CALM: come back when the freeing may have stopped; no protect meanwhile. */
        d->wp_delay_ms = d->wp_orphan_quiet_ms;
    }
    more = more || d->wp_orphans->len;
    /*
     * The duty gate is checked here and not only in the pacing timer: the step
     * is also scheduled straight away by faults, untracks and a set reaching its
     * streak, and those runs used to protect another chunk regardless — up to
     * 715 ms of vCPU pause in one second on full-screen transitions even with
     * the delay rounded correctly (qemu-log 20261002-205648).
     */
    if (!more && budget > 0 && d->wp_mode && qatomic_read(&d->wp_active) &&
        get_clock() < d->wp_next_ok_ns) {
        int64_t wait_ns = d->wp_next_ok_ns - get_clock();

        d->wp_delay_ms = wait_ns > 0 ? ((uint64_t)wait_ns + 999999) / 1000000 : 0;
        more = true;
        budget = 0;
    }
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

            /*
             * Keep the guest paused at most `wp_duty` percent of the time. From
             * nanoseconds and rounded up: whole milliseconds truncated every
             * pause under 1 ms to no delay at all, and with 4-run chunks (~1-3 ms
             * pauses) the guest sat paused 30-100 % of a second on full-screen
             * transitions (qemu-log 20261002-204324: pause_us 1001 ms in one s).
             */
            d->wp_delay_ms = paused <= 0 ? 0 :
                ((uint64_t)paused * (100 - d->wp_duty) / d->wp_duty + 999999) / 1000000;
            d->wp_next_ok_ns = get_clock() +
                (paused <= 0 ? 0 : (int64_t)((uint64_t)paused * (100 - d->wp_duty) / d->wp_duty));
            reims_wp_lock(d, __LINE__);
        } else if (s) {
            reims_vgpu_dirty_wp_finish(d, d->wp_cur, s);
        }
        qemu_mutex_lock(&d->lock);
        more = d->wp_cur || d->wp_queue->len;
        qemu_mutex_unlock(&d->lock);
    }
    reims_wp_in_step = false;
    reims_wp_cause = WP_CAUSE_OTHER;
    qemu_mutex_unlock(&d->wp_mutex);
    return more;
}

/*
 * TEMP lab (b461 freeze watch): every 500 ms, whether `wp_mutex` has been held more than a second - by which line and
 * thread, doing what (reims_wp_stage) - and how many store faults wait. A frozen guest whose vCPU is parked in the
 * store-fault hook then names the lock it is parked behind.
 */
static void *reims_vgpu_dirty_wp_watch_fn(void *opaque)
{
    ReimsVgpuDirty *d = opaque;
    int64_t last_report = 0;

    for (;;) {
        int64_t now, held;

        g_usleep(500 * 1000);
        if (qemu_mutex_trylock(&d->wp_mutex) == 0) {
            qemu_mutex_unlock(&d->wp_mutex);
            continue;
        }
        now = get_clock();
        held = now - qatomic_read(&reims_wp_holder_ns);
        if (held >= NANOSECONDS_PER_SECOND && now - last_report >= NANOSECONDS_PER_SECOND) {
            static const char *const stage[] = { "-", "query-ticket wait", "hypervisor remap", "marks" };
            int st = qatomic_read(&reims_wp_stage);

            fprintf(stderr, "reims-dirty-wp-watch: wp_mutex held %" PRId64 " ms from line %d by thread %d (%s), "
                    "faults waiting %d, tickets in flight %d\n", held / SCALE_MS, qatomic_read(&reims_wp_holder_line),
                    qatomic_read(&reims_wp_holder_tid), stage[st < 4 && st >= 0 ? st : 0],
                    qatomic_read(&d->wp_waiters), qatomic_read(&d->q_ninflight));
            last_report = now;
        }
    }
    return NULL;
}

/* TEMP lab DIRTY_WP_THREAD: the thread that runs the protect step, woken by the BH. */
/*
 * TEMP (lab, REIMS_VGPU_LAB_DEVICE_PRIO=on): run the calling device thread above normal priority, ahead of the
 * vCPU threads. With 24 vCPUs on a 28-thread host the device's helpers compete with the guest for cores; n217's
 * heavy-glass profile had the drain waiting on them a quarter of its time. The Rust side does the same for its
 * pool, recorder, submit and stamp threads (lab_cores::device_thread_priority).
 */
void reims_vgpu_lab_device_prio(const char *who)
{
#if defined(_WIN32)
    static int on = -1;

#ifndef REIMS_DIRTY_STRESS
    whpx_lab_pin_ccd(who);   /* TEMP lab DEVICE_CCD */
#endif
    if (on < 0) {
        const char *e = getenv("REIMS_VGPU_LAB_DEVICE_PRIO");

        on = e && strcmp(e, "on") == 0;
    }
    if (on && SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL)) {
        fprintf(stderr, "reims-lab: device-prio %s" "\n", who);
    }
#else
    (void)who;
#endif
}

static void *reims_vgpu_dirty_wp_thread_fn(void *opaque)
{
    ReimsVgpuDirty *d = opaque;

    rcu_register_thread();
    reims_vgpu_lab_device_prio("wp");
    for (;;) {
        qemu_event_wait(&d->wp_ev);
        qemu_event_reset(&d->wp_ev);   /* before the step: a wake during it runs another */
        for (;;) {
            int64_t t0 = get_clock();
            bool more = reims_vgpu_dirty_wp_step(d);
            int64_t spent = get_clock() - t0;
            int64_t duty = qatomic_read(&reims_vgpu_dirty_wp_thread_duty_live) ?: d->wp_thread_duty;
            int64_t rest_us;

            if (!more) {
                break;
            }
            if (d->wp_giving_back) {
                /*
                 * Give back first: pages a guest is writing fault one by one. TEMP lab
                 * (REIMS_VGPU_LAB_DIRTY_WP_GIVEBACK_US, default 50, 2026-10-14): the rest between give-back
                 * steps. b446: the guest's audio doorbells paused 50-190 ms exactly while ~2.5k orphan
                 * remaps went back to back (1.2 s) when Safari left full screen - is it the remap burst?
                 */
                static int64_t giveback_us = -1;

                if (giveback_us < 0) {
                    const char *e = getenv("REIMS_VGPU_LAB_DIRTY_WP_GIVEBACK_US");
                    giveback_us = e ? MAX(0, atoi(e)) : 50;
                }
                g_usleep(giveback_us);
                continue;
            }
            rest_us = spent * (100 - duty) / duty / 1000;
            g_usleep(MAX(rest_us, MAX((int64_t)d->wp_delay_ms * 1000, 250)));
        }
    }
    return NULL;
}

/*
 * Main loop (BH or pacing timer), BQL held: one step and a reschedule while anything is left, or
 * (DIRTY_WP_THREAD) a wake of the thread that runs the steps.
 */
static void reims_vgpu_dirty_wp_bh(void *opaque)
{
    ReimsVgpuDirty *d = opaque;

    if (d->wp_thread_mode) {
        qemu_event_set(&d->wp_ev);
        return;
    }
    if (reims_vgpu_dirty_wp_step(d)) {
        REIMS_WP_PACE(d);
    }
}

/* TEMP lab UNTRACK_POPULATE: a thread that fills in writable entries for untracked sets' pages. */
static void *reims_vgpu_dirty_up_thread_fn(void *opaque)
{
    ReimsVgpuDirty *d = opaque;
    GArray *batch = g_array_new(FALSE, FALSE, sizeof(ReimsVgpuDirtyRun));

    rcu_register_thread();
    for (;;) {
        GArray *q;
        guint i;
        int64_t now, was;

        qemu_mutex_lock(&d->up_mutex);
        while (!d->up_queue->len) {
            qemu_cond_wait(&d->up_cond, &d->up_mutex);
        }
        q = d->up_queue;
        d->up_queue = batch;
        batch = q;
        qemu_mutex_unlock(&d->up_mutex);
        for (i = 0; i < batch->len; i += 256) {
            unsigned n = MIN(256, batch->len - i);
            uint64_t pages = 0;
            int64_t t0 = get_clock();
            guint k;
            int rc = 0;

            for (k = 0; k < n; k++) {
                pages += g_array_index(batch, ReimsVgpuDirtyRun, i + k).size >> qemu_target_page_bits();
            }
#ifndef REIMS_DIRTY_STRESS
            rc = whpx_populate_write((const uint64_t *)&g_array_index(batch, ReimsVgpuDirtyRun, i), n);
#endif
            qatomic_add(&d->lab_up_ns, (uint64_t)(get_clock() - t0));
            qatomic_inc(&d->lab_up_calls);
            qatomic_add(&d->lab_up_runs, n);
            qatomic_add(&d->lab_up_pages, pages);
            if (rc < 0) {
                qatomic_inc(&d->lab_up_fail);
            }
        }
        g_array_set_size(batch, 0);
        now = get_clock();
        was = qatomic_read(&d->lab_up_last_ns);
        if (now - was >= NANOSECONDS_PER_SECOND && qatomic_cmpxchg(&d->lab_up_last_ns, was, now) == was) {
            fprintf(stderr, "reims-dirty-up sets=%" PRIu64 " runs=%" PRIu64 " pages=%" PRIu64
                    " calls=%" PRIu64 " fail=%" PRIu64 " us=%" PRIu64 " dropped=%" PRIu64 "\n",
                    qatomic_xchg(&d->lab_up_sets, 0), qatomic_xchg(&d->lab_up_runs, 0),
                    qatomic_xchg(&d->lab_up_pages, 0), qatomic_xchg(&d->lab_up_calls, 0),
                    qatomic_xchg(&d->lab_up_fail, 0), qatomic_xchg(&d->lab_up_ns, 0) / 1000,
                    qatomic_xchg(&d->lab_up_dropped, 0));
        }
    }
    return NULL;
}

/*
 * TEMP lab UNTRACK_POPULATE: queue the writable runs of `s`, a queried set being untracked, for the
 * populate threads. `lock` held; the WP bits are read without `wp_mutex` - a page a remap turns
 * read-only meanwhile only gets its populate refused, which loses nothing.
 */
static void reims_vgpu_dirty_up_queue(ReimsVgpuDirty *d, const ReimsVgpuDirtySet *s)
{
    uint64_t pg = 1ULL << qemu_target_page_bits();
    size_t p = 0;
    guint added = 0;

    qemu_mutex_lock(&d->up_mutex);
    while (p < s->count) {
        uint64_t end, a, start;
        size_t q = reims_vgpu_dirty_wp_run(s, p, &end);

        for (a = s->pages[p]; a < end;) {
            while (a < end && reims_vgpu_dirty_wp_ro(d, a)) {
                a += pg;
            }
            start = a;
            while (a < end && !reims_vgpu_dirty_wp_ro(d, a)) {
                a += pg;
            }
            if (a > start) {
                if (d->up_queue->len >= (1u << 20)) {
                    qatomic_inc(&d->lab_up_dropped);
                } else {
                    ReimsVgpuDirtyRun r = { .gpa = start, .size = a - start };

                    g_array_append_val(d->up_queue, r);
                    added++;
                }
            }
        }
        p = q;
    }
    if (added) {
        qatomic_inc(&d->lab_up_sets);
        qemu_cond_broadcast(&d->up_cond);
    }
    qemu_mutex_unlock(&d->up_mutex);
}

void reims_vgpu_dirty_untrack(ReimsVgpuDirty *d, uint64_t token)
{
    bool bury = false;

    if (!d || token == 0) {
        return;
    }
    if (d->wp_bridge && d->wp_list && qatomic_read(&d->wp_bits)) {
        /*
         * TEMP lab DIRTY_WP_BRIDGE: an unlisted set's pages may be read-only in another set's bridged span; they
         * go back on the orphan list with it. A racy look first, so most untracks take no `wp_mutex`.
         */
        ReimsVgpuDirtySet *us;
        bool any;

        qemu_mutex_lock(&d->lock);
        us = g_hash_table_lookup(d->sets, &token);
        any = us && !us->wp_listed && us->wp_state == WP_NONE && reims_vgpu_dirty_wp_any_ro(d, us, 0, us->count);
        qemu_mutex_unlock(&d->lock);
        if (any) {
            reims_wp_lock(d, __LINE__);
            qemu_mutex_lock(&d->lock);
            us = g_hash_table_lookup(d->sets, &token);
            any = us && !us->wp_listed && us->wp_state == WP_NONE;
            qemu_mutex_unlock(&d->lock);
            if (any) {
                reims_vgpu_dirty_wp_orphan(d, us);   /* alive: only this untrack frees it */
                d->lab_wp_br_orphaned++;
                REIMS_WP_BH_SCHEDULE(d->wp_bh);
            }
            qemu_mutex_unlock(&d->wp_mutex);
        }
    }
    if (d->wp_list && !d->wp_calm) {   /* lab DIRTY_WP: its read-only runs outlive it, the step makes them writable */
        guint i;

        reims_wp_lock(d, __LINE__);
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
                    reims_vgpu_dirty_wp_count(d, ws, false);   /* TEMP lab DIRTY_WP_OVERLAP */
                    reims_vgpu_dirty_wp_orphan(d, ws);
                    reims_vgpu_dirty_wp_holes_clear(d, ws);   /* TEMP lab DIRTY_WP_HOLE */
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
    {
        /*
         * A reader may hold the set without `lock` — a lock-free generation read
         * (TEMP lab GEN_LOCKFREE), or a sync's prep reading the page list (lab
         * DIRTY_PREPLOCK) — so it is unpublished first and freed once every
         * reader that could have found it has left its read-side section.
         */
        gpointer okey, oval;

        if (g_hash_table_steal_extended(d->sets, &token, &okey, &oval)) {
            if (d->lf_mode) {
                reims_vgpu_dirty_tab_del(d, token);
            }
            reims_vgpu_dirty_trk_add(d, oval, -1);   /* TEMP lab DIRTY_TIGHT */
            reims_vgpu_dirty_wp_hold_add(d, oval, -1);   /* TEMP lab DIRTY_WP_BRIDGE */
            g_free(okey);
            d->lab_hash_sets -= ((ReimsVgpuDirtySet *)oval)->hash != NULL;   /* lab DIRTY_HASH */
            if (d->up_ready && (reims_vgpu_lab_live("UNTRACK_POPULATE") < 0 ? d->up_on
                                : reims_vgpu_lab_live("UNTRACK_POPULATE") == 1) &&
                !(((ReimsVgpuDirtySet *)oval)->hash && d->hash_mode)) {
                reims_vgpu_dirty_up_queue(d, oval);   /* TEMP lab UNTRACK_POPULATE */
            }
            if (d->wp_calm && d->wp_dead && ((ReimsVgpuDirtySet *)oval)->wp_listed) {
                /*
                 * Lab DIRTY_WP_CALM: the WP step may be holding it; it orphans the set's read-only
                 * runs and frees it. Unpublished already, so no reader or fault finds it again.
                 */
                g_ptr_array_add(d->wp_dead, oval);
                d->lab_wp_dead++;
                bury = true;
            } else {
                call_rcu((ReimsVgpuDirtySet *)oval, reims_vgpu_dirty_set_free_rcu, rcu);
            }
        }
    }
    qemu_mutex_unlock(&d->sets_lock);
    qemu_mutex_unlock(&d->lock);
    if (bury) {
        REIMS_WP_BH_SCHEDULE(d->wp_bh);
    }
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

bool reims_vgpu_dirty_tail_unnoted(ReimsVgpuDirty *d, uint32_t channel, uint32_t tail)
{
    bool unnoted;

    if (!d || d->hole_mode != 3 || channel >= ARRAY_SIZE(d->ring_tail)) {
        return false;
    }
    qemu_mutex_lock(&d->lock);
    unnoted = !d->ring_tail_valid[channel] || d->ring_tail[channel] != tail;
    qemu_mutex_unlock(&d->lock);
    return unnoted;
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
    qatomic_set(&d->work_chan, scope < ARRAY_SIZE(d->chan_epoch) && scope < 63 ? scope : 63);
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
 *
 * The value pinned is re-read at the install, not the `live` the caller once
 * read: the lock-free peek walks the token table before it gets here, and a
 * sync can evaluate the set in that window. Pinning the older generation under
 * the newer `needed` would answer every later hit of that epoch less than a
 * read had already been told mid-window, and a generation an answer went 27 ->
 * 24 (stress, ring + exact3; the committed pre-PREPLOCK tree fails the same
 * rows, so the race is the memo's own, not the prep unlock's). A straggler
 * (a newer epoch's answer already installed) answers the generation the set
 * holds now, for the same reason, and raises the pin to it: an answer above
 * the pin has been served, and no later hit may be told less than that.
 */
static void reims_vgpu_dirty_memo_raise(ReimsVgpuDirtySet *s, uint64_t ans)
{
    uint64_t m = qatomic_read(&s->memo);

    if ((ans >> 32) || ans == 0) {
        return;   /* nothing that does not fit the value half can be raised to */
    }
    for (;;) {
        uint64_t mn = m >> 32, old;

        if (m == 0 || (uint32_t)m >= ans) {
            return;
        }
        old = qatomic_cmpxchg(&s->memo, m, (mn << 32) | ans);
        if (old == m) {
            return;
        }
        m = old;   /* somebody moved it: raise against what is there now */
    }
}

static uint64_t reims_vgpu_dirty_memo_answer(ReimsVgpuDirty *d, ReimsVgpuDirtySet *s,
                                             uint64_t needed, uint64_t live)
{
    uint64_t m;

    if (!d->gen_memo_mode || live == 0 || (live >> 32) || (needed >> 32)) {
        return live;   /* "cannot tell" is never memoized; nor is what does not fit */
    }
    m = qatomic_read(&s->memo);
    for (;;) {
        uint64_t mn = m >> 32, old;

        if (m != 0 && mn == needed) {
            uint64_t memo = (uint32_t)m;

            qatomic_inc(&d->lab_memo_hit);
            if (memo != live) {
                qatomic_inc(&d->lab_memo_diff);
                if (memo > live) {
                    /* the pin leads a stale peek: a straggler raised it since */
                    qatomic_inc(&d->lab_memo_bad);
                }
            }
            return d->gen_memo_mode == 1 ? memo : live;
        }
        if (m != 0 && mn > needed) {
            uint64_t now = qatomic_read(&s->gen);   /* this read is a straggler */

            reims_vgpu_dirty_memo_raise(s, now);
            return now;   /* answer what the set holds now, never the stale `live` */
        }
        live = qatomic_read(&s->gen);
        if (live == 0 || (live >> 32)) {
            return live;   /* disarmed meanwhile, or too big to pin */
        }
        old = qatomic_cmpxchg(&s->memo, m, (needed << 32) | live);
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
/* TEMP lab DIRTY_CHAN_PREFETCH: note which channel's work read `s`. Racy by design (a hint). */
static inline void reims_vgpu_dirty_note_read_chan(ReimsVgpuDirty *d, ReimsVgpuDirtySet *s)
{
    uint64_t now, bit;

    if (!d->chan_prefetch) {
        return;
    }
    now = qatomic_read(&d->epoch);
    bit = 1ULL << MIN(qatomic_read(&d->work_chan), 63u);
    if (qatomic_read(&s->rd_chans_epoch) + d->od_hot < now) {
        qatomic_set(&s->rd_chans, bit);
        qatomic_set(&s->rd_chans_epoch, now);
    } else if (!(qatomic_read(&s->rd_chans) & bit)) {
        qatomic_or(&s->rd_chans, bit);
    }
}

static inline void reims_vgpu_dirty_note_read_epoch(ReimsVgpuDirty *d,
                                                    ReimsVgpuDirtySet *s,
                                                    uint64_t needed)
{
    uint64_t last;

    reims_vgpu_dirty_note_read_chan(d, s);
    if ((!d->period_mode && !(d->hash_due && s->hash)) || needed == 0) {
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

/*
 * Lab DIRTY_HASH_DUE: whether the background should hash `s` for `epoch`: the old window's width
 * (REIMS_VGPU_DIRTY_HASH_HOT doorbells), starting at its shortest recent read gap past its last
 * read. `-1` when it has no history (the caller keeps the window right after the read then).
 * n188: a window reaching twice the longest gap hashed rarely read sets for up to ~500 doorbells
 * in a row — twice the hashing of the old window.
 */
static int reims_vgpu_dirty_hash_due(const ReimsVgpuDirtySet *s, uint64_t epoch)
{
    uint32_t gaps = qatomic_read(&s->rd_gaps), lo = 255, g;
    uint64_t last = qatomic_read(&s->rd_epoch);
    int k;

    if (gaps == 0 || last == 0) {
        return -1;
    }
    for (k = 0; k < 4; k++) {
        g = (gaps >> (8 * k)) & 0xff;
        if (g) {
            lo = MIN(lo, g);
        }
    }
    return epoch >= last + lo && epoch < last + lo + REIMS_VGPU_DIRTY_HASH_HOT;
}

/*
 * TEMP lab DIRTY_HASH_BG_ONCE: whether `s` is in its DIRTY_HASH_DUE window for `epoch` and nothing has
 * synced it since the window opened - the one background hash a predicted read gets.
 */
static bool reims_vgpu_dirty_hash_due_open(const ReimsVgpuDirtySet *s, uint64_t epoch)
{
    uint32_t gaps = qatomic_read(&s->rd_gaps), lo = 255, g;
    uint64_t last = qatomic_read(&s->rd_epoch);
    int k;

    if (gaps == 0 || last == 0) {
        return false;
    }
    for (k = 0; k < 4; k++) {
        g = (gaps >> (8 * k)) & 0xff;
        if (g) {
            lo = MIN(lo, g);
        }
    }
    return epoch >= last + lo && epoch < last + lo + REIMS_VGPU_DIRTY_HASH_HOT &&
           qatomic_read(&s->synced_epoch) < last + lo;
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
    if ((lagged || d->lab_nosync) && qatomic_read(&d->ondemand)) {
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
        reims_vgpu_dirty_sync_one(d, token, true, needed_pre, &lw, &lq);
    }
    qemu_mutex_lock(&d->lock);
    if (stale) {
        d->lab_od_fg_n++;
        d->lab_od_fg_ns += get_clock() - fg0;
    }
    s = g_hash_table_lookup(d->sets, &token);
    if (stale && s && s->count * (s->page_size >> qemu_target_page_bits()) >= 4096) {   /* TEMP lab */
        unsigned ch = MIN(qatomic_read(&d->work_chan), 7u);

        d->lab_fgbig_n[ch]++;
        d->lab_fgbig_ns[ch] += get_clock() - fg0;
    }
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
        } else if (d->gen_memo_mode) {
            /*
             * It does not vouch for it (a write-protection fault reset the set
             * mid-read): the fresh generation goes out unmemoized, and the pin
             * must not answer a later hit below it.
             */
            reims_vgpu_dirty_memo_raise(s, gen);
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
        uint64_t ws_needed = reims_vgpu_dirty_needed_epoch(d);

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
        reims_vgpu_dirty_sync_one(d, token, true, ws_needed, &lw, &lq);
        qemu_mutex_lock(&d->lock);
        d->lab_od_fg_n++;
        d->lab_od_fg_ns += get_clock() - fg0;
        s = g_hash_table_lookup(d->sets, &token);
        if (s && s->count * (s->page_size >> qemu_target_page_bits()) >= 4096) {   /* TEMP lab */
            unsigned ch = MIN(qatomic_read(&d->work_chan), 7u);

            d->lab_fgbig_n[ch]++;
            d->lab_fgbig_ns[ch] += get_clock() - fg0;
        }
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

static void reims_vgpu_dirty_qep_free(ReimsVgpuDirtyQep *q)
{
    g_free(q);
}

/* Grow the page stamps to cover target page frames below `end_pfn`. d->lock held. */
static void reims_vgpu_dirty_page_seq_cover(ReimsVgpuDirty *d, uint64_t end_pfn)
{
    ReimsVgpuDirtyQep *oq = d->qep, *nq;
    uint64_t n;

    if (end_pfn <= d->page_seq_n) {
        return;
    }
    n = MAX(end_pfn, d->page_seq_n * 2);
    d->page_seq = g_renew(uint64_t, d->page_seq, n);
    memset(d->page_seq + d->page_seq_n, 0, (n - d->page_seq_n) * sizeof(uint64_t));
    /* Copied, not reallocated in place: a prep may be reading the old one. */
    nq = g_malloc0(sizeof(*nq) + n * sizeof(uint64_t));
    nq->n = n;
    if (oq) {
        memcpy(nq->e, oq->e, d->page_seq_n * sizeof(uint64_t));
    }
    qatomic_rcu_set(&d->qep, nq);
    if (oq) {
        call_rcu(oq, reims_vgpu_dirty_qep_free, rcu);
    }
    d->page_seq_n = n;
}

/* Stamp target page frames [pfn, end) as queried in `epoch`. d->lock held. */
static void reims_vgpu_dirty_qep_stamp(ReimsVgpuDirty *d, uint64_t pfn, uint64_t end,
                                       uint64_t epoch)
{
    uint64_t *e;

    reims_vgpu_dirty_page_seq_cover(d, end);
    e = d->qep->e;
    for (; pfn < end; pfn++) {
        if (e[pfn] < epoch) {
            qatomic_set(&e[pfn], epoch);
        }
    }
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
    bool was_armed;   /* TEMP lab DIRTY_VERIFY */
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
    /*
     * TEMP lab DIRTY_VERIFY: an armed set's stamped pages count only if their bytes moved; an unarmed set's baseline
     * takes them in. Before the generation is rewritten, so a set whose stamped pages all held their bytes keeps it.
     */
    was_armed = s->gen != 0;
    if (s->vhash && s->vhash_base && hit->len) {
        if (!was_armed) {
            reims_vgpu_dirty_verify_refresh(s, slices, n, hit);
        } else if (reims_vgpu_dirty_verify_on()) {
            any = reims_vgpu_dirty_verify_filter(d, s, slices, n, hit);
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
        (s->count * (s->page_size >> shift) >= d->wp_min_pages ||
         /*
          * TEMP lab DIRTY_WP_BRIDGE: a queried set below the minimum too, from 16 pages, while no sync has found it
          * written. b450 canvas 5K: the costly 256/57-page surfaces were queried sets (few runs, so the cost model
          * did not hash them) and never reached the 2048-page minimum - 0 clean-streak, never protected.
          */
         (d->wp_bridge && s->gen <= 1 && !s->wp_lost && s->count * (s->page_size >> shift) >= 16))) {
        /* lab DIRTY_WP */
        uint32_t need = (uint32_t)(qatomic_read(&reims_vgpu_dirty_wp_streak_live) ?: d->wp_streak_need)
                        << MIN(s->wp_lost, 6);   /* TEMP lab live DIRTY_WP_STREAK */

        need = reims_vgpu_dirty_wp_need(d, s, need);   /* TEMP lab DIRTY_WP_BRIDGE */

        s->wp_streak = any ? 0 : s->wp_streak + 1;
        /*
         * Queued when the streak reaches the need and, with WP_RETRY, again every `need` clean
         * syncs after: the step passes over a set it cannot take at that moment (its pages share a
         * read-only page with a set being protected, the bitmap is not up yet), and a set queued
         * only at `streak == need` was then never queued again — n145's two hottest foreground sets
         * were 5K surfaces clean for 5 300 syncs in a row and never protected.
         */
        if (s->wp_streak == need ||
            ((d->wp_retry || d->wp_calm) && s->wp_streak > need && (s->wp_streak - need) % need == 0)) {
            if (d->wp_calm && get_clock() - s->born_ns < reims_vgpu_dirty_wp_age_need(d, s)) {
                d->lab_wp_young++;   /* lab DIRTY_WP_CALM: asked again `need` clean syncs later */
            } else {
                if (s->wp_streak != need) {
                    d->lab_wp_requeued++;
                }
                g_array_append_val(d->wp_queue, s->token);
                REIMS_WP_BH_SCHEDULE(d->wp_bh);
            }
        }
    } else if (d->wp_hole && d->wp_active && !by_harvest && s->wp_state == WP_HOLED) {
        /* TEMP lab DIRTY_WP_HOLE: the heal's streak, doubled per fault up to 8x. */
        uint32_t need = d->wp_streak_need << MIN(s->wp_lost, 3);

        s->wp_streak = any ? 0 : s->wp_streak + 1;
        if (s->wp_streak == need || (s->wp_streak > need && (s->wp_streak - need) % need == 0)) {
            d->lab_wp_heal_queued++;
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
    uint64_t live = qatomic_read(&reims_vgpu_dirty_gap_live);

    if (live) {
        return live;   /* TEMP lab live DIRTY_GAP */
    }
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
        qatomic_set(&reims_wp_stage, 1);   /* TEMP lab (b461 freeze watch): seen only if `wp_mutex` is held too */
        qemu_cond_wait(&d->q_cond, &d->lock);
        qatomic_set(&reims_wp_stage, 0);
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
/* TEMP lab DIRTY_PSTAMP: stamp the pending piece ranges no in-flight ticket can still owe. d->lock. */
static void reims_vgpu_dirty_pstamp_pop(ReimsVgpuDirty *d)
{
    const int shift = qemu_target_page_bits();
    uint64_t low = UINT64_MAX;
    int i;

    if (!d->pstamps || d->pstamps_head >= d->pstamps->len) {
        return;
    }
    for (i = 0; i < d->q_ninflight; i++) {
        low = MIN(low, d->q_inflight[i]);
    }
    while (d->pstamps_head < d->pstamps->len) {
        const ReimsVgpuDirtyPstamp *ps = &g_array_index(d->pstamps, ReimsVgpuDirtyPstamp,
                                                        d->pstamps_head);

        guint w;

        if (ps->upto >= low) {
            break;
        }
        for (w = ps->wa; w < ps->wb; w++) {
            const MemoryRegionRange *r = &g_array_index(d->split_walked, MemoryRegionRange, w);
            uint64_t a = MAX(r->start, ps->gpa), b = MIN(r->start + r->len, ps->end);

            if (a < b) {
                reims_vgpu_dirty_qep_stamp(d, a >> shift, b >> shift, ps->epoch);
                d->lab_pstamp_pages += (b - a) >> shift;
            }
        }
        d->lab_pstamps++;
        d->pstamps_head++;
    }
    if (d->pstamps_head == d->pstamps->len) {
        g_array_set_size(d->pstamps, 0);
        d->pstamps_head = 0;
    }
}

/*
 * TEMP lab DIRTY_PSTAMP: a batch piece (`c`, region offsets) has deposited its bits; queue the
 * batch's set pages inside it for stamping once every ticket begun so far has ended. d->lock.
 */
static void reims_vgpu_dirty_pstamp_piece(ReimsVgpuDirty *d, const ReimsVgpuDirtyChunk *c)
{
    const GArray *w = d->split_walked;
    int i;

    for (i = 0; i < d->split_nslices; i++) {
        const ReimsVgpuDirtySlice *sl = &d->split_slices[i];
        uint64_t lo, hi, g0, g1;
        guint a, b;

        if (sl->mr != c->mr) {
            continue;
        }
        lo = MAX(c->r.start, sl->offset);
        hi = MIN(c->r.start + c->r.len, sl->offset + sl->len);
        if (lo >= hi) {
            continue;
        }
        g0 = sl->gpa + (lo - sl->offset);
        g1 = sl->gpa + (hi - sl->offset);
        /* The first walked run ending past g0 (runs are disjoint and in GPA order). */
        a = 0;
        b = w->len;
        while (a < b) {
            guint m = (a + b) / 2;
            const MemoryRegionRange *r = &g_array_index(w, MemoryRegionRange, m);

            if (r->start + r->len <= g0) {
                a = m + 1;
            } else {
                b = m;
            }
        }
        b = a;
        while (b < w->len && g_array_index(w, MemoryRegionRange, b).start < g1) {
            b++;
        }
        if (b > a) {
            ReimsVgpuDirtyPstamp ps = {
                .upto = d->q_next, .gpa = g0, .end = g1, .wa = a, .wb = b, .epoch = d->split_epoch,
            };

            g_array_append_val(d->pstamps, ps);
        }
    }
}

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
    reims_vgpu_dirty_pstamp_pop(d);   /* TEMP lab DIRTY_PSTAMP */
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
static bool reims_vgpu_dirty_split_pending(ReimsVgpuDirty *d)
{
    return d->fjob_open != 0 || d->split_next < d->split_total;
}

/*
 * Cut `chunks` into pieces of at most 1024 target pages into `out` (emptied
 * first), so a set of a few long runs still spreads over every helper.
 */
static void reims_vgpu_dirty_split_cut(GArray *out, const GArray *chunks, int shift)
{
    const uint64_t piece = 1024ULL << shift;
    guint k;

    g_array_set_size(out, 0);
    for (k = 0; k < chunks->len; k++) {
        ReimsVgpuDirtyChunk c = g_array_index(chunks, ReimsVgpuDirtyChunk, k);

        while (c.r.len > piece) {
            ReimsVgpuDirtyChunk head = { c.mr, { c.r.start, piece } };

            g_array_append_val(out, head);
            c.r.start += piece;
            c.r.len -= piece;
        }
        g_array_append_val(out, c);
    }
}

/*
 * The open foreground job a helper takes its next piece from: the first open
 * one at or after the round-robin cursor, which then moves past it, so that
 * concurrent foreground syncs progress side by side. `fjob_open` nonzero.
 * d->lock held.
 */
static int reims_vgpu_dirty_fjob_pick(ReimsVgpuDirty *d)
{
    uint32_t open = d->fjob_open;
    uint32_t from = open & ~((1u << d->fjob_rr) - 1);
    int k = ctz32(from ? from : open);

    d->fjob_rr = (k + 1) % REIMS_VGPU_DIRTY_FGSLOTS_MAX;
    return k;
}

/*
 * Query split pieces until none is left that this thread may take. `own` >= 0
 * is the owner of foreground job `own` (lab DIRTY_FGSLOTS): it takes only that
 * job's pieces, since a piece of another job (or of the batch's) could outlast
 * its own job and hold its sync up. `own` < 0 is a helper: one piece at a time
 * from the open foreground jobs in turn, then the batch's. d->lock held.
 */
static void reims_vgpu_dirty_split_help_from(ReimsVgpuDirty *d, int own)
{
    for (;;) {
        GArray *arr;
        guint *next, *done, total;
        ReimsVgpuDirtyChunk c;
        uint64_t pt;
        int k = -1;

        if (own >= 0) {
            if (!(d->fjob_open & (1u << own))) {
                return;
            }
            k = own;
        } else if (d->fjob_open) {
            k = reims_vgpu_dirty_fjob_pick(d);
        }
        if (k >= 0) {
            ReimsVgpuDirtySplitJob *j = &d->fjob[k];

            arr = j->chunks;
            next = &j->next;
            done = &j->done;
            total = j->total;
        } else if (d->split_next < d->split_total) {
            arr = d->split_chunks;
            next = &d->split_next;
            done = &d->split_done;
            total = d->split_total;
        } else {
            return;
        }
        /* A job is reset only after all its pieces are done, this one included. */
        c = g_array_index(arr, ReimsVgpuDirtyChunk, (*next)++);
        if (k >= 0 && *next == total) {
            d->fjob_open &= ~(1u << k);
        }
        pt = d->od_nopticket ? 0 : reims_vgpu_dirty_query_begin(d);
        reims_vgpu_dirty_query_footprint(d, pt, &c, 1);
        qemu_mutex_unlock(&d->lock);
        memory_region_sync_dirty_ranges(c.mr, &c.r, 1);
        qemu_mutex_lock(&d->lock);
        if (k < 0 && d->split_pstamp && pt) {
            reims_vgpu_dirty_pstamp_piece(d, &c);   /* TEMP lab DIRTY_PSTAMP; before its own end */
        }
        if (pt) {
            reims_vgpu_dirty_query_end(d, pt);
        }
        if (++*done == total) {
            qemu_cond_broadcast(&d->split_cond);
        }
    }
}

static void reims_vgpu_dirty_split_help(ReimsVgpuDirty *d)
{
    reims_vgpu_dirty_split_help_from(d, -1);
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
    const bool tight = reims_vgpu_dirty_tight_on(d);   /* TEMP lab DIRTY_TIGHT */
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
                        next - last - 1 > REIMS_VGPU_DIRTY_SYNC_GAP_PAGES ||
                        !reims_vgpu_dirty_bridge_ok(d, tight, last + 1, next)) {
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

            if (set->hash && d->hash_mode) {
                continue;   /* lab DIRTY_HASH: answered by its own hashes */
            }
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
    const bool tight = reims_vgpu_dirty_tight_on(d);   /* TEMP lab DIRTY_TIGHT */
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
            (tk != token && set->last_query_epoch + 1 < epoch) ||
            (set->hash && d->hash_mode)) {   /* lab DIRTY_HASH */
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
                    REIMS_VGPU_DIRTY_SYNC_GAP_PAGES * target >= off &&
                    reims_vgpu_dirty_bridge_ok(d, tight, (sj->gpa + (r.start + r.len - sj->offset)) >> shift,
                                               gpa >> shift)) {
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

/*
 * TEMP lab REIMS_VGPU_LAB_PREP_SYNC: sync_batch for the sets a caller names — the ones the reads it
 * is about to make will ask for (the drain, ahead of a pass's planned binds, whose plan items would
 * otherwise each sync their own set on their own thread, every one of them settling behind the
 * others' queries; n156: 20 % of the drain inside those items, and n159 found one merged list costs
 * a third of the hypervisor time of per-set queries). Brought up to date for the current epoch in
 * one merged range list, one ticket, one settle. Skipped: unknown tokens, unarmed sets (their
 * arming sync stays per set, with its log flush), protected or already synced ones, and read-only
 * pages (DIRTY_RO_SKIP's argument).
 */
void reims_vgpu_dirty_sync_tokens(ReimsVgpuDirty *d, const uint64_t *want, size_t count)
{
    uint64_t qticket;
    const int shift = qemu_target_page_bits();
    const bool tight = reims_vgpu_dirty_tight_on(d);   /* TEMP lab DIRTY_TIGHT */
    const uint64_t target = 1ULL << shift;
    ReimsVgpuDirtySlice *slices = NULL;
    int cap = 0, n, i;
    g_autoptr(GArray) tokens = g_array_new(FALSE, FALSE, sizeof(uint64_t));
    g_autoptr(GArray) pages = g_array_new(FALSE, FALSE, sizeof(uint64_t));
    g_autoptr(GArray) ranges = NULL;
    g_autoptr(GArray) written = NULL;
    g_autoptr(GArray) hit = NULL;
    uint64_t epoch;
    bool ro_skip;
    guint t, pi;
    size_t k;
    int64_t t0 = get_clock();

    int64_t tq0, tq1, ts1, tl0;

    if (!d || !want || count == 0 || !qatomic_read(&d->ondemand)) {
        return;
    }
    tl0 = get_clock();
    qemu_mutex_lock(&d->lock);
    d->lab_prep_lock_ns += get_clock() - tl0;
    d->lab_prep_asked += count;
    epoch = d->epoch;
    ro_skip = d->ro_skip && qatomic_read(&d->wp_bits);
    for (k = 0; k < count; k++) {
        uint64_t tk = want[k] & ~REIMS_VGPU_DIRTY_TOKEN_LAGGED;
        ReimsVgpuDirtySet *set;
        size_t p;
        uint64_t off;
        bool dup = false;

        if (tk == 0) {
            continue;
        }
        set = g_hash_table_lookup(d->sets, &tk);
        if (!set || set->gen == 0 || set->synced_epoch >= epoch || set->wp_state == WP_ON) {
            continue;
        }
        for (t = 0; t < tokens->len; t++) {
            if (g_array_index(tokens, uint64_t, t) == tk) {
                dup = true;
                break;
            }
        }
        if (dup) {
            continue;
        }
        g_array_append_val(tokens, tk);
        for (p = 0; p < set->count; p++) {
            for (off = 0; off < set->page_size; off += target) {
                uint64_t g = set->pages[p] + off;

                if (ro_skip && reims_vgpu_dirty_wp_ro(d, g)) {
                    continue;
                }
                g_array_append_val(pages, g);
            }
        }
    }
    qemu_mutex_unlock(&d->lock);
    if (tokens->len == 0) {
        return;
    }
    g_array_sort(pages, reims_vgpu_dirty_cmp_u64_asc);

    n = reims_vgpu_dirty_ram_slices_into(&slices, &cap);
    qemu_mutex_lock(&d->lock);
    for (i = 0; i < n; i++) {
        slices[i].logged = g_hash_table_contains(d->logged, slices[i].mr);
    }
    qticket = reims_vgpu_dirty_query_begin(d);
    qemu_mutex_unlock(&d->lock);
    tq0 = get_clock();

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
                uint64_t o;

                if (gpa < sj->gpa || gpa - sj->gpa >= sj->len) {
                    continue;
                }
                o = sj->offset + (gpa - sj->gpa);
                if (open && o < r.start + r.len) {
                    continue;   /* the same page, from a second set */
                }
                if (open && r.start + r.len +
                    REIMS_VGPU_DIRTY_SYNC_GAP_PAGES * target >= o &&
                    reims_vgpu_dirty_bridge_ok(d, tight, (sj->gpa + (r.start + r.len - sj->offset)) >> shift,
                                               gpa >> shift)) {
                    r.len = o + target - r.start;
                    continue;
                }
                if (open) {
                    g_array_append_val(ranges, r);
                }
                r.start = o;
                r.len = target;
                open = true;
            }
            if (open) {
                g_array_append_val(ranges, r);
            }
        }
        if (ranges->len) {
            memory_region_sync_dirty_ranges(slices[i].mr,
                                            (MemoryRegionRange *)ranges->data,
                                            ranges->len);
        }
    }

    written = g_array_new(FALSE, FALSE, sizeof(ReimsVgpuDirtyWritten));
    hit = g_array_new(FALSE, FALSE, sizeof(size_t));
    tq1 = get_clock();
    qemu_mutex_lock(&d->lock);
    reims_vgpu_dirty_query_settle(d, qticket);
    ts1 = get_clock();
    {
        uint64_t pass_seq = d->global_seq;

        for (t = 0; t < tokens->len; t++) {
            ReimsVgpuDirtySet *set = g_hash_table_lookup(d->sets,
                                        &g_array_index(tokens, uint64_t, t));
            if (set) {
                reims_vgpu_dirty_consume_set(d, set, slices, n, pass_seq, written);
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
    d->lab_prep_query_ns += tq1 - tq0;
    d->lab_prep_settle_ns += ts1 - tq1;
    d->lab_prep_fold_ns += get_clock() - ts1;
    d->lab_prep_n++;
    d->lab_prep_sets += tokens->len;
    d->lab_prep_pages += pages->len;
    d->lab_prep_ns += get_clock() - t0;
    qemu_mutex_unlock(&d->lock);
    reims_vgpu_dirty_clear_written(written, slices, n);
    g_free(slices);
}

/*
 * The pages of a set a sync for `epoch` queries: all of `spages` but those
 * already queried in `epoch` or later (all of them when `arming`: an arming
 * sync queries every page itself, after its own flush). Returns how many, in a
 * new array `*out`; `*skipped` counts the rest. Either under `lock`, or (lab
 * DIRTY_PREPLOCK) inside an RCU read-side section with the set claimed: `qep`
 * may then be stale, which only skips fewer pages.
 */
static size_t reims_vgpu_dirty_page_list(ReimsVgpuDirty *d, const uint64_t *spages,
                                         size_t scount, uint64_t epoch, bool arming,
                                         uint64_t **out, uint64_t *skipped, uint64_t *ro_out)
{
    const int shift = qemu_target_page_bits();
    const ReimsVgpuDirtyQep *q = arming ? NULL : qatomic_rcu_read(&d->qep);
    uint64_t *pages = g_new(uint64_t, MAX(scount, 1));
    size_t p, count = 0, ro = 0;
    /*
     * TEMP lab DIRTY_RO_SKIP: a read-only page takes no store without a fault, and the fault makes
     * it writable (and clears its bit) before the store lands, so a page whose bit is set when the
     * list is built — after the epoch it is for opened — has no hypervisor bit this query could
     * find. Whatever the protect step's own query moved into the VGA bitmap is consumed by every
     * set's consume pass, which walks all of a set's pages, not this list. A set sharing its pages
     * with a protected one is thereby protected for free.
     */
    bool ro_skip = d->ro_skip && !arming && qatomic_read(&d->wp_bits);

    for (p = 0; p < scount; p++) {
        uint64_t pfn = spages[p] >> shift;

        if (q && pfn < q->n && qatomic_read(&q->e[pfn]) >= epoch) {
            continue;
        }
        if (ro_skip && reims_vgpu_dirty_wp_ro(d, spages[p])) {
            ro++;
            continue;
        }
        pages[count++] = spages[p];
    }
    if (ro) {
        qatomic_add(&d->lab_ro_skipped, ro);
    }
    *skipped = scount - count;
    *ro_out = ro;
    *out = pages;
    return count;
}

/*
 * Lab DIRTY_HASH: hash pages [p0, p1) of `s` into its `hash`, flagging in `moved` the pages whose
 * content moved (when `base`) or that lie outside guest RAM. The caller holds the RCU read lock and
 * owns these pages' slots (the set's `hash_busy` holder, or a chunk of its job).
 */
static void reims_vgpu_dirty_hash_range(ReimsVgpuDirty *d, ReimsVgpuDirtySet *s,
                                        const ReimsVgpuDirtySlice *slices, int n,
                                        size_t p0, size_t p1, bool base, uint8_t *moved)
{
    int si = -1;
    size_t p;

    for (p = p0; p < p1; p++) {
        uint64_t gpa = s->pages[p], h;

        if (si < 0 || gpa < slices[si].gpa || gpa + s->page_size > slices[si].gpa + slices[si].len) {
            si = reims_vgpu_dirty_slice_of(slices, n, gpa);
            if (si >= 0 && gpa + s->page_size > slices[si].gpa + slices[si].len) {
                si = -1;   /* a page straddling two ranges: not one host mapping */
            }
        }
        if (si < 0) {
            qatomic_inc(&d->lab_hash_noram);
            moved[p] = 1;
            continue;
        }
        h = reims_vgpu_dirty_hash_bytes((const uint8_t *)memory_region_get_ram_ptr(slices[si].mr) +
                                        slices[si].offset + (gpa - slices[si].gpa),
                                        s->page_size);
        if (base && h != s->hash[p]) {
            moved[p] = 1;
        }
        s->hash[p] = h;
    }
}

/*
 * TEMP lab REBASE_SPLIT: hash pages `idx[i0..i1)` of `s` into `out` (no comparison). The caller holds the RCU read
 * lock. A page outside guest RAM keeps its old hash, as the serial rebase leaves it.
 */
static void reims_vgpu_dirty_rehash_idx(ReimsVgpuDirtySet *s, const ReimsVgpuDirtySlice *slices, int n,
                                        const size_t *idx, size_t i0, size_t i1, uint64_t *out)
{
    size_t i;

    for (i = i0; i < i1; i++) {
        size_t p = idx[i];

        reims_vgpu_dirty_page_hash(slices, n, s->pages[p], s->page_size, &out[p]);
    }
}

/* Lab DIRTY_HASH: whether a running hash job has a chunk left to take. d->lock held. */
static bool reims_vgpu_dirty_hash_pending(ReimsVgpuDirty *d)
{
    int j;

    for (j = 0; j < REIMS_VGPU_DIRTY_HJOBS; j++) {
        if (d->hjob[j].busy && d->hjob[j].next < d->hjob[j].total) {
            return true;
        }
    }
    return false;
}

/*
 * Lab DIRTY_HASH: take one chunk of a running hash job (preferring slot `first`), if any is left.
 * d->lock held (dropped while hashing). The slot's fields stay put until its owner sees every chunk
 * done, so they are read again after the lock is retaken.
 */
static bool reims_vgpu_dirty_hash_help_one(ReimsVgpuDirty *d, int first)
{
    int i;

    for (i = 0; i < REIMS_VGPU_DIRTY_HJOBS; i++) {
        int j = (first + i) % REIMS_VGPU_DIRTY_HJOBS;

        if (d->hjob[j].busy && d->hjob[j].next < d->hjob[j].total) {
            guint k = d->hjob[j].next++;
            size_t p0 = (size_t)k * d->hjob[j].chunk;
            size_t lim = d->hjob[j].idx ? d->hjob[j].nidx : d->hjob[j].s->count;
            size_t p1 = MIN(p0 + d->hjob[j].chunk, lim);

            qemu_mutex_unlock(&d->lock);
            rcu_read_lock();
            if (d->hjob[j].idx) {
                reims_vgpu_dirty_rehash_idx(d->hjob[j].s, d->hjob[j].slices, d->hjob[j].n, d->hjob[j].idx,
                                            p0, p1, d->hjob[j].out);
            } else {
                reims_vgpu_dirty_hash_range(d, d->hjob[j].s, d->hjob[j].slices, d->hjob[j].n, p0, p1,
                                            d->hjob[j].base, d->hjob[j].moved);
            }
            rcu_read_unlock();
            qemu_mutex_lock(&d->lock);
            if (++d->hjob[j].done == d->hjob[j].total) {
                qemu_cond_broadcast(&d->split_cond);
            }
            return true;
        }
    }
    return false;
}

/* Lab DIRTY_HASH: take chunks of the running hash jobs until none is left. d->lock held. */
static void reims_vgpu_dirty_hash_help(ReimsVgpuDirty *d)
{
    while (reims_vgpu_dirty_hash_help_one(d, 0)) {
    }
}

/* Pages past which a set's hash is cut into chunks for the workers, and the chunk. */
#define REIMS_VGPU_DIRTY_HASH_SPLIT_MIN 256
#define REIMS_VGPU_DIRTY_HASH_CHUNK 128

/*
 * TEMP lab REBASE_SPLIT (live REBASE_SPLIT=on|off, else REIMS_VGPU_LAB_REBASE_SPLIT=off to disable): whether a
 * rebase of many pages hashes them on the workers. b287: a landed 5K frame's rebase hashed ~14 000 pages on the
 * drain under `lock`, 41 % of the landing and ~6 % of the busy drain on full-screen Maps.
 */
static bool reims_vgpu_dirty_rebase_split_on(void)
{
    static int env = -1;
    int v = reims_vgpu_lab_live("REBASE_SPLIT");

    if (v >= 0) {
        return v == 1;
    }
    if (qatomic_read(&env) < 0) {
        const char *e = getenv("REIMS_VGPU_LAB_REBASE_SPLIT");

        qatomic_set(&env, !(e && strcmp(e, "off") == 0));
    }
    return qatomic_read(&env) == 1;
}

/*
 * TEMP lab REBASE_SPLIT: hash pages `idx[0..nidx)` of `s` into `out`, split over the workers (and this thread) when
 * there are many and a job slot is free, else here. Called without `lock`, under the RCU read lock; the caller owns
 * those slots of `out` for the duration (the set's `hash_busy` for `hash`; for `vhash`, a concurrent evaluation
 * only ever stores the same page's current hash there).
 */
static void reims_vgpu_dirty_rehash(ReimsVgpuDirty *d, ReimsVgpuDirtySet *s, const ReimsVgpuDirtySlice *slices,
                                    int n, const size_t *idx, size_t nidx, uint64_t *out)
{
    bool split = false;

    if (nidx >= REIMS_VGPU_DIRTY_HASH_SPLIT_MIN && reims_vgpu_dirty_rebase_split_on()) {
        int j;

        qemu_mutex_lock(&d->lock);
        if (d->od_nworkers > 0) {
            for (j = 0; j < REIMS_VGPU_DIRTY_HJOBS && d->hjob[j].busy; j++) {
            }
            if (j < REIMS_VGPU_DIRTY_HJOBS) {
                d->hjob[j].s = s;
                d->hjob[j].slices = slices;
                d->hjob[j].n = n;
                d->hjob[j].moved = NULL;
                d->hjob[j].base = false;
                d->hjob[j].idx = idx;
                d->hjob[j].nidx = nidx;
                d->hjob[j].out = out;
                d->hjob[j].chunk = REIMS_VGPU_DIRTY_HASH_CHUNK;
                d->hjob[j].next = d->hjob[j].done = 0;
                d->hjob[j].total = (nidx + REIMS_VGPU_DIRTY_HASH_CHUNK - 1) / REIMS_VGPU_DIRTY_HASH_CHUNK;
                d->hjob[j].busy = true;
                split = true;
                d->lab_rebase_split++;
                qemu_cond_broadcast(&d->od_work_cond);
                while (d->hjob[j].done < d->hjob[j].total) {
                    if (!reims_vgpu_dirty_hash_help_one(d, j)) {
                        qemu_cond_wait(&d->split_cond, &d->lock);
                    }
                }
                d->hjob[j].busy = false;
                d->hjob[j].s = NULL;
                d->hjob[j].idx = NULL;
                d->hjob[j].out = NULL;
                d->hjob[j].nidx = 0;
                d->hjob[j].total = d->hjob[j].done = d->hjob[j].next = 0;
            }
        }
        qemu_mutex_unlock(&d->lock);
    }
    if (!split) {
        reims_vgpu_dirty_rehash_idx(s, slices, n, idx, 0, nidx, out);
    }
}

/*
 * Lab DIRTY_HASH: bring hashed set `s` (claimed for `epoch` by the caller) up to date by comparing
 * its pages' content hashes with those of its last sync. Called and returning with `lock` held.
 *
 * Every store ordered before the doorbell that opened `epoch` is in guest RAM before the hashes
 * are taken (the doorbell is a trapping write the tracker saw first), so a page whose content
 * moved since the previous sync shows a new hash; a store racing the walk shows now or at the next
 * sync. One sync of a set at a time (`hash_busy`): a second, for a newer epoch, waits for it and
 * then hashes again. A page outside guest RAM reads as written, as in an evaluation.
 */
static void reims_vgpu_dirty_hash_sync(ReimsVgpuDirty *d, ReimsVgpuDirtySet *s,
                                       uint64_t token, uint64_t epoch, int64_t *lab_query)
{
    const int shift = qemu_target_page_bits();
    ReimsVgpuDirtySlice *slices = NULL;
    g_autoptr(GArray) moved = NULL;
    uint8_t *flags;
    int cap = 0, n;
    int64_t t0;
    size_t p;
    bool base, split;

    while (s && s->hash_busy) {
        d->lab_hash_waits++;
        qemu_cond_wait(&d->synced_cond, &d->lock);
        s = g_hash_table_lookup(d->sets, &token);
    }
    if (!s) {
        return;
    }
    if (s->synced_epoch >= epoch) {
        /* A sync for this epoch or a newer one finished while this one waited. */
        if (s->claim_epoch == epoch) {
            s->claim_epoch = 0;
        }
        if (s->tclaim == epoch) {
            s->tclaim = 0;
        }
        qemu_cond_broadcast(&d->synced_cond);
        return;
    }
    s->hash_busy = true;
    base = s->hash_base;
    qemu_mutex_unlock(&d->lock);

    t0 = get_clock();
    moved = g_array_new(FALSE, FALSE, sizeof(size_t));
    flags = g_malloc0(MAX(s->count, 1));
    n = reims_vgpu_dirty_ram_slices_into(&slices, &cap);
    rcu_read_lock();   /* the set (RCU-freed) and guest RAM stay put while hashing */
    split = false;
    if (s->count * (s->page_size >> shift) >= REIMS_VGPU_DIRTY_HASH_SPLIT_MIN && d->od_nworkers > 0) {
        int j;

        qemu_mutex_lock(&d->lock);
        for (j = 0; j < REIMS_VGPU_DIRTY_HJOBS && d->hjob[j].busy; j++) {
        }
        if (j < REIMS_VGPU_DIRTY_HJOBS) {
            /* The workers and any waiter take chunks while this thread takes its own share. */
            d->hjob[j].s = s;
            d->hjob[j].slices = slices;
            d->hjob[j].n = n;
            d->hjob[j].moved = flags;
            d->hjob[j].base = base;
            d->hjob[j].chunk = REIMS_VGPU_DIRTY_HASH_CHUNK;
            d->hjob[j].next = d->hjob[j].done = 0;
            d->hjob[j].total = (s->count + REIMS_VGPU_DIRTY_HASH_CHUNK - 1) / REIMS_VGPU_DIRTY_HASH_CHUNK;
            d->hjob[j].busy = true;
            split = true;
            d->lab_hash_split++;
            qemu_cond_broadcast(&d->od_work_cond);
            /* Its own chunks first, then any other job's, then wait for the chunks others took. */
            while (d->hjob[j].done < d->hjob[j].total) {
                if (!reims_vgpu_dirty_hash_help_one(d, j)) {
                    qemu_cond_wait(&d->split_cond, &d->lock);
                }
            }
            d->hjob[j].busy = false;
            d->hjob[j].s = NULL;
            d->hjob[j].total = d->hjob[j].done = d->hjob[j].next = 0;
        }
        qemu_mutex_unlock(&d->lock);
    }
    if (!split) {
        reims_vgpu_dirty_hash_range(d, s, slices, n, 0, s->count, base, flags);
    }
    rcu_read_unlock();
    g_free(slices);
    for (p = 0; p < s->count; p++) {
        if (flags[p]) {
            g_array_append_val(moved, p);
        }
    }
    g_free(flags);
    if (lab_query) {   /* TEMP lab (FGTRACE): the hash time stands for the query */
        *lab_query = get_clock() - t0;
    }

    qemu_mutex_lock(&d->lock);
    s = g_hash_table_lookup(d->sets, &token);
    if (s) {
        guint m;

        /* Stored atomically: the generation fast path reads it without `lock`. */
        qatomic_set(&s->eval_seq, s->eval_seq + 1);   /* odd: rewriting gen and the stamps */
        smp_wmb();
        if (s->gen == 0) {
            qatomic_set(&s->gen, 1);   /* the baseline: what a reader gathers from now on */
        } else if (moved->len) {
            qatomic_set(&s->gen, s->gen + 1);
            for (m = 0; m < moved->len; m++) {
                qatomic_set(&s->page_gen[g_array_index(moved, size_t, m)], s->gen);
            }
        }
        smp_wmb();
        qatomic_set(&s->eval_seq, s->eval_seq + 1);   /* even again */
        s->hash_base = true;
        s->hash_busy = false;
        qatomic_store_release(&s->synced_epoch, MAX(s->synced_epoch, epoch));
        if (qatomic_read(&reims_vgpu_dirty_wp_hashed_live) && d->wp_mode && d->wp_active &&
            s->wp_state == WP_NONE && s->count * (s->page_size >> shift) >=
                (qatomic_read(&reims_vgpu_dirty_wp_hashed_min_live) ?: d->wp_min_pages)) {
            /* TEMP lab live DIRTY_WP_HASHED: the clean streak and the queueing of the query path. */
            uint32_t need = (uint32_t)(qatomic_read(&reims_vgpu_dirty_wp_hashed_streak_live) ?:
                                       qatomic_read(&reims_vgpu_dirty_wp_streak_live) ?: d->wp_streak_need)
                            << MIN(s->wp_lost, 6);

            need = reims_vgpu_dirty_wp_need(d, s, need);   /* TEMP lab DIRTY_WP_BRIDGE */

            s->wp_streak = moved->len ? 0 : s->wp_streak + 1;
            if (s->wp_streak == need ||
                ((d->wp_retry || d->wp_calm) && s->wp_streak > need && (s->wp_streak - need) % need == 0)) {
                if (d->wp_calm && get_clock() - s->born_ns < reims_vgpu_dirty_wp_age_need(d, s)) {
                    d->lab_wp_young++;
                } else {
                    g_array_append_val(d->wp_queue, s->token);
                    REIMS_WP_BH_SCHEDULE(d->wp_bh);
                }
            }
        }
        if (s->claim_epoch == epoch) {
            s->claim_epoch = 0;
        }
        if (s->tclaim == epoch) {
            s->tclaim = 0;
        }
        d->lab_hash_syncs++;
        d->lab_hash_pages += s->count * (s->page_size >> shift);
        d->lab_hash_dirty += moved->len != 0;
    }
    d->lab_hash_ns += get_clock() - t0;
    qemu_cond_broadcast(&d->synced_cond);
}

/*
 * Bring one set up to date: on return it is synced for `needed` or later (or
 * gone). `needed` is the epoch the caller's read must see (0 = the epoch at
 * entry); a sync this one starts is for the epoch at entry, never older. A sync
 * already running for `needed` or later (a thread's, or a batch's that `fg`
 * does not bypass) is waited for instead, helping split pieces meanwhile.
 */
static void reims_vgpu_dirty_sync_one(ReimsVgpuDirty *d, uint64_t token, bool fg,
                                      uint64_t needed, int64_t *lab_wait,
                                      int64_t *lab_query)
{
    const int shift = qemu_target_page_bits();
    const bool tight = reims_vgpu_dirty_tight_on(d);   /* TEMP lab DIRTY_TIGHT */
    ReimsVgpuDirtySlice *slices = NULL;
    int cap = 0, n, i;
    ReimsVgpuDirtySet *s;
    g_autofree uint64_t *pages = NULL;
    g_autoptr(GArray) ranges = NULL;
    g_autoptr(GArray) walked = NULL;
    g_autoptr(GArray) written = NULL;
    g_autoptr(GArray) hit = NULL;
    size_t count, p;
    uint64_t page_size, epoch, need, qticket, walked_pages = 0;
    int slot;             /* lab DIRTY_FGSLOTS: the foreground job this sync owns, or -1 */
    int64_t ph = 0, ph_prep_end = 0, ph_query_end = 0, ph_settle_end = 0;
    bool stole = false;   /* TEMP lab STEAL */
    bool joined = false;  /* TEMP lab: waited for a thread that stole the set for us */
    bool arming = false;  /* lab ARM_FLUSH */
    g_autoptr(GArray) chunks = g_array_new(FALSE, FALSE, sizeof(ReimsVgpuDirtyChunk));
    int64_t t0;
    int64_t lk_wait, lk_prep, lk_list, lk_split, lk_post, lk_t;   /* TEMP lab: d->lock wait/hold */
    uint64_t set_pages;   /* TEMP lab: target pages of the set */
    const uint64_t *spages;   /* the set's page list (immutable; RCU-freed) */
    size_t scount;
    uint64_t skipped = 0;
    uint64_t ro_listed = 0;   /* lab DIRTY_SETTLE_SKIP: pages left out for being read-only */
    bool nothing = false;     /* lab DIRTY_SETTLE_SKIP: no page to query, no ticket, no settle */
    bool unlocked;        /* lab DIRTY_PREPLOCK */

    if (d->od_batch && reims_vgpu_dirty_sync_batch(d, token)) {
        return;
    }
    t0 = get_clock();
    qemu_mutex_lock(&d->lock);
    lk_t = get_clock();
    lk_wait = lk_t - t0;
    s = g_hash_table_lookup(d->sets, &token);
    if (!s) {
        qemu_mutex_unlock(&d->lock);
        return;
    }
    epoch = d->epoch;
    /*
     * The epoch this read must see. A read that needs an older epoch than the
     * newest (the drain's work epoch lags the doorbells) is served by any sync
     * claimed for that epoch or later — it began after that doorbell, so it
     * vouches for every store before it — and waits for one already running
     * rather than starting its own beside it. Several foreground readers of
     * one set (the drain and the device's worker threads) then share one sync.
     */
    need = needed && d->fgjoin_mode ? MIN(needed, epoch) : epoch;
    /*
     * Up to date already, or being brought up to date for this epoch by the
     * other thread (drain or prefetch): wait for that one rather than repeat
     * its hypervisor calls. The claimant holds no lock while it syncs.
     */
    ph = get_clock();
    while (s && s->synced_epoch < need && s->claim_epoch >= need) {
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
         * *thread's* sync (`bclaim` 0) is still waited for, and so is one a
         * thread already stole for an epoch this read can use (`tclaim`): one
         * bypass per batch claim, not one per reader.
         */
        if (d->steal_mode && fg && s->bclaim >= need) {
            if (s->tclaim < need || !d->fgjoin_mode) {
                stole = true;
                d->lab_steals++;
                break;
            }
            joined = true;   /* TEMP lab */
        }
        /* Waiting for the claimant: query its split chunks meanwhile. */
        if (reims_vgpu_dirty_split_pending(d)) {
            reims_vgpu_dirty_split_help(d);
        } else {
            qemu_cond_wait(&d->synced_cond, &d->lock);
        }
        s = g_hash_table_lookup(d->sets, &token);
    }
    d->lab_ph_wait += get_clock() - ph;
    d->lab_steal_joined += joined;
    if (lab_wait) {   /* TEMP lab (FGTRACE) */
        *lab_wait = get_clock() - ph;
    }
    if (!s || s->synced_epoch >= need) {
        qemu_mutex_unlock(&d->lock);
        return;
    }
    ph = get_clock();
    /* Not below a newer batch claim this read bypasses: that batch still runs. */
    s->claim_epoch = d->fgjoin_mode ? MAX(s->claim_epoch, epoch) : epoch;
    if (s->tclaim >= (needed ? MIN(needed, epoch) : epoch)) {   /* TEMP lab */
        d->lab_dup_inflight++;
    } else if (s->tclaim) {
        d->lab_ov_older++;
    }
    s->tclaim = MAX(s->tclaim, epoch);
    d->epoch_queried = true;   /* a hash sync observes this epoch too: no coalescing past it */
    if (s->hash && d->hash_mode) {   /* lab DIRTY_HASH: no hypervisor call, no ticket */
        int64_t h0 = get_clock();

        reims_vgpu_dirty_hash_sync(d, s, token, epoch, lab_query);
        if (fg) {
            d->lab_hash_fg++;
            d->lab_hash_fg_ns += get_clock() - h0;
        }
        qemu_mutex_unlock(&d->lock);
        return;
    }
    page_size = s->page_size;
    set_pages = s->count * (page_size >> shift);
    arming = d->arm_flush && qatomic_read(&s->gen) == 0 && memory_dirty_log_flush_hook;
    spages = s->pages;
    scount = s->count;
    unlocked = d->prep_unlocked;
    lk_list = 0;
    if (!unlocked) {
        int64_t l0 = get_clock();

        count = reims_vgpu_dirty_page_list(d, spages, scount, epoch, arming, &pages, &skipped,
                                           &ro_listed);
        lk_list = get_clock() - l0;   /* TEMP lab */
    }
    {
        int64_t wait = d->doorbell_ns + d->lab_od_delay_ns - get_clock();

        bool flush = d->od_flush && d->flushed_epoch < epoch &&
                     memory_dirty_log_flush_hook;

        if (flush) {
            d->flushed_epoch = epoch;
        }
        if (arming) {
            d->lab_arm_flushes++;
        }
        lk_prep = get_clock() - ph;   /* TEMP lab: held from the claim on */
        if (unlocked) {
            /* Lab DIRTY_PREPLOCK: the page list under RCU, after the unlock. */
            int64_t l0;

            rcu_read_lock();
            qemu_mutex_unlock(&d->lock);
            l0 = get_clock();
            count = reims_vgpu_dirty_page_list(d, spages, scount, epoch, arming, &pages,
                                               &skipped, &ro_listed);
            lk_list = get_clock() - l0;   /* TEMP lab: built, but not held */
            rcu_read_unlock();
        } else {
            qemu_mutex_unlock(&d->lock);
        }
        if (d->lab_od_delay_ns > 0 && wait > 0) {
            g_usleep(wait / 1000);
        }
        if (flush || arming) {
            int64_t f0 = get_clock();

            memory_dirty_log_flush_hook();
            if (arming) {
                qatomic_add(&d->lab_arm_flush_ns, get_clock() - f0);
            }
        }
    }

    n = reims_vgpu_dirty_ram_slices_into(&slices, &cap);
    lk_t = get_clock();
    qemu_mutex_lock(&d->lock);
    lk_wait += get_clock() - lk_t;
    lk_t = get_clock();
    for (i = 0; i < n; i++) {
        slices[i].logged = g_hash_table_contains(d->logged, slices[i].mr);
    }
    d->lab_od_skipped += skipped;
    if (count == 0 && !arming) {   /* TEMP lab DIRTY_SETTLE_SKIP */
        d->lab_settle_skip_cand++;
        nothing = d->settle_skip && (ro_listed == 0 || d->wp_tickets_open == 0);
    }
    /*
     * Lab DIRTY_RSETTLE, DIRTY_PREPLOCK: the ticket is taken below, with its
     * footprint, once the ranges are known and before any query: a ticket held
     * through the walk that builds them has no footprint yet, and every settle
     * meanwhile has to wait for it.
     */
    qticket = d->rsettle_mode || unlocked || nothing ? 0 : reims_vgpu_dirty_query_begin(d);
    lk_prep += get_clock() - lk_t;
    qemu_mutex_unlock(&d->lock);

    /* Contiguous runs of this set's pages, per logged region. */
    ranges = g_array_new(FALSE, FALSE, sizeof(MemoryRegionRange));
    walked = g_array_new(FALSE, FALSE, sizeof(MemoryRegionRange));  /* GPA spans */
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
                    (1ULL << shift) >= sj->offset + (gpa - sj->gpa) &&
                    reims_vgpu_dirty_bridge_ok(d, tight, (sj->gpa + (r.start + r.len - sj->offset)) >> shift,
                                               gpa >> shift)) {
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
    lk_t = get_clock();
    qemu_mutex_lock(&d->lock);
    lk_wait += get_clock() - lk_t;
    lk_t = get_clock();
    if ((d->rsettle_mode || unlocked) && !nothing) {
        qticket = reims_vgpu_dirty_query_begin(d);
    }
    reims_vgpu_dirty_query_footprint(d, qticket, (const ReimsVgpuDirtyChunk *)chunks->data,
                                     chunks->len);
    slot = -1;
    if (fg && d->fgslots && d->od_split && d->od_nworkers > 0 && walked_pages >= d->fg_split_min) {
        /* Lab DIRTY_FGSLOTS: a free foreground job, else the batch's, else alone. */
        for (i = 0; i < (int)d->fgslots; i++) {
            if (!d->fjob[i].busy) {
                slot = i;
                break;
            }
        }
        if (slot < 0) {
            d->lab_fg_noslot++;
        }
    }
    if (slot >= 0) {
        ReimsVgpuDirtySplitJob *j = &d->fjob[slot];

        /* The pieces carry their own tickets; this one has queried nothing. */
        if (!d->od_nopticket) {
            reims_vgpu_dirty_query_end(d, qticket);
            qticket = 0;
        }
        reims_vgpu_dirty_split_cut(j->chunks, chunks, shift);
        j->next = j->done = 0;
        j->total = j->chunks->len;
        j->busy = true;
        if (j->total) {
            d->fjob_open |= 1u << slot;
        }
        d->fjob_busy++;
        d->lab_fg_slots_max = MAX(d->lab_fg_slots_max, d->fjob_busy);
        d->lab_fg_slots_multi += d->fjob_busy > 1;
        if (d->split_busy) {
            d->lab_fg_split_shared++;
        }
        qemu_cond_broadcast(&d->od_work_cond);
        qemu_cond_broadcast(&d->synced_cond);   /* claim waiters help too */
        lk_split = get_clock() - lk_t;
        reims_vgpu_dirty_split_help_from(d, slot);
        while (j->done < j->total) {
            qemu_cond_wait(&d->split_cond, &d->lock);
        }
        j->total = j->done = j->next = 0;
        j->busy = false;
        d->fjob_busy--;
        d->lab_fg_split++;   /* TEMP lab (FGTRACE) */
        d->lab_fg_split_slot++;
        qemu_mutex_unlock(&d->lock);
    } else if (d->od_split && d->od_nworkers > 0 && walked_pages >= d->fg_split_min &&
        !d->split_busy && (fg || !d->bigsplit_mode)) {
        /* The pieces carry their own tickets; this one has queried nothing. */
        if (!d->od_nopticket) {
            reims_vgpu_dirty_query_end(d, qticket);
            qticket = 0;
        }
        reims_vgpu_dirty_split_cut(d->split_chunks, chunks, shift);
        d->split_next = d->split_done = 0;
        d->split_total = d->split_chunks->len;
        d->split_busy = true;
        qemu_cond_broadcast(&d->od_work_cond);
        qemu_cond_broadcast(&d->synced_cond);   /* claim waiters help too */
        lk_split = get_clock() - lk_t;
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

        bool tb = d->ticket_batch && qticket && !d->rsettle_mode &&
                  chunks->len > d->ticket_batch;   /* TEMP lab DIRTY_TICKET_BATCH */

        if (fg && walked_pages >= d->fg_split_min) {
            d->lab_fg_big_alone++;   /* TEMP lab: a large foreground sync not split */
        }
        lk_split = get_clock() - lk_t;
        if (tb) {
            /* Nothing queried under it yet: the batches below carry their own. */
            reims_vgpu_dirty_query_end(d, qticket);
            qticket = 0;
            d->lab_tb_syncs++;
        }
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
            if (tb) {
                guint b;

                for (b = 0; b < ranges->len; b += d->ticket_batch) {
                    guint nb = MIN(d->ticket_batch, ranges->len - b);
                    uint64_t pt;

                    qemu_mutex_lock(&d->lock);
                    pt = reims_vgpu_dirty_query_begin(d);
                    d->lab_tb_batches++;
                    qemu_mutex_unlock(&d->lock);
                    memory_region_sync_dirty_ranges(mr,
                        &g_array_index(ranges, MemoryRegionRange, b), nb);
                    qemu_mutex_lock(&d->lock);
                    reims_vgpu_dirty_query_end(d, pt);
                    qemu_mutex_unlock(&d->lock);
                }
            } else {
                memory_region_sync_dirty_ranges(mr, (MemoryRegionRange *)ranges->data,
                                                ranges->len);
            }
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
    lk_t = get_clock();
    qemu_mutex_lock(&d->lock);
    ph_query_end = get_clock();
    lk_wait += ph_query_end - lk_t;
    if (lab_query) {   /* TEMP lab (FGTRACE) */
        *lab_query = ph_query_end - ph_prep_end;
    }
    if (nothing) {
        d->lab_settle_skips++;   /* TEMP lab DIRTY_SETTLE_SKIP */
    } else {
        reims_vgpu_dirty_query_settle_fp(d, qticket, (const ReimsVgpuDirtyChunk *)chunks->data,
                                         chunks->len);
    }
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

        reims_vgpu_dirty_qep_stamp(d, w->start >> shift, (w->start + w->len) >> shift, epoch);
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
        if (s->tclaim == epoch) {
            s->tclaim = 0;
        }
    }
    qemu_cond_broadcast(&d->synced_cond);
    if (stole && d->steal_mode == 2) {
        /* Lab STEAL verify: answer as the original did, after the batch. */
        uint64_t g1 = s ? s->gen : 0;

        while ((s = g_hash_table_lookup(d->sets, &token)) && s->bclaim >= need) {
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

        lk_post = now - ph_settle_end;   /* TEMP lab: stamp, consume, eval */
        d->lab_hold_prep += lk_prep;
        d->lab_hold_list += lk_list;
        d->lab_hold_split += lk_split;
        d->lab_hold_post += lk_post;
        d->lab_lock_wait += lk_wait;
        d->lab_hold_n++;
        if (set_pages >= 4096) {
            d->lab_holdb_n++;
            d->lab_holdb_prep += lk_prep;
            d->lab_holdb_list += lk_list;
            d->lab_holdb_post += lk_post;
        }
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
                    " bp_groups=%" PRIu64 " frontier=%u fr_child=%" PRIu64
                    " fgslot=%u fg_split_slot=%" PRIu64 " fg_split_shared=%" PRIu64
                    " fg_big_alone=%" PRIu64 " fgslots=%u fg_slots_max=%" PRIu64
                    " fg_slots_multi=%" PRIu64 " fg_noslot=%" PRIu64 " lk_n=%" PRIu64
                    " lk_wait_us=%" PRId64 " lk_prep_us=%" PRId64 " lk_list_us=%" PRId64
                    " lk_split_us=%" PRId64 " lk_post_us=%" PRId64 " lkb_n=%" PRIu64
                    " lkb_prep_us=%" PRId64 " lkb_list_us=%" PRId64 " lkb_post_us=%" PRId64
                    " fgjoin=%u dup_inflight=%" PRIu64 " ov_older=%" PRIu64
                    " settle_skip=%u settle_skips=%" PRIu64 " settle_skip_cand=%" PRIu64
                    " tb=%u tb_syncs=%" PRIu64 " tb_batches=%" PRIu64 "\n",
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
                    d->lab_frontier_child, d->fgslots != 0, d->lab_fg_split_slot,
                    d->lab_fg_split_shared, d->lab_fg_big_alone, d->fgslots,
                    d->lab_fg_slots_max, d->lab_fg_slots_multi, d->lab_fg_noslot,
                    d->lab_hold_n, d->lab_lock_wait / 1000, d->lab_hold_prep / 1000,
                    d->lab_hold_list / 1000, d->lab_hold_split / 1000,
                    d->lab_hold_post / 1000, d->lab_holdb_n, d->lab_holdb_prep / 1000,
                    d->lab_holdb_list / 1000, d->lab_holdb_post / 1000, d->fgjoin_mode,
                    d->lab_dup_inflight, d->lab_ov_older, d->settle_skip,
                    d->lab_settle_skips, d->lab_settle_skip_cand, d->ticket_batch,
                    d->lab_tb_syncs, d->lab_tb_batches);
            reims_vgpu_dirty_gap_refresh();   /* TEMP lab live DIRTY_GAP */
            {   /* TEMP lab: who syncs the large sets in the foreground */
                int c;

                fprintf(stderr, "reims-dirty-fgbig");
                for (c = 0; c < 8; c++) {
                    fprintf(stderr, " ch%d=%" PRIu64 "/%" PRIu64 "us", c, d->lab_fgbig_n[c],
                            d->lab_fgbig_ns[c] / 1000);
                    d->lab_fgbig_n[c] = d->lab_fgbig_ns[c] = 0;
                }
                fputc('\n', stderr);
            }
            if (d->trk_on) {   /* TEMP lab DIRTY_TIGHT */
                fprintf(stderr, "reims-dirty-tight on=%d gap=%" PRIu64 " kept=%" PRIu64 " cut=%" PRIu64
                        " cut_pages=%" PRIu64 "\n", reims_vgpu_dirty_tight_on(d),
                        reims_vgpu_dirty_gap_pages(),
                        qatomic_xchg(&d->lab_tight_kept, 0), qatomic_xchg(&d->lab_tight_cut, 0),
                        qatomic_xchg(&d->lab_tight_cut_pages, 0));
            }
            d->lab_settle_skips = d->lab_settle_skip_cand = 0;
            d->lab_tb_syncs = d->lab_tb_batches = 0;
            d->lab_frontier_child = 0;
            d->lab_fg_split_slot = d->lab_fg_split_shared = d->lab_fg_big_alone = 0;
            d->lab_fg_slots_max = d->lab_fg_slots_multi = d->lab_fg_noslot = 0;
            d->lab_hold_n = 0;
            d->lab_lock_wait = d->lab_hold_prep = d->lab_hold_list = 0;
            d->lab_hold_split = d->lab_hold_post = 0;
            d->lab_holdb_n = 0;
            d->lab_holdb_prep = d->lab_holdb_list = d->lab_holdb_post = 0;
            d->lab_dup_inflight = d->lab_ov_older = 0;
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
                        " pause_us=%" PRId64 " pause_max_us=%" PRId64
                        " rej_state=%" PRIu64 " rej_gen=%" PRIu64 " rej_bits=%" PRIu64
                        " rej_ro=%" PRIu64 " requeued=%" PRIu64 " ro_skipped=%" PRIu64
                        " hole=%u holed=%" PRIu64 " heal_queued=%" PRIu64 " heals=%" PRIu64
                        " heal_skipped=%" PRIu64 " hole_drops=%" PRIu64 " calm=%u young=%" PRIu64
                        " dead=%" PRIu64 " yields=%" PRIu64 " merge=%u mspans=%" PRIu64
                        " mruns=%" PRIu64 " mpages=%" PRIu64 " mfault=%" PRIu64 "\n",
                        d->wp_active, d->wp_list ? d->wp_list->len : 0, d->wp_cur != 0,
                        d->lab_wp_protects, d->lab_wp_chunks, d->lab_wp_faults, d->lab_wp_lost,
                        d->lab_wp_dropped, d->wp_orphans ? d->wp_orphans->len : 0, d->lab_wp_remaps,
                        d->lab_wp_refused, (int64_t)d->lab_wp_remap_ns / 1000,
                        (int64_t)d->lab_wp_pause_ns / 1000, (int64_t)d->lab_wp_pause_max_ns / 1000,
                        d->lab_wp_rej_state, d->lab_wp_rej_gen, d->lab_wp_rej_bits, d->lab_wp_rej_ro,
                        d->lab_wp_requeued, qatomic_xchg(&d->lab_ro_skipped, 0), d->wp_hole,
                        d->lab_wp_holed, d->lab_wp_heal_queued, d->lab_wp_heals,
                        d->lab_wp_heal_skipped, d->lab_wp_hole_drops, d->wp_calm, d->lab_wp_young,
                        d->lab_wp_dead, d->lab_wp_yields, d->wp_merge, d->lab_wp_merge_spans,
                        d->lab_wp_merge_runs, d->lab_wp_merge_pages, d->lab_wp_merge_fault);
                fprintf(stderr, "reims-dirty-wp-rm n1=%" PRIu64 " us1=%" PRIu64 " n16=%" PRIu64
                        " us16=%" PRIu64 " n256=%" PRIu64 " us256=%" PRIu64 " nbig=%" PRIu64
                        " usbig=%" PRIu64 "\n",   /* TEMP lab (b357) */
                        d->lab_wp_rm_n[0], d->lab_wp_rm_ns[0] / 1000, d->lab_wp_rm_n[1],
                        d->lab_wp_rm_ns[1] / 1000, d->lab_wp_rm_n[2], d->lab_wp_rm_ns[2] / 1000,
                        d->lab_wp_rm_n[3], d->lab_wp_rm_ns[3] / 1000);
                memset(d->lab_wp_rm_n, 0, sizeof(d->lab_wp_rm_n));
                memset(d->lab_wp_rm_ns, 0, sizeof(d->lab_wp_rm_ns));
                fprintf(stderr, "reims-dirty-wp-fault fair=%u n=%" PRIu64 " wait_us=%" PRIu64 " wait_max_us=%" PRIu64
                        " hold_us=%" PRIu64 " hold_max_us=%" PRIu64 " slow4ms=%" PRIu64 "\n",   /* TEMP lab DIRTY_WP_FAIR */
                        d->wp_fair, d->lab_wp_faults, d->lab_wp_fwait_ns / 1000, d->lab_wp_fwait_max_ns / 1000,
                        d->lab_wp_fhold_ns / 1000, d->lab_wp_fhold_max_ns / 1000, d->lab_wp_fslow);
                d->lab_wp_fwait_ns = d->lab_wp_fwait_max_ns = d->lab_wp_fhold_ns = d->lab_wp_fhold_max_ns = 0;
                d->lab_wp_fslow = 0;
                if (d->wp_bridge) {   /* TEMP lab DIRTY_WP_BRIDGE */
                    fprintf(stderr, "reims-dirty-wp-bridge spans=%" PRIu64 " runs=%" PRIu64 " gap_ro=%" PRIu64
                            " gap_hashed=%" PRIu64 " stop_log=%" PRIu64 " stop_free=%" PRIu64 " stop_far=%" PRIu64
                            " orphaned=%" PRIu64 " bridged=%u\n",
                            d->lab_wp_br_spans, d->lab_wp_br_runs, d->lab_wp_br_ro, d->lab_wp_br_hashed,
                            d->lab_wp_br_stop_log, d->lab_wp_br_stop_free, d->lab_wp_br_stop_far,
                            d->lab_wp_br_orphaned, d->wp_bridged->len);
                    d->lab_wp_br_spans = d->lab_wp_br_runs = d->lab_wp_br_ro = d->lab_wp_br_hashed = 0;
                    d->lab_wp_br_stop_log = d->lab_wp_br_stop_free = d->lab_wp_br_stop_far = 0;
                    d->lab_wp_br_orphaned = 0;
                }
                d->lab_wp_young = d->lab_wp_dead = d->lab_wp_yields = 0;
                d->lab_wp_merge_spans = d->lab_wp_merge_runs = 0;   /* TEMP lab DIRTY_WP_MERGE */
                d->lab_wp_merge_pages = d->lab_wp_merge_fault = 0;
                d->lab_wp_holed = d->lab_wp_heal_queued = d->lab_wp_heals = 0;
                d->lab_wp_heal_skipped = d->lab_wp_hole_drops = 0;
                d->lab_wp_rej_state = d->lab_wp_rej_gen = d->lab_wp_rej_bits = 0;
                d->lab_wp_rej_ro = d->lab_wp_requeued = 0;
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
            if (d->lab_prep_asked) {   /* TEMP lab PREP_SYNC */
                fprintf(stderr, "reims-dirty-prep n=%" PRIu64 " asked=%" PRIu64 " sets=%" PRIu64
                        " pages=%" PRIu64 " us=%" PRIu64 " lock_us=%" PRIu64 " query_us=%" PRIu64
                        " settle_us=%" PRIu64 " fold_us=%" PRIu64 "\n", d->lab_prep_n,
                        d->lab_prep_asked, d->lab_prep_sets, d->lab_prep_pages, d->lab_prep_ns / 1000,
                        d->lab_prep_lock_ns / 1000, d->lab_prep_query_ns / 1000,
                        d->lab_prep_settle_ns / 1000, d->lab_prep_fold_ns / 1000);
                d->lab_prep_n = d->lab_prep_asked = d->lab_prep_sets = 0;
                d->lab_prep_pages = d->lab_prep_ns = 0;
                d->lab_prep_lock_ns = d->lab_prep_query_ns = 0;
                d->lab_prep_settle_ns = d->lab_prep_fold_ns = 0;
            }
            if (d->arm_flush && (d->lab_arm_flushes || d->lab_pb_arm_flushes)) {
                /* TEMP lab ARM_FLUSH */
                fprintf(stderr, "reims-dirty-armflush epoch_ms=%" PRId64 " n=%" PRIu64
                        " pb=%" PRIu64
                        " us=%" PRId64 "\n", g_get_real_time() / 1000, d->lab_arm_flushes,
                        qatomic_xchg(&d->lab_pb_arm_flushes, 0),
                        qatomic_xchg(&d->lab_arm_flush_ns, 0) / 1000);
                d->lab_arm_flushes = 0;
            }
            if (d->steal_mode || d->gen_memo_mode || d->epoch_named ||
                d->lf_mode) {   /* TEMP lab gates */
                fprintf(stderr, "reims-dirty-lab epoch_ms=%" PRId64 " steal=%u memo=%u"
                        " steals=%" PRIu64 " steal_same=%" PRIu64 " steal_late=%" PRIu64
                        " steal_joined=%" PRIu64 " memo_hit=%" PRIu64 " memo_fresh=%" PRIu64
                        " memo_diff=%" PRIu64 " memo_bad=%" PRIu64 " named=%u unnamed_skipped=%" PRIu64 " lf=%u lf_hit=%" PRIu64 " lf_hitdiff=%" PRIu64
                        " lf_gendiff=%" PRIu64 " lf_race=%" PRIu64 " lf_fast=%" PRIu64 " lfw_hit=%" PRIu64
                        " lfw_miss=%" PRIu64 " lfw_diff=%" PRIu64 " lfw_race=%" PRIu64 "\n",
                        g_get_real_time() / 1000, d->steal_mode, d->gen_memo_mode,
                        d->lab_steals, d->lab_steal_same, d->lab_steal_late,
                        d->lab_steal_joined, qatomic_xchg(&d->lab_memo_hit, 0),
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
                d->lab_steal_joined = 0;
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
        if (!d->od_stop && d->ahead_next < d->ahead->len) {
            /*
             * Lab SYNC_AHEAD: a set the device named is read sooner than a guessed hot one, so
             * the named queue goes first even in the middle of an epoch's list.
             */
            token = g_array_index(d->ahead, uint64_t, d->ahead_next++);
            if (d->ahead_next == d->ahead->len) {
                g_array_set_size(d->ahead, 0);
                d->ahead_next = 0;
            }
            qemu_mutex_unlock(&d->lock);
            reims_vgpu_dirty_sync_one(d, token, false, 0, NULL, NULL);
            qemu_mutex_lock(&d->lock);
            d->lab_ahead_done++;
            qemu_mutex_unlock(&d->lock);
            continue;
        }
        if (d->od_stop || d->od_work_epoch != epoch || d->epoch != epoch ||
            d->od_work_next >= d->od_work->len) {
            qemu_mutex_unlock(&d->lock);
            return;
        }
        token = g_array_index(d->od_work, uint64_t, d->od_work_next++);
        qemu_mutex_unlock(&d->lock);
        reims_vgpu_dirty_sync_one(d, token, false, 0, NULL, NULL);
    }
}

static void *reims_vgpu_dirty_od_worker(void *opaque)
{
    ReimsVgpuDirty *d = opaque;
    uint64_t done = 0;

    rcu_register_thread();   /* sync_one walks the FlatView under RCU */
    reims_vgpu_lab_device_prio("dirty-od");
    qemu_mutex_lock(&d->lock);
    while (!d->od_stop) {
        uint64_t epoch = d->od_work_epoch;

        if (reims_vgpu_dirty_hash_pending(d)) {
            d->lab_hash_split_helped++;
            reims_vgpu_dirty_hash_help(d);   /* lab DIRTY_HASH */
            continue;
        }
        if (reims_vgpu_dirty_split_pending(d)) {
            reims_vgpu_dirty_split_help(d);
            continue;
        }
        if (d->ahead_next < d->ahead->len) {   /* lab SYNC_AHEAD: named reads first */
            uint64_t token = g_array_index(d->ahead, uint64_t, d->ahead_next++);

            if (d->ahead_next == d->ahead->len) {
                g_array_set_size(d->ahead, 0);
                d->ahead_next = 0;
            }
            qemu_mutex_unlock(&d->lock);
            reims_vgpu_dirty_sync_one(d, token, false, 0, NULL, NULL);
            qemu_mutex_lock(&d->lock);
            d->lab_ahead_done++;
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

/* Most tokens the ahead queue holds; past it a pass's names wait for its own reads. */
#define REIMS_VGPU_DIRTY_AHEAD_MAX 1024

/*
 * TEMP lab SYNC_AHEAD: queue the sets of `tokens` for the prefetch workers to sync for the epoch
 * current when a worker reaches them, and return. The device names the sets a pass's later chunks
 * are about to read (their planned binds), so by the time those reads run they find the sets synced
 * — a hash or a query on a worker instead of on the reader, and no wait behind another thread's.
 * Skipped as sync_tokens skips them: unknown tokens, unarmed sets (their arming sync stays with the
 * read), sets already synced for the reads' epoch, protected ones; and sets a sync is already
 * hashing. Never waits; without workers it does nothing. Safe from any thread.
 */
void reims_vgpu_dirty_sync_ahead(ReimsVgpuDirty *d, const uint64_t *tokens, size_t count)
{
    uint64_t needed;
    size_t k;

    if (!d || !tokens || count == 0 || !qatomic_read(&d->ondemand)) {
        return;
    }
    qemu_mutex_lock(&d->lock);
    if (d->od_nworkers == 0 || d->od_stop) {
        qemu_mutex_unlock(&d->lock);
        return;
    }
    needed = reims_vgpu_dirty_needed_epoch(d);
    d->lab_ahead_req += count;
    for (k = 0; k < count; k++) {
        uint64_t tk = tokens[k] & ~REIMS_VGPU_DIRTY_TOKEN_LAGGED;
        ReimsVgpuDirtySet *s = tk ? g_hash_table_lookup(d->sets, &tk) : NULL;

        if (!s || s->gen == 0 || s->wp_state == WP_ON) {
            continue;
        }
        if (qatomic_load_acquire(&s->synced_epoch) >= needed) {
            d->lab_ahead_hit++;
            continue;
        }
        if (s->hash_busy) {
            d->lab_ahead_busy++;
            continue;
        }
        if (d->ahead->len - d->ahead_next >= REIMS_VGPU_DIRTY_AHEAD_MAX) {
            d->lab_ahead_full++;
            break;
        }
        g_array_append_val(d->ahead, tk);
        d->lab_ahead_q++;
    }
    qemu_cond_broadcast(&d->od_work_cond);
    qemu_mutex_unlock(&d->lock);
}

/*
 * TEMP lab REBASE_SPLIT: the indices into `s->pages` of `gpas[0..count)` (sorted `pages`, binary search), into
 * `idx` (sized `count`); returns how many were found.
 */
static size_t reims_vgpu_dirty_rebase_indices(const ReimsVgpuDirtySet *s, const uint64_t *gpas, size_t count,
                                              size_t *idx)
{
    size_t k, m = 0;

    for (k = 0; k < count; k++) {
        uint64_t gpa = gpas[k] & ~(s->page_size - 1);
        size_t lo = 0, hi = s->count;

        while (lo < hi) {   /* `pages` is sorted */
            size_t mid = lo + (hi - lo) / 2;

            if (s->pages[mid] < gpa) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (lo < s->count && s->pages[lo] == gpa) {
            idx[m++] = lo;
        }
    }
    return m;
}

/*
 * The device itself just stored into these pages of `token`'s set (a writeback landing the frame
 * the GPU rendered there). A hashed set cannot tell those stores from the guest's: its next sync
 * sees the pages' content move and reports the device's own frame as the guest writing the
 * surface, and the reader then refuses the resident that frame came from. Take the pages' hashes
 * again now, as the set's baseline, so the generation moves only for stores made after this one.
 * A set the hypervisor answers for needs nothing: the device's stores reach neither the VGA bitmap
 * (reims_vgpu_shim_write_gpa) nor a protected page (they go through the host's mapping). Pages
 * outside the set, or outside guest RAM, are left alone. Waits for a running sync of the set.
 * Safe from any thread.
 */
void reims_vgpu_dirty_rebase(ReimsVgpuDirty *d, uint64_t token, const uint64_t *gpas, size_t count)
{
    ReimsVgpuDirtySlice *slices = NULL;
    ReimsVgpuDirtySet *s;
    size_t *idx = NULL;
    size_t m;
    int cap = 0, n;

    token &= ~REIMS_VGPU_DIRTY_TOKEN_LAGGED;
    if (!d || !token || !gpas || count == 0) {
        return;
    }
    qemu_mutex_lock(&d->lock);
    d->lab_rebase_calls++;
    s = g_hash_table_lookup(d->sets, &token);
    while (s && s->hash_busy) {
        d->lab_hash_waits++;
        qemu_cond_wait(&d->synced_cond, &d->lock);
        s = g_hash_table_lookup(d->sets, &token);
    }
    if (s && !s->hash && s->vhash && s->vhash_base) {
        /*
         * TEMP lab DIRTY_VERIFY: a hypervisor-answered set never sees the device's stores, but its baseline must
         * hold their bytes, or a guest store of those same bytes later reads as moved content. The indices are
         * taken under `lock`; the hashes (REBASE_SPLIT: on the workers when there are many) outside it, under RCU,
         * which keeps the set and its `vhash` alive. An evaluation meanwhile stores only a page's current hash
         * into its slot, the same value this rebase takes, so neither can hide a guest store from the other.
         */
        qemu_mutex_unlock(&d->lock);
        n = reims_vgpu_dirty_ram_slices_into(&slices, &cap);
        idx = g_new(size_t, count);
        rcu_read_lock();
        qemu_mutex_lock(&d->lock);
        s = g_hash_table_lookup(d->sets, &token);
        if (s && s->vhash && s->vhash_base) {
            uint64_t *out = s->vhash;

            m = reims_vgpu_dirty_rebase_indices(s, gpas, count, idx);
            qemu_mutex_unlock(&d->lock);
            reims_vgpu_dirty_rehash(d, s, slices, n, idx, m, out);
            qemu_mutex_lock(&d->lock);
            d->lab_vhash_rebase += m;
        }
        qemu_mutex_unlock(&d->lock);
        rcu_read_unlock();
        g_free(idx);
        g_free(slices);
        return;
    }
    if (!s || !s->hash || !s->hash_base) {
        /* Not hashed, or no baseline yet: the first sync takes one, over these stores too. */
        d->lab_rebase_unhashed++;
        qemu_mutex_unlock(&d->lock);
        return;
    }
    s->hash_busy = true;
    qemu_mutex_unlock(&d->lock);

    n = reims_vgpu_dirty_ram_slices_into(&slices, &cap);
    idx = g_new(size_t, count);
    rcu_read_lock();   /* the set (RCU-freed) and guest RAM stay put while hashing */
    m = reims_vgpu_dirty_rebase_indices(s, gpas, count, idx);
    /* A page outside guest RAM, or straddling two ranges, keeps its hash: its sync keeps reading it as written. */
    reims_vgpu_dirty_rehash(d, s, slices, n, idx, m, s->hash);
    rcu_read_unlock();
    g_free(idx);
    g_free(slices);

    qemu_mutex_lock(&d->lock);
    s = g_hash_table_lookup(d->sets, &token);
    if (s) {
        s->hash_busy = false;
    }
    d->lab_rebase_pages += m;
    qemu_cond_broadcast(&d->synced_cond);
    qemu_mutex_unlock(&d->lock);
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
        d->arm_flush = !(getenv("REIMS_VGPU_LAB_ARM_FLUSH") &&      /* lab A/B */
                         strcmp(getenv("REIMS_VGPU_LAB_ARM_FLUSH"), "off") == 0);
        d->pb_arm_flush = !(getenv("REIMS_VGPU_LAB_PB_ARM_FLUSH") &&  /* lab A/B */
                            strcmp(getenv("REIMS_VGPU_LAB_PB_ARM_FLUSH"), "off") == 0);
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
        /* On by default (n78b: cold foreground syncs 12.3-13.4k -> 6.8-7.0k a boot,
         * drain sync time 12.4-14.0 s -> 7.9-11.1 s); =off restores lazy arming. */
        d->hot_new = !(getenv("REIMS_VGPU_LAB_DIRTY_HOT_NEW") &&        /* lab A/B */
                       strcmp(getenv("REIMS_VGPU_LAB_DIRTY_HOT_NEW"), "off") == 0);
        d->od_hot = getenv("REIMS_VGPU_DIRTY_OD_HOT") ?                /* lab A/B */
                    MAX(1, atoi(getenv("REIMS_VGPU_DIRTY_OD_HOT"))) : 24;
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
        d->lab_nosync = getenv("REIMS_VGPU_LAB_DIRTY_NOSYNC") &&      /* lab, diagnostic */
                        strcmp(getenv("REIMS_VGPU_LAB_DIRTY_NOSYNC"), "on") == 0;
        d->gen_fast = !(getenv("REIMS_VGPU_DIRTY_GEN_FAST") &&        /* lab A/B */
                        strcmp(getenv("REIMS_VGPU_DIRTY_GEN_FAST"), "off") == 0);
        d->fgjoin_mode = !(getenv("REIMS_VGPU_LAB_DIRTY_FGJOIN") &&    /* lab A/B */
                           strcmp(getenv("REIMS_VGPU_LAB_DIRTY_FGJOIN"), "off") == 0);
        d->prep_unlocked = !(getenv("REIMS_VGPU_LAB_DIRTY_PREPLOCK") &&  /* lab A/B */
                             strcmp(getenv("REIMS_VGPU_LAB_DIRTY_PREPLOCK"), "off") == 0);
        {   /* lab A/B: DIRTY_FGSLOT=off, DIRTY_FGSLOTS=1..8 (default 4) */
            const char *fs = getenv("REIMS_VGPU_LAB_DIRTY_FGSLOTS");
            bool off = getenv("REIMS_VGPU_LAB_DIRTY_FGSLOT") &&
                       strcmp(getenv("REIMS_VGPU_LAB_DIRTY_FGSLOT"), "off") == 0;
            int want = fs ? atoi(fs) : 4;

            d->fgslots = off ? 0 : MIN(MAX(want, 1), REIMS_VGPU_DIRTY_FGSLOTS_MAX);
        }
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
        d->hash_ns = getenv("REIMS_VGPU_LAB_DIRTY_HASH_NS") ?
                     MAX(1, atoi(getenv("REIMS_VGPU_LAB_DIRTY_HASH_NS"))) : 250;
        /*
         * 16384 by default (was 4096; hq085, AMD, 5K, a boot an arm x2): the 5K surfaces (10.8k pages) are hashed
         * instead of queried - a hypervisor dirty query write-protects every page it walks again and changes nested
         * page tables the guest pays for (35 phases: WindowServer's __bzero ~ query count r=0.54, ~ WP remap time
         * 0.66), while a hash touches nothing of the guest's. Hypervisor calls 27k -> 3k a second, __bzero 1880 ->
         * 1499; with DIRTY_HASH_BG_MAX's 4096 a large hashed set is hashed when read, not at every doorbell.
         * The call stays priced at the host's ~5 us (DIRTY_CALL_NS): at 50 us the mid-size sets went to hashing
         * too and Safari's transitions lost a quarter of their frames (hq082/hq084).
         */
        /*
         * Not one number for both lab hosts. The hash moves work from the guest (the re-protection a query leaves
         * behind) onto the device, which pays off where the guest is the limiter (AMD 7950X, hq085: +17 % transition
         * frames) and not where the drain is (Intel 14700K, p009/p010, a boot an arm): there the guest's fresh page
         * got cheaper too (0.88 against 1.46 us) but the drain's foreground syncs grew (41 against 18 ms a second),
         * windowed glass ran 160.9-175.1 Hz over seven hashed boots against 192.5 / 178.4 with the sets queried,
         * transitions equal. The CPU vendor stands for that measured split; an Intel host keeps the old bounds
         * (4096, no background bound). REIMS_VGPU_LAB_DIRTY_HASH_MAX / _HASH_BG_MAX still override either way.
         */
        d->host_intel = __builtin_cpu_is("intel");
        d->hash_max = getenv("REIMS_VGPU_LAB_DIRTY_HASH_MAX") ?
                      MAX(1, atoi(getenv("REIMS_VGPU_LAB_DIRTY_HASH_MAX"))) : d->host_intel ? 4096 : 16384;
        if (d->host_intel) {
            reims_vgpu_dirty_hash_bg_max_default = 0;
        }
        if (getenv("REIMS_VGPU_LAB_DIRTY_HASH_BG_MAX")) {
            reims_vgpu_dirty_hash_bg_max_default = (uint64_t)MAX(0, atoi(getenv("REIMS_VGPU_LAB_DIRTY_HASH_BG_MAX")));
        }
        qatomic_set(&reims_vgpu_dirty_hash_bg_max_live, reims_vgpu_dirty_hash_bg_max_default);
        reims_vgpu_dirty_wp_hashed_default = d->host_intel;
        reims_vgpu_dirty_wp_hashed_min_default = d->host_intel ? 64 : 0;
        if (getenv("REIMS_VGPU_LAB_DIRTY_WP_HASHED")) {
            reims_vgpu_dirty_wp_hashed_default = strcmp(getenv("REIMS_VGPU_LAB_DIRTY_WP_HASHED"), "on") == 0;
        }
        if (getenv("REIMS_VGPU_LAB_DIRTY_WP_HASHED_MIN")) {
            reims_vgpu_dirty_wp_hashed_min_default =
                (uint64_t)MAX(0, atoi(getenv("REIMS_VGPU_LAB_DIRTY_WP_HASHED_MIN")));
        }
        qatomic_set(&reims_vgpu_dirty_wp_hashed_live, reims_vgpu_dirty_wp_hashed_default);
        qatomic_set(&reims_vgpu_dirty_wp_hashed_min_live, reims_vgpu_dirty_wp_hashed_min_default);
        fprintf(stderr, "reims-dirty-hash-config hash_max=%" PRIu64 " bg_max=%" PRIu64 " hash_ns=%" PRIu64
                " host=%s wp_hashed=%d wp_hashed_min=%" PRIu64 "\n", (uint64_t)d->hash_max,
                reims_vgpu_dirty_hash_bg_max_default, (uint64_t)d->hash_ns, d->host_intel ? "intel" : "other",
                reims_vgpu_dirty_wp_hashed_default, reims_vgpu_dirty_wp_hashed_min_default);
        /* TEMP lab DIRTY_TIGHT (opt-in for its A/B; the counts are kept when a live file may set it). */
        d->tight = getenv("REIMS_VGPU_LAB_DIRTY_TIGHT") &&
                   strcmp(getenv("REIMS_VGPU_LAB_DIRTY_TIGHT"), "on") == 0;
        d->trk_on = d->tight || getenv("REIMS_VGPU_LAB_LIVE_FILE");
        d->tight_slack = getenv("REIMS_VGPU_LAB_DIRTY_TIGHT_SLACK") ?
                         (uint64_t)MAX(0, atoi(getenv("REIMS_VGPU_LAB_DIRTY_TIGHT_SLACK"))) : 0;
#ifndef REIMS_DIRTY_STRESS
        /* TEMP lab UNTRACK_POPULATE (opt-in for its A/B). */
        d->up_on = getenv("REIMS_VGPU_LAB_UNTRACK_POPULATE") &&
                   strcmp(getenv("REIMS_VGPU_LAB_UNTRACK_POPULATE"), "on") == 0;
        if (d->up_on || getenv("REIMS_VGPU_LAB_LIVE_FILE")) {
            int t;

            d->up_ready = true;

            d->up_threads = getenv("REIMS_VGPU_LAB_UNTRACK_POPULATE_THREADS") ?
                            MIN(8, MAX(1, atoi(getenv("REIMS_VGPU_LAB_UNTRACK_POPULATE_THREADS")))) : 2;
            qemu_mutex_init(&d->up_mutex);
            qemu_cond_init(&d->up_cond);
            d->up_queue = g_array_new(FALSE, FALSE, sizeof(ReimsVgpuDirtyRun));
            for (t = 0; t < d->up_threads; t++) {
                QemuThread th;

                qemu_thread_create(&th, "reims-dirty-up", reims_vgpu_dirty_up_thread_fn, d,
                                   QEMU_THREAD_DETACHED);
            }
            fprintf(stderr, "reims-dirty-up on: threads=%d\n", d->up_threads);
        }
#endif
        /*
         * On by default (2026-10-05, interleaved x2 on both hosts with the scene modes pinned:
         * Intel n222 glass windowed 155.4/155.3 against 146.1/154.8 Hz and full screen 130.9/146.5
         * against 131.8/136.7; AMD hq042 193.9/196.3 against 189.0/191.6 and 142.0/123.1 against
         * 120.2/117.2; transitions alike within a mode; hash syncs -26-35 %, waits -34-46 %);
         * `=off` restores hashing a hot set at each of the 4 doorbells after a read.
         */
        d->hash_due = !(getenv("REIMS_VGPU_LAB_DIRTY_HASH_DUE") &&
                        strcmp(getenv("REIMS_VGPU_LAB_DIRTY_HASH_DUE"), "off") == 0);
        d->hash_all = getenv("REIMS_VGPU_LAB_DIRTY_HASH_ALL") &&
                      strcmp(getenv("REIMS_VGPU_LAB_DIRTY_HASH_ALL"), "on") == 0;
        /*
         * On by default (n176/n177, 5K Intel, def vs hash x3 each: glass 84.5 -> 93.9 and 85.9 ->
         * 112.2 Hz, scroll 86.2 -> 111.6 and 84.6 -> 107.1 Hz); `=off` restores the queried path.
         */
        qatomic_set(&d->hash_mode, !(getenv("REIMS_VGPU_LAB_DIRTY_HASH") &&   /* lab A/B */
                                     strcmp(getenv("REIMS_VGPU_LAB_DIRTY_HASH"), "off") == 0));
        d->chan_prefetch = getenv("REIMS_VGPU_LAB_DIRTY_CHAN_PREFETCH") &&   /* lab A/B */
                           strcmp(getenv("REIMS_VGPU_LAB_DIRTY_CHAN_PREFETCH"), "on") == 0;
        /* Opt-in: n170 +8 % glass, n172 -2 % (PSTAMP=off 87.6 / 101.6 against 97.7 / 87.1) — noise. */
        d->pstamp = getenv("REIMS_VGPU_LAB_DIRTY_PSTAMP") &&   /* lab A/B */
                    strcmp(getenv("REIMS_VGPU_LAB_DIRTY_PSTAMP"), "on") == 0;
        d->fg_split_min = getenv("REIMS_VGPU_LAB_DIRTY_FG_SPLIT_MIN") ?   /* lab A/B */
                          MAX(1, atoi(getenv("REIMS_VGPU_LAB_DIRTY_FG_SPLIT_MIN"))) : 4096;
        d->ticket_batch = getenv("REIMS_VGPU_LAB_DIRTY_TICKET_BATCH") ?   /* lab A/B */
                          MAX(0, atoi(getenv("REIMS_VGPU_LAB_DIRTY_TICKET_BATCH"))) : 0;
        d->settle_skip = !(getenv("REIMS_VGPU_LAB_DIRTY_SETTLE_SKIP") &&   /* lab A/B */
                           strcmp(getenv("REIMS_VGPU_LAB_DIRTY_SETTLE_SKIP"), "off") == 0);
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
            d->wp_retry = !(getenv("REIMS_VGPU_LAB_DIRTY_WP_RETRY") &&
                            strcmp(getenv("REIMS_VGPU_LAB_DIRTY_WP_RETRY"), "off") == 0);
            d->ro_skip = !(getenv("REIMS_VGPU_LAB_DIRTY_RO_SKIP") &&
                           strcmp(getenv("REIMS_VGPU_LAB_DIRTY_RO_SKIP"), "off") == 0);
            d->wp_streak_need = getenv("REIMS_VGPU_LAB_DIRTY_WP_STREAK") ?
                                MAX(1, atoi(getenv("REIMS_VGPU_LAB_DIRTY_WP_STREAK"))) : 16;
            d->wp_min_pages = getenv("REIMS_VGPU_LAB_DIRTY_WP_MIN") ?
                              MAX(1, atoi(getenv("REIMS_VGPU_LAB_DIRTY_WP_MIN"))) : 2048;
            /*
             * 4 runs a pause (~2-7 ms with every vCPU stopped), not 24: at 24 a pause
             * was 10-29 ms, which full-screen transitions (new 5K surfaces every frame,
             * protected and dropped again) hit 10-30 times a second. Same duty, same
             * drag (120 Hz, max gap 18.5 ms against 29); transitions 17-18 % of frames
             * over 25 ms against 15-28 %.
             */
            d->wp_chunk = getenv("REIMS_VGPU_LAB_DIRTY_WP_CHUNK") ?
                          MAX(1, atoi(getenv("REIMS_VGPU_LAB_DIRTY_WP_CHUNK"))) : 4;
            d->wp_duty = getenv("REIMS_VGPU_LAB_DIRTY_WP_DUTY") ?
                         MIN(100, MAX(1, atoi(getenv("REIMS_VGPU_LAB_DIRTY_WP_DUTY")))) : 10;
            d->wp_pace_ms = getenv("REIMS_VGPU_LAB_DIRTY_WP_PACE_MS") ?
                            MAX(1, atoi(getenv("REIMS_VGPU_LAB_DIRTY_WP_PACE_MS"))) : 4;
            d->wp_nopause = getenv("REIMS_VGPU_LAB_DIRTY_WP_NOPAUSE") &&
                            strcmp(getenv("REIMS_VGPU_LAB_DIRTY_WP_NOPAUSE"), "on") == 0;
#ifndef REIMS_DIRTY_STRESS
            /*
             * On by default (n153, heavy glass scene, frames of ~120 draws: 104 / 115 Hz and 11.1 /
             * 12.0 k draws/s against 91 / 9.8 k with the paused main-loop step and 75 / 8.0 k with the
             * old queueing); `=off` restores the paused main-loop step.
             */
            d->wp_thread_mode = !(getenv("REIMS_VGPU_LAB_DIRTY_WP_THREAD") &&
                                  strcmp(getenv("REIMS_VGPU_LAB_DIRTY_WP_THREAD"), "off") == 0);
#endif
            d->wp_run_ticket = !(getenv("REIMS_VGPU_LAB_DIRTY_WP_RUN_TICKET") &&
                                 strcmp(getenv("REIMS_VGPU_LAB_DIRTY_WP_RUN_TICKET"), "off") == 0);
            d->wp_thread_duty = getenv("REIMS_VGPU_LAB_DIRTY_WP_THREAD_DUTY") ?
                                MIN(100, MAX(1, atoi(getenv("REIMS_VGPU_LAB_DIRTY_WP_THREAD_DUTY")))) : 30;
            if (d->wp_thread_mode) {
                d->wp_nopause = true;
                if (!getenv("REIMS_VGPU_LAB_DIRTY_WP_CHUNK")) {
                    d->wp_chunk = 32;   /* no pause to bound: the duty paces the thread */
                }
                if (!getenv("REIMS_VGPU_LAB_DIRTY_WP_STREAK")) {
                    /*
                     * ~0.7 s of a set read every epoch: transition surfaces live shorter, and each
                     * one protected is a remap per run now and again when it is freed (n151 t256:
                     * orphans released at ~0.3-0.9 s of remaps a second).
                     */
                    d->wp_streak_need = 128;
                }
            }
            d->wp_hole =
#ifndef REIMS_DIRTY_STRESS
                         d->wp_thread_mode &&
#endif
                         d->ro_skip && getenv("REIMS_VGPU_LAB_DIRTY_WP_HOLE") &&
                         strcmp(getenv("REIMS_VGPU_LAB_DIRTY_WP_HOLE"), "on") == 0;
            d->wp_hole_max = getenv("REIMS_VGPU_LAB_DIRTY_WP_HOLE_MAX") ?
                             MAX(1, atoi(getenv("REIMS_VGPU_LAB_DIRTY_WP_HOLE_MAX"))) : 16;
            qemu_mutex_init(&d->wp_mutex);
            d->wp_queue = g_array_new(FALSE, FALSE, sizeof(uint64_t));
            d->wp_list = g_array_new(FALSE, FALSE, sizeof(uint64_t));
            d->wp_orphans = g_array_new(FALSE, FALSE, sizeof(ReimsVgpuDirtyRun));
            d->wp_dead = g_ptr_array_new();
            /* On by default since p006 (faults 12 556 -> 4, WindowServer kernel CPU -18 %); =off restores. */
            d->wp_merge = !(getenv("REIMS_VGPU_LAB_DIRTY_WP_MERGE") &&
                            strcmp(getenv("REIMS_VGPU_LAB_DIRTY_WP_MERGE"), "off") == 0);
            d->wp_merge_gap = getenv("REIMS_VGPU_LAB_DIRTY_WP_MERGE_GAP") ?
                              (uint32_t)MAX(0, atoi(getenv("REIMS_VGPU_LAB_DIRTY_WP_MERGE_GAP"))) : 512;
            d->wp_merge_runs = g_array_new(FALSE, FALSE, sizeof(ReimsVgpuDirtyRun));
            /*
             * TEMP lab DIRTY_WP_OVERLAP (see reims_vgpu_dirty_wp_kept): on by default since b110 (Intel, a boot an
             * arm, A O O A): rej_ro 600-750 -> 0, drain us a draw reveal-desktop 11.95 -> 11.15, Safari 21.55 -> 19.0,
             * Weather and Finder -2 %, Safari window fps 140 -> 146, no artifacts; the stress gate's overlap rows 0.
             * `=off` restores disjoint protection.
             */
            d->wp_overlap = !(getenv("REIMS_VGPU_LAB_DIRTY_WP_OVERLAP") &&
                              strcmp(getenv("REIMS_VGPU_LAB_DIRTY_WP_OVERLAP"), "off") == 0);
            d->wp_layout = getenv("REIMS_VGPU_LAB_DIRTY_WP_LAYOUT") != NULL;   /* TEMP lab (b357) */
            /*
             * TEMP lab DIRTY_WP_BRIDGE (see `wp_bridge`): on by default since b453 (Intel, ABAB, work USB config):
             * canvas 5K 36.5/32.9 -> 47.1/40.8 fps (+27 %), Maps 31.9/28.5 -> 32.2/32.9 (+8 %), 0 artifacts,
             * 0 panics; dirty-stress bridge rows 0 violations. `=off` restores the previous WP admission.
             */
            d->wp_bridge = d->wp_mode && d->wp_overlap &&
                           !(getenv("REIMS_VGPU_LAB_DIRTY_WP_BRIDGE") &&
                             strcmp(getenv("REIMS_VGPU_LAB_DIRTY_WP_BRIDGE"), "off") == 0);
            d->wp_fast_streak = getenv("REIMS_VGPU_LAB_DIRTY_WP_FAST_STREAK") ?
                                MAX(1, atoi(getenv("REIMS_VGPU_LAB_DIRTY_WP_FAST_STREAK"))) : 16;
            if (d->wp_bridge) {
                uint64_t np = 1ULL << (36 - qemu_target_page_bits());

                d->wp_hc = g_new0(uint16_t, np);
                d->wp_lc = g_new0(uint16_t, np);
                d->wp_bridged = g_array_new(FALSE, FALSE, sizeof(ReimsVgpuDirtyRun));
                d->wp_bridge_marks = g_array_new(FALSE, FALSE, sizeof(ReimsVgpuDirtyRun));
                if (!getenv("REIMS_VGPU_LAB_DIRTY_WP_HASHED_MIN")) {
                    reims_vgpu_dirty_wp_hashed_min_default = 16;
                    qatomic_set(&reims_vgpu_dirty_wp_hashed_min_live, 16);
                }
                fprintf(stderr, "reims-dirty-wp bridge on: gap<=%u fast_streak=%u hashed_min=%" PRIu64 "\n",
                        d->wp_merge_gap, d->wp_fast_streak, reims_vgpu_dirty_wp_hashed_min_default);
            }
            d->wp_calm = !(getenv("REIMS_VGPU_LAB_DIRTY_WP_CALM") &&
                           strcmp(getenv("REIMS_VGPU_LAB_DIRTY_WP_CALM"), "off") == 0);
            d->wp_fair = !(getenv("REIMS_VGPU_LAB_DIRTY_WP_FAIR") &&   /* TEMP lab DIRTY_WP_FAIR */
                           strcmp(getenv("REIMS_VGPU_LAB_DIRTY_WP_FAIR"), "off") == 0) ? 15 : 0;
            if (getenv("REIMS_VGPU_LAB_DIRTY_WP_FAIR_MASK")) {   /* TEMP lab: which parts (stress bisect) */
                d->wp_fair &= atoi(getenv("REIMS_VGPU_LAB_DIRTY_WP_FAIR_MASK"));
            }
            /*
             * TEMP lab, on by default since b457 (ABAB x3, Safari canvas 5K + leaving full screen): worst second of
             * guest exit time 1.8/2.0/2.4 -> 1.6/0.56/1.5 s, mean exit there 207/222/343 -> 96/39/80 us, a store
             * fault's time inside 15-17 -> 2.5 ms, fps equal (35.3/39.6/36.0 vs 35.9/39.2/37.8). `=off` restores.
             */
            d->wp_orphan_calm = !(getenv("REIMS_VGPU_LAB_DIRTY_WP_ORPHAN_CALM") &&
                                  strcmp(getenv("REIMS_VGPU_LAB_DIRTY_WP_ORPHAN_CALM"), "off") == 0);
            d->wp_orphan_near = getenv("REIMS_VGPU_LAB_DIRTY_WP_ORPHAN_NEAR_KB") ?
                                (uint64_t)MAX(0, atoi(getenv("REIMS_VGPU_LAB_DIRTY_WP_ORPHAN_NEAR_KB"))) << 10 : 0;
            d->wp_orphan_quiet_ms = getenv("REIMS_VGPU_LAB_DIRTY_WP_ORPHAN_QUIET_MS") ?
                                    MAX(1, atoi(getenv("REIMS_VGPU_LAB_DIRTY_WP_ORPHAN_QUIET_MS"))) : 300;
            d->wp_orphan_hold_ms = getenv("REIMS_VGPU_LAB_DIRTY_WP_ORPHAN_HOLD_MS") ?
                                   MAX(1, atoi(getenv("REIMS_VGPU_LAB_DIRTY_WP_ORPHAN_HOLD_MS"))) : 3000;
            if (d->wp_orphan_calm) {
                fprintf(stderr, "reims-dirty-wp orphan calm on: quiet %u ms, hold <= %u ms\n",
                        d->wp_orphan_quiet_ms, d->wp_orphan_hold_ms);
            }
            /* 1 s plus 2 ms a run (its protect and give-back cost ~0.7 ms); the stress rows use 0. */
            d->wp_age_ms = 1000;
            d->wp_age_run_us = 2000;
            if (getenv("REIMS_VGPU_LAB_DIRTY_WP_AGE_MS")) {
                d->wp_age_ms = atoi(getenv("REIMS_VGPU_LAB_DIRTY_WP_AGE_MS"));
                d->wp_age_run_us = 0;
            }
            if (d->wp_mode) {
#ifndef REIMS_DIRTY_STRESS
                d->wp_timer = timer_new_ms(QEMU_CLOCK_REALTIME, reims_vgpu_dirty_wp_bh, d);
#endif
                d->wp_bh = REIMS_WP_BH_NEW(reims_vgpu_dirty_wp_bh, d);
                if (d->wp_thread_mode) {
                    qemu_event_init(&d->wp_ev, false);
                    qemu_thread_create(&d->wp_thread, "reims-wp", reims_vgpu_dirty_wp_thread_fn, d,
                                       QEMU_THREAD_DETACHED);
                    {   /* TEMP lab (b461 freeze watch) */
                        QemuThread watch;

                        qemu_thread_create(&watch, "reims-wp-watch", reims_vgpu_dirty_wp_watch_fn, d,
                                           QEMU_THREAD_DETACHED);
                    }
                    fprintf(stderr, "reims-dirty-wp thread on: chunk=%u duty=%u%%\n", d->wp_chunk,
                            d->wp_thread_duty);
                }
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
    {
        /*
         * TEMP lab: the live switches are otherwise re-read only beside the on-demand census, which prints
         * only in seconds with foreground queries (every 30-60 s on a hashed Intel boot): a live A/B arm
         * then ran on the previous arm's values (b099). A doorbell re-reads them at most every 250 ms.
         */
        static int64_t live_next_ns;
        int64_t now = get_clock(), next = qatomic_read(&live_next_ns);

        if (now >= next && qatomic_cmpxchg(&live_next_ns, next, now + 250 * SCALE_MS) == next) {
            reims_vgpu_dirty_gap_refresh();
        }
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
    d->db_chans |= channel >= 0 && channel < 63 ? 1ULL << channel : 1ULL << 63;
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
    /* TEMP lab DIRTY_WP_HOLE; `wp_bits` is published after `wp_bits_n` (the step allocates it late). */
    const bool ro_check = d->wp_hole && !d->rsettle_mode && qatomic_read(&d->wp_bits);
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

        /*
         * TEMP lab DIRTY_WP_HOLE: a read-only page has no bit a query could find (RO_SKIP's
         * argument); the batch settles behind every earlier ticket, a protect's included.
         */
        if (ro_check && reims_vgpu_dirty_wp_ro(d, s->pages[p])) {
            continue;
        }
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
    const bool tight = reims_vgpu_dirty_tight_on(d);   /* TEMP lab DIRTY_TIGHT */
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
    bool arming = false;   /* lab ARM_FLUSH: this pass sets some set's baseline */

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
        if (qatomic_read(&set->gen) == 0) {
            arming = true;
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
    /*
     * Lab ARM_FLUSH, the batch's half of the 72c5c3e5 invariant: a set this
     * pass takes from gen 0 gets its baseline here, and a guest store still
     * sitting in a running vCPU's log is invisible to this pass's queries —
     * it would surface with the *next* epoch's query as a write after the
     * Store, and the device would reseed the page from guest zeros (the
     * fullscreen-enter stripe class). The hook is global, so one flush covers
     * the whole union; the pass queries the union with no qep skips of its
     * own, so completeness is already there. Same gate as sync_one's arming.
     */
    if (arming && d->arm_flush && d->pb_arm_flush &&
        memory_dirty_log_flush_hook) {   /* lab ARM_FLUSH; PB_ARM_FLUSH=off: the control */
        int64_t f0 = get_clock();

        memory_dirty_log_flush_hook();
        qatomic_add(&d->lab_arm_flush_ns, get_clock() - f0);
        qatomic_inc(&d->lab_pb_arm_flushes);
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
                REIMS_VGPU_DIRTY_SYNC_GAP_PAGES * target >= off &&
                reims_vgpu_dirty_bridge_ok(d, tight, (slices[i].gpa + (last->r.start + last->r.len -
                                                      slices[i].offset)) >> shift, gpa >> shift)) {
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
        if (d->pstamp && !d->od_nopticket) {   /* TEMP lab DIRTY_PSTAMP */
            d->split_pstamp = true;
            d->split_slices = slices;
            d->split_nslices = n;
            d->split_walked = walked;
            d->split_epoch = epoch;
        }
        qemu_cond_broadcast(&d->od_work_cond);
        qemu_cond_broadcast(&d->synced_cond);
        reims_vgpu_dirty_split_help(d);
        while (d->split_done < d->split_total) {
            qemu_cond_wait(&d->split_cond, &d->lock);
        }
        d->split_pstamp = false;
        d->split_slices = NULL;
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
    reims_vgpu_dirty_pstamp_pop(d);   /* TEMP lab DIRTY_PSTAMP: all of this batch's are safe now */
    if (d->pstamps && d->pstamps_head < d->pstamps->len) {
        /* A settle that waited only for overlapping tickets (RSETTLE): the stamps below cover them. */
        g_array_set_size(d->pstamps, 0);
        d->pstamps_head = 0;
    }
    d->split_walked = NULL;   /* its runs are this pass's `walked`, freed on return */
    /*
     * Stamp the set pages as queried for this epoch. The gap pages between
     * them were queried too, but no set of this batch names them, and leaving
     * them unstamped only lets a later sync of some other set query them
     * again — the harmless direction. Stamping them was three quarters of
     * this loop, run under `lock`.
     */
    for (pi = 0; pi < walked->len; pi++) {
        const MemoryRegionRange *w = &g_array_index(walked, MemoryRegionRange, pi);

        reims_vgpu_dirty_qep_stamp(d, w->start >> shift, (w->start + w->len) >> shift, epoch);
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
                " eval_us=%" PRId64 " pstamps=%" PRIu64 " pstamp_pages=%" PRIu64
                " cpf_taken=%" PRIu64 " cpf_skipped=%" PRIu64 "\n",
                g_get_real_time() / 1000, d->lab_pb_n, d->lab_pb_sets,
                d->lab_pb_setpages, d->lab_pb_pages, d->lab_pb_chunks,
                d->lab_pb_lag / 1000, d->lab_pb_build / 1000, d->lab_pb_query / 1000,
                d->lab_pb_settle / 1000, d->lab_pb_mark / 1000,
                d->lab_pb_consume / 1000, d->lab_pb_eval / 1000, d->lab_pstamps,
                d->lab_pstamp_pages, d->lab_cpf_taken, d->lab_cpf_skipped);
        d->lab_pstamps = d->lab_pstamp_pages = 0;
        d->lab_cpf_taken = d->lab_cpf_skipped = 0;
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

/* Lab DIRTY_HASH: the per-second line (hash syncs rarely pass the query path's report). d->lock held. */
static void reims_vgpu_dirty_hash_report(ReimsVgpuDirty *d)
{
    int64_t now = get_clock();

    if (!d->hash_mode || now - d->lab_hash_last < NANOSECONDS_PER_SECOND) {
        return;
    }
    d->lab_hash_last = now;
    fprintf(stderr, "reims-dirty-hash sets=%" PRIu64 " syncs=%" PRIu64 " pages=%" PRIu64
            " us=%" PRIu64 " dirty=%" PRIu64 " fg=%" PRIu64 " fg_us=%" PRIu64
            " waits=%" PRIu64 " noram=%" PRIu64 " pf=%" PRIu64 " split=%" PRIu64 " helped=%" PRIu64
            " due_skipped=%" PRIu64 " ahead_req=%" PRIu64 " ahead_q=%" PRIu64 " ahead_done=%" PRIu64
            " ahead_hit=%" PRIu64 " ahead_busy=%" PRIu64 " ahead_full=%" PRIu64
            " rebase=%" PRIu64 " rebase_pages=%" PRIu64 " rebase_unhashed=%" PRIu64
            " verify_sets=%" PRIu64 " verify_base_us=%" PRIu64 " verify_same=%" PRIu64
            " verify_moved=%" PRIu64 " verify_rebase=%" PRIu64 " rebase_split=%" PRIu64 "\n",
            d->lab_hash_sets, d->lab_hash_syncs, d->lab_hash_pages,
            d->lab_hash_ns / 1000, d->lab_hash_dirty, d->lab_hash_fg,
            d->lab_hash_fg_ns / 1000, d->lab_hash_waits, qatomic_read(&d->lab_hash_noram),
            d->lab_hash_pf, d->lab_hash_split, d->lab_hash_split_helped, d->lab_hash_due_skipped,
            d->lab_ahead_req, d->lab_ahead_q, d->lab_ahead_done, d->lab_ahead_hit,
            d->lab_ahead_busy, d->lab_ahead_full,
            d->lab_rebase_calls, d->lab_rebase_pages, d->lab_rebase_unhashed,
            qatomic_read(&d->lab_vhash_sets), qatomic_read(&d->lab_vhash_base_ns) / 1000,
            d->lab_vhash_same, d->lab_vhash_moved, d->lab_vhash_rebase, d->lab_rebase_split);
    d->lab_rebase_calls = d->lab_rebase_pages = d->lab_rebase_unhashed = 0;
    d->lab_rebase_split = 0;
    qatomic_set(&d->lab_vhash_sets, 0);
    qatomic_set(&d->lab_vhash_base_ns, 0);
    d->lab_vhash_same = d->lab_vhash_moved = d->lab_vhash_rebase = 0;
    d->lab_ahead_req = d->lab_ahead_q = d->lab_ahead_done = d->lab_ahead_hit = 0;
    d->lab_ahead_busy = d->lab_ahead_full = 0;
    d->lab_hash_split = d->lab_hash_split_helped = d->lab_hash_due_skipped = 0;
    d->lab_hash_syncs = d->lab_hash_pages = d->lab_hash_ns = d->lab_hash_dirty = 0;
    d->lab_hash_fg = d->lab_hash_fg_ns = d->lab_hash_waits = d->lab_hash_pf = 0;
    qatomic_set(&d->lab_hash_noram, 0);
}

void reims_vgpu_dirty_background(ReimsVgpuDirty *d)
{
    g_autoptr(GArray) tokens = NULL;
    g_autoptr(GArray) order = NULL;
    g_autoptr(GArray) hot_hash = NULL;   /* lab DIRTY_HASH */
    g_autoptr(GArray) hot_ord = NULL;    /* lab DIRTY_HASH: the same, with the read order */
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
    reims_vgpu_dirty_hash_report(d);   /* lab DIRTY_HASH */
    full = !d->ondemand || !d->od_prefetch || d->log_wanted ||
           g_hash_table_size(d->logged) == 0;
    qemu_mutex_unlock(&d->lock);
    if (full) {
        reims_vgpu_dirty_harvest(d);
        return;
    }
    tokens = g_array_new(FALSE, FALSE, sizeof(uint64_t));
    order = g_array_new(FALSE, FALSE, sizeof(ReimsVgpuDirtyOrdered));
    hot_hash = g_array_new(FALSE, FALSE, sizeof(uint64_t));
    hot_ord = g_array_new(FALSE, FALSE, sizeof(ReimsVgpuDirtyOrdered));
    qemu_mutex_lock(&d->lock);
    epoch = d->epoch;
    {
        /* TEMP lab DIRTY_CHAN_PREFETCH: the channels rung since the last batch, plus All-scope readers. */
        uint64_t rung = d->db_chans | (1ULL << 63);

        d->db_chans = 0;
        g_hash_table_iter_init(&it, d->sets);
        while (g_hash_table_iter_next(&it, &key, &val)) {
            ReimsVgpuDirtySet *set = val;

            if (set->hash && d->hash_mode) {
                uint64_t bgmax = qatomic_read(&reims_vgpu_dirty_hash_bg_max_live);

                if (bgmax && set->count * (set->page_size >> qemu_target_page_bits()) > bgmax &&
                    !(qatomic_read(&reims_vgpu_dirty_hash_bg_once_live) &&
                      reims_vgpu_dirty_hash_due_open(set, epoch))) {
                    continue;   /* TEMP lab live DIRTY_HASH_BG_MAX: hashed when read (or BG_ONCE) */
                }
                /*
                 * Lab DIRTY_HASH: its own hashes, on a worker, never a batch's query — and only for
                 * a set read within about a frame (~3.5 doorbells). A query's bitmap costs ~26 ns a
                 * page, so the batch keeps `od_hot` (24) epochs of sets; a hash costs ~0.3-0.5 us a
                 * page, and n177 hashed 2 TB a boot (13 GB/s, ~1.7 cores) re-hashing sets read
                 * frames ago at every doorbell. A colder set is hashed by its next read instead.
                 */
                if (set->synced_epoch < epoch) {
                    int due = d->hash_due ? reims_vgpu_dirty_hash_due(set, epoch) : -1;

                    if (due < 0 ? set->last_query_epoch + REIMS_VGPU_DIRTY_HASH_HOT >= epoch
                                : due > 0) {
                        ReimsVgpuDirtyOrdered ho = { set->read_order, *(uint64_t *)key };

                        g_array_append_val(hot_ord, ho);
                    } else if (due == 0 && set->last_query_epoch + REIMS_VGPU_DIRTY_HASH_HOT >= epoch) {
                        d->lab_hash_due_skipped++;   /* the old window would have hashed it */
                    }
                }
                continue;
            }
            if (d->chan_prefetch && set->rd_chans && !(set->rd_chans & rung) &&
                set->rd_chans_epoch + d->od_hot >= epoch) {
                d->lab_cpf_skipped++;   /* its readers' channels did not ring */
                continue;
            }
            if (set->synced_epoch < epoch && set->last_query_epoch + d->od_hot >= epoch) {
                ReimsVgpuDirtyOrdered o = { set->read_order, *(uint64_t *)key };

                d->lab_cpf_taken += d->chan_prefetch;

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
    /*
     * Lab DIRTY_HASH: the workers take the hashed sets in the order the drain first read them last
     * time, as the batch below takes the queried ones — in hash-table order the drain's first reads
     * waited for hashes the workers ran last.
     */
    g_array_sort(hot_ord, reims_vgpu_dirty_cmp_ordered);
    for (t = 0; t < hot_ord->len; t++) {
        uint64_t tk = g_array_index(hot_ord, ReimsVgpuDirtyOrdered, t).token;

        g_array_append_val(hot_hash, tk);
    }
    d->lab_hash_pf += hot_hash->len;
    if (hot_hash->len && d->od_pbatch && d->od_nworkers > 0) {
        /* Lab DIRTY_HASH: published before the batch, so the workers hash while it walks. */
        g_array_set_size(d->od_work, 0);
        g_array_append_vals(d->od_work, hot_hash->data, hot_hash->len);
        d->od_work_next = 0;
        d->od_work_epoch = epoch;
        qemu_cond_broadcast(&d->od_work_cond);
    } else if (hot_hash->len && !d->od_pbatch) {
        g_array_append_vals(tokens, hot_hash->data, hot_hash->len);
    }
    if (d->od_pbatch && d->bigsplit_mode && d->od_nworkers > 0) {
        g_autoptr(GArray) small = g_array_new(FALSE, FALSE, sizeof(uint64_t));
        guint nbig = 0;

        g_array_set_size(d->od_work, 0);
        g_array_append_vals(d->od_work, hot_hash->data, hot_hash->len);   /* lab DIRTY_HASH */
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
        if (nbig || hot_hash->len) {
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
