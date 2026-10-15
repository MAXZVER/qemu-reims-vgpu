/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef REIMS_VGPU_DIRTY_QUERY_H
#define REIMS_VGPU_DIRTY_QUERY_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/* end is exclusive. Subtraction avoids overflowing end + gap. Querying the
 * intervening RAM adds dirty observations; it never removes a tracked page. */
static inline bool reims_dirty_query_joins(uint64_t end, uint64_t next,
                                          uint64_t max_gap)
{
    return next <= end || next - end <= max_gap;
}

/*
 * The per-set range planner (FIX 1). Kept here, free of QEMU types, so the
 * unit test can hold it against the bitmap scan it replaces for one set.
 *
 * One guest-RAM slice as the planner sees it: the region identity (compared,
 * never dereferenced), the GPA span, where that span starts in its region,
 * and whether the region is being logged. A copy of the slice table a plan
 * was built from is its layout signature.
 */
typedef struct ReimsDirtyQuerySlice {
    const void *mr;
    uint64_t gpa;
    uint64_t len;
    uint64_t offset;
    bool logged;
} ReimsDirtyQuerySlice;

typedef struct ReimsDirtyQueryRange {
    uint64_t start;         /* offset within the group's region */
    uint64_t len;
} ReimsDirtyQueryRange;

/* One batch per region: `slice` is the region's first logged slice. */
typedef struct ReimsDirtyQueryGroup {
    int slice;
    size_t first;
    size_t count;
} ReimsDirtyQueryGroup;

static inline void reims_dirty_query_emit(ReimsDirtyQueryRange *ranges,
                                          size_t cap, size_t *total,
                                          const ReimsDirtyQuerySlice *sl,
                                          uint64_t first, uint64_t last,
                                          unsigned shift)
{
    if (*total < cap) {
        /* The same arithmetic as the bitmap scan, so the same bytes. */
        ranges[*total].start = sl->offset + ((first << shift) - sl->gpa);
        ranges[*total].len = (last - first + 1) << shift;
    }
    (*total)++;
}

/*
 * The ranges one tracked set's sync covers, per region, without a bitmap of
 * guest-physical space. `pages` is the set's page list: sorted, deduplicated,
 * aligned to `page_size`, which is a power of two no smaller than the target
 * page (1 << shift) -- exactly what reims_vgpu_dirty_track stores.
 *
 * The result equals what marking that one set into a bitmap and scanning it
 * produced: regions in order of their first logged slice, every logged slice
 * of a region in table order, target pages of the slice in ascending order,
 * merged while reims_dirty_query_joins(last + 1, next, max_gap) holds, and
 * never across a slice boundary. Pages in no logged slice are never queried.
 *
 * Writes at most `ranges_cap` ranges and returns the total, so a caller can
 * size its array and call again. `groups` must have room for `n` entries and
 * always receives the complete group list; `*ngroups` is set. A region with
 * no tracked page still gets a group with count 0, as the scan issued a
 * zero-length batch for it.
 */
static inline size_t reims_dirty_query_plan(const uint64_t *pages,
                                            size_t count, uint64_t page_size,
                                            unsigned shift,
                                            const ReimsDirtyQuerySlice *slices,
                                            int n, uint64_t max_gap,
                                            ReimsDirtyQueryRange *ranges,
                                            size_t ranges_cap,
                                            ReimsDirtyQueryGroup *groups,
                                            int *ngroups)
{
    const uint64_t sub = page_size >> shift;    /* target pages per page */
    size_t total = 0;
    int i, j, ng = 0;

    for (i = 0; i < n; i++) {
        bool seen = false;

        for (j = 0; j < i; j++) {
            seen |= slices[j].logged && slices[j].mr == slices[i].mr;
        }
        if (!slices[i].logged || seen) {
            continue;
        }
        groups[ng].slice = i;
        groups[ng].first = total;
        for (j = i; j < n; j++) {
            const ReimsDirtyQuerySlice *sl = &slices[j];
            uint64_t lo = sl->gpa >> shift;
            uint64_t end = (sl->gpa + sl->len) >> shift;
            uint64_t first = 0, last = 0;
            bool open = false;
            size_t a = 0, b = count;

            if (!sl->logged || sl->mr != slices[i].mr) {
                continue;
            }
            /* First page whose last target page reaches the slice. */
            while (a < b) {
                size_t m = a + (b - a) / 2;

                if ((pages[m] >> shift) + sub - 1 < lo) {
                    a = m + 1;
                } else {
                    b = m;
                }
            }
            for (; a < count && (pages[a] >> shift) < end; a++) {
                uint64_t base = pages[a] >> shift, k;

                for (k = 0; k < sub; k++) {
                    uint64_t pfn = base + k;

                    if (pfn < lo) {
                        continue;
                    }
                    if (pfn >= end) {
                        break;
                    }
                    if (open && reims_dirty_query_joins(last + 1, pfn,
                                                        max_gap)) {
                        last = pfn;
                        continue;
                    }
                    if (open) {
                        reims_dirty_query_emit(ranges, ranges_cap, &total,
                                               sl, first, last, shift);
                    }
                    first = last = pfn;
                    open = true;
                }
            }
            if (open) {
                reims_dirty_query_emit(ranges, ranges_cap, &total, sl,
                                       first, last, shift);
            }
        }
        groups[ng].count = total - groups[ng].first;
        ng++;
    }
    *ngroups = ng;
    return total;
}

/*
 * A set's cached plan. Derived, never maintained: it is rebuilt from the
 * set's own page list whenever the key it was built against -- the page list
 * (pointer, count, page size), the target page shift, the gap, and the full
 * slice table -- differs from the caller's, and the slice table is re-read
 * from the flat view at every sync. A set's pages never change after it is
 * tracked (a new mapping incarnation is a new set), and retiring the mapping
 * frees the set and this cache with it.
 */
typedef struct ReimsDirtyQueryCache {
    ReimsDirtyQueryRange *ranges;
    size_t nranges;
    size_t ranges_cap;
    ReimsDirtyQueryGroup *groups;
    int ngroups;
    ReimsDirtyQuerySlice *layout;
    int nlayout;
    int layout_cap;
    const uint64_t *pages;
    size_t count;
    uint64_t page_size;
    unsigned shift;
    uint64_t max_gap;
    bool valid;
} ReimsDirtyQueryCache;

enum {
    REIMS_DIRTY_QUERY_CACHE_HIT,
    REIMS_DIRTY_QUERY_CACHE_BUILT,      /* first plan for this set */
    REIMS_DIRTY_QUERY_CACHE_REBUILT,    /* key changed or invalidated */
    REIMS_DIRTY_QUERY_CACHE_REFUSED,    /* allocation failed: no plan */
};

static inline void reims_dirty_query_cache_invalidate(ReimsDirtyQueryCache *c)
{
    c->valid = false;
}

static inline void reims_dirty_query_cache_free(ReimsDirtyQueryCache *c)
{
    free(c->ranges);
    free(c->groups);
    free(c->layout);
    c->ranges = NULL;
    c->groups = NULL;
    c->layout = NULL;
    c->ranges_cap = 0;
    c->layout_cap = 0;
    c->valid = false;
}

static inline bool reims_dirty_query_cache_matches(
    const ReimsDirtyQueryCache *c, const uint64_t *pages, size_t count,
    uint64_t page_size, unsigned shift, const ReimsDirtyQuerySlice *slices,
    int n, uint64_t max_gap)
{
    int i;

    if (!c->valid || c->pages != pages || c->count != count ||
        c->page_size != page_size || c->shift != shift ||
        c->max_gap != max_gap || c->nlayout != n) {
        return false;
    }
    for (i = 0; i < n; i++) {
        const ReimsDirtyQuerySlice *x = &c->layout[i], *y = &slices[i];

        if (x->mr != y->mr || x->gpa != y->gpa || x->len != y->len ||
            x->offset != y->offset || x->logged != y->logged) {
            return false;
        }
    }
    return true;
}

/*
 * Make `c` the plan for this key, rebuilding it when the key differs. Returns
 * one of the REIMS_DIRTY_QUERY_CACHE_* values; on REFUSED the cache is left
 * invalid and the caller must plan some other way.
 */
static inline int reims_dirty_query_cache_get(ReimsDirtyQueryCache *c,
                                              const uint64_t *pages,
                                              size_t count,
                                              uint64_t page_size,
                                              unsigned shift,
                                              const ReimsDirtyQuerySlice *slices,
                                              int n, uint64_t max_gap)
{
    bool built_before = c->groups != NULL;
    size_t total;
    int i, ng;

    if (reims_dirty_query_cache_matches(c, pages, count, page_size, shift,
                                        slices, n, max_gap)) {
        return REIMS_DIRTY_QUERY_CACHE_HIT;
    }
    c->valid = false;
    if (n > c->layout_cap) {
        ReimsDirtyQuerySlice *layout = realloc(c->layout,
                                               (size_t)n * sizeof(*layout));
        ReimsDirtyQueryGroup *groups;

        if (!layout) {
            return REIMS_DIRTY_QUERY_CACHE_REFUSED;
        }
        c->layout = layout;
        groups = realloc(c->groups, (size_t)n * sizeof(*groups));
        if (!groups) {
            return REIMS_DIRTY_QUERY_CACHE_REFUSED;
        }
        c->groups = groups;
        c->layout_cap = n;
    }
    if (!c->groups) {
        /* n == 0: still mark the cache built so a hit can follow. */
        c->groups = malloc(sizeof(*c->groups));
        if (!c->groups) {
            return REIMS_DIRTY_QUERY_CACHE_REFUSED;
        }
    }
    total = reims_dirty_query_plan(pages, count, page_size, shift, slices, n,
                                   max_gap, c->ranges, c->ranges_cap,
                                   c->groups, &ng);
    if (total > c->ranges_cap) {
        ReimsDirtyQueryRange *ranges = realloc(c->ranges,
                                               total * sizeof(*ranges));

        if (!ranges) {
            return REIMS_DIRTY_QUERY_CACHE_REFUSED;
        }
        c->ranges = ranges;
        c->ranges_cap = total;
        total = reims_dirty_query_plan(pages, count, page_size, shift, slices,
                                       n, max_gap, c->ranges, c->ranges_cap,
                                       c->groups, &ng);
    }
    for (i = 0; i < n; i++) {
        c->layout[i] = slices[i];
    }
    c->nranges = total;
    c->ngroups = ng;
    c->nlayout = n;
    c->pages = pages;
    c->count = count;
    c->page_size = page_size;
    c->shift = shift;
    c->max_gap = max_gap;
    c->valid = true;
    return built_before ? REIMS_DIRTY_QUERY_CACHE_REBUILT
                        : REIMS_DIRTY_QUERY_CACHE_BUILT;
}
#endif