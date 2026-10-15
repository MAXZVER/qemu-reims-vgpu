#include <assert.h>
#include <limits.h>
#include <string.h>
#include "reims-vgpu-dirty-query.h"

/*
 * Reference: the bitmap scan reims_vgpu_dirty_sync_tracked ran for a single
 * set before FIX 1 (mark the set's target pages, scan each logged slice of
 * each region with find_next_bit, merge, zero). The cached planner must give
 * these ranges and groups exactly.
 */
static uint64_t ref_next(const unsigned char *bm, uint64_t end, uint64_t from)
{
    while (from < end && !bm[from]) {
        from++;
    }
    return from < end ? from : end;
}

static size_t ref_plan(const uint64_t *pages, size_t count, uint64_t page_size,
                       unsigned shift, const ReimsDirtyQuerySlice *slices,
                       int n, uint64_t gap, ReimsDirtyQueryRange *out,
                       ReimsDirtyQueryGroup *groups, int *ngroups)
{
    uint64_t bits = 0, p, off;
    unsigned char *bm;
    size_t total = 0;
    int i, j, ng = 0;

    for (i = 0; i < n; i++) {
        if (slices[i].logged && (slices[i].gpa + slices[i].len) >> shift > bits) {
            bits = (slices[i].gpa + slices[i].len) >> shift;
        }
    }
    *ngroups = 0;
    if (bits == 0) {
        return 0;
    }
    bm = calloc(bits, 1);
    assert(bm);
    for (p = 0; p < count; p++) {
        for (off = 0; off < page_size; off += UINT64_C(1) << shift) {
            uint64_t pfn = (pages[p] + off) >> shift;
            if (pfn < bits) {
                bm[pfn] = 1;
            }
        }
    }
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
            uint64_t end = (sl->gpa + sl->len) >> shift;
            uint64_t pfn, first, last, next;
            if (!sl->logged || sl->mr != slices[i].mr) {
                continue;
            }
            pfn = ref_next(bm, end, sl->gpa >> shift);
            while (pfn < end) {
                first = last = pfn;
                for (;;) {
                    next = ref_next(bm, end, last + 1);
                    if (next >= end || !reims_dirty_query_joins(last + 1, next, gap)) {
                        break;
                    }
                    last = next;
                }
                out[total].start = sl->offset + ((first << shift) - sl->gpa);
                out[total].len = (last - first + 1) << shift;
                total++;
                pfn = next;
            }
        }
        groups[ng].count = total - groups[ng].first;
        ng++;
    }
    *ngroups = ng;
    free(bm);
    return total;
}

static uint64_t rng_state = 0x9e3779b97f4a7c15ull;
static uint64_t rng(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static int cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

/* A set as reims_vgpu_dirty_track stores it: aligned, sorted, deduplicated. */
static size_t make_set(uint64_t *pages, size_t want, uint64_t page_size,
                       uint64_t limit, int kind)
{
    size_t i, n = 0;
    uint64_t cursor = (rng() % (limit / page_size)) * page_size;
    for (i = 0; i < want; i++) {
        uint64_t g;
        if (kind == 0) {            /* scattered over all of guest space */
            g = rng() % limit;
        } else if (kind == 1) {     /* surface-like: runs with small holes */
            cursor += page_size * (1 + (rng() % 8 == 0 ? rng() % 300 : 0));
            g = cursor % limit;
        } else {                    /* strided rows straddling slice edges */
            g = (0x9F000 + i * page_size * 37 + (i & 1) * 0x7FF00000ull) % limit;
        }
        pages[i] = g & ~(page_size - 1);
    }
    qsort(pages, want, sizeof(*pages), cmp_u64);
    for (i = 0; i < want; i++) {
        if (n == 0 || pages[n - 1] != pages[i]) {
            pages[n++] = pages[i];
        }
    }
    return n;
}

static int A, B, C;     /* region identities */

static void q35_layout(ReimsDirtyQuerySlice *s, int *n, int variant)
{
    int k = 0;
    if (variant == 4) {
        /* A 4 KiB slice inside a 16 KiB page: planning must start mid-page. */
        s[k++] = (ReimsDirtyQuerySlice){ &A, 0, 0x9F000, 0, true };
        s[k++] = (ReimsDirtyQuerySlice){ &C, 0x9F000, 0x1000, 0, true };
    } else {
        s[k++] = (ReimsDirtyQuerySlice){ &A, 0, 0xA0000, 0, true };
    }
    s[k++] = (ReimsDirtyQuerySlice){ &B, 0xC0000, 0x40000, 0, variant == 2 };
    s[k++] = (ReimsDirtyQuerySlice){ &A, 0x100000, 0x7FF00000, 0x100000,
                                     variant != 1 };
    if (variant == 3) {
        s[k++] = (ReimsDirtyQuerySlice){ &C, 0xFD000000, 0x1000000, 0, true };
    }
    s[k++] = (ReimsDirtyQuerySlice){ &A, 0x100000000ull, 0x40000000,
                                     0x80000000, true };
    *n = k;
}

static void expect_equal(const ReimsDirtyQueryRange *r, size_t nr,
                         const ReimsDirtyQueryGroup *g, int ng,
                         const ReimsDirtyQueryRange *rr, size_t nrr,
                         const ReimsDirtyQueryGroup *rg, int nrg)
{
    size_t i;
    int k;
    assert(nr == nrr);
    assert(ng == nrg);
    for (i = 0; i < nr; i++) {
        assert(r[i].start == rr[i].start && r[i].len == rr[i].len);
    }
    for (k = 0; k < ng; k++) {
        assert(g[k].slice == rg[k].slice && g[k].first == rg[k].first &&
               g[k].count == rg[k].count);
    }
}

#define MAXP 6000
#define MAXR (MAXP * 4 + 16)

static uint64_t pages[MAXP], pages2[MAXP];
static ReimsDirtyQueryRange rr[MAXR];

static void planner_against_reference(void)
{
    static const uint64_t gaps[] = { 0, 1, 256, 1024 };
    static const size_t sizes[] = { 1, 50, 5000 };
    ReimsDirtyQuerySlice sl[8];
    ReimsDirtyQueryGroup g[8], rg[8];
    unsigned geo;
    for (geo = 0; geo < 3; geo++) {
        unsigned shift = geo == 2 ? 14 : 12;
        uint64_t page_size = geo == 0 ? 4096 : 16384;
        for (int variant = 0; variant < 5; variant++) {
            int n;
            q35_layout(sl, &n, variant);
            for (unsigned gi = 0; gi < 4; gi++) {
                for (unsigned si = 0; si < 3; si++) {
                    for (int kind = 0; kind < 3; kind++) {
                        size_t cnt = make_set(pages, sizes[si], page_size,
                                              0x150000000ull, kind);
                        int ng, nrg;
                        size_t want = ref_plan(pages, cnt, page_size, shift, sl,
                                               n, gaps[gi], rr, rg, &nrg);
                        /* Count first with no room, as the cache does. */
                        size_t got = reims_dirty_query_plan(pages, cnt, page_size,
                                                            shift, sl, n, gaps[gi],
                                                            NULL, 0, g, &ng);
                        ReimsDirtyQueryRange *r = malloc((got + 1) * sizeof(*r));
                        assert(got == want);
                        got = reims_dirty_query_plan(pages, cnt, page_size, shift,
                                                     sl, n, gaps[gi], r, got, g, &ng);
                        expect_equal(r, got, g, ng, rr, want, rg, nrg);
                        free(r);
                    }
                }
            }
        }
    }
}

/* The cache equals a fresh plan after building, hitting, every kind of key
 * change, and explicit invalidation; and reports which one happened. */
static void cache_against_fresh(void)
{
    ReimsDirtyQueryCache c;
    ReimsDirtyQuerySlice sl[8], sl2[8];
    ReimsDirtyQueryGroup rg[8];
    int n, n2, nrg;
    size_t cnt, cnt2, want;

    memset(&c, 0, sizeof(c));
    q35_layout(sl, &n, 0);
    cnt = make_set(pages, 3000, 4096, 0x150000000ull, 1);
    cnt2 = make_set(pages2, 2000, 16384, 0x150000000ull, 2);

#define CHECK(p, k, ps, sh, s, nn, gp)                                        \
    do {                                                                    \
        want = ref_plan(p, k, ps, sh, s, nn, gp, rr, rg, &nrg);             \
        expect_equal(c.ranges, c.nranges, c.groups, c.ngroups, rr, want,    \
                     rg, nrg);                                              \
    } while (0)

    assert(reims_dirty_query_cache_get(&c, pages, cnt, 4096, 12, sl, n, 256) ==
           REIMS_DIRTY_QUERY_CACHE_BUILT);
    CHECK(pages, cnt, 4096, 12, sl, n, 256);
    assert(reims_dirty_query_cache_get(&c, pages, cnt, 4096, 12, sl, n, 256) ==
           REIMS_DIRTY_QUERY_CACHE_HIT);
    CHECK(pages, cnt, 4096, 12, sl, n, 256);

    /* Slice layout changes: logging dropped, a region added, offset moved. */
    for (int variant = 1; variant <= 3; variant++) {
        q35_layout(sl2, &n2, variant);
        assert(reims_dirty_query_cache_get(&c, pages, cnt, 4096, 12, sl2, n2,
                                           256) == REIMS_DIRTY_QUERY_CACHE_REBUILT);
        CHECK(pages, cnt, 4096, 12, sl2, n2, 256);
    }
    /* Each single-field change of an otherwise identical table rebuilds. */
    assert(reims_dirty_query_cache_get(&c, pages, cnt, 4096, 12, sl, n, 256) ==
           REIMS_DIRTY_QUERY_CACHE_REBUILT);
    memcpy(sl2, sl, sizeof(sl));
    sl2[2].mr = &C;
    assert(reims_dirty_query_cache_get(&c, pages, cnt, 4096, 12, sl2, n, 256) ==
           REIMS_DIRTY_QUERY_CACHE_REBUILT);
    CHECK(pages, cnt, 4096, 12, sl2, n, 256);
    assert(reims_dirty_query_cache_get(&c, pages, cnt, 4096, 12, sl, n, 256) ==
           REIMS_DIRTY_QUERY_CACHE_REBUILT);
    sl2[2] = sl[2];
    sl2[2].gpa += 0x1000;
    sl2[2].len -= 0x1000;
    assert(reims_dirty_query_cache_get(&c, pages, cnt, 4096, 12, sl2, n, 256) ==
           REIMS_DIRTY_QUERY_CACHE_REBUILT);
    CHECK(pages, cnt, 4096, 12, sl2, n, 256);
    assert(reims_dirty_query_cache_get(&c, pages, cnt, 4096, 12, sl, n, 256) ==
           REIMS_DIRTY_QUERY_CACHE_REBUILT);
    sl2[2] = sl[2];
    sl2[2].offset += 0x1000;
    assert(reims_dirty_query_cache_get(&c, pages, cnt, 4096, 12, sl2, n, 256) ==
           REIMS_DIRTY_QUERY_CACHE_REBUILT);
    CHECK(pages, cnt, 4096, 12, sl2, n, 256);
    assert(reims_dirty_query_cache_get(&c, pages, cnt, 4096, 12, sl, n, 256) ==
           REIMS_DIRTY_QUERY_CACHE_REBUILT);
    sl2[2] = sl[2];
    sl2[3].len -= 0x200000;
    assert(reims_dirty_query_cache_get(&c, pages, cnt, 4096, 12, sl2, n, 256) ==
           REIMS_DIRTY_QUERY_CACHE_REBUILT);
    CHECK(pages, cnt, 4096, 12, sl2, n, 256);

    /* Page changes: a different page list, then fewer of the same pages. */
    assert(reims_dirty_query_cache_get(&c, pages2, cnt2, 16384, 12, sl, n, 256) ==
           REIMS_DIRTY_QUERY_CACHE_REBUILT);
    CHECK(pages2, cnt2, 16384, 12, sl, n, 256);
    assert(reims_dirty_query_cache_get(&c, pages2, cnt2 / 2, 16384, 12, sl, n,
                                       256) == REIMS_DIRTY_QUERY_CACHE_REBUILT);
    CHECK(pages2, cnt2 / 2, 16384, 12, sl, n, 256);
    assert(reims_dirty_query_cache_get(&c, pages2, cnt2 / 2, 16384, 12, sl, n,
                                       256) == REIMS_DIRTY_QUERY_CACHE_HIT);

    /* Gap change and explicit invalidation. */
    assert(reims_dirty_query_cache_get(&c, pages2, cnt2 / 2, 16384, 12, sl, n,
                                       0) == REIMS_DIRTY_QUERY_CACHE_REBUILT);
    CHECK(pages2, cnt2 / 2, 16384, 12, sl, n, 0);
    reims_dirty_query_cache_invalidate(&c);
    assert(reims_dirty_query_cache_get(&c, pages2, cnt2 / 2, 16384, 12, sl, n,
                                       0) == REIMS_DIRTY_QUERY_CACHE_REBUILT);
    CHECK(pages2, cnt2 / 2, 16384, 12, sl, n, 0);

    /* No logged slice at all: an empty plan, and it still caches. */
    for (int k = 0; k < n; k++) {
        sl2[k] = sl[k];
        sl2[k].logged = false;
    }
    assert(reims_dirty_query_cache_get(&c, pages, cnt, 4096, 12, sl2, n, 256) ==
           REIMS_DIRTY_QUERY_CACHE_REBUILT);
    assert(c.nranges == 0 && c.ngroups == 0);
    assert(reims_dirty_query_cache_get(&c, pages, cnt, 4096, 12, sl2, n, 256) ==
           REIMS_DIRTY_QUERY_CACHE_HIT);
    reims_dirty_query_cache_free(&c);
    assert(!c.valid && !c.ranges && !c.groups && !c.layout);
#undef CHECK
}

int main(void)
{
    assert(reims_dirty_query_joins(10, 10, 0));
    assert(reims_dirty_query_joins(10, 1034, 1024));
    assert(!reims_dirty_query_joins(10, 1035, 1024));
    assert(reims_dirty_query_joins(UINT64_MAX - 2, UINT64_MAX, 2));
    assert(!reims_dirty_query_joins(UINT64_MAX - 2, UINT64_MAX, 1));
    /* Both 4 KiB and 16 KiB geometries use the same exclusive-gap rule. */
    for (unsigned shift = 12; shift <= 14; shift += 2) {
        uint64_t page = UINT64_C(1) << shift;
        assert(reims_dirty_query_joins(5 * page, 1029 * page, 1024 * page));
        assert(!reims_dirty_query_joins(5 * page, 1030 * page, 1024 * page));
    }
    /* Exercise the real predicate in a range planner. Every listed page must
     * remain covered for exact, old, and wider coalescing; gap pages may join. */
    for (unsigned gap = 0; gap <= 1024; gap += 256) {
        uint64_t starts[100], ends[100], pages[100], cursor = 0;
        unsigned n = 0;
        for (unsigned i = 0; i < 100; ++i) {
            cursor += 1 + (i * 997u) % 1600;
            pages[i] = cursor;
            if (n && reims_dirty_query_joins(ends[n - 1], cursor, gap)) {
                ends[n - 1] = cursor + 1;
            } else {
                starts[n] = cursor;
                ends[n++] = cursor + 1;
            }
        }
        for (unsigned i = 0; i < 100; ++i) {
            bool covered = false;
            for (unsigned j = 0; j < n; ++j) {
                covered |= pages[i] >= starts[j] && pages[i] < ends[j];
            }
            assert(covered);
        }
    }
    planner_against_reference();
    cache_against_fresh();
    return 0;
}