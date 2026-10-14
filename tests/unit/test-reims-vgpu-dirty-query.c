#include <assert.h>
#include <limits.h>
#include "reims-vgpu-dirty-query.h"

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
    return 0;
}
