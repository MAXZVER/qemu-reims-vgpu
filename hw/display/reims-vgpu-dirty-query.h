/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef REIMS_VGPU_DIRTY_QUERY_H
#define REIMS_VGPU_DIRTY_QUERY_H
#include <stdbool.h>
#include <stdint.h>

/* end is exclusive. Subtraction avoids overflowing end + gap. Querying the
 * intervening RAM adds dirty observations; it never removes a tracked page. */
static inline bool reims_dirty_query_joins(uint64_t end, uint64_t next,
                                          uint64_t max_gap)
{
    return next <= end || next - end <= max_gap;
}
#endif
