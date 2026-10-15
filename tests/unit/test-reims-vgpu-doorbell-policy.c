/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * The cursor-doorbell epoch policy (reims-vgpu-doorbell-policy.h).
 * Header-only and assert-based, like test-reims-vgpu-dirty-query.c.
 */
#include <assert.h>
#include <stdio.h>
#include "reims-vgpu-doorbell-policy.h"

/* Every other register the guest writes, and the work doorbells in
 * particular, must keep bumping in both modes. */
static const uint64_t others[] = {
    0x1000, 0x1004, 0x1008, 0x100c, 0x1010, 0x1014, 0x1018, 0x101c,
    0x1020, 0x1024, 0x1028, 0x102c, 0x1030, 0x1034,
    0x1200, 0x1204, 0x1208, 0x120c, 0x1210, 0x1214, 0x1218, 0x121c,
    0x1224, 0x1228, 0x122c, 0x1234,
    0x121f, 0x1221, 0x1222, 0x1223, 0x11220, 0x0, UINT64_MAX,
};

int main(void)
{
    size_t i;

    assert(REIMS_VGPU_GFX_REG_EFI_DISPLAY_IRQ == 0x1220u);

    /* Mode parsing: only the exact word "bump" restores the bump. */
    assert(reims_dirty_cursor_epoch_mode(NULL) == REIMS_DIRTY_CURSOR_EPOCH_SKIP);
    assert(reims_dirty_cursor_epoch_mode("") == REIMS_DIRTY_CURSOR_EPOCH_SKIP);
    assert(reims_dirty_cursor_epoch_mode("skip") == REIMS_DIRTY_CURSOR_EPOCH_SKIP);
    assert(reims_dirty_cursor_epoch_mode("BUMP") == REIMS_DIRTY_CURSOR_EPOCH_SKIP);
    assert(reims_dirty_cursor_epoch_mode("bumpx") == REIMS_DIRTY_CURSOR_EPOCH_SKIP);
    assert(reims_dirty_cursor_epoch_mode("bump") == REIMS_DIRTY_CURSOR_EPOCH_BUMP);

    /* The cursor doorbell: skipped by default, noted when forced. */
    assert(reims_dirty_doorbell_action(0x1220, REIMS_DIRTY_CURSOR_EPOCH_SKIP) ==
           REIMS_DIRTY_DOORBELL_SKIP_CURSOR);
    assert(reims_dirty_doorbell_action(0x1220, REIMS_DIRTY_CURSOR_EPOCH_BUMP) ==
           REIMS_DIRTY_DOORBELL_NOTE_CURSOR_FORCED);

    for (i = 0; i < sizeof(others) / sizeof(others[0]); i++) {
        assert(reims_dirty_doorbell_action(others[i], REIMS_DIRTY_CURSOR_EPOCH_SKIP) ==
               REIMS_DIRTY_DOORBELL_NOTE);
        assert(reims_dirty_doorbell_action(others[i], REIMS_DIRTY_CURSOR_EPOCH_BUMP) ==
               REIMS_DIRTY_DOORBELL_NOTE);
    }
    printf("test-reims-vgpu-doorbell-policy: ok\n");
    return 0;
}