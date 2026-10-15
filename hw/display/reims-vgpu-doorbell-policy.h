/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef REIMS_VGPU_DOORBELL_POLICY_H
#define REIMS_VGPU_DOORBELL_POLICY_H
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

/*
 * Which guest gfx-register writes advance the dirty observation epoch.
 *
 * Kept here, free of QEMU types, so tests/unit can hold the decision
 * (test-reims-vgpu-doorbell-policy.c) and the device (reims-vgpu-pci.c) calls
 * the same function rather than a second spelling of it.
 *
 * The register owner is the Rust model: GFX_REG_EFI_DISPLAY_IRQ in
 * crates/reims-vgpu/src/model/regs.rs:48. C has no other gfx register
 * defines, so this is the single C spelling, and regs.rs
 * (`the_cursor_doorbell_offset_matches_the_c_define`) reads it back out of
 * this file, so the two cannot drift.
 */
#define REIMS_VGPU_GFX_REG_EFI_DISPLAY_IRQ 0x1220u

/*
 * Why 0x1220 carries no work handoff.
 *
 * The epoch exists so that every guest store ordered before the guest hands
 * the device work is observed before that work can reuse a host-side copy of
 * the pages (a surface-currency verdict, a gather witness). 0x1220 hands over
 * no such work. Its whole handler (crates/reims-vgpu/src/runtime/mmio.rs, the
 * GFX_REG_EFI_DISPLAY_IRQ arm) does three things:
 *   - ORs bit `val` into the display interrupt status;
 *   - reads the 4-byte cursor position at display shared page + 0xe00 with a
 *     direct guest-physical read (read_gpa), and no cached copy is involved;
 *   - enqueues CursorUpdate + IrqGfx and wakes the BH.
 * It reads no surface, mapping, FIFO or ring memory. macOS writes it on every
 * hardware-cursor move (measured at 190-290 writes/s while the pointer moves),
 * and bumping the epoch on each one invalidated every currency verdict ~300
 * times a second and queued a background prefetch with each.
 *
 * What stays true when it is skipped. Every other gfx write still bumps,
 * exactly as before. This includes the work doorbells (FIFO written 0x1008,
 * child 0x1020/0x1028 and main kick 0x1024), so no cached Clean is served
 * across a work doorbell without being re-evaluated. Every cache hit is still
 * backed by generation + epoch + dirty witness; this only stops one register
 * from advancing the epoch. Epoch 0 still disables caching; that is the
 * tracker's rule and is unchanged.
 *
 * REIMS_VGPU_DIRTY_CURSOR_EPOCH=bump restores the old behaviour (lab A/B and
 * escape hatch). Any other value, or none, means skip.
 */
typedef enum ReimsDirtyCursorEpoch {
    REIMS_DIRTY_CURSOR_EPOCH_SKIP = 0,
    REIMS_DIRTY_CURSOR_EPOCH_BUMP = 1,
} ReimsDirtyCursorEpoch;

static inline ReimsDirtyCursorEpoch reims_dirty_cursor_epoch_mode(const char *env)
{
    return (env && strcmp(env, "bump") == 0) ? REIMS_DIRTY_CURSOR_EPOCH_BUMP
                                             : REIMS_DIRTY_CURSOR_EPOCH_SKIP;
}

typedef enum ReimsDirtyDoorbellAction {
    /* Every gfx write except the one below: note the doorbell, as always. */
    REIMS_DIRTY_DOORBELL_NOTE = 0,
    /* 0x1220 under the default mode: no epoch bump, no prefetch request.
     * Slug: dirty_epoch_skip_cursor_doorbell. */
    REIMS_DIRTY_DOORBELL_SKIP_CURSOR = 1,
    /* 0x1220 with REIMS_VGPU_DIRTY_CURSOR_EPOCH=bump: noted as before.
     * Slug: dirty_epoch_cursor_bump_forced. */
    REIMS_DIRTY_DOORBELL_NOTE_CURSOR_FORCED = 2,
} ReimsDirtyDoorbellAction;

/* Decided on the offset alone, as the Rust handler matches it. */
static inline ReimsDirtyDoorbellAction
reims_dirty_doorbell_action(uint64_t offset, ReimsDirtyCursorEpoch mode)
{
    if (offset != REIMS_VGPU_GFX_REG_EFI_DISPLAY_IRQ) {
        return REIMS_DIRTY_DOORBELL_NOTE;
    }
    return mode == REIMS_DIRTY_CURSOR_EPOCH_BUMP
               ? REIMS_DIRTY_DOORBELL_NOTE_CURSOR_FORCED
               : REIMS_DIRTY_DOORBELL_SKIP_CURSOR;
}

#endif /* REIMS_VGPU_DOORBELL_POLICY_H */