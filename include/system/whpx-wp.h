/*
 * Lab REIMS DIRTY_WP: write-protected guest pages (hw/display/reims-vgpu-dirty.c,
 * accel/whpx/whpx-common.c, target/i386/whpx/whpx-all.c).
 */
#ifndef QEMU_WHPX_WP_H
#define QEMU_WHPX_WP_H

extern bool (*whpx_wp_fault_hook)(uint64_t gpa);
extern void (*whpx_wp_invalidate_hook)(void);
int whpx_wp_remap(uint64_t gpa, uint64_t size, bool writable);
/*
 * Lab REIMS_VGPU_LAB_UNTRACK_POPULATE: fill in writable second-level entries for `n` guest-physical ranges
 * (`ranges`: n pairs of uint64_t {gpa, size}, the layout of WHV_MEMORY_RANGE_ENTRY), as if the guest had
 * written them. 0, or < 0 when the hypervisor refused or the call is unavailable.
 */
int whpx_populate_write(const uint64_t *ranges, unsigned n);
void whpx_lab_pin_ccd(const char *who);
/* Lab REIMS_VGPU_LAB_PCORES=on: prefer the hybrid CPU's performance cores for the calling thread. */
void whpx_lab_pin_pcores(const char *who);

#endif
