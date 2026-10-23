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
 * TEMP lab LAZY_SYNC: take [gpa, gpa+size) away from the guest (`closed`) or give it back writable and tracked;
 * a guest access to a closed range exits to `whpx_fence_fault_hook`, which returns true once it may run again.
 */
extern bool (*whpx_fence_fault_hook)(uint64_t gpa);
int whpx_fence_remap(uint64_t gpa, uint64_t size, bool closed);
/*
 * TEMP lab LAZY_SYNC: the lookup the remaps do (under QEMU's RCU), and remaps of a range looked up before - for a
 * thread outside QEMU's RCU, while the range cannot move (a fence holds QEMU's RAM remaps off).
 */
int whpx_ram_lookup(uint64_t gpa, uint64_t size, void **mem, uint64_t *ram_addr);
int whpx_fence_remap_at(uint64_t gpa, uint64_t size, bool closed, void *mem, uint64_t ram_addr);
int whpx_wp_remap_at(uint64_t gpa, uint64_t size, bool writable, void *mem);
/* TEMP lab LAZY_SYNC: system/physmem.c calls it (when set) before QEMU itself reads or writes guest RAM. */
extern void (*qemu_lab_ram_fence_hook)(uint64_t addr, uint64_t len);
/* TEMP lab LAZY_SYNC: around QEMU's own remaps of guest RAM - begin waits the fences out and holds new ones off. */
extern void (*whpx_fence_remap_guard_hook)(bool begin);
/* TEMP lab (b454): the DIRTY_WP remaps since the last call - count, time, worst, 4 KiB pages; resets them. */
void whpx_lab_wp_remap_take(uint64_t *n, uint64_t *ns, uint64_t *max_ns, uint64_t *pages);
/*
 * Lab REIMS_VGPU_LAB_UNTRACK_POPULATE: fill in writable second-level entries for `n` guest-physical ranges
 * (`ranges`: n pairs of uint64_t {gpa, size}, the layout of WHV_MEMORY_RANGE_ENTRY), as if the guest had
 * written them. 0, or < 0 when the hypervisor refused or the call is unavailable.
 */
int whpx_populate_write(const uint64_t *ranges, unsigned n);
void whpx_lab_pin_ccd(const char *who);
/* Lab REIMS_VGPU_LAB_PCORES=on: prefer the hybrid CPU's performance cores for the calling thread. */
void whpx_lab_pin_pcores(const char *who);
/* TEMP lab REIMS_VGPU_LAB_EXCL_CORE=on: the drain alone on P-core 0 (whpx-common.c). */
void whpx_lab_excl_core_drain(void);

#endif
