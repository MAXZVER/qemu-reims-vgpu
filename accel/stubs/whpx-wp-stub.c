/*
 * Lab REIMS hooks into accel/whpx (system/whpx-wp.h) for hosts without WHPX.
 *
 * hw/display/reims-vgpu-dirty.c and reims-vgpu-pci.c call the WHPX write-protect,
 * populate and core-pinning hooks, which exist only in accel/whpx/whpx-common.c.
 * Elsewhere (Linux/KVM) nothing can protect, populate or pin through them, so
 * each refuses the way the Windows calls do when the hypervisor declines: the
 * tracker then never protects a set and populates nothing, as with the lab
 * switches off. A separate object from whpx-stub.c so a WHPX build, which defines
 * every one of these, never links it.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "system/whpx-wp.h"

bool (*whpx_wp_fault_hook)(uint64_t gpa);
void (*whpx_wp_invalidate_hook)(void);

int whpx_wp_remap(uint64_t gpa, uint64_t size, bool writable)
{
    return -ENOSYS;
}

int whpx_populate_write(const uint64_t *ranges, unsigned n)
{
    return -ENOSYS;
}

void whpx_lab_pin_ccd(const char *who)
{
}

void whpx_lab_pin_pcores(const char *who)
{
}
