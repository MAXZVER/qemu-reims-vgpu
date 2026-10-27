/*
 * TEMP lab: one write per stderr line.
 *
 * On mingw (__USE_MINGW_ANSI_STDIO) fprintf to the unbuffered stderr goes through __mingw_pformat, which emits the
 * line a character at a time and each fputc flushes: one WriteFile per character. a1150 (AMD, Spaces scene, drain
 * sampler, slow tranches only): 1.4 % of the drain inside reims_vgpu_dirty_sync_one's once-a-second stats lines
 * (~400 characters each), printed under the dirty tracker's lock. Formatting into a buffer first and writing it
 * with one fwrite (the CRT writes an unbuffered stream's block in one call) keeps the text and its order.
 */
#ifndef QEMU_LAB_ELOG_H
#define QEMU_LAB_ELOG_H

static inline void G_GNUC_PRINTF(1, 2) lab_elog(const char *fmt, ...)
{
    char buf[8192];
    va_list ap;
    int n;

    va_start(ap, fmt);
    n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n <= 0) {
        return;
    }
    if ((size_t)n >= sizeof(buf)) {
        n = sizeof(buf) - 1;
    }
    fwrite(buf, 1, (size_t)n, stderr);
}

#endif
