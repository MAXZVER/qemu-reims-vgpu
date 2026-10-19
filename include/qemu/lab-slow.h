/* TEMP lab: see util/lab-slow.c */
#ifndef QEMU_LAB_SLOW_H
#define QEMU_LAB_SLOW_H

extern int64_t qemu_lab_slow_ns;
int64_t qemu_lab_slow_threshold(void);
void qemu_lab_slow(const char *what, const char *name, const void *fn,
                   int64_t start, int64_t end);

/* The start stamp, or 0 when the lab switch is off. */
static inline int64_t qemu_lab_slow_begin(void)
{
    return qemu_lab_slow_threshold() > 0 ? get_clock() : 0;
}

static inline void qemu_lab_slow_end(const char *what, const char *name,
                                     const void *fn, int64_t start)
{
    if (start) {
        int64_t end = get_clock();
        if (end - start > qemu_lab_slow_ns) {
            qemu_lab_slow(what, name, fn, start, end);
        }
    }
}

#endif
