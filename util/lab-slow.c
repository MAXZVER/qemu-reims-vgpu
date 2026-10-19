/*
 * TEMP lab (2026-10-07, devices agent): who holds QEMU's main loop.
 *
 * QEMU_LAB_SLOW_MS=<n> logs every BQL hold, bottom half, timer callback and
 * main-loop wait-object/fd handler that runs longer than n ms, with the
 * get_clock() nanosecond stamp (the clock of QEMU_USB_HOST_TRACE) and, for
 * callbacks, the function's offset in the executable for addr2line.
 * The passed-through mouse's stutter was the main loop stalling 100+ ms.
 */
#include "qemu/osdep.h"
#include "qemu/timer.h"
#include "qemu/lab-slow.h"

int64_t qemu_lab_slow_ns = -1;

int64_t qemu_lab_slow_threshold(void)
{
    if (qemu_lab_slow_ns < 0) {
        const char *e = getenv("QEMU_LAB_SLOW_MS");
        qemu_lab_slow_ns = e && *e ? (int64_t)atoi(e) * 1000000 : 0;
    }
    return qemu_lab_slow_ns;
}

void qemu_lab_slow(const char *what, const char *name, const void *fn,
                   int64_t start, int64_t end)
{
    uintptr_t off = (uintptr_t)fn;
#ifdef _WIN32
    if (fn) {
        off = (uintptr_t)fn - (uintptr_t)GetModuleHandle(NULL);
    }
#endif
    fprintf(stderr, "lab-slow %s %.1f ms at %" PRId64 " %s fn+0x%" PRIxPTR "\n",
            what, (end - start) / 1e6, start, name ? name : "-", off);
}

#ifdef _WIN32
/*
 * TEMP lab: QEMU_LAB_TIMER_1MS=1 asks Windows for a 1 ms timer resolution for
 * this process (NtSetTimerResolution) and turns its power throttling off,
 * including "ignore timer resolution", which Windows 11 applies to processes
 * without a visible window. Tests whether the guest's late answers to the
 * mouse's interrupts (20-60 ms at 24 vCPUs) are host timer ticks.
 */
static void __attribute__((constructor)) qemu_lab_timer_1ms(void)
{
    typedef LONG (NTAPI *SetResFn)(ULONG, BOOLEAN, PULONG);
    typedef BOOL (WINAPI *SetInfoFn)(HANDLE, int, LPVOID, DWORD);
    struct { ULONG Version, ControlMask, StateMask; } st = { 1, 0x5u, 0 };
    const char *e = getenv("QEMU_LAB_TIMER_1MS");
    HMODULE nt = GetModuleHandleA("ntdll.dll");
    HMODULE k32 = GetModuleHandleA("kernel32.dll");
    SetResFn setres;
    SetInfoFn setinfo;
    ULONG actual = 0;
    LONG rc = -1;
    BOOL ok = FALSE;

    if (!e || strcmp(e, "1") != 0) {
        return;
    }
    setinfo = k32 ? (SetInfoFn)(void *)GetProcAddress(k32, "SetProcessInformation") : NULL;
    if (setinfo) {
        ok = setinfo(GetCurrentProcess(), 4 /* ProcessPowerThrottling */, &st, sizeof(st));
    }
    setres = nt ? (SetResFn)(void *)GetProcAddress(nt, "NtSetTimerResolution") : NULL;
    if (setres) {
        rc = setres(10000, TRUE, &actual);
    }
    fprintf(stderr, "lab-timer-1ms throttling_off=%d rc=%ld resolution_100ns=%lu\n",
            ok ? 1 : 0, (long)rc, (unsigned long)actual);
}
#endif
