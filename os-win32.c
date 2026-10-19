/*
 * os-win32.c
 *
 * Copyright (c) 2003-2008 Fabrice Bellard
 * Copyright (c) 2010 Red Hat, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "qemu/osdep.h"
#include <windows.h>
#include <mmsystem.h>
#include <tlhelp32.h>
#include "system/runstate.h"

static BOOL WINAPI qemu_ctrl_handler(DWORD type)
{
    qemu_system_shutdown_request(SHUTDOWN_CAUSE_HOST_SIGNAL);
    /* Windows 7 kills application when the function returns.
       Sleep here to give QEMU a try for closing.
       Sleep period is 10000ms because Windows kills the program
       after 10 seconds anyway. */
    Sleep(10000);

    return TRUE;
}

static TIMECAPS mm_tc;

static void os_undo_timer_resolution(void)
{
    timeEndPeriod(mm_tc.wPeriodMin);
}

/*
 * Keep the timer resolution asked for above, and full execution speed, when
 * our windows are hidden.
 *
 * Since Windows 11 the system stops honouring a process's timeBeginPeriod()
 * once none of its windows is visible (occluded, minimised) and falls back to
 * the 15.6 ms default tick, and may run the process under EcoQoS. Every
 * sub-tick wait in QEMU then rounds up to 15.6 ms: a device thread polling
 * every 4 ms runs at 64 Hz. With reims-vgpu that is the display VBL, and a
 * macOS guest whose display comes up while the QEMU window is covered latches
 * its compositor to ~60 Hz for the whole boot (measured: 64.3 Hz delivered
 * against 120 Hz on boots where the window happened to be visible).
 *
 * Opting out is per process and needs no privilege; on older Windows the call
 * fails harmlessly.
 */
static void os_keep_timer_resolution(void)
{
    const char *off = getenv("QEMU_WIN_TIMER_THROTTLE_OK");   /* lab A/B */
    PROCESS_POWER_THROTTLING_STATE state = {
        .Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        .ControlMask = PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION |
                       PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
        .StateMask = 0,     /* honour timer requests; no EcoQoS */
    };

    if (off && strcmp(off, "1") == 0) {
        return;
    }
    SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling,
                          &state, sizeof(state));
}

/*
 * TEMP lab (Opus 2026-10-19, work VM crashes in nvoglv64.dll and whole-VM
 * freezes): stack traces to stderr - the faulting thread on a crash, every
 * thread on a crash or a vCPU stall - and a minidump of a crash in
 * QEMU_CRASH_DUMP_DIR (default E:/reims-lab/dumps). Frames print as
 * module+offset; ours resolve with addr2line on the exe (0x140000000 + offset).
 */
#define LAB_STACK_MAX 48
#define LAB_THREADS_MAX 512

static int lab_walk(const CONTEXT *start, DWORD64 *pcs, int max)
{
    CONTEXT c = *start;
    int n = 0;

    while (n < max && c.Rip > 0x10000 && c.Rsp > 0x10000 && !(c.Rsp & 7)) {
        DWORD64 base = 0;
        PRUNTIME_FUNCTION f;

        pcs[n++] = c.Rip;
        f = RtlLookupFunctionEntry(c.Rip, &base, NULL);
        if (f) {
            PVOID handler_data;
            DWORD64 frame;
            RtlVirtualUnwind(UNW_FLAG_NHANDLER, base, c.Rip, f, &c, &handler_data, &frame, NULL);
        } else {   /* leaf: the return address is on top of the stack */
            c.Rip = *(DWORD64 *)c.Rsp;
            c.Rsp += 8;
        }
    }
    return n;
}

static void lab_print_frames(const char *what, DWORD tid, const DWORD64 *pcs, int n)
{
    char line[LAB_STACK_MAX * 48 + 128];
    int len = snprintf(line, sizeof(line), "lab-stack %s tid %lu:", what, (unsigned long)tid);
    int i;

    for (i = 0; i < n && len < (int)sizeof(line) - 64; i++) {
        HMODULE mod = NULL;
        char path[MAX_PATH] = "?";
        const char *name = path;

        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               (LPCSTR)(uintptr_t)pcs[i], &mod) && mod &&
            GetModuleFileNameA(mod, path, sizeof(path))) {
            const char *slash = strrchr(path, '\\');
            name = slash ? slash + 1 : path;
            len += snprintf(line + len, sizeof(line) - len, " %s+%#llx", name,
                            (unsigned long long)(pcs[i] - (DWORD64)(uintptr_t)mod));
        } else {
            len += snprintf(line + len, sizeof(line) - len, " %#llx", (unsigned long long)pcs[i]);
        }
    }
    fprintf(stderr, "%s\n", line);
}

/*
 * Every thread but the caller: suspend, walk, resume - addresses only while a
 * thread is suspended (it may hold the heap or loader lock), names afterwards.
 */
void qemu_lab_dump_thread_stacks(const char *why)
{
    static DWORD tids[LAB_THREADS_MAX];
    static DWORD64 pcs[LAB_THREADS_MAX][LAB_STACK_MAX];
    static int npcs[LAB_THREADS_MAX];
    DWORD self = GetCurrentThreadId(), pid = GetCurrentProcessId();
    THREADENTRY32 te = { .dwSize = sizeof(te) };
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    int n = 0, i;

    if (snap == INVALID_HANDLE_VALUE) {
        return;
    }
    for (BOOL ok = Thread32First(snap, &te); ok && n < LAB_THREADS_MAX; ok = Thread32Next(snap, &te)) {
        HANDLE th;
        CONTEXT ctx = { .ContextFlags = CONTEXT_FULL };

        if (te.th32OwnerProcessID != pid || te.th32ThreadID == self) {
            continue;
        }
        th = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT, FALSE, te.th32ThreadID);
        if (!th) {
            continue;
        }
        if (SuspendThread(th) != (DWORD)-1) {
            if (GetThreadContext(th, &ctx)) {
                tids[n] = te.th32ThreadID;
                npcs[n] = lab_walk(&ctx, pcs[n], LAB_STACK_MAX);
                n++;
            }
            ResumeThread(th);
        }
        CloseHandle(th);
    }
    CloseHandle(snap);
    fprintf(stderr, "lab-stacks begin (%s): %d threads\n", why, n);
    for (i = 0; i < n; i++) {
        lab_print_frames(why, tids[i], pcs[i], npcs[i]);
    }
    fprintf(stderr, "lab-stacks end (%s)\n", why);
}

typedef BOOL (WINAPI *LabMiniDumpWriteDump)(HANDLE, DWORD, HANDLE, int, PVOID, PVOID, PVOID);

static LONG WINAPI lab_crash_filter(EXCEPTION_POINTERS *ep)
{
    static LONG once;
    DWORD64 pcs[LAB_STACK_MAX];
    int n;

    if (InterlockedExchange(&once, 1)) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    fprintf(stderr, "lab-crash: exception %#lx at %p (access %llu address %#llx)\n",
            (unsigned long)ep->ExceptionRecord->ExceptionCode, ep->ExceptionRecord->ExceptionAddress,
            ep->ExceptionRecord->NumberParameters > 1 ? (unsigned long long)ep->ExceptionRecord->ExceptionInformation[0] : 0ULL,
            ep->ExceptionRecord->NumberParameters > 1 ? (unsigned long long)ep->ExceptionRecord->ExceptionInformation[1] : 0ULL);
    n = lab_walk(ep->ContextRecord, pcs, LAB_STACK_MAX);
    lab_print_frames("crash", GetCurrentThreadId(), pcs, n);
    qemu_lab_dump_thread_stacks("crash");

    {
        const char *dir = getenv("QEMU_CRASH_DUMP_DIR");
        HMODULE dbghelp = LoadLibraryA("dbghelp.dll");
        LabMiniDumpWriteDump write = dbghelp ?
            (LabMiniDumpWriteDump)(void *)GetProcAddress(dbghelp, "MiniDumpWriteDump") : NULL;
        char path[MAX_PATH];
        HANDLE f;

        if (!dir || !*dir) {
            dir = "E:/reims-lab/dumps";
        }
        CreateDirectoryA(dir, NULL);
        snprintf(path, sizeof(path), "%s/qemu-%lu-%llu.dmp", dir, (unsigned long)GetCurrentProcessId(),
                 (unsigned long long)GetTickCount64());
        f = CreateFileA(path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
        if (write && f != INVALID_HANDLE_VALUE) {
            struct {
                DWORD ThreadId;
                EXCEPTION_POINTERS *ExceptionPointers;
                BOOL ClientPointers;
            } info = { GetCurrentThreadId(), ep, FALSE };
            /* MiniDumpWithIndirectlyReferencedMemory | MiniDumpWithThreadInfo */
            BOOL ok = write(GetCurrentProcess(), GetCurrentProcessId(), f, 0x40 | 0x1000, &info, NULL, NULL);
            fprintf(stderr, "lab-crash: minidump %s %s\n", path, ok ? "written" : "failed");
        }
        if (f != INVALID_HANDLE_VALUE) {
            CloseHandle(f);
        }
    }
    fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}

void os_setup_early_signal_handling(void)
{
    SetUnhandledExceptionFilter(lab_crash_filter);
    SetConsoleCtrlHandler(qemu_ctrl_handler, TRUE);
    timeGetDevCaps(&mm_tc, sizeof(mm_tc));
    timeBeginPeriod(mm_tc.wPeriodMin);
    atexit(os_undo_timer_resolution);
    os_keep_timer_resolution();
}

void os_set_line_buffering(void)
{
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
}
