/*
 * os-win32.c
 *
 * Copyright (c) 2003-2008 Fabrice Bellard
 * Copyright (c) 2010-2016 Red Hat, Inc.
 *
 * QEMU library functions for win32 which are shared between QEMU and
 * the QEMU tools.
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
#include "qemu/atomic.h"
#include <windows.h>
#include "qapi/error.h"
#include "qemu/main-loop.h"
#include "trace.h"
#include "qemu/sockets.h"
#include "qemu/cutils.h"
#include "qemu/error-report.h"
#include "qemu/timer.h"
#include <malloc.h>

static int get_allocation_granularity(void)
{
    SYSTEM_INFO system_info;

    GetSystemInfo(&system_info);
    return system_info.dwAllocationGranularity;
}

/*
 * TEMP (lab): REIMS_VGPU_LAB_RAM_LARGE=on backs each guest RAM block of 1 GiB or more with large pages. Guest
 * RAM then sits in the hypervisor's second-level tables as 2 MiB mappings (one walk and one TLB entry per 2 MiB
 * instead of per 4 KiB) and is never trimmed out of the working set (large pages are not pageable). Needs the
 * account's "Lock pages in memory" right (SeLockMemoryPrivilege), which this only enables in the process token;
 * without it, or without enough contiguous free memory, the block falls back to ordinary pages. One line a block.
 * On by default (n209, Intel, 40 GB guest, x2 interleaved: a guest memcpy of 1 GiB through a reused buffer 0.015 s
 * against 0.022 s; full-screen transitions and stripes unchanged within noise; a fresh guest page costs the same,
 * the guest's own pages stay 4 KiB). REIMS_VGPU_LAB_RAM_LARGE=off turns it off. whpx_lab_resident's VirtualLock
 * then refuses (87) and needs not succeed: large pages are never paged.
 */
static void *qemu_lab_large_ram_alloc(size_t size)
{
    static int on = -1;
    static bool priv;
    SIZE_T large = GetLargePageMinimum();
    size_t want;
    int64_t t0;
    void *ptr;
    DWORD err;

    if (on < 0) {
        const char *e = getenv("REIMS_VGPU_LAB_RAM_LARGE");
        HANDLE tok;

        on = !(e && strcmp(e, "off") == 0);
        if (on && OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) {
            TOKEN_PRIVILEGES tp = { .PrivilegeCount = 1 };

            if (LookupPrivilegeValueA(NULL, "SeLockMemoryPrivilege", &tp.Privileges[0].Luid)) {
                tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
                /* Succeeds with ERROR_NOT_ALL_ASSIGNED when the account does not hold the right. */
                priv = AdjustTokenPrivileges(tok, FALSE, &tp, 0, NULL, NULL) &&
                       GetLastError() == ERROR_SUCCESS;
            }
            CloseHandle(tok);
        }
    }
    if (!on || size < (1ULL << 30) || large == 0) {
        return NULL;
    }
    want = ROUND_UP(size, large);
    t0 = get_clock();
    ptr = priv ? VirtualAlloc(NULL, want, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE) : NULL;
    err = ptr ? 0 : GetLastError();
    fprintf(stderr, "reims-lab: large-page RAM size_mb=%zu page_kb=%zu priv=%d ok=%d err=%lu ms=%" PRId64 "\n",
            size >> 20, (size_t)(large >> 10), priv, ptr != NULL, priv ? err : 1314UL,
            (get_clock() - t0) / 1000000);
    return ptr;
}

/*
 * TEMP (lab, QEMU_WHPX_WP_VIEW=on): guest RAM of 1 GiB and more as one pagefile-backed section mapped twice. QEMU and
 * the device use the first view (the pointer returned); the hypervisor maps the second (qemu_win_ram_hv_view), and
 * write protection is that view's page protection (VirtualProtect), not the hypervisor's GPA map. A store from the
 * guest to a page read-only there exits like one to a GPA mapped read-only; the host's own stores go through the
 * first view and never fault. Why: a WHvMapGpaRange remap costs ~22 ns for every read-only range already in the
 * partition (whpfrag2.c, 2026-10-15: 5.5 us with none, 327 us with 16k, 740 us with 32k; mappings split in chunks do
 * not help), the WP thread remaps at 0.5-1 ms in the VM, and a canvas scene's ~10k protects take 16 s at its duty.
 * VirtualProtect on the view: 1.4 us on a guest-written page, flat to 100k protected regions (whpwp.c, whpsem.c).
 * Not with large pages (4K protection inside a large page is impossible).
 */
typedef struct QemuWinRamView {
    char *host, *hv;
    size_t size;
    HANDLE section;
} QemuWinRamView;

static QemuWinRamView qemu_win_ram_views[8];
static int qemu_win_ram_nviews;

static bool qemu_win_wp_view_on(void)
{
    static int on = -1;

    if (on < 0) {
        const char *e = getenv("QEMU_WHPX_WP_VIEW");

        /* On by default since b476 (canvas 5K 56.9 -> 66.4 fps, no 16 s warm-up); `=off` restores. */
        on = !(e && strcmp(e, "off") == 0);
    }
    return on;
}

static void *qemu_win_ram_view_alloc(size_t size)
{
    QemuWinRamView *v;
    HANDLE h;
    void *a, *b;

    if (!qemu_win_wp_view_on() || size < (1ULL << 30) ||
        qemu_win_ram_nviews == ARRAY_SIZE(qemu_win_ram_views)) {
        return NULL;
    }
    h = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE | SEC_COMMIT,
                           (DWORD)((uint64_t)size >> 32), (DWORD)size, NULL);
    if (!h) {
        fprintf(stderr, "reims-lab: wp-view section of %zu MB refused (%lu)\n", size >> 20, GetLastError());
        return NULL;
    }
    a = MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, size);
    b = a ? MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, size) : NULL;
    if (!b) {
        fprintf(stderr, "reims-lab: wp-view views of %zu MB refused (%lu)\n", size >> 20, GetLastError());
        if (a) {
            UnmapViewOfFile(a);
        }
        CloseHandle(h);
        return NULL;
    }
    v = &qemu_win_ram_views[qemu_win_ram_nviews];
    v->host = a;
    v->hv = b;
    v->size = size;
    v->section = h;
    qatomic_store_release(&qemu_win_ram_nviews, qemu_win_ram_nviews + 1);
    fprintf(stderr, "reims-lab: wp-view RAM size_mb=%zu host=%p hv=%p\n", size >> 20, a, b);
    return a;
}

void *qemu_win_ram_hv_view(void *host, size_t len)
{
    int i, n = qatomic_load_acquire(&qemu_win_ram_nviews);

    for (i = 0; i < n; i++) {
        QemuWinRamView *v = &qemu_win_ram_views[i];

        if ((char *)host >= v->host && (char *)host + len <= v->host + v->size) {
            return v->hv + ((char *)host - v->host);
        }
    }
    return host;
}

/*
 * TEMP (lab, WP_VIEW): one linear window over scattered pages of guest RAM - the Windows form of the POSIX packed
 * alias (MAP_SHARED | MAP_FIXED per page) a device builds when a task's virtually linear surface sits on
 * guest-physical pages that are not. Guest RAM under WP_VIEW is one pagefile section, so each page can be mapped
 * once more: reserve a placeholder for the whole window, split it per page and replace each piece with a view of
 * the section at that page's offset. An alias, not a copy - guest CPU stores, the device's stores and the GPU's
 * reads all address the same pages. Measured (lab/tools/session-2026-10-15/whp/placeholder4k.c): 1.8 us a page,
 * page-aligned section offsets accepted, write-through exact.
 *
 * The three calls are looked up at run time: the build targets an older Windows API level than the one that
 * declares them (Windows 10 1803).
 */
#ifndef MEM_RESERVE_PLACEHOLDER
#define MEM_RESERVE_PLACEHOLDER 0x00040000
#endif
#ifndef MEM_REPLACE_PLACEHOLDER
#define MEM_REPLACE_PLACEHOLDER 0x00004000
#endif
#ifndef MEM_PRESERVE_PLACEHOLDER
#define MEM_PRESERVE_PLACEHOLDER 0x00000002
#endif

typedef PVOID (WINAPI *QemuVirtualAlloc2Fn)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, void *, ULONG);
typedef PVOID (WINAPI *QemuMapViewOfFile3Fn)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T, ULONG, ULONG, void *, ULONG);
typedef BOOL (WINAPI *QemuUnmapViewOfFile2Fn)(HANDLE, PVOID, ULONG);

static QemuVirtualAlloc2Fn qemu_win_va2;
static QemuMapViewOfFile3Fn qemu_win_mv3;
static QemuUnmapViewOfFile2Fn qemu_win_uv2;

static bool qemu_win_placeholder_api(void)
{
    static int state = -1;   /* benign race: every thread resolves the same three pointers */

    if (qatomic_read(&state) < 0) {
        HMODULE k = GetModuleHandleW(L"kernelbase.dll");

        if (k) {
            qemu_win_va2 = (QemuVirtualAlloc2Fn)(void *)GetProcAddress(k, "VirtualAlloc2");
            qemu_win_mv3 = (QemuMapViewOfFile3Fn)(void *)GetProcAddress(k, "MapViewOfFile3");
            qemu_win_uv2 = (QemuUnmapViewOfFile2Fn)(void *)GetProcAddress(k, "UnmapViewOfFile2");
        }
        qatomic_set(&state, qemu_win_va2 && qemu_win_mv3 && qemu_win_uv2);
    }
    return qatomic_read(&state) > 0;
}

/* The WP_VIEW section and byte offset behind `len` bytes of guest RAM at `host` (QEMU's view). */
static bool qemu_win_ram_section_of(const void *host, size_t len, HANDLE *section, uint64_t *offset)
{
    int i, n = qatomic_load_acquire(&qemu_win_ram_nviews);

    for (i = 0; i < n; i++) {
        QemuWinRamView *v = &qemu_win_ram_views[i];

        if (v->section && (const char *)host >= v->host && (const char *)host + len <= v->host + v->size) {
            *section = v->section;
            *offset = (uint64_t)((const char *)host - v->host);
            return true;
        }
    }
    return false;
}

void *qemu_win_ram_alias_pages(void *const *hosts, size_t count, size_t page, unsigned long *err,
                               size_t *failed_index)
{
    HANDLE proc = GetCurrentProcess();
    char *view;
    size_t i = 0;
    bool split = false;

    *err = 0;
    *failed_index = SIZE_MAX;
    if (count == 0 || !qemu_win_placeholder_api()) {
        *err = ERROR_NOT_SUPPORTED;
        return NULL;
    }
    view = qemu_win_va2(proc, NULL, count * page, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, NULL, 0);
    if (!view) {
        *err = GetLastError();
        return NULL;
    }
    for (i = 0; i < count; i++) {
        HANDLE section;
        uint64_t offset;

        split = false;
        if (!qemu_win_ram_section_of(hosts[i], page, &section, &offset) || (offset & (page - 1))) {
            *err = ERROR_INVALID_ADDRESS;
            goto fail;
        }
        /* Split this page off the placeholder that still spans [i, count); the last page is already alone. */
        if (i + 1 < count) {
            if (!VirtualFree(view + i * page, page, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
                *err = GetLastError();
                goto fail;
            }
            split = true;
        }
        if (!qemu_win_mv3(section, proc, view + i * page, offset, page, MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE,
                          NULL, 0)) {
            *err = GetLastError();
            goto fail;
        }
    }
    return view;

fail:
    *failed_index = i;
    {
        size_t j;

        for (j = 0; j < i; j++) {
            qemu_win_uv2(proc, view + j * page, 0);
        }
    }
    /* Page i's own placeholder (or the unsplit rest from i), then the rest after a split. */
    VirtualFree(view + i * page, 0, MEM_RELEASE);
    if (split && i + 1 < count) {
        VirtualFree(view + (i + 1) * page, 0, MEM_RELEASE);
    }
    return NULL;
}

void qemu_win_ram_alias_free(void *view, size_t count, size_t page)
{
    HANDLE proc = GetCurrentProcess();
    size_t i;

    if (!view || !qemu_win_placeholder_api()) {
        return;
    }
    for (i = 0; i < count; i++) {
        qemu_win_uv2(proc, (char *)view + i * page, 0);
    }
}

void *qemu_anon_ram_alloc(size_t size, uint64_t *align, bool shared,
                          bool noreserve)
{
    void *ptr;

    if (noreserve) {
        /*
         * We need a MEM_COMMIT before accessing any memory in a MEM_RESERVE
         * area; we cannot easily mimic POSIX MAP_NORESERVE semantics.
         */
        error_report("Skipping reservation of swap space is not supported.");
        return NULL;
    }

    ptr = qemu_win_ram_view_alloc(size);
    if (!ptr) {
        ptr = qemu_lab_large_ram_alloc(size);
    }
    if (!ptr) {
        ptr = VirtualAlloc(NULL, size, MEM_COMMIT, PAGE_READWRITE);
    }
    trace_qemu_anon_ram_alloc(size, ptr);

    if (ptr && align) {
        *align = MAX(get_allocation_granularity(), getpagesize());
    }
    return ptr;
}

void qemu_anon_ram_free(void *ptr, size_t size)
{
    int i;

    trace_qemu_anon_ram_free(ptr, size);
    for (i = 0; ptr && i < qemu_win_ram_nviews; i++) {
        QemuWinRamView *v = &qemu_win_ram_views[i];

        if (v->host == ptr && v->section) {   /* TEMP (lab, QEMU_WHPX_WP_VIEW) */
            UnmapViewOfFile(v->hv);
            UnmapViewOfFile(v->host);
            CloseHandle(v->section);
            v->section = NULL;
            v->size = 0;
            return;
        }
    }
    if (ptr) {
        VirtualFree(ptr, 0, MEM_RELEASE);
    }
}

#ifndef _POSIX_THREAD_SAFE_FUNCTIONS
/* FIXME: add proper locking */
struct tm *gmtime_r(const time_t *timep, struct tm *result)
{
    struct tm *p = gmtime(timep);
    memset(result, 0, sizeof(*result));
    if (p) {
        *result = *p;
        p = result;
    }
    return p;
}

/* FIXME: add proper locking */
struct tm *localtime_r(const time_t *timep, struct tm *result)
{
    struct tm *p = localtime(timep);
    memset(result, 0, sizeof(*result));
    if (p) {
        *result = *p;
        p = result;
    }
    return p;
}
#endif /* _POSIX_THREAD_SAFE_FUNCTIONS */

static int socket_error(void)
{
    switch (WSAGetLastError()) {
    case 0:
        return 0;
    case WSAEINTR:
        return EINTR;
    case WSAEINVAL:
        return EINVAL;
    case WSA_INVALID_HANDLE:
        return EBADF;
    case WSA_NOT_ENOUGH_MEMORY:
        return ENOMEM;
    case WSA_INVALID_PARAMETER:
        return EINVAL;
    case WSAENAMETOOLONG:
        return ENAMETOOLONG;
    case WSAENOTEMPTY:
        return ENOTEMPTY;
    case WSAEWOULDBLOCK:
         /* not using EWOULDBLOCK as we don't want code to have
          * to check both EWOULDBLOCK and EAGAIN */
        return EAGAIN;
    case WSAEINPROGRESS:
        return EINPROGRESS;
    case WSAEALREADY:
        return EALREADY;
    case WSAENOTSOCK:
        return ENOTSOCK;
    case WSAEDESTADDRREQ:
        return EDESTADDRREQ;
    case WSAEMSGSIZE:
        return EMSGSIZE;
    case WSAEPROTOTYPE:
        return EPROTOTYPE;
    case WSAENOPROTOOPT:
        return ENOPROTOOPT;
    case WSAEPROTONOSUPPORT:
        return EPROTONOSUPPORT;
    case WSAEOPNOTSUPP:
        return EOPNOTSUPP;
    case WSAEAFNOSUPPORT:
        return EAFNOSUPPORT;
    case WSAEADDRINUSE:
        return EADDRINUSE;
    case WSAEADDRNOTAVAIL:
        return EADDRNOTAVAIL;
    case WSAENETDOWN:
        return ENETDOWN;
    case WSAENETUNREACH:
        return ENETUNREACH;
    case WSAENETRESET:
        return ENETRESET;
    case WSAECONNABORTED:
        return ECONNABORTED;
    case WSAECONNRESET:
        return ECONNRESET;
    case WSAENOBUFS:
        return ENOBUFS;
    case WSAEISCONN:
        return EISCONN;
    case WSAENOTCONN:
        return ENOTCONN;
    case WSAETIMEDOUT:
        return ETIMEDOUT;
    case WSAECONNREFUSED:
        return ECONNREFUSED;
    case WSAELOOP:
        return ELOOP;
    case WSAEHOSTUNREACH:
        return EHOSTUNREACH;
    default:
        return EIO;
    }
}

bool qemu_set_blocking(int fd, bool block, Error **errp)
{
    unsigned long opt = block ? 0 : 1;

    if (block) {
        qemu_socket_unselect_nofail(fd);
    }

    if (ioctlsocket(fd, FIONBIO, &opt) != NO_ERROR) {
        error_setg_errno(errp, socket_error(),
                         "Can't set file descriptor %d %s", fd,
                         block ? "blocking" : "non-blocking");
        return false;
    }

    return true;
}

int socket_set_fast_reuse(int fd)
{
    /* Enabling the reuse of an endpoint that was used by a socket still in
     * TIME_WAIT state is usually performed by setting SO_REUSEADDR. On Windows
     * fast reuse is the default and SO_REUSEADDR does strange things. So we
     * don't have to do anything here. More info can be found at:
     * http://msdn.microsoft.com/en-us/library/windows/desktop/ms740621.aspx */
    return 0;
}

int inet_aton(const char *cp, struct in_addr *ia)
{
    uint32_t addr = inet_addr(cp);
    if (addr == 0xffffffff) {
        return 0;
    }
    ia->s_addr = addr;
    return 1;
}

void qemu_set_cloexec(int fd)
{
}

void qemu_clear_cloexec(int fd)
{
}

int qemu_get_thread_id(void)
{
    return GetCurrentThreadId();
}

char *
qemu_get_local_state_dir(void)
{
    const char * const *data_dirs = g_get_system_data_dirs();

    g_assert(data_dirs && data_dirs[0]);

    return g_strdup(data_dirs[0]);
}

void qemu_set_tty_echo(int fd, bool echo)
{
    HANDLE handle = (HANDLE)_get_osfhandle(fd);
    DWORD dwMode = 0;

    if (handle == INVALID_HANDLE_VALUE) {
        return;
    }

    GetConsoleMode(handle, &dwMode);

    if (echo) {
        SetConsoleMode(handle, dwMode | ENABLE_ECHO_INPUT | ENABLE_LINE_INPUT);
    } else {
        SetConsoleMode(handle,
                       dwMode & ~(ENABLE_ECHO_INPUT | ENABLE_LINE_INPUT));
    }
}

int getpagesize(void)
{
    /*
     * GetSystemInfo is a system call (NtQueryInformationProcess) and this is
     * asked on every dirty-log sync (physical_memory_set_dirty_lebitmap via
     * qemu_real_host_page_size), hundreds of thousands of times a second
     * under reims-vgpu's on-demand dirty tracking. The page size cannot
     * change while the process runs.
     */
    static int page_size;
    int ps = qatomic_read(&page_size);

    if (!ps) {
        SYSTEM_INFO system_info;

        GetSystemInfo(&system_info);
        ps = system_info.dwPageSize;
        qatomic_set(&page_size, ps);
    }
    return ps;
}

bool qemu_prealloc_mem(int fd, char *area, size_t sz, int max_threads,
                       ThreadContext *tc, bool async, Error **errp)
{
    int i;
    size_t pagesize = qemu_real_host_page_size();

    sz = (sz + pagesize - 1) & -pagesize;
    for (i = 0; i < sz / pagesize; i++) {
        memset(area + pagesize * i, 0, 1);
    }

    return true;
}

bool qemu_finish_async_prealloc_mem(Error **errp)
{
    /* async prealloc not supported, there is nothing to finish */
    return true;
}

char *qemu_get_pid_name(pid_t pid)
{
    /* XXX Implement me */
    abort();
}


bool qemu_socket_select(int sockfd, WSAEVENT hEventObject,
                        long lNetworkEvents, Error **errp)
{
    SOCKET s = _get_osfhandle(sockfd);

    if (s == INVALID_SOCKET) {
        error_setg(errp, "invalid socket fd=%d", sockfd);
        return false;
    }

    if (WSAEventSelect(s, hEventObject, lNetworkEvents) != 0) {
        error_setg_win32(errp, WSAGetLastError(), "failed to WSAEventSelect()");
        return false;
    }

    return true;
}

bool qemu_socket_unselect(int sockfd, Error **errp)
{
    return qemu_socket_select(sockfd, NULL, 0, errp);
}

void qemu_socket_select_nofail(int sockfd, WSAEVENT hEventObject,
                               long lNetworkEvents)
{
    Error *err = NULL;

    if (!qemu_socket_select(sockfd, hEventObject, lNetworkEvents, &err)) {
        warn_report_err(err);
    }
}

void qemu_socket_unselect_nofail(int sockfd)
{
    Error *err = NULL;

    if (!qemu_socket_unselect(sockfd, &err)) {
        warn_report_err(err);
    }
}

int qemu_socketpair(int domain, int type, int protocol, int sv[2])
{
    struct sockaddr_un addr = {
        0,
    };
    socklen_t socklen;
    int listener = -1;
    int client = -1;
    int server = -1;
    g_autofree char *path = NULL;
    int tmpfd;
    u_long arg;
    int ret = -1;

    g_return_val_if_fail(sv != NULL, -1);

    addr.sun_family = AF_UNIX;
    socklen = sizeof(addr);

    tmpfd = g_file_open_tmp(NULL, &path, NULL);
    if (tmpfd == -1 || !path) {
        errno = EACCES;
        goto out;
    }

    close(tmpfd);

    if (strlen(path) >= sizeof(addr.sun_path)) {
        errno = EINVAL;
        goto out;
    }

    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);

    listener = socket(domain, type, protocol);
    if (listener == -1) {
        goto out;
    }

    if (DeleteFile(path) == 0 && GetLastError() != ERROR_FILE_NOT_FOUND) {
        errno = EACCES;
        goto out;
    }
    g_clear_pointer(&path, g_free);

    if (bind(listener, (struct sockaddr *)&addr, socklen) == -1) {
        goto out;
    }

    if (listen(listener, 1) == -1) {
        goto out;
    }

    client = socket(domain, type, protocol);
    if (client == -1) {
        goto out;
    }

    arg = 1;
    if (ioctlsocket(client, FIONBIO, &arg) != NO_ERROR) {
        goto out;
    }

    if (connect(client, (struct sockaddr *)&addr, socklen) == -1 &&
        WSAGetLastError() != WSAEWOULDBLOCK) {
        goto out;
    }

    server = accept(listener, NULL, NULL);
    if (server == -1) {
        goto out;
    }

    arg = 0;
    if (ioctlsocket(client, FIONBIO, &arg) != NO_ERROR) {
        goto out;
    }

    arg = 0;
    if (ioctlsocket(client, SIO_AF_UNIX_GETPEERPID, &arg) != NO_ERROR) {
        goto out;
    }

    if (arg != GetCurrentProcessId()) {
        errno = EPERM;
        goto out;
    }

    sv[0] = server;
    server = -1;
    sv[1] = client;
    client = -1;
    ret = 0;

out:
    if (listener != -1) {
        close(listener);
    }
    if (client != -1) {
        close(client);
    }
    if (server != -1) {
        close(server);
    }
    if (path) {
        DeleteFile(path);
    }
    return ret;
}

#undef connect
int qemu_connect_wrap(int sockfd, const struct sockaddr *addr,
                      socklen_t addrlen)
{
    int ret;
    SOCKET s = _get_osfhandle(sockfd);

    if (s == INVALID_SOCKET) {
        return -1;
    }

    ret = connect(s, addr, addrlen);
    if (ret < 0) {
        if (WSAGetLastError() == WSAEWOULDBLOCK) {
            errno = EINPROGRESS;
        } else {
            errno = socket_error();
        }
    }
    return ret;
}


#undef listen
int qemu_listen_wrap(int sockfd, int backlog)
{
    int ret;
    SOCKET s = _get_osfhandle(sockfd);

    if (s == INVALID_SOCKET) {
        return -1;
    }

    ret = listen(s, backlog);
    if (ret < 0) {
        errno = socket_error();
    }
    return ret;
}


#undef bind
int qemu_bind_wrap(int sockfd, const struct sockaddr *addr,
                   socklen_t addrlen)
{
    int ret;
    SOCKET s = _get_osfhandle(sockfd);

    if (s == INVALID_SOCKET) {
        return -1;
    }

    ret = bind(s, addr, addrlen);
    if (ret < 0) {
        errno = socket_error();
    }
    return ret;
}

QEMU_USED EXCEPTION_DISPOSITION
win32_close_exception_handler(struct _EXCEPTION_RECORD *exception_record,
                              void *registration, struct _CONTEXT *context,
                              void *dispatcher)
{
    return EXCEPTION_EXECUTE_HANDLER;
}

#undef close
int qemu_close_socket_osfhandle(int fd)
{
    SOCKET s = _get_osfhandle(fd);
    DWORD flags = 0;

    /*
     * If we were to just call _close on the descriptor, it would close the
     * HANDLE, but it wouldn't free any of the resources associated to the
     * SOCKET, and we can't call _close after calling closesocket, because
     * closesocket has already closed the HANDLE, and _close would attempt to
     * close the HANDLE again, resulting in a double free. We can however
     * protect the HANDLE from actually being closed long enough to close the
     * file descriptor, then close the socket itself.
     */
    if (!GetHandleInformation((HANDLE)s, &flags)) {
        errno = EACCES;
        return -1;
    }

    if (!SetHandleInformation((HANDLE)s, HANDLE_FLAG_PROTECT_FROM_CLOSE, HANDLE_FLAG_PROTECT_FROM_CLOSE)) {
        errno = EACCES;
        return -1;
    }

    __try1(win32_close_exception_handler) {
        /*
         * close() returns EBADF since we PROTECT_FROM_CLOSE the underlying
         * handle, but the FD is actually freed
         */
        if (close(fd) < 0 && errno != EBADF) {
            return -1;
        }
    }
    __except1 {
    }

    if (!SetHandleInformation((HANDLE)s, flags, flags)) {
        errno = EACCES;
        return -1;
    }

    return 0;
}

int qemu_close_wrap(int fd)
{
    SOCKET s = INVALID_SOCKET;
    int ret = -1;

    if (!fd_is_socket(fd)) {
        return close(fd);
    }

    s = _get_osfhandle(fd);
    qemu_close_socket_osfhandle(fd);

    ret = closesocket(s);
    if (ret < 0) {
        errno = socket_error();
    }

    return ret;
}


#undef socket
int qemu_socket_wrap(int domain, int type, int protocol)
{
    SOCKET s;
    int fd;

    s = socket(domain, type, protocol);
    if (s == -1) {
        errno = socket_error();
        return -1;
    }

    fd = _open_osfhandle(s, _O_BINARY);
    if (fd < 0) {
        closesocket(s);
        /* _open_osfhandle may not set errno, and closesocket() may override it */
        errno = ENOMEM;
    }

    return fd;
}


#undef accept
int qemu_accept_wrap(int sockfd, struct sockaddr *addr,
                     socklen_t *addrlen)
{
    int fd;
    SOCKET s = _get_osfhandle(sockfd);

    if (s == INVALID_SOCKET) {
        return -1;
    }

    s = accept(s, addr, addrlen);
    if (s == -1) {
        errno = socket_error();
        return -1;
    }

    fd = _open_osfhandle(s, _O_BINARY);
    if (fd < 0) {
        closesocket(s);
        /* _open_osfhandle may not set errno, and closesocket() may override it */
        errno = ENOMEM;
    }

    return fd;
}


#undef shutdown
int qemu_shutdown_wrap(int sockfd, int how)
{
    int ret;
    SOCKET s = _get_osfhandle(sockfd);

    if (s == INVALID_SOCKET) {
        return -1;
    }

    ret = shutdown(s, how);
    if (ret < 0) {
        errno = socket_error();
    }
    return ret;
}


#undef ioctlsocket
int qemu_ioctlsocket_wrap(int fd, int req, void *val)
{
    int ret;
    SOCKET s = _get_osfhandle(fd);

    if (s == INVALID_SOCKET) {
        return -1;
    }

    ret = ioctlsocket(s, req, val);
    if (ret < 0) {
        errno = socket_error();
    }
    return ret;
}


#undef getsockopt
int qemu_getsockopt_wrap(int sockfd, int level, int optname,
                         void *optval, socklen_t *optlen)
{
    int ret;
    SOCKET s = _get_osfhandle(sockfd);

    if (s == INVALID_SOCKET) {
        return -1;
    }

    ret = getsockopt(s, level, optname, optval, optlen);
    if (ret < 0) {
        errno = socket_error();
    }
    return ret;
}


#undef setsockopt
int qemu_setsockopt_wrap(int sockfd, int level, int optname,
                         const void *optval, socklen_t optlen)
{
    int ret;
    SOCKET s = _get_osfhandle(sockfd);

    if (s == INVALID_SOCKET) {
        return -1;
    }

    ret = setsockopt(s, level, optname, optval, optlen);
    if (ret < 0) {
        errno = socket_error();
    }
    return ret;
}


#undef getpeername
int qemu_getpeername_wrap(int sockfd, struct sockaddr *addr,
                          socklen_t *addrlen)
{
    int ret;
    SOCKET s = _get_osfhandle(sockfd);

    if (s == INVALID_SOCKET) {
        return -1;
    }

    ret = getpeername(s, addr, addrlen);
    if (ret < 0) {
        errno = socket_error();
    }
    return ret;
}


#undef getsockname
int qemu_getsockname_wrap(int sockfd, struct sockaddr *addr,
                          socklen_t *addrlen)
{
    int ret;
    SOCKET s = _get_osfhandle(sockfd);

    if (s == INVALID_SOCKET) {
        return -1;
    }

    ret = getsockname(s, addr, addrlen);
    if (ret < 0) {
        errno = socket_error();
    }
    return ret;
}


#undef send
ssize_t qemu_send_wrap(int sockfd, const void *buf, size_t len, int flags)
{
    int ret;
    SOCKET s = _get_osfhandle(sockfd);

    if (s == INVALID_SOCKET) {
        return -1;
    }

    ret = send(s, buf, len, flags);
    if (ret < 0) {
        errno = socket_error();
    }
    return ret;
}


#undef sendto
ssize_t qemu_sendto_wrap(int sockfd, const void *buf, size_t len, int flags,
                         const struct sockaddr *addr, socklen_t addrlen)
{
    int ret;
    SOCKET s = _get_osfhandle(sockfd);

    if (s == INVALID_SOCKET) {
        return -1;
    }

    ret = sendto(s, buf, len, flags, addr, addrlen);
    if (ret < 0) {
        errno = socket_error();
    }
    return ret;
}


#undef recv
ssize_t qemu_recv_wrap(int sockfd, void *buf, size_t len, int flags)
{
    int ret;
    SOCKET s = _get_osfhandle(sockfd);

    if (s == INVALID_SOCKET) {
        return -1;
    }

    ret = recv(s, buf, len, flags);
    if (ret < 0) {
        errno = socket_error();
    }
    return ret;
}


#undef recvfrom
ssize_t qemu_recvfrom_wrap(int sockfd, void *buf, size_t len, int flags,
                           struct sockaddr *addr, socklen_t *addrlen)
{
    int ret;
    SOCKET s = _get_osfhandle(sockfd);

    if (s == INVALID_SOCKET) {
        return -1;
    }

    ret = recvfrom(s, buf, len, flags, addr, addrlen);
    if (ret < 0) {
        errno = socket_error();
    }
    return ret;
}

bool qemu_write_pidfile(const char *filename, Error **errp)
{
    char buffer[128];
    int len;
    HANDLE file;
    OVERLAPPED overlap;
    BOOL ret;
    memset(&overlap, 0, sizeof(overlap));

    file = CreateFile(filename, GENERIC_WRITE, FILE_SHARE_READ, NULL,
                      OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

    if (file == INVALID_HANDLE_VALUE) {
        error_setg(errp, "Failed to create PID file");
        return false;
    }
    len = snprintf(buffer, sizeof(buffer), FMT_pid "\n", (pid_t)getpid());
    ret = WriteFile(file, (LPCVOID)buffer, (DWORD)len,
                    NULL, &overlap);
    CloseHandle(file);
    if (ret == 0) {
        error_setg(errp, "Failed to write PID file");
        return false;
    }
    return true;
}

size_t qemu_get_host_physmem(void)
{
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);

    if (GlobalMemoryStatusEx(&statex)) {
        return statex.ullTotalPhys;
    }
    return 0;
}

int qemu_msync(void *addr, size_t length, int fd)
{
    /**
     * Perform the sync based on the file descriptor
     * The sync range will most probably be wider than the one
     * requested - but it will still get the job done
     */
    return qemu_fdatasync(fd);
}

void *qemu_win32_map_alloc(size_t size, HANDLE *h, Error **errp)
{
    void *bits;

    trace_win32_map_alloc(size);

    *h = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0,
                          size, NULL);
    if (*h == NULL) {
        error_setg_win32(errp, GetLastError(), "Failed to CreateFileMapping");
        return NULL;
    }

    bits = MapViewOfFile(*h, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (bits == NULL) {
        error_setg_win32(errp, GetLastError(), "Failed to MapViewOfFile");
        CloseHandle(*h);
        return NULL;
    }

    return bits;
}

void qemu_win32_map_free(void *ptr, HANDLE h, Error **errp)
{
    trace_win32_map_free(ptr, h);

    if (UnmapViewOfFile(ptr) == 0) {
        error_setg_win32(errp, GetLastError(), "Failed to UnmapViewOfFile");
    }
    CloseHandle(h);
}

int qemu_shm_alloc(size_t size, Error **errp)
{
    error_setg(errp, "Shared memory is not supported.");
    return -1;
}

int qemu_ftruncate64(int fd, int64_t length)
{
    LARGE_INTEGER li;
    DWORD dw;
    LONG high;
    HANDLE h;
    BOOL res;

    if ((GetVersion()&0x80000000UL) && (length >> 32) != 0) {
        return -1;
    }

    h = (HANDLE)_get_osfhandle(fd);

    /* get current position, ftruncate do not change position */
    li.HighPart = 0;
    li.LowPart = SetFilePointer (h, 0, &li.HighPart, FILE_CURRENT);
    if (li.LowPart == INVALID_SET_FILE_POINTER && GetLastError() != NO_ERROR) {
        return -1;
    }

    high = length >> 32;
    dw = SetFilePointer(h, (DWORD) length, &high, FILE_BEGIN);
    if (dw == INVALID_SET_FILE_POINTER && GetLastError() != NO_ERROR) {
        return -1;
    }
    res = SetEndOfFile(h);

    /* back to old position */
    SetFilePointer(h, li.LowPart, &li.HighPart, FILE_BEGIN);
    return res ? 0 : -1;
}
