# qemu-reims-vgpu: Windows edition

[![Host: Windows 11](https://img.shields.io/badge/host-Windows%2011-0078D4)](#build-on-windows)
[![Accelerator: WHPX](https://img.shields.io/badge/accelerator-WHPX-E65100)](#what-this-is)
[![Guest: macOS 13 Ventura](https://img.shields.io/badge/guest-macOS%2013%20Ventura-555555)](#results)
[![Status: experimental](https://img.shields.io/badge/status-experimental-yellow)](#status)
[![License: GPL-2.0](https://img.shields.io/badge/license-GPL--2.0-blue)](#license)

**The QEMU half of [reims-vgpu](https://github.com/MAXZVER/reims-vgpu) on a Windows 11 PC.**
This fork of [steelbrain/qemu-reims-vgpu](https://github.com/steelbrain/qemu-reims-vgpu)
collects the QEMU changes that make a macOS guest with the reims-vgpu paravirtual GPU fast on
Windows 11 with QEMU's WHPX accelerator (Windows Hypervisor Platform). On one lab PC, a
fullscreen CSS animation in a macOS 13 guest at 1920x1080 went from **5–7 fps to a median of
~93 fps** over the course of this work.

The default branch, `windows`, is what you build. Nearly every change in it is also offered to
the fork it came from as a pull request; the one exception is marked in
[What changed](#what-changed). This README is the only file that exists just for this fork;
QEMU's own README is [README.rst](README.rst), unchanged.

## Latest results (2026-09-29)

These come from the same lab PC (RTX 4060, Windows 11 + WHPX), with a macOS Ventura 13.7.8
guest. The figures are median present rates in Hz. Boot-to-boot variance in the lab is large, so
read them as ranges. They were measured on the lab's development trees, which carry the changes
below plus temporary diagnostics; the published branches have been compile-checked, not
re-benchmarked on their own.

| Guest | CSS animation | Full-screen CSS scroll | Wheel scroll |
|---|---|---|---|
| 1920x1080 | ~92–97 | ~116–118 | ~88–96 |
| 5120x2160 | ~98–106 | ~47–51 | ~63–71 |

What is new:

- **LLP64 dirty-bitmap fix.** `physical_memory_set_dirty_lebitmap()` used the pointer width as
  the width of the bitmap's `unsigned long` words. On Windows `long` is 32 bits, so in every
  unaligned range the bits past the first 32 pages were dropped or landed on the wrong page. The
  dropped bits were the stale tile pages; on full-screen scroll the streak detector now reads 0
  stale-page streaks.
  The misplaced bits had been marking random surfaces as written, which cost a lot at 5K; fixing
  them was a large part of the 5K speedup. The fix is in
  [steelbrain#10](https://github.com/steelbrain/qemu-reims-vgpu/pull/10).
- **On-demand dirty sync is the default.** The drain no longer waits for a dirty-log harvest per
  doorbell: each generation read syncs only its own surface's pages.
  `REIMS_VGPU_DIRTY_ONDEMAND=off` restores the per-doorbell harvest.
- **reims-vgpu's page-diff writeback is the default** on its `windows` branch. A presented
  framebuffer is written back into guest RAM page by page, with only the pages that changed and
  the pages the guest CPU wrote since the last write-back. At 5K this raised CSS animation from
  ~93 to ~100–106, wheel scroll from ~64 to ~71 and window drag from ~23 to ~28–30; 1080p is
  unchanged. `REIMS_VGPU_PAY_DIFF=off` writes whole frames back instead.

Known limits:

- 5K does not reach 120 Hz yet. Full-screen scrolling is the slowest case at ~47–51.
- At 5K the remaining cost is the per-doorbell hypervisor dirty-log queries, at about 300
  doorbells a second. Work on running the prefetch in parallel is in progress.
- A macOS 26 Tahoe guest has not been tested in the lab yet.

The older tables below are kept for comparison.

## What this is

[reims-vgpu](https://github.com/steelbrain/reims-vgpu) implements the device that macOS's
built-in `AppleParavirtGPU.kext` drives. It decodes the guest's Metal command stream and runs it
on the host GPU through Vulkan. The device has two halves:

- a Rust crate (the device model and the GPU backend), in the reims-vgpu repository;
- a thin QEMU device, `reims-vgpu-pci`, plus the QEMU tree it lives in. That is this repository,
  which reims-vgpu vendors as its `vendor/qemu` submodule.

`windows` combines:

- steelbrain's `host-reims-vgpu-vmapple` branch, which carries the reims-vgpu shims and
  [@Hi-Jiajun](https://github.com/Hi-Jiajun)'s Windows host build port;
- Hi-Jiajun's WHPX fixes (`whpx-win-stability`, open upstream as
  [steelbrain#4](https://github.com/steelbrain/qemu-reims-vgpu/pull/4)), which make macOS boot
  under WHPX at all;
- our Windows/WHPX performance and correctness changes, listed below.

## Results

The number is the present rate: frames per second the guest actually presents, from
reims-vgpu's census log (`present_hz`, one-second windows).

| Scenario | Before (same PC, morning of the bring-up) | `windows` ¹ | `experimental/direct-irq` (1 ms heartbeat) ² |
|---|---|---|---|
| CSS keyframe animation, fullscreen Safari | 5–7 | median ~93–96 (max ~114) | median ~106–112 (max ~118) |
| Safari scroll | 5–7 | median ~76–96, mean ~66–76 | median ~75–84 |
| Window drag | 5–7 | median ~40 | not settled |

**Test conditions**

- Host: Intel Core i7-14700K, 96 GB RAM, NVIDIA GeForce RTX 4060 8 GB, Windows 11 Pro.
- Guest: x86_64 macOS 13 Ventura (OSX-KVM OpenCore) at 1920x1080, 8 vCPUs, 12 GB RAM,
  QEMU + WHPX, `reims-vgpu-pci` with its host window, `REIMS_VGPU_GUEST_IMPORT=off`.
- Scripted workloads over QMP and SSH, 15–35 s per scenario, several boots per configuration.
  Ranges span boots.
- ¹ Measured on the development tree `windows` was cut from, with the reims-vgpu changes from
  [MAXZVER/reims-vgpu](https://github.com/MAXZVER/reims-vgpu). That tree also compiled in the
  code from `experimental/direct-irq` (with direct IRQ switched off for these numbers) and some
  diagnostics. `windows` itself has been compile-checked, not re-benchmarked on its own.
- ² Branch `experimental/direct-irq` with its defaults: the display heartbeat polls every 1 ms
  instead of 4 ms, and direct IRQ delivery is off. Only a few boots so far. Direct IRQ delivery
  (opt-in there) raised the animation median to ~107–110 with the 4 ms heartbeat but cut the
  scroll mean to ~57–59, and with the 1 ms heartbeat it lost outright.

These are single-machine lab numbers at 1080p. **A steady 120 fps is the goal and has not been
reached.** The reims-vgpu README has the device-side part of the story.

## What changed

Relative to steelbrain's `host-reims-vgpu-vmapple`, `windows` carries the following. Each row is
a pull request to the fork whose tree holds the code it touches.

| Change | Where it is proposed | What it does |
|---|---|---|
| WHPX fixes by Hi-Jiajun: INIT IPIs, page-fault injection, SYSENTER MSRs, CPUID leaf 4, TSC-deadline, MISC_ENABLE, XSAVE write-back, MMIO under the BQL | [steelbrain#4](https://github.com/steelbrain/qemu-reims-vgpu/pull/4) (not ours) | Gets macOS guests booting and stable under WHPX. |
| `os-win32`: keep timer resolution and full speed while QEMU's windows are hidden | [steelbrain#6](https://github.com/steelbrain/qemu-reims-vgpu/pull/6) | Windows 11 stops honouring `timeBeginPeriod()` for occluded processes. The display heartbeat then ran at 64 Hz and the guest compositor latched to 60 Hz. |
| `reims-vgpu-pci`: harvest the dirty log off the vCPU | [steelbrain#7](https://github.com/steelbrain/qemu-reims-vgpu/pull/7) | A doorbell used to hold the vCPU and the BQL for a ~10 ms dirty-log sync. Scroll median went from ~52 to ~80–96 Hz. Needs reims-vgpu ABI v21 ([reims-vgpu#110](https://github.com/steelbrain/reims-vgpu/pull/110)). |
| WHPX dirty-page tracking, `memory_region_sync_dirty_ranges()`, and a harvest that syncs only tracked pages | [steelbrain#10](https://github.com/steelbrain/qemu-reims-vgpu/pull/10) | WHPX used to report all guest RAM as written on every sync. Device copies out of guest memory fell from ~32 GB to 0.05–0.2 GB per 20 s; a harvest sync from ~22 ms to ~12 ms. |
| `reims-vgpu-shim`: keep the device's own stores out of the VGA dirty log | [steelbrain#9](https://github.com/steelbrain/qemu-reims-vgpu/pull/9) | The device's writebacks read back as guest stores and forced re-reads. |
| WHPX: serve plain MMIO loads from lockless regions without emulation | [Hi-Jiajun#1](https://github.com/Hi-Jiajun/qemu-reims-vgpu/pull/1) | A polled register read costs ~1.3–2 µs instead of ~8.5 µs. |
| `reims-vgpu-pci`: serve BAR0 reads without the BQL | [steelbrain#11](https://github.com/steelbrain/qemu-reims-vgpu/pull/11) | Lets the guest's polling reads skip the BQL, and the WHPX fast path serve them. |
| Windows build fixes: copy instead of symlink without Developer Mode; guard `munmap` | [steelbrain#8](https://github.com/steelbrain/qemu-reims-vgpu/pull/8) | `host-reims-vgpu-vmapple` does not configure or compile on Windows without them. |
| `physmem`: use `BITS_PER_LONG` in the lebitmap slow path (LLP64 hosts) | [steelbrain#10](https://github.com/steelbrain/qemu-reims-vgpu/pull/10) (fifth commit) | Fixes the dirty bits that the ranged WHPX syncs lost or misplaced on Windows. See [Latest results](#latest-results-2026-09-29). |
| `reims-vgpu-dirty`/`-pci`: on-demand dirty sync (the default) with a prefetch at each doorbell, a harvest without the BQL that takes bits atomically, and a 1 ms display heartbeat | Not proposed yet. Fork branch `windows-dirty-ondemand` | The drain stops waiting for a harvest per doorbell. `REIMS_VGPU_DIRTY_ONDEMAND=off` restores the harvest. |

The pull requests have the measurements, the caveats and how each was tested.

### Branches

- **`windows`** (default): the integration branch above. Build this.
- **`windows-dirty-ondemand`**: the on-demand dirty-sync series, merged into `windows`.
- **`experimental/direct-irq`**: `windows` plus work that is still being measured, clearly
  labelled in each commit: a 1 ms display heartbeat (tunable), a lockless WHPX APIC MSI region
  with opt-in direct MSI delivery from the device (needs reims-vgpu ABI v22, which is not
  published yet, so this branch does not build against the reims-vgpu `windows` branch), an
  opt-in lock-free vCPU pre-run, and a page-fault-intercept switch. Not for general use.
- The other branches are the pull-request branches, and steelbrain's upstream branches.

## Build on Windows

> **Verification status.** The configure flags and build script below are the ones our lab uses.
> `windows` configures, compiles (GCC 16.2, `-Werror`, no warnings) and links with those flags in a
> separate build tree against the reims-vgpu ABI v21 header. The steps as written, from a fresh
> clone, have **not** been run end to end. If a step fails for you, please open an issue.

This tree is built through reims-vgpu's build script, not on its own: its `meson.build` looks for
the reims-vgpu crate two directories above the QEMU source, so it has to sit at
`<reims-vgpu>/vendor/qemu`.

### Prerequisites

- Windows 11 with the **Windows Hypervisor Platform** optional feature enabled.
- A Vulkan driver for your GPU (the lab uses NVIDIA's on an RTX 4060).
- [MSYS2](https://www.msys2.org/), **UCRT64** environment, with the package set of the lab
  machine (not trimmed to a minimal set):

  ```sh
  pacman -S --needed git make bison flex diffutils sed grep patch unzip \
    mingw-w64-ucrt-x86_64-{gcc,binutils,pkgconf,ninja,meson,python,glib2,pixman,zstd} \
    mingw-w64-ucrt-x86_64-{libslirp,SDL2,curl,rust,vulkan-headers,vulkan-loader} \
    mingw-w64-ucrt-x86_64-{llvm-tools,spirv-tools,qemu-image-util}
  ```

  `llvm-tools` and `spirv-tools` are needed at run time: reims-vgpu's shader translator calls
  `llvm-dis` and `spirv-val`. Without them on `PATH` the guest shows a black screen.

### Steps

1. Get reims-vgpu's Windows branch and put this branch in its submodule:

   ```sh
   git clone --recurse-submodules https://github.com/MAXZVER/reims-vgpu.git
   cd reims-vgpu/vendor/qemu
   git remote add maxzver https://github.com/MAXZVER/qemu-reims-vgpu.git
   git fetch maxzver windows
   git checkout -B windows maxzver/windows
   cd ../..
   ```

   `windows` needs reims-vgpu ABI v21 (`harvests_settled`), which the reims-vgpu `windows` branch
   has. Check `REIMS_VGPU_QEMU_ABI_VERSION` in `crates/reims-vgpu/include/reims_vgpu_qemu_abi.h`
   if you mix other branches.

2. Build, in an MSYS2 UCRT64 shell at the reims-vgpu root:

   ```sh
   REIMS_VGPU_BACKEND=vulkan scripts/qemu-build/qemu-build.sh --target x86_64 --backend vulkan
   ```

   This configures `vendor/qemu` with

   ```sh
   ./configure --target-list=x86_64-softmmu --disable-hvf --disable-cocoa --disable-docs \
     --disable-bsd-user --disable-linux-user --disable-tools -Dreims_vgpu_backend=vulkan
   ```

   builds the reims-vgpu Rust staticlib with cargo, and links
   `vendor/qemu/build/qemu-system-x86_64.exe`.

   Without Developer Mode, configure copies files into `build/qemu-bundle` instead of
   symlinking them (one of the fixes here). Build outputs that do not exist yet at configure time
   are then missing from the bundle, so run the binary with `-L <reims-vgpu>/vendor/qemu/pc-bios`,
   as our lab does.

3. Guest setup and boot are described in the
   [reims-vgpu README](https://github.com/MAXZVER/reims-vgpu#build-and-run-on-windows). In short,
   the lab runs `qemu-system-x86_64.exe -accel whpx -machine q35 -device reims-vgpu-pci,...`
   with `C:\msys64\ucrt64\bin` on `PATH`, `REIMS_VGPU_WINDOW=on` and
   `REIMS_VGPU_GUEST_IMPORT=off`.

No binaries or releases are published here.

## Status

- **Experimental.** reims-vgpu calls itself alpha, and so is this.
- **One machine.** Only an Intel CPU with an NVIDIA RTX 4060 has been measured. Hi-Jiajun
  validated his WHPX fixes on an AMD Ryzen host; our changes have not been tried on AMD.
- **macOS 13 only**, at 1920x1080 and, since 2026-09-29, 5120x2160. 5K does not reach 120 Hz
  yet.

### Roadmap

1. A steady 120 fps at 1920x1080, including window drag.
2. A 5120x2160 guest.
3. A macOS 26 Tahoe guest.

Each step goes to the upstream forks as pull requests, as before.

## Related

- **[MAXZVER/reims-vgpu](https://github.com/MAXZVER/reims-vgpu)**, branch `windows`: the device
  side, with its own README, results and guest setup.
- [steelbrain/reims-vgpu](https://github.com/steelbrain/reims-vgpu) and
  [steelbrain/qemu-reims-vgpu](https://github.com/steelbrain/qemu-reims-vgpu): upstream.
- [Hi-Jiajun/qemu-reims-vgpu](https://github.com/Hi-Jiajun/qemu-reims-vgpu): the Windows port
  and WHPX fixes.

## Credits

- **The [QEMU](https://www.qemu.org/) project and its contributors**, including the WHPX
  accelerator. Everything here is built on their work.
- **[steelbrain](https://github.com/steelbrain)** (Anees Iqbal) wrote reims-vgpu and the QEMU
  shims in this tree.
- **[@Hi-Jiajun](https://github.com/Hi-Jiajun)** (Jiajun Liang) did the Windows host port and
  the WHPX fixes that got macOS guests running under WHPX.
- The other contributors to steelbrain/qemu-reims-vgpu whose work is in the base branch.

## License

QEMU's licensing is unchanged: QEMU as a whole is released under the GNU General Public License,
version 2, and individual files carry their own compatible licenses. See [LICENSE](LICENSE),
[COPYING](COPYING) and [COPYING.LIB](COPYING.LIB). Every license file and copyright notice is kept
as it is, and our changes are contributed under the terms of the files they touch.

## Legal note

Apple's macOS license agreement permits running macOS in a virtual machine only on
Apple-branded hardware. Running a macOS guest on a Windows PC falls outside those terms. This is
a research and interoperability project about GPU virtualization; it ships no Apple software, and
you are responsible for complying with the licenses of whatever you run. It is not affiliated
with, sponsored by, or endorsed by Apple Inc. macOS and Metal are trademarks of Apple Inc.

## AI-assisted development

Development in this fork was AI-assisted: the investigation, the code and the pull-request
write-ups were done with Claude (Anthropic) through Claude Code, directed by the repository
owner, and measured on the owner's machine. Commits say so in a `Co-Authored-By` trailer.

QEMU's own contribution policy
([docs/devel/code-provenance.rst](docs/devel/code-provenance.rst)) declines AI-generated
contributions, so none of this is sent to upstream QEMU. The pull requests go only to the
reims-vgpu forks this branch is built from.
