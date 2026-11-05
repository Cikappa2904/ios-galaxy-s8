# Galaxy S8+ (SM-G955F, dream2lte)

The S8 port runs on the Exynos S8+ with three changes; everything else in
[s8-install.md](s8-install.md) applies as written. Tested on a G955F with
bootloader G955FXXUCDVI1 and TWRP 3.7.0_9-0 (dream2lte): iOS 16.7.16 reaches
the home screen, buttons and touch work.

Snapdragon models (G955U/W/0) are not supported.

## What differs from the S8

| | S8+ G955F | S8 G950F |
|---|---|---|
| SoC, BOOT `sda7`, USERDATA `sda24`, EFS `sda3` | same | same |
| S-Boot framebuffer `0xcc000000` 1440x2960, DECON `0x12860000` | same | same |
| Battery MAX77865 on `hsi2c@10970000` | same | same |
| Touch | STMicro **FTS8CD56 @0x49** on `hsi2c@108E0000` | S6SY761 @0x48, same bus |
| Touch rails (S2MPS17) | LDO34 1.85 V, **LDO35 3.3 V** | LDO34 1.85 V, LDO35 3.0 V |

Handled by:
- `xnu/patches/0140-dream2lte-fts-touch.patch` (sources:
  `tools/dreamlte/dream2lte_fts.c`, `tools/dreamlte/apply-dream2lte-fts.py`):
  when nothing answers at 0x48, the touch start falls back to the FTS.
  Protocol from Samsung's `stm/fts_ts.c` (driver 0132). The kernel log shows
  `S8+ TOUCH: ...` lines.
- `shim/dreamlte/pmic.c`: LDO35 at 3.3 V.

## Speed

This branch carries the s8-network work (0125-0136): eight cores (0131-0133),
CPU clocks through ACPM (0127), `s8noexec` (0130). The ACPM DVFS plugin is a
Samsung binary: `xnu/patches/0137-stage-acpm-dvfs-fw.sh` fetches
`exynos8895_acpm_dvfs.fw.ihex` from Samsung's GPL kernel source at a fixed
commit and checksum during the CI build, so it is not in the repository.

The loader no longer runs `mngs_probe` (it powers the Mongoose cluster; with
`cpus=8` XNU does that).

## BOOT image

```bash
bash tools/dreamlte/make-boot-s8plus.sh xnu/out/kernel.development.a5xelte backup/BOOT.img s8plus-boot.img fast
```

`fast` is the tested default (8 cores, 1690/2314 MHz, 1 GiB, s8noexec, touch).
`safe` (one core, 512 MiB) and `notouch` are fallbacks for debugging.

## Host tool notes (2026)

- CI: GitHub retired the `macos-14` runners on 2026-11-02; the workflow uses
  `macos-15`, whose oldest Xcode (16.0) builds this tree unchanged. The LSE
  scan's `1 <K256>` is a false positive (SHA-256 constants table).
- `ipsw img4 extract` (v20.08.x) has no `-o` and leaves
  `DeviceTree.d22ap` LZFSE-compressed (`bvx2`): decompress it (lzfse from
  apfs-fuse's `3rdparty/lzfse`) into `research/ipsw-x/d22ap.adt`.
- `mkiosroot.sh` stops at `install -o 272` with current coreutils: run
  `tools/dreamlte/finish-iosroot.sh` afterwards (as the S8 guide says).
- WSL2 has no HFS+: build `hfsplus.ko` from `microsoft/WSL2-Linux-Kernel` at
  the running tag (`CONFIG_HFSPLUS_FS=m`); with a GCC different from
  Microsoft's it loads only with `insmod -f`.
- apfs-fuse with GCC 15 / CMake 4:
  `-DCMAKE_POLICY_VERSION_MINIMUM=3.5 "-DCMAKE_CXX_FLAGS=-include cstdint"`.
