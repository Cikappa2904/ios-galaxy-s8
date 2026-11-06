#!/bin/bash
# Galaxy S8+ (SM-G955F) BOOT image with the boot-args tested on the phone.
#   make-boot-s8plus.sh KERNEL STOCK_BOOT.img OUT.img [fast|gpu|metal|safe|notouch]
#
# fast (default, tested 2026-11-05): eight cores, A53 1690 / Mongoose 2314 MHz
#   through ACPM (needs the 0137-staged DVFS firmware in the kernel), 1 GiB
#   for XNU, the crash-looping daemons refused at exec (0130), FTS touch.
# gpu (s8plus-gpu branch): fast + the GPU present at 30 frames/s, 1 GiB.
# safe: one A53 at the S-Boot clock, 512 MiB, touch on (first S8+ boots).
# notouch: as safe with the touch driver off (s8touch=0).
# MIF/INT are left alone: s8mif froze the S8, s8int=667000 caused aborts.
set -euo pipefail
[[ $# -ge 3 ]] || { sed -n '2,12p' "$0"; exit 2; }
kernel=$1 stock=$2 out=$3 mode=${4:-fast}
here=$(dirname "$(realpath "$0")")

common=(debug=0xc serial=3 keepsyms=1 -enable_kprintf_spam -noprogress
	rd=ufsroot0 ufsroot=USERDATA ufsrw=1 rootrw=1
	unentitled_ios_sim_launch=1 ignition_level=1 launchd_unsecure_cache=1
	wdt=-1 wdt_test=no_user_checkin
	a5touchprobe=0 a5fb_keep_on=1 a5fb_noconsole=1 sweetfb_native=1)
noexec=vmd,bluetoothd,fairplayd.H2,nanotimekitcompaniond,locationd,WirelessRadioManagerd,CommCenterMobileHelper

case $mode in
fast)
	export S8_DRAM_MIB=${S8_DRAM_MIB:-1024}   # the high bank holds 2048
	args=(cpus=8 s8cpufreq=1690000 s8cpufreq_big=2314000 "s8noexec=$noexec")
	;;
gpu)
	# s8plus-gpu branch, tested on the G955F 2026-11-06: fast + the GPU
	# present (the Mali stretches each iOS frame over the panel, ~6.5 ms,
	# instead of the CPU copy), so 30 frames/s instead of the default 10.
	# s8gpu=1: the loader powers BUCK6 vdd_g3d and runs Samsung's g3d_on
	# (without it every kernel job stays ACTIVE). s8mngs=0: the merged loader
	# neither probes nor caps the Mongoose cores. 1 GiB: patch 0141 moved the
	# blit blob/page tables to 0x8c0000000 and the GPU heap to 0x8c1000000,
	# right above XNU's RAM. No s8metal: the Metal draws (stage 3) hang the
	# Mali on this phone (js ACTIVE), which also stalled the present.
	# No s8g3d: changing the G3D clock through ACPM after setup broke jobs.
	export S8_DRAM_MIB=${S8_DRAM_MIB:-1024}
	args=(cpus=8 s8mngs=0 s8cpufreq=1690000 s8cpufreq_big=2314000 "s8noexec=$noexec"
		s8gpu=1 sweetfb_gpu=1 sweetfb_hz=30)
	;;
metal)
	# gpu + the Metal driver (S8_OVERLAY = metal/stage3.ovl), for debugging
	# stage 3 only: the Metal job chains hang the Mali (screen black). The
	# matrix/shader tests of the loader (s8gpu=2) write 0x8a0000000+, so 512 MiB.
	export S8_DRAM_MIB=512
	: "${S8_OVERLAY:?S8_OVERLAY must name the Metal overlay (stage3.ovl)}"
	args=(cpus=8 s8mngs=0 s8cpufreq=1690000 s8cpufreq_big=2314000 "s8noexec=$noexec"
		s8gpu=2 sweetfb_gpu=1 s8metal=1 sweetfb_hz=30)
	;;
safe)    args=(cpus=1) ;;
notouch) args=(cpus=1 s8touch=0) ;;
*) echo "mode: fast, gpu, metal, safe or notouch" >&2; exit 2 ;;
esac

bash "$here/make-boot.sh" "$kernel" "$stock" "$out" "${common[@]}" "${args[@]}"
sha256sum "$out" 2>/dev/null || shasum -a 256 "$out"
