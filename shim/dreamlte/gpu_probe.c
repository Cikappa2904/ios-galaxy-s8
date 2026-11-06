/* SPDX-License-Identifier: GPL-2.0-only
 * Galaxy S8 Mali-G71 bring-up probe, run by the loader (MMU off), boot-arg
 * s8gpu=1. Step 1 of a GPU for iOS: power the GPU and read its ID.
 *
 * Sources (Samsung universal8895 kernel): exynos8895.dtsi mali@13900000
 * (pd_g3d), exynos8895-pm-domains.dtsi pd-g3d@164840e0, cal-if/exynos8895
 * pmucal_local_exynos8895.h g3d_on: CMU_G3D 0x13800000 + 0x148 :=
 * 0x30000003, + 0x140 bit 4, + 0x100 bit 4 (MUX_CLKCMU_EMBEDDED_G3D_USER),
 * PMU 0x40E0 := 0xF, wait PMU 0x40E4 & 0xF == 0xF. The GPU registers are
 * read only once the domain reports on, so a gated block is never touched.
 * Mali registers (mali_kbase_gpu_regmap.h): GPU_ID 0x000, L2_FEATURES
 * 0x004, GPU_STATUS 0x034, SHADER_PRESENT_LO 0x100, L2_PRESENT_LO 0x120.
 *
 * Boot 36: GPU_ID 0x60a00001 (product 0x60a0 = TMIx, Mali-G71 r0p0),
 * SHADER_PRESENT 0x17771777 (20 cores), L2_PRESENT 1, pd-g3d already on.
 * Step 2, the first GPU jobs: soft reset, power L2/tiler/shaders, address
 * space 0 in IDENTITY mode (GPU VA = PA, no page tables; kbase
 * AS_TRANSCFG_ADRMODE_IDENTITY 2), then on job slot 0 a NULL job (type 1)
 * and a WRITE_VALUE job (type 2, immediate 32-bit) that must store a magic
 * word in RAM. Registers and the 32-byte job header follow Samsung's kbase
 * r0p0 (mali_midg_regmap.h, struct job_descriptor_header); the write-value
 * payload (address, value type 6 = immediate 32, immediate) follows Mesa's
 * Bifrost job description.
 */
#include <stdint.h>
#include "gpu_probe.h"

#define PMU		0x16480000UL
#define G3D_CFG		(PMU + 0x40e0)
#define G3D_STAT	(PMU + 0x40e4)
#define CMU_G3D		0x13800000UL
#define MALI		0x13900000UL
#define TIMER_HZ	26000000UL

#define GPU_IRQ_RAWSTAT	0x020
#define GPU_IRQ_CLEAR	0x024
#define GPU_COMMAND	0x030
#define RESET_DONE	(1u << 8)
#define JOB_IRQ_RAWSTAT	0x1000
#define JOB_IRQ_CLEAR	0x1004
#define JS0		0x1800
#define AS0		0x2400

/* Job descriptors and the write target: MMU off, so CPU and GPU see RAM
 * uncached; GPU addresses are physical (IDENTITY). */
static volatile uint8_t gpu_mem[1024] __attribute__((aligned(256)));

/*
 * Step 3, the GPU draws: a fragment job (type 9) whose single render target
 * is a 256x256 square of the S-Boot framebuffer that the DECON scans out,
 * cleared to light grey by the GPU's tile writeback. Layouts from Mesa
 * 25.2.8 src/panfrost/genxml/v6.xml (Bifrost): Framebuffer Parameters
 * (64 bytes) + padding (64) + one RGB Render Target (64), Fragment Job
 * Payload, Framebuffer pointer (MFBD bit, RT count - 1). Values as
 * Panfrost's pan_emit_fbd/pan_emit_rt for a clear-only batch: no tiler
 * context (Panfrost emits one only for draws), 16x16 tiles (effective tile
 * size log2 256 = 8, colour buffer allocation 1 KiB), single-sample
 * positions from pan_samples.c (128, 128), R8G8B8A8 internal and writeback,
 * linear block format, identity swizzle (R,G,B,A = 0,1,2,3).
 */
#define FB_PHYS		0xcc000000UL
#define FB_STRIDE	5760u
#define SQ_X		592u
#define SQ_Y		2496u
#define SQ		256u
#define SQ_BASE		(FB_PHYS + SQ_Y * FB_STRIDE + SQ_X * 4)
#define GREY		0xffc0c0c0u
static volatile uint8_t gpu_fb[1024] __attribute__((aligned(256)));

static void fput32(unsigned off, uint32_t v)
{
	*(volatile uint32_t *)(gpu_fb + off) = v;
}

static inline uint32_t rd(uintptr_t a) { return *(volatile uint32_t *)a; }
static inline void wr(uintptr_t a, uint32_t v) { *(volatile uint32_t *)a = v; }

static uint64_t ticks(void)
{
	uint64_t v;
	__asm__ volatile("isb; mrs %0, cntpct_el0" : "=r"(v));
	return v;
}

static int wait_mask(uintptr_t a, uint32_t mask, uint32_t want, unsigned ms)
{
	uint64_t end = ticks() + TIMER_HZ / 1000 * ms;
	while (ticks() < end)
		if ((rd(a) & mask) == want)
			return 1;
	return 0;
}

static void put32(unsigned off, uint32_t v)
{
	*(volatile uint32_t *)(gpu_mem + off) = v;
}

static uint32_t get32(unsigned off)
{
	return *(volatile uint32_t *)(gpu_mem + off);
}

static void put64(unsigned off, uint64_t v)
{
	put32(off, (uint32_t)v);
	put32(off + 4, (uint32_t)(v >> 32));
}

/* Runs the one-job chain at gpu_mem + 0 on slot 0; returns JOB_IRQ_RAWSTAT. */
static uint32_t run_job(uint32_t shaders)
{
	uint64_t job = (uintptr_t)gpu_mem;
	wr(MALI + JOB_IRQ_CLEAR, 0xffffffffu);
	wr(MALI + JS0 + 0x40, (uint32_t)job);		/* JS_HEAD_NEXT */
	wr(MALI + JS0 + 0x44, (uint32_t)(job >> 32));
	wr(MALI + JS0 + 0x50, shaders);			/* JS_AFFINITY_NEXT */
	wr(MALI + JS0 + 0x54, 0);
	/* JS_CONFIG: AS 0, flush clean+invalidate at start and end, prio 8. */
	wr(MALI + JS0 + 0x58, (8u << 16) | (3u << 12) | (3u << 8));
	wr(MALI + JS0 + 0x60, 1);			/* JS_COMMAND_START */
	uint64_t end = ticks() + TIMER_HZ / 10;
	while (ticks() < end && !(rd(MALI + JOB_IRQ_RAWSTAT) & ((1u << 16) | 1u)))
		;
	return rd(MALI + JOB_IRQ_RAWSTAT);
}

static void job_header(unsigned type);

static int gpu_draw(void (*kv)(const char *, uint64_t), void (*out)(const char *), uint32_t shaders)
{
	out("GPU draw (fragment job, clear 256x256)\n");
	for (unsigned i = 0; i < sizeof(gpu_fb); i += 4)
		fput32(i, 0);
	/* Sample positions (0x000): single-sampled position and origin. */
	fput32(0x00, 128u | 128u << 16);
	fput32(0x80, 128u | 128u << 16);		/* positions[32] = origin */
	/* Framebuffer descriptor (0x100): parameters, padding, RT at +128. */
	unsigned fbd = 0x100, rt = fbd + 128;
	uint64_t sp = (uintptr_t)gpu_fb;
	fput32(fbd + 4 * 4, (uint32_t)sp);		/* sample locations */
	fput32(fbd + 5 * 4, (uint32_t)(sp >> 32));
	fput32(fbd + 8 * 4, (SQ - 1) | (SQ - 1) << 16);	/* width, height */
	fput32(fbd + 10 * 4, (SQ - 1) | (SQ - 1) << 16);	/* bound max */
	/* sample count log2 0, pattern 0, tile 2^8 px, 1 RT, cbuf 1 KiB */
	fput32(fbd + 11 * 4, (8u << 9) | (0u << 19) | (1u << 24));
	fput32(rt + 0 * 4, 1u << 26);			/* internal R8G8B8A8 */
	fput32(rt + 1 * 4, 1u | (19u << 3) | (2u << 10) | (0x688u << 16) | (1u << 31));
	fput32(rt + 8 * 4, (uint32_t)SQ_BASE);		/* writeback base */
	fput32(rt + 9 * 4, (uint32_t)((uint64_t)SQ_BASE >> 32));
	fput32(rt + 10 * 4, FB_STRIDE);			/* row stride */
	for (unsigned c = 0; c < 4; c++)
		fput32(rt + (12 + c) * 4, GREY);	/* clear colour */
	/* Black square first, so a GPU write is visible and checkable. */
	for (unsigned y = 0; y < SQ; y++)
		for (unsigned x = 0; x < SQ; x++)
			wr(SQ_BASE + y * FB_STRIDE + x * 4, 0xff000000u);

	job_header(9);					/* FRAGMENT */
	put32(0x20, 0);					/* bound min tile 0,0 */
	put32(0x24, (SQ / 16 - 1) | (SQ / 16 - 1) << 16);	/* bound max tile */
	uint64_t ptr = ((uintptr_t)gpu_fb + fbd) | 1u;	/* MFBD, 1 RT */
	put64(0x28, ptr);
	uint32_t irq = run_job(shaders);
	kv("FRAG job irq", irq);
	kv("js0 status", rd(MALI + JS0 + 0x24));
	kv("exc", get32(0x00));
	kv("AS0 fault", rd(MALI + AS0 + 0x1c));
	out("\n");
	uint32_t p0 = rd(SQ_BASE), p1 = rd(SQ_BASE + (SQ - 1) * FB_STRIDE + (SQ - 1) * 4);
	unsigned grey = 0;
	for (unsigned y = 0; y < SQ; y += 16)
		for (unsigned x = 0; x < SQ; x += 16)
			grey += (rd(SQ_BASE + y * FB_STRIDE + x * 4) & 0xffffff) == 0xc0c0c0;
	kv("pixel 0,0", p0);
	kv("pixel 255,255", p1);
	kv("grey samples/256", grey);
	out("\n");
	kv("GPU DRAW WORKS", grey == 256);
	out("\n");
	return grey == 256;
}

/* Job header (kbase struct job_descriptor_header), 64-bit descriptors. */
static void job_header(unsigned type)
{
	for (unsigned i = 0; i < 0x40; i += 4)
		put32(i, 0);
	put32(0x10, 1u | (type << 1) | (1u << 16));	/* size 64, type, job_index 1 */
}

/*
 * Address matrix (Galaxy S8+ G955F, 2026-11-06). The jobs above all live in
 * low RAM (the loader image, below 4 GiB) with AS0 in IDENTITY mode, and
 * they work. XNU's GPU code keeps its job descriptors, page tables and GPU
 * heap in the high bank (0x8a0000000+) with AS0 in AARCH64_4K mode (the
 * same setup as Samsung's kbase r2p0), and on the S8+ every such job stays
 * ACTIVE with no fault. One WRITE_VALUE job per case separates the two
 * factors: where the job/target are, where the page tables are, and the
 * translation mode. A stuck job is hard-stopped and the GPU soft-reset and
 * re-powered before the next case. MMU off: CPU accesses are uncached.
 */
#define MX_HIGH		0x8a0100000ULL	/* above XNU's 512 MiB (s8gpu=2 boots only) */
#define MX_HIGH_PT	MX_HIGH			/* 4 tables: L0..L3 */
#define MX_HIGH_JOB	(MX_HIGH + 0x10000)
static volatile uint64_t mx_low_pt[4 * 512] __attribute__((aligned(4096)));
static volatile uint8_t mx_low_job[4096] __attribute__((aligned(4096)));

static inline void w32(uint64_t a, uint32_t v) { *(volatile uint32_t *)(uintptr_t)a = v; }
static inline uint32_t r32(uint64_t a) { return *(volatile uint32_t *)(uintptr_t)a; }
static inline void w64(uint64_t a, uint64_t v) { *(volatile uint64_t *)(uintptr_t)a = v; }

static void mx_power(uint32_t shaders)
{
	wr(MALI + GPU_IRQ_CLEAR, 0xffffffffu);
	wr(MALI + GPU_COMMAND, 1);			/* SOFT_RESET */
	wait_mask(MALI + GPU_IRQ_RAWSTAT, RESET_DONE, RESET_DONE, 100);
	wr(MALI + GPU_IRQ_CLEAR, 0xffffffffu);
	uint32_t l2 = rd(MALI + 0x120), tiler = rd(MALI + 0x110);
	wr(MALI + 0x1a0, l2);
	wait_mask(MALI + 0x160, l2, l2, 100);
	wr(MALI + 0x190, tiler);
	wait_mask(MALI + 0x150, tiler, tiler, 100);
	wr(MALI + 0x180, shaders);
	wait_mask(MALI + 0x140, shaders, shaders, 100);
}

/* AS0: transcfg 2 (IDENTITY) or 6 (AARCH64_4K, table at pt), as the kernel. */
static int mx_as(uint32_t transcfg, uint64_t pt)
{
	wr(MALI + AS0 + 0x00, (uint32_t)pt);
	wr(MALI + AS0 + 0x04, (uint32_t)(pt >> 32));
	wr(MALI + AS0 + 0x08, 0x88888888u);
	wr(MALI + AS0 + 0x0c, 0x88888888u);
	wr(MALI + AS0 + 0x30, transcfg);
	wr(MALI + AS0 + 0x34, 0);
	wr(MALI + AS0 + 0x18, 1);			/* AS_COMMAND_UPDATE */
	return wait_mask(MALI + AS0 + 0x28, 1, 0, 100);
}

/* 4-level AArch64 4K tables at pt (4 pages), mapping the 4 KiB page of va
 * VA = PA, entries exactly as s8_gpu_present.inc / kbase r2p0 write them. */
static void mx_map(uint64_t pt, uint64_t va)
{
	for (unsigned i = 0; i < 4 * 512; i++)
		w64(pt + i * 8, 0);
	for (unsigned lv = 0; lv < 3; lv++) {
		unsigned idx = (unsigned)(va >> (39 - 9 * lv)) & 511;
		w64(pt + lv * 4096 + idx * 8, (pt + (lv + 1) * 4096) | (1ULL << 10) | 3);
	}
	w64(pt + 3 * 4096 + ((va >> 12) & 511) * 8,
	    (va & 0xfffffffff000ULL) | (1ULL << 6) | (1ULL << 10) | 3);
	__asm__ volatile("dsb sy" ::: "memory");
}

static void mx_case(void (*kv)(const char *, uint64_t), void (*out)(const char *),
    uint32_t shaders, const char *name, uint64_t job, uint64_t pt)
{
	const uint32_t magic = 0x5a5a0000u | (uint32_t)(job >> 20 & 0xffff);
	out(name);
	out("\n");
	for (unsigned i = 0; i < 0x40; i += 4)
		w32(job + i, 0);
	w32(job + 0x10, 1u | (2u << 1) | (1u << 16));	/* 64-bit header, WRITE_VALUE */
	w64(job + 0x20, job + 0x100);			/* target */
	w32(job + 0x28, 6);				/* immediate 32 */
	w64(job + 0x30, magic);
	w32(job + 0x100, 0xdeadbeefu);
	if (pt)
		mx_map(pt, job);
	kv("as0 update", mx_as(pt ? 6 : 2, pt));
	__asm__ volatile("dsb sy" ::: "memory");
	wr(MALI + JOB_IRQ_CLEAR, 0xffffffffu);
	wr(MALI + JS0 + 0x40, (uint32_t)job);
	wr(MALI + JS0 + 0x44, (uint32_t)(job >> 32));
	wr(MALI + JS0 + 0x50, shaders);
	wr(MALI + JS0 + 0x54, 0);
	wr(MALI + JS0 + 0x58, (8u << 16) | (3u << 12) | (3u << 8));
	wr(MALI + JS0 + 0x60, 1);
	uint64_t end = ticks() + TIMER_HZ / 10;
	while (ticks() < end && !(rd(MALI + JOB_IRQ_RAWSTAT) & ((1u << 16) | 1u)))
		;
	uint32_t irq = rd(MALI + JOB_IRQ_RAWSTAT), js = rd(MALI + JS0 + 0x24);
	kv("irq", irq);
	kv("js", js);
	kv("fault", rd(MALI + AS0 + 0x1c));
	kv("faddr", ((uint64_t)rd(MALI + AS0 + 0x24) << 32) | rd(MALI + AS0 + 0x20));
	out("\n");
	kv("value ok", r32(job + 0x100) == magic);
	out("\n");
	if (js == 0x8 || !(irq & 1u)) {		/* still ACTIVE or no DONE: recover */
		wr(MALI + JS0 + 0x60, 3);		/* JS_COMMAND_HARD_STOP */
		wait_mask(MALI + JOB_IRQ_RAWSTAT, (1u << 16) | 1u, (1u << 16), 50);
		mx_power(shaders);
	}
	wr(MALI + JOB_IRQ_CLEAR, 0xffffffffu);
}

static void bl_case(void (*kv)(const char *, uint64_t), void (*out)(const char *),
    uint32_t shaders, const char *name, int tables);

static void gpu_matrix(void (*kv)(const char *, uint64_t), void (*out)(const char *), uint32_t shaders)
{
	uint64_t lo_job = (uintptr_t)mx_low_job, lo_pt = (uintptr_t)mx_low_pt;
	out("GPU address matrix (job / page tables / mode)\n");
	kv("low job", lo_job);
	kv("low pt", lo_pt);
	out("\n");
	/* The image is linked at 0 and runs where S-Boot put it: the 4 KiB
	 * alignment of the low arrays is only real if that address is aligned. */
	int lo_ok = !(lo_pt & 0xfff) && !(lo_job & 0xfff);
	kv("low aligned", lo_ok);
	out("\n");
	mx_case(kv, out, shaders, "MX B: high job, IDENTITY", MX_HIGH_JOB, 0);
	if (lo_ok)
		mx_case(kv, out, shaders, "MX C: low job, low pt", lo_job, lo_pt);
	mx_case(kv, out, shaders, "MX D: high job, high pt (kernel)", MX_HIGH_JOB, MX_HIGH_PT);
	if (lo_ok) {
		mx_case(kv, out, shaders, "MX E: high job, low pt", MX_HIGH_JOB, lo_pt);
		mx_case(kv, out, shaders, "MX F: low job, high pt", lo_job, MX_HIGH_PT);
	}
	bl_case(kv, out, shaders, "MX G: XNU present job (shader), IDENTITY", 0);
	bl_case(kv, out, shaders, "MX H: XNU present job (shader), page tables", 1);
	mx_as(2, 0);					/* back to IDENTITY */
	out("GPU address matrix done\n");
}

/*
 * Cases G/H (2026-11-06): every job the loader ran so far is fixed-function
 * (NULL, WRITE_VALUE, a clear-only fragment job), and they all work, also
 * with the kernel's page-table layout (case D). Every job XNU runs executes
 * a shader (the present blit, Metal draws) and stays ACTIVE. G runs XNU's
 * own present job (s8_mali_blit.h, byte for byte, at its fixed address)
 * from here: a 64x64 green source stretched over the whole panel, AS0 in
 * IDENTITY. H is the same job through page tables built as the kernel's.
 */
#include "../../tools/dreamlte/s8_mali_blit.h"

#define BL_SRC		(S8_BLIT_BASE_PA + 0x200000)	/* 64x64 source, 256-byte rows */
#define BL_PT		(S8_BLIT_BASE_PA + 0x400000)	/* table pool, 64 pages */
#define BL_PT_PAGES	64u
#define BL_PANEL_SIZE	(5760ULL * 2960)
static unsigned bl_pt_used;

static uint32_t bl_float(uint32_t v)
{
	if (v == 0)
		return 0;
	uint32_t e = 31 - (uint32_t)__builtin_clz(v);
	uint32_t m = e > 23 ? (v >> (e - 23)) : (v << (23 - e));
	return ((e + 127) << 23) | (m & 0x7fffff);
}

/* Same walk and entries as s8gpu_pte/s8gpu_map in s8_gpu_present.inc. */
static int bl_map(uint64_t va, uint64_t len)
{
	for (uint64_t o = 0; o < len; o += 4096) {
		unsigned table = 0;
		for (unsigned lv = 0; lv < 3; lv++) {
			unsigned idx = (unsigned)((va + o) >> (39 - 9 * lv)) & 511;
			uint64_t e = BL_PT + (uint64_t)table * 4096 + idx * 8;
			uint64_t v = *(volatile uint64_t *)(uintptr_t)e;
			if ((v & 3) != 3) {
				if (bl_pt_used == BL_PT_PAGES)
					return 0;
				table = bl_pt_used++;
				w64(e, (BL_PT + (uint64_t)table * 4096) | (1ULL << 10) | 3);
			} else {
				table = (unsigned)(((v & 0xfffffffff000ULL) - BL_PT) / 4096);
			}
		}
		w64(BL_PT + (uint64_t)table * 4096 + (((va + o) >> 12) & 511) * 8,
		    ((va + o) & 0xfffffffff000ULL) | (1ULL << 6) | (1ULL << 10) | 3);
	}
	return 1;
}

static void bl_case(void (*kv)(const char *, uint64_t), void (*out)(const char *),
    uint32_t shaders, const char *name, int tables)
{
	const uint64_t base = S8_BLIT_BASE_PA;
	const uint32_t w = 64, h = 64, bpr = 256;
	out(name);
	out("\n");
	for (unsigned i = 0; i < S8_BLIT_SIZE; i += 4) {
		uint32_t v = (uint32_t)s8_blit_blob[i] | (uint32_t)s8_blit_blob[i + 1] << 8 |
		    (uint32_t)s8_blit_blob[i + 2] << 16 | (uint32_t)s8_blit_blob[i + 3] << 24;
		w32(base + i, v);
	}
	for (unsigned i = 0; i < h * bpr; i += 4)
		w32(BL_SRC + i, 0xff00ff00u);		/* opaque green in BGRA and RGBA */
	w32(base + S8_BLIT_TEX_SIZE, (w - 1) | ((h - 1) << 16));
	w64(base + S8_BLIT_SURF_PTR, BL_SRC);
	w32(base + S8_BLIT_SURF_STRIDE, bpr);
	uint32_t tc = S8_BLIT_TEXCOORD, fw = bl_float(w), fh = bl_float(h);
	w32(base + tc + 0, 0);  w32(base + tc + 4, 0);
	w32(base + tc + 16, fw); w32(base + tc + 20, 0);
	w32(base + tc + 32, 0);  w32(base + tc + 36, fh);
	w32(base + tc + 48, fw); w32(base + tc + 52, fh);
	uint64_t mid = FB_PHYS + 1480ULL * FB_STRIDE + 720 * 4;
	w32(mid, 0xff000000u);
	int as_ok;
	if (tables) {
		for (unsigned i = 0; i < BL_PT_PAGES * 512; i++)
			w64(BL_PT + i * 8, 0);
		bl_pt_used = 1;
		int m = bl_map(base, S8_BLIT_SIZE) && bl_map(BL_SRC, h * bpr) &&
		    bl_map(FB_PHYS, (BL_PANEL_SIZE + 4095) & ~4095ULL);
		kv("mapped", m);
		kv("pt pages", bl_pt_used);
		__asm__ volatile("dsb sy" ::: "memory");
		as_ok = mx_as(6, BL_PT);
	} else {
		as_ok = mx_as(2, 0);
	}
	kv("as0 update", as_ok);
	out("\n");
	__asm__ volatile("dsb sy" ::: "memory");
	uint64_t job = base + S8_BLIT_JOB;
	wr(MALI + JOB_IRQ_CLEAR, 0xffffffffu);
	wr(MALI + JS0 + 0x40, (uint32_t)job);
	wr(MALI + JS0 + 0x44, (uint32_t)(job >> 32));
	wr(MALI + JS0 + 0x50, shaders);
	wr(MALI + JS0 + 0x54, 0);
	wr(MALI + JS0 + 0x58, (8u << 16) | (3u << 12) | (3u << 8));
	wr(MALI + JS0 + 0x60, 1);
	uint64_t end = ticks() + TIMER_HZ / 5;		/* 200 ms, as the kernel */
	while (ticks() < end && !(rd(MALI + JOB_IRQ_RAWSTAT) & ((1u << 16) | 1u)))
		;
	uint32_t irq = rd(MALI + JOB_IRQ_RAWSTAT), js = rd(MALI + JS0 + 0x24);
	kv("irq", irq);
	kv("js", js);
	kv("exc", r32(job));
	kv("fault", rd(MALI + AS0 + 0x1c));
	kv("faddr", ((uint64_t)rd(MALI + AS0 + 0x24) << 32) | rd(MALI + AS0 + 0x20));
	out("\n");
	kv("GPU_STATUS", rd(MALI + 0x034));
	kv("shader ready", rd(MALI + 0x140));
	kv("gpu irq", rd(MALI + GPU_IRQ_RAWSTAT));
	kv("fault st", rd(MALI + 0x03c));
	out("\n");
	kv("mid pixel", r32(mid));
	kv("SHADER JOB WORKS", (irq & 1u) && !(irq & (1u << 16)) && ((r32(mid) >> 8) & 0xff) == 0xff);
	out("\n");
	if (js == 0x8 || !(irq & 1u)) {
		wr(MALI + JS0 + 0x60, 3);		/* HARD_STOP */
		wait_mask(MALI + JOB_IRQ_RAWSTAT, (1u << 16) | 1u, (1u << 16), 50);
		mx_power(shaders);
	}
	wr(MALI + JOB_IRQ_CLEAR, 0xffffffffu);
}

/*
 * Cases R1/R2 (2026-11-06): with the kernel's present job working, iOS's
 * Metal vertex/tiler chains (s8mtl_draw.c) still stay ACTIVE with no fault
 * and wedge the GPU. tools/dreamlte/metal/test/build-replay.sh runs that
 * draw path on the host with the test shaders and records its job chains
 * (s8_mali_replay.h: 'P' 4 KiB pages of GPU memory at RP_BASE, 'S' submits,
 * 'E' end with the bytes used). Here they are replayed on the GPU, in
 * IDENTITY (R1) and through page tables built as the kernel's (R2): per
 * chain the JS status, the job the slot is on (JS_HEAD) and every job's
 * exception status, so the stalling job type shows up. The first chain is
 * also recorded as its vertex job alone and its tiler job alone.
 */
/* Generated, not committed (tools/dreamlte/metal/test/build-replay.sh,
 * dump2replay.py, gen-suite): without it the replay is an empty table. */
#if __has_include("../../tools/dreamlte/s8_mali_replay.h")
#include "../../tools/dreamlte/s8_mali_replay.h"
#else
#define S8_REPLAY_SIZE 16u
/* one 'E' record; sized past the reads the parser could do */
static const unsigned char s8_replay[256] __attribute__((aligned(16))) = { 'E' };
#endif

/* Range base: a 'B' record (dump2replay.py: iOS's memory at 0x8c1000000+),
 * else replay-gen.c's S8T_BASE. The table pool follows the range. */
#define RP_BASE_DEFAULT	0x8b0000000ULL
#define RP_PT_PAGES	128u
static uint64_t rp_base = RP_BASE_DEFAULT, rp_pt;
static unsigned rp_pt_used;

static uint32_t rp32(unsigned off) { return *(const volatile uint32_t *)(s8_replay + off); }
static uint64_t rp64(unsigned off) { return *(const volatile uint64_t *)(s8_replay + off); }

static int rp_map(uint64_t va, uint64_t len)
{
	for (uint64_t o = 0; o < len; o += 4096) {
		unsigned table = 0;
		for (unsigned lv = 0; lv < 3; lv++) {
			unsigned idx = (unsigned)((va + o) >> (39 - 9 * lv)) & 511;
			uint64_t e = rp_pt + (uint64_t)table * 4096 + idx * 8;
			uint64_t v = *(volatile uint64_t *)(uintptr_t)e;
			if ((v & 3) != 3) {
				if (rp_pt_used == RP_PT_PAGES)
					return 0;
				table = rp_pt_used++;
				w64(e, (rp_pt + (uint64_t)table * 4096) | (1ULL << 10) | 3);
			} else {
				table = (unsigned)(((v & 0xfffffffff000ULL) - rp_pt) / 4096);
			}
		}
		w64(rp_pt + (uint64_t)table * 4096 + (((va + o) >> 12) & 511) * 8,
		    ((va + o) & 0xfffffffff000ULL) | (1ULL << 6) | (1ULL << 10) | 3);
	}
	return 1;
}

static int rp_as(int tables)
{
	return tables ? mx_as(6, rp_pt) : mx_as(2, 0);
}

static void rp_run(void (*kv)(const char *, uint64_t), void (*out)(const char *),
    uint32_t shaders, int tables)
{
	out(tables ? "MX R2: Metal replay, page tables\n" : "MX R1: Metal replay, IDENTITY\n");
	uint64_t used = 0;
	for (unsigned off = 0; off + 16 <= S8_REPLAY_SIZE;) {
		uint32_t tag = rp32(off);
		if (tag == 'B') {
			rp_base = rp64(off + 8);
			off += 16;
		} else if (tag == 'P') {
			off += 16 + 4096;
		} else if (tag == 'S') {
			off += 128;
		} else {
			used = tag == 'E' ? rp64(off + 8) : 0;
			break;
		}
	}
	rp_pt = (rp_base + used + 0xfffff) & ~0xfffffULL;
	kv("base", rp_base);
	kv("bytes", used);
	if (used == 0 || used > (96ULL << 20)) {
		out("bad replay table\n");
		return;
	}
	for (uint64_t o = 0; o < used; o += 8)
		w64(rp_base + o, 0);
	if (tables) {
		for (unsigned i = 0; i < RP_PT_PAGES * 512; i++)
			w64(rp_pt + i * 8, 0);
		rp_pt_used = 1;
		kv("mapped", rp_map(rp_base, (used + 4095) & ~4095ULL));
		kv("pt pages", rp_pt_used);
		__asm__ volatile("dsb sy" ::: "memory");
	}
	kv("as0 update", rp_as(tables));
	out("\n");
	unsigned chains = 0, ok = 0;
	for (unsigned off = 0; off + 16 <= S8_REPLAY_SIZE;) {
		uint32_t tag = rp32(off);
		if (tag == 'B') {
			off += 16;
			continue;
		}
		if (tag == 'P') {
			uint64_t a = rp64(off + 8);
			for (unsigned i = 0; i < 4096; i += 8)
				w64(a + i, rp64(off + 16 + i));
			off += 16 + 4096;
			continue;
		}
		if (tag != 'S')
			break;
		unsigned slot = rp32(off + 4), nj = rp32(off + 24);
		uint64_t jc = rp64(off + 8), probe = rp64(off + 16);
		char label[33];
		for (unsigned i = 0; i < 32; i++)
			label[i] = (char)s8_replay[off + 96 + i];
		label[32] = 0;
		chains++;
		__asm__ volatile("dsb sy" ::: "memory");
		uintptr_t js = MALI + JS0 + 0x80 * slot;
		uint32_t done = 1u << slot, fail = 1u << (16 + slot);
		wr(MALI + JOB_IRQ_CLEAR, 0xffffffffu);
		wr(js + 0x40, (uint32_t)jc);			/* JS_HEAD_NEXT */
		wr(js + 0x44, (uint32_t)(jc >> 32));
		wr(js + 0x50, shaders);				/* JS_AFFINITY_NEXT */
		wr(js + 0x54, 0);
		wr(js + 0x58, (8u << 16) | (3u << 12) | (3u << 8));
		wr(js + 0x60, 1);				/* JS_COMMAND_START */
		/* 50 ms: a probe takes tens of us, a 64x64 suite pass a few ms;
		 * 270 chains must not keep the loader long if many hang */
		uint64_t t0 = ticks(), end = t0 + TIMER_HZ / 20;
		while (ticks() < end && !(rd(MALI + JOB_IRQ_RAWSTAT) & (done | fail)))
			;
		uint64_t us = (ticks() - t0) / (TIMER_HZ / 1000000);
		uint32_t irq = rd(MALI + JOB_IRQ_RAWSTAT), st = rd(js + 0x24);
		uint64_t head = ((uint64_t)rd(js + 0x04) << 32) | rd(js + 0x00);
		uint32_t fault = rd(MALI + AS0 + 0x1c);
		uint64_t faddr = ((uint64_t)rd(MALI + AS0 + 0x24) << 32) | rd(MALI + AS0 + 0x20);
		int good = (irq & done) && !(irq & fail), stopped = 0;
		if (!good) {
			wr(js + 0x60, 3);			/* JS_COMMAND_HARD_STOP */
			stopped = wait_mask(MALI + JOB_IRQ_RAWSTAT, done | fail, done | fail, 50);
		}
		/* Two lines per chain: XNU echoes only 120 loader lines. */
		out("R ");
		out(label);
		out(": ");
		kv("irq", irq);
		kv("js", st);
		kv("us", us);
		kv("head", head);
		kv("fault", fault);
		kv("faddr", faddr);
		out("\n ");
		for (unsigned i = 0; i < nj && i < 8; i++) {
			uint64_t j = rp64(off + 32 + i * 8);
			kv(" t", (r32(j + 0x10) >> 1) & 0x7f);
			kv("e", r32(j));
		}
		if (probe)
			kv(" pixel", r32(probe));
		if (!good) {
			kv(" stop", stopped);
			kv("js", rd(js + 0x24));
		}
		out("\n");
		if (good) {
			ok++;
		} else {
			mx_power(shaders);
			rp_as(tables);
		}
		wr(MALI + JOB_IRQ_CLEAR, 0xffffffffu);
		off += 128;
	}
	kv("chains", chains);
	kv("done", ok);
	out("\n");
}

static void gpu_jobs(void (*kv)(const char *, uint64_t), void (*out)(const char *), uint32_t shaders,
    int matrix)
{
	out("GPU jobs\n");
	wr(MALI + GPU_IRQ_CLEAR, 0xffffffffu);
	wr(MALI + GPU_COMMAND, 1);			/* SOFT_RESET */
	kv("reset", wait_mask(MALI + GPU_IRQ_RAWSTAT, RESET_DONE, RESET_DONE, 100));
	wr(MALI + GPU_IRQ_CLEAR, 0xffffffffu);
	uint32_t l2 = rd(MALI + 0x120), tiler = rd(MALI + 0x110);
	wr(MALI + 0x1a0, l2);				/* L2_PWRON */
	kv("l2 ready", wait_mask(MALI + 0x160, l2, l2, 100));
	wr(MALI + 0x190, tiler);			/* TILER_PWRON */
	kv("tiler ready", wait_mask(MALI + 0x150, tiler, tiler, 100));
	wr(MALI + 0x180, shaders);			/* SHADER_PWRON */
	kv("shaders ready", wait_mask(MALI + 0x140, shaders, shaders, 100));
	out("\n");
	kv("shader ready", rd(MALI + 0x140));
	kv("GPU_STATUS", rd(MALI + 0x034));
	out("\n");

	wr(MALI + AS0 + 0x30, 2);			/* AS_TRANSCFG: IDENTITY */
	wr(MALI + AS0 + 0x34, 0);
	wr(MALI + AS0 + 0x08, 0x88888888u);		/* AS_MEMATTR: impl. defined */
	wr(MALI + AS0 + 0x0c, 0x88888888u);
	wr(MALI + AS0 + 0x00, 0);
	wr(MALI + AS0 + 0x04, 0);
	wr(MALI + AS0 + 0x18, 1);			/* AS_COMMAND_UPDATE */
	kv("as0 update", wait_mask(MALI + AS0 + 0x28, 1, 0, 100));
	out("\n");

	job_header(1);					/* NULL job */
	uint32_t irq = run_job(shaders);
	kv("NULL job irq", irq);
	kv("js0 status", rd(MALI + JS0 + 0x24));
	kv("exc", get32(0x00));
	out("\n");

	const uint32_t magic = 0x5a5a1234u;
	put32(0x100, 0xdeadbeefu);
	job_header(2);					/* WRITE_VALUE */
	put64(0x20, (uintptr_t)gpu_mem + 0x100);	/* address */
	put32(0x28, 6);					/* immediate 32 */
	put64(0x30, magic);
	irq = run_job(shaders);
	kv("WRITE job irq", irq);
	kv("js0 status", rd(MALI + JS0 + 0x24));
	kv("exc", get32(0x00));
	kv("value", get32(0x100));
	out("\n");
	kv("AS0 fault", rd(MALI + AS0 + 0x1c));
	kv("GPU_IRQ", rd(MALI + GPU_IRQ_RAWSTAT));
	out("\n");
	kv("GPU JOB WORKS", get32(0x100) == magic);
	out("\n");
	if (get32(0x100) == magic) {
		gpu_draw(kv, out, shaders);
		if (matrix) {
			if (matrix > 1)			/* s8gpu=3 */
				gpu_matrix(kv, out, shaders);
			/* 'B' record flags bit 0: page tables only (one run, as XNU) */
			if (!(rp32(0) == 'B' && (rp32(4) & 1)))
				rp_run(kv, out, shaders, 0);
			rp_run(kv, out, shaders, 1);
			mx_as(2, 0);			/* back to IDENTITY */
		}
	}
}

int gpu_probe(void (*kv)(const char *, uint64_t), void (*out)(const char *), int matrix)
{
	out("GPU probe (Mali-G71, pd-g3d)\n");
	kv("g3d cfg", rd(G3D_CFG));
	kv("stat", rd(G3D_STAT));
	out("\n");
	if ((rd(G3D_STAT) & 0xf) != 0xf) {
		wr(CMU_G3D + 0x148, 0x30000003);
		wr(CMU_G3D + 0x140, rd(CMU_G3D + 0x140) | (1u << 4));
		wr(CMU_G3D + 0x100, rd(CMU_G3D + 0x100) | (1u << 4));
		wr(G3D_CFG, rd(G3D_CFG) | 0xf);
		uint64_t end = ticks() + TIMER_HZ / 20;
		while (ticks() < end && (rd(G3D_STAT) & 0xf) != 0xf)
			;
		kv("g3d on: stat", rd(G3D_STAT));
		out("\n");
	}
	if ((rd(G3D_STAT) & 0xf) != 0xf) {
		out("GPU: power domain stays off\n");
		return 0;
	}
	kv("cmu g3d pll", rd(CMU_G3D + 0x120));
	kv("0x140", rd(CMU_G3D + 0x140));
	kv("mux", rd(CMU_G3D + 0x100));
	out("\n");
	uint32_t id = rd(MALI + 0x000);
	kv("GPU_ID", id);
	kv("L2_FEATURES", rd(MALI + 0x004));
	kv("GPU_STATUS", rd(MALI + 0x034));
	out("\n");
	kv("SHADER_PRESENT", rd(MALI + 0x100));
	kv("L2_PRESENT", rd(MALI + 0x120));
	out("\n");
	/* Thread storage sizing (Mesa pan_query_thread_tls_alloc: TLS_ALLOC,
	 * else MAX_THREADS); s8mtl_draw.c assumed 256 threads per core. */
	kv("THREAD_MAX_THREADS", rd(MALI + 0x0a0));
	kv("WG", rd(MALI + 0x0a4));
	kv("FEATURES", rd(MALI + 0x0ac));
	kv("TLS_ALLOC", rd(MALI + 0x310));
	out("\n");
	/* kbase GPU_ID2_PRODUCT_TMIX = MAKE(6, 0, 10, 0) = 0x60a0 (Mali-G71). */
	int g71 = (id >> 16) == 0x60a0;
	kv("MALI G71 FOUND", g71);
	out("\n");
	if (g71)
		gpu_jobs(kv, out, rd(MALI + 0x100), matrix);
	out("GPU probe done\n");
	return g71;
}
