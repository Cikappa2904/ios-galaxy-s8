/*
 * Mali replay generator: runs the S8 Metal draw path (s8mtl_draw.c) on the
 * host exactly as draw-test.c does, but instead of only decoding the job
 * chains it records them for the loader (shim/dreamlte/gpu_probe.c, MX R),
 * which replays them on the real Mali-G71 before XNU starts:
 *
 *   per S8MALI_SUBMIT: every non-zero 4 KiB page of GPU memory at that
 *   moment ('P' records), then the submit itself ('S': slot, chain head,
 *   the chain's job addresses, a probe address, a label).
 *
 * The first vertex/tiler chain is also recorded cut down to its first
 * VERTEX job alone and to its first TILER job alone (dependencies cleared),
 * so one boot tells which job type the hardware stalls on. GPU memory is
 * zero-filled here, as S8MALI_ALLOC does on the phone. Every chain is still
 * checked by pandecode.
 *
 *   replay-gen CACHE.s8sc OUT.h
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "wrap.h"
#include "s8mtl_core.h"
#include "s8mtl_draw.h"
#include "../../s8mali_ioctl.h"

#define S8T_BASE  0x8b0000000ULL
#define S8T_SIZE  (96u << 20)
#define MAXJOBS   8

static uint8_t *mem;
static uint64_t used;
static struct pandecode_context *dec;
static int submits, failures;
static FILE *blob;
static unsigned npages, nsubmits;
static uint64_t probe_next;             /* probe address for the next fragment submit */
static int vt_seen;

void s8_log(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	printf("S8 MALI| ");
	vprintf(fmt, ap);
	printf("\n");
	va_end(ap);
	if (strstr(fmt, "failed") || strstr(fmt, "bad") || strstr(fmt, "no GPU pipeline"))
		failures++;
}

int s8_fd(void) { return 3; }
uint64_t s8_now_ns(void) { return 0; }

int s8_mem_alloc(s8_mem *m, uint64_t size, uint64_t align)
{
	if (align < 64)
		align = 64;
	uint64_t at = (used + align - 1) & ~(align - 1);
	if (at + size > S8T_SIZE)
		return -1;
	used = at + size;
	memset(mem + at, 0, size);      /* as S8MALI_ALLOC */
	m->cpu = mem + at;
	m->gpu = S8T_BASE + at;
	m->size = size;
	m->big = 1;
	return 0;
}

void s8_mem_free(s8_mem *m) { (void)m; }

/* the chains never fail here, so s8mtl_draw.c's dump never runs */
unsigned s8_mem_regions(uint64_t *gpu, void **cpu, uint64_t *size, unsigned max)
{
	(void)gpu; (void)cpu; (void)size; (void)max;
	return 0;
}

static uint8_t *host(uint64_t gpu) { return mem + (gpu - S8T_BASE); }
static uint32_t h32(uint64_t gpu) { uint32_t v; memcpy(&v, host(gpu), 4); return v; }
static uint64_t h64(uint64_t gpu) { uint64_t v; memcpy(&v, host(gpu), 8); return v; }
static void s32(uint64_t gpu, uint32_t v) { memcpy(host(gpu), &v, 4); }
static void s64(uint64_t gpu, uint64_t v) { memcpy(host(gpu), &v, 8); }

static void put(const void *p, size_t n) { fwrite(p, 1, n, blob); }
static void put32(uint32_t v) { put(&v, 4); }
static void put64(uint64_t v) { put(&v, 8); }

/* Record: the memory as it is now, then the submit. */
static void record(unsigned slot, uint64_t jc, uint64_t probe, const char *label)
{
	static const uint8_t zero[4096];
	for (uint64_t o = 0; o < used; o += 4096) {
		if (memcmp(mem + o, zero, 4096) == 0)
			continue;
		put32('P');
		put32(0);
		put64(S8T_BASE + o);
		put(mem + o, 4096);
		npages++;
	}
	uint64_t jobs[MAXJOBS] = { 0 };
	unsigned n = 0;
	for (uint64_t j = jc; j && n < MAXJOBS; j = h64(j + 0x18))
		jobs[n++] = j;
	char name[32] = { 0 };
	snprintf(name, sizeof(name), "%s", label);
	put32('S');
	put32(slot);
	put64(jc);
	put64(probe);
	put32(n);
	put32(0);
	put(jobs, sizeof(jobs));
	put(name, sizeof(name));
	nsubmits++;
	printf("==== record %u: %s slot %u jc 0x%llx, %u jobs:", nsubmits, name, slot,
	    (unsigned long long)jc, n);
	for (unsigned i = 0; i < n; i++)
		printf(" 0x%llx(type %u idx %u dep %u/%u)", (unsigned long long)jobs[i],
		    (h32(jobs[i] + 0x10) >> 1) & 0x7f, h32(jobs[i] + 0x10) >> 16,
		    h32(jobs[i] + 0x14) & 0xffff, h32(jobs[i] + 0x14) >> 16);
	printf("\n");
}

static void vt_variants(uint64_t jc)
{
	/* first VERTEX job alone */
	uint64_t next = h64(jc + 0x18);
	s64(jc + 0x18, 0);
	record(1, jc, 0, "vertex job alone");
	s64(jc + 0x18, next);
	/* first TILER job alone, no dependencies */
	uint64_t t = next;
	if (t && ((h32(t + 0x10) >> 1) & 0x7f) == 7 /* Tiler */) {
		uint32_t deps = h32(t + 0x14);
		uint64_t tn = h64(t + 0x18);
		s32(t + 0x14, 0);
		s64(t + 0x18, 0);
		record(1, t, 0, "tiler job alone");
		s32(t + 0x14, deps);
		s64(t + 0x18, tn);
	} else {
		printf("==== second job is not a tiler job\n");
		failures++;
	}
}

/* s8mtl_draw.c's /dev/s8mali requests. */
int ioctl(int fd, unsigned long req, ...)
{
	va_list ap;
	va_start(ap, req);
	void *arg = va_arg(ap, void *);
	va_end(ap);
	(void)fd;
	if (req == S8MALI_INFO) {
		struct s8mali_info *i = arg;
		memset(i, 0, sizeof(*i));
		i->gpu_id = 0x60a00001;
		i->shader_present = 0x17771777;
		i->tiler_features = 0x809;
		i->ready = 1;
		i->heap_free = S8T_SIZE;
		return 0;
	}
	if (req == S8MALI_SUBMIT) {
		struct s8mali_submit *s = arg;
		printf("---- submit slot %u jc 0x%llx\n", s->slot, (unsigned long long)s->jc);
		pandecode_jc(dec, s->jc, 0x60a00001);
		if (s->slot == 1 && !vt_seen++)
			vt_variants(s->jc);
		record(s->slot, s->jc, s->slot == 0 ? probe_next : 0,
		    s->slot == 0 ? "fragment chain" : "vertex/tiler chain");
		s->result = 0;
		submits++;
		return 0;
	}
	return -1;
}

static struct s8d_pipeline *pipeline(void)
{
	struct s8d_pipeline_desc d = { 0 };
	d.vs_name = "@test_vs";
	d.fs_name = "@test_fs";
	d.vs_constants = d.fs_constants = "";
	d.color_format = 80;                    /* BGRA8Unorm */
	d.blending = 1;
	d.write_mask = 0xf;
	d.rgb_op = d.alpha_op = 0;
	d.src_rgb = d.src_alpha = 1;            /* one, one-minus-source-alpha */
	d.dst_rgb = d.dst_alpha = 5;
	d.attribs[0] = (struct s8d_vertex_attrib){ 29, 0, 0 };   /* Float2 */
	d.attribs[1] = (struct s8d_vertex_attrib){ 29, 8, 0 };
	d.layouts[0] = (struct s8d_vertex_layout){ 16, 1, 1 };
	return s8d_pipeline_new(&d, 1);
}

/* One pass over rt: load 1 (preload) / 2 (clear), with or without the two draws. */
static void pass(struct s8d_pipeline *p, const struct s8d_texture *rt, unsigned load, int draws,
    s8_mem *vbuf, const struct s8d_texture *tex)
{
	struct s8d_sampler smp = { .min_linear = 1, .mag_linear = 1, .normalized = 1 };
	float scale[4] = { 1, 1, 1, 1 }, colour[4] = { 1, 0.5f, 0.25f, 1 };
	struct s8d_cmdbuf *cb = s8d_cmdbuf_new();
	double clear[4] = { 0, 0, 1, 1 };
	int idx = s8d_pass_begin(cb, rt, load, clear);
	if (draws) {
		s8d_set_pipeline(cb, p);
		s8d_set_buffer(cb, S8D_STAGE_VERTEX, 0, vbuf->gpu, vbuf->cpu, 64);
		s8d_set_bytes(cb, S8D_STAGE_VERTEX, 1, scale, sizeof(scale));
		s8d_set_bytes(cb, S8D_STAGE_FRAGMENT, 0, colour, sizeof(colour));
		s8d_set_texture(cb, S8D_STAGE_FRAGMENT, 0, tex);
		s8d_set_sampler(cb, S8D_STAGE_FRAGMENT, 0, &smp);
		s8d_set_viewport(cb, 0, 0, rt->width, rt->height, 0, 1);
		s8d_draw(cb, 4, 0, 4, 1, 0, 0, NULL);            /* triangle strip */
		uint16_t ix[6] = { 0, 1, 2, 2, 1, 3 };
		s8d_draw(cb, 3, 0, 6, 1, 2, 0, ix);              /* indexed triangles */
	}
	s8d_pass_end(cb);
	probe_next = rt->gpu + (uint64_t)(rt->height / 2) * rt->row_bytes + (rt->width / 2) * 4;
	if (s8d_pass_execute(cb, idx) != 0)
		failures++;
	s8d_cmdbuf_free(cb);
}

int main(int argc, char **argv)
{
	if (argc != 3) {
		fprintf(stderr, "usage: %s CACHE.s8sc OUT.bin\n", argv[0]);
		return 2;
	}
	setenv("S8_TEST_CACHE", argv[1], 1);
	blob = fopen(argv[2], "wb");
	if (blob == NULL)
		return 2;
	mem = aligned_alloc(4096, S8T_SIZE);
	memset(mem, 0, S8T_SIZE);
	dec = pandecode_create_context(true);  /* decode to stderr */
	pandecode_inject_mmap(dec, S8T_BASE, mem, S8T_SIZE, "s8mali");
	if (s8d_init() != 0)
		return 1;
	struct s8d_pipeline *p = pipeline();
	if (p == NULL)
		return 1;

	s8_mem small, big, texmem, vbuf;
	s8_mem_alloc(&small, 256 * 128 * 4, 4096);
	s8_mem_alloc(&big, 1440 * 2960 * 4, 4096);
	s8_mem_alloc(&texmem, 64 * 64 * 4, 64);
	s8_mem_alloc(&vbuf, 4 * 16, 64);
	for (unsigned i = 0; i < 64 * 64; i++)
		((uint32_t *)texmem.cpu)[i] = 0xff00ff00u;      /* opaque green */
	float v[16] = { -1, -1, 0, 1,   1, -1, 1, 1,   -1, 1, 0, 0,   1, 1, 1, 0 };
	memcpy(vbuf.cpu, v, sizeof(v));
	struct s8d_texture rt_s = { small.gpu, 256, 128, 256 * 4, 80 };
	struct s8d_texture rt_b = { big.gpu, 1440, 2960, 1440 * 4, 80 };
	struct s8d_texture tex = { texmem.gpu, 64, 64, 64 * 4, 80 };

	pass(p, &rt_s, 2, 0, &vbuf, &tex);      /* fragment only: clear */
	pass(p, &rt_s, 2, 1, &vbuf, &tex);      /* clear + 2 draws */
	pass(p, &rt_s, 1, 1, &vbuf, &tex);      /* preload + 2 draws */
	pass(p, &rt_b, 2, 1, &vbuf, &tex);      /* panel size, clear + 2 draws */

	put32('E');
	put32(0);
	put64(used);
	fclose(blob);
	printf("==== %d submits, %u records, %u pages, GPU memory used %llu KiB, %d failures\n",
	    submits, nsubmits, npages, (unsigned long long)(used >> 10), failures);
	return failures ? 1 : 0;
}
