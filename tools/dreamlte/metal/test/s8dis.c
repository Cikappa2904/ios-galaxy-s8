/*
 * Disassemble every shader of an S8SC cache (the code air2nir wrote, as the
 * phone runs it): "=== NAME stage S code N" then Mesa's Bifrost listing.
 *
 *   s8dis CACHE.s8sc [NAME]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "wrap.h"
#include "pan_shader.h"

static uint32_t rd32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }

int main(int argc, char **argv)
{
	if (argc < 2)
		return 2;
	FILE *f = fopen(argv[1], "rb");
	if (f == NULL)
		return 2;
	fseek(f, 0, SEEK_END);
	long size = ftell(f);
	fseek(f, 0, SEEK_SET);
	uint8_t *b = malloc(size);
	if (fread(b, 1, size, f) != (size_t)size)
		return 2;
	fclose(f);
	if (size < 12 || rd32(b) != 0x43533853)        /* "S8SC" */
		return 3;
	unsigned n = rd32(b + 8);
	for (unsigned i = 0; i < n; i++) {
		const uint8_t *e = b + 12 + i * 72;
		char name[65] = { 0 };
		memcpy(name, e, 64);
		if (argc > 2 && strcmp(name, argv[2]) != 0)
			continue;
		uint32_t off = rd32(e + 64), len = rd32(e + 68);
		const uint8_t *s = b + off;
		uint32_t stage = rd32(s + 8), code = rd32(s + 12);
		printf("=== %s stage %u code %u\n", name, stage, code);
		fflush(stdout);
		pan_shader_disassemble(stdout, s + len - code, code, 0x60a00001, false);
		fflush(stdout);
	}
	return 0;
}
