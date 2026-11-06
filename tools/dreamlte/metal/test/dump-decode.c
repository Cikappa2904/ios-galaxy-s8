/*
 * pandecode over a GPU memory image rebuilt by undump.py from the phone's
 * log (s8mtl_draw.c s8_dump): decodes the job chain at JC.
 *
 *   dump-decode IMAGE BASE JC
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "wrap.h"

int main(int argc, char **argv)
{
	if (argc != 4) {
		fprintf(stderr, "usage: %s IMAGE BASE JC\n", argv[0]);
		return 2;
	}
	FILE *f = fopen(argv[1], "rb");
	if (f == NULL)
		return 2;
	fseek(f, 0, SEEK_END);
	long size = ftell(f);
	fseek(f, 0, SEEK_SET);
	void *img = malloc(size);
	if (img == NULL || fread(img, 1, size, f) != (size_t)size)
		return 2;
	fclose(f);
	uint64_t base = strtoull(argv[2], NULL, 0), jc = strtoull(argv[3], NULL, 0);
	struct pandecode_context *dec = pandecode_create_context(true);
	pandecode_inject_mmap(dec, base, img, size, "dump");
	pandecode_jc(dec, jc, 0x60a00001);
	pandecode_destroy_context(dec);
	return 0;
}
