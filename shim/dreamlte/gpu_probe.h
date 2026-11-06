#pragma once
#include <stdint.h>

/* Powers pd-g3d and reads the Mali ID; 1 when a Mali-G71 answers. */
int gpu_probe(void (*kv)(const char *, uint64_t), void (*out)(const char *),
    int matrix);
/* matrix: also the address/shader matrix (boot-arg s8gpu=2). It writes the
 * high bank at 0x8a0000000+, so only with S8_DRAM_MIB=512 (XNU's RAM must
 * not reach it, the payload may live there before the jump). */
