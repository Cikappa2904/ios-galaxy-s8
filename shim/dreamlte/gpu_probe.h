#pragma once
#include <stdint.h>

/* Powers pd-g3d and reads the Mali ID; 1 when a Mali-G71 answers. */
int gpu_probe(void (*kv)(const char *, uint64_t), void (*out)(const char *),
    int matrix);
/* matrix 1 (boot-arg s8gpu=2): replay the Metal job chains of
 * s8_mali_replay.h; 2 (s8gpu=3): the address/shader matrix first. They write
 * the high bank at 0x8a0000000+, so only with S8_DRAM_MIB=512 (XNU's RAM must
 * not reach it, the payload may live there before the jump). */
