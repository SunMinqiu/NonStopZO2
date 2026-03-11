/*
 * generate_cpu.c — CPU kernel for deterministic normal RNG
 *
 * Uses Random123 Philox4x32-10 for PRNG + deterministic Box-Muller transform.
 * OpenMP parallel over counter values (embarrassingly parallel).
 */
#include <stdint.h>
#include <Random123/philox.h>
#include "deterministic_math.h"

void zo_rng_generate_cpu(int64_t seed, int64_t counter, int64_t n, float *output) {
    /* Derive Philox key from seed (split 64-bit seed into 2×32) */
    philox4x32_key_t key;
    key.v[0] = (uint32_t)(seed & 0xFFFFFFFFu);
    key.v[1] = (uint32_t)((uint64_t)seed >> 32);

    int64_t num_blocks = (n + 3) / 4;

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < num_blocks; i++) {
        int64_t ctr_val = counter + i;

        philox4x32_ctr_t ctr;
        ctr.v[0] = (uint32_t)(ctr_val & 0xFFFFFFFFu);
        ctr.v[1] = (uint32_t)((uint64_t)ctr_val >> 32);
        ctr.v[2] = 0;
        ctr.v[3] = 0;

        philox4x32_ctr_t result = philox4x32(ctr, key);

        uint32_t raw[4] = {result.v[0], result.v[1], result.v[2], result.v[3]};
        float normals[4];
        uint32x4_to_normal4(raw, normals);

        int64_t base = i * 4;
        int64_t remaining = n - base;
        int count = (remaining >= 4) ? 4 : (int)remaining;
        for (int j = 0; j < count; j++) {
            output[base + j] = normals[j];
        }
    }
}
