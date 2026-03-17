/*
 * generate_cuda.cu — CUDA kernel for deterministic normal RNG
 *
 * Same algorithm as generate_cpu.c: Philox4x32-10 + deterministic Box-Muller.
 * Uses grid-stride loop pattern; one thread per Philox counter value (4 floats).
 */
#include <stdint.h>
#include <Random123/philox.h>
#include "deterministic_math.h"

__global__ void zo_rng_kernel(uint32_t key_lo, uint32_t key_hi,
                              int64_t counter, int64_t n, float *output) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t num_blocks = (n + 3) / 4;

    philox4x32_key_t key;
    key.v[0] = key_lo;
    key.v[1] = key_hi;

    for (int64_t i = idx; i < num_blocks; i += stride) {
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

extern "C"
void zo_rng_generate_cuda(int64_t seed, int64_t counter, int64_t n,
                           float *output, void *stream) {
    uint32_t key_lo = (uint32_t)(seed & 0xFFFFFFFFu);
    uint32_t key_hi = (uint32_t)((uint64_t)seed >> 32);

    int64_t num_blocks = (n + 3) / 4;
    int threads = 256;
    int blocks = (int)((num_blocks + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    cudaStream_t cu_stream = (cudaStream_t)stream;
    zo_rng_kernel<<<blocks, threads, 0, cu_stream>>>(
        key_lo, key_hi, counter, n, output);
}

/*
 * High-level entry point called from bindings.cpp.
 * Handles device setup, tensor allocation, and stream management
 * so that bindings.cpp doesn't need any CUDA headers.
 */
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

extern "C"
torch::Tensor zo_rng_generate_cuda_tensor(int64_t seed, int64_t counter,
                                           int64_t n, int device_idx) {
    c10::cuda::CUDAGuard guard(device_idx);
    auto output = torch::empty({n}, torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .device(torch::kCUDA, device_idx));
    auto stream = c10::cuda::getCurrentCUDAStream(device_idx).stream();
    zo_rng_generate_cuda(seed, counter, n,
                          output.data_ptr<float>(), (void *)stream);
    return output;
}
