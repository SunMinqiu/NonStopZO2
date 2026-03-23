/*
 * bindings.cpp — PyTorch C++ extension bindings for zo_rng
 *
 * Dispatches to CPU or CUDA kernel based on requested device.
 * No CUDA headers needed here — CUDA logic lives in generate_cuda.cu.
 */
#include <torch/extension.h>
#include <stdint.h>
#include "zo_rng_pool.h"

/* CPU kernel (compiled from generate_cpu.cpp) */
extern "C" void zo_rng_generate_cpu(int64_t seed, int64_t counter,
                                     int64_t n, float *output, int pool_id);

#ifdef ZO_RNG_WITH_CUDA
/* CUDA entry point (compiled from generate_cuda.cu) */
extern "C" torch::Tensor zo_rng_generate_cuda_tensor(
    int64_t seed, int64_t counter, int64_t n, int device_idx);
#endif

torch::Tensor generate_normal(int64_t seed, int64_t counter, int64_t n,
                               const std::string &device_str, int pool_id) {
    if (device_str == "cpu") {
        auto output = torch::empty({n}, torch::TensorOptions()
                                            .dtype(torch::kFloat32)
                                            .device(torch::kCPU));
        zo_rng_generate_cpu(seed, counter, n, output.data_ptr<float>(), pool_id);
        return output;
    }

#ifdef ZO_RNG_WITH_CUDA
    if (device_str.rfind("cuda", 0) == 0) {
        int device_idx = 0;
        if (device_str.size() > 5) {
            device_idx = std::stoi(device_str.substr(5));
        }
        /* CUDA has its own parallelism — pool_id not used */
        return zo_rng_generate_cuda_tensor(seed, counter, n, device_idx);
    }
#endif

    throw std::runtime_error("zo_rng: unsupported device '" + device_str +
                              "'. Build with CUDA support for GPU generation, "
                              "or use device='cpu'.");
}

bool has_cuda() {
#ifdef ZO_RNG_WITH_CUDA
    return true;
#else
    return false;
#endif
}

/* Default pool operations (backward compatible) */
void pool_set_num_threads(int n) {
    ZoRngPoolRegistry::instance().set_num_threads(n);
}

int pool_get_num_threads() {
    return ZoRngPoolRegistry::instance().get_num_threads();
}

/* Multi-pool operations */
int pool_create(int num_threads) {
    return ZoRngPoolRegistry::instance().create_pool(num_threads);
}

void pool_destroy(int pool_id) {
    ZoRngPoolRegistry::instance().destroy_pool(pool_id);
}

PYBIND11_MODULE(_ext_impl, m) {
    m.def("generate_normal", &generate_normal,
          "Generate deterministic normal-distributed random numbers",
          py::arg("seed"), py::arg("counter"), py::arg("n"),
          py::arg("device"), py::arg("pool_id") = 0);
    m.def("has_cuda", &has_cuda,
          "Whether the extension was built with CUDA support");
    m.def("set_num_threads", &pool_set_num_threads,
          "Set default pool's thread count (independent of OMP)",
          py::arg("n"));
    m.def("get_num_threads", &pool_get_num_threads,
          "Get default pool's current thread count");
    m.def("create_pool", &pool_create,
          "Create an independent thread pool, returns pool_id",
          py::arg("num_threads"));
    m.def("destroy_pool", &pool_destroy,
          "Destroy a thread pool by pool_id",
          py::arg("pool_id"));
}
