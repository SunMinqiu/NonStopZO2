/*
 * bindings.cpp — PyTorch C++ extension bindings for zo_rng
 *
 * Dispatches to CPU or CUDA kernel based on requested device.
 */
#include <torch/extension.h>
#include <stdint.h>

/* CPU kernel (compiled from generate_cpu.c) */
extern "C" void zo_rng_generate_cpu(int64_t seed, int64_t counter,
                                     int64_t n, float *output);

#ifdef ZO_RNG_WITH_CUDA
/* CUDA kernel (compiled from generate_cuda.cu) */
extern "C" void zo_rng_generate_cuda(int64_t seed, int64_t counter,
                                      int64_t n, float *output, void *stream);
#endif

torch::Tensor generate_normal(int64_t seed, int64_t counter, int64_t n,
                               const std::string &device_str) {
    if (device_str == "cpu") {
        auto output = torch::empty({n}, torch::TensorOptions()
                                            .dtype(torch::kFloat32)
                                            .device(torch::kCPU));
        zo_rng_generate_cpu(seed, counter, n, output.data_ptr<float>());
        return output;
    }

#ifdef ZO_RNG_WITH_CUDA
    if (device_str.rfind("cuda", 0) == 0) {
        /* Parse device index from "cuda" or "cuda:N" */
        int device_idx = 0;
        if (device_str.size() > 5) {
            device_idx = std::stoi(device_str.substr(5));
        }

        c10::cuda::set_device(device_idx);
        auto output = torch::empty({n}, torch::TensorOptions()
                                            .dtype(torch::kFloat32)
                                            .device(torch::kCUDA, device_idx));

        /* Use PyTorch's current CUDA stream for proper synchronization */
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_idx).stream();
        zo_rng_generate_cuda(seed, counter, n,
                              output.data_ptr<float>(), (void *)stream);
        return output;
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

PYBIND11_MODULE(_ext_impl, m) {
    m.def("generate_normal", &generate_normal,
          "Generate deterministic normal-distributed random numbers",
          py::arg("seed"), py::arg("counter"), py::arg("n"),
          py::arg("device"));
    m.def("has_cuda", &has_cuda,
          "Whether the extension was built with CUDA support");
}
