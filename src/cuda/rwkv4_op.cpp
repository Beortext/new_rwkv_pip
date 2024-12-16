#include <torch/extension.h>
#include "ATen/ATen.h"
#include <iostream>
#include <c10/cuda/CUDAGuard.h>

typedef at::Half fp16;

template <typename F>
void cuda_wkv_forward(int B, int T, int C,
                      float *w, float *u, F *k, F *v, F *y,
                      float *aa, float *bb, float *pp);

void wkv_forward(int64_t B, int64_t T, int64_t C,
                 torch::Tensor &w, torch::Tensor &u,
                 torch::Tensor &k, torch::Tensor &v, torch::Tensor &y,
                 torch::Tensor &aa, torch::Tensor &bb, torch::Tensor &pp)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(w));
    switch (k.scalar_type())
    {
    case c10::ScalarType::Half:
        cuda_wkv_forward(B, T, C,
                         w.data_ptr<float>(), u.data_ptr<float>(),
                         k.data_ptr<fp16>(), v.data_ptr<fp16>(), y.data_ptr<fp16>(),
                         aa.data_ptr<float>(), bb.data_ptr<float>(), pp.data_ptr<float>());
        break;
    case c10::ScalarType::Float:
        cuda_wkv_forward(B, T, C,
                         w.data_ptr<float>(), u.data_ptr<float>(),
                         k.data_ptr<float>(), v.data_ptr<float>(), y.data_ptr<float>(),
                         aa.data_ptr<float>(), bb.data_ptr<float>(), pp.data_ptr<float>());
        break;
    default:
        assert(false && "Only FP16 and FP32 are currently supported");
    }
}

using torch::Tensor;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("wkv_forward", &wkv_forward, "wkv forward");
}

TORCH_LIBRARY(rwkv4, m)
{
    m.def("wkv_forward", wkv_forward);
}
