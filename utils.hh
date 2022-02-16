#pragma once 

#include <torch/extension.h>

torch::Tensor check_float_vec(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.dim() == 1, "tensor of dim 1 required");
    TORCH_CHECK(tensor.is_contiguous(), "contiguous tensor required");
    TORCH_CHECK(tensor.dtype() == torch::kFloat32, "Float32 tensor required");
    return tensor;
}