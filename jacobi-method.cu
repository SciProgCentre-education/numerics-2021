#define HAVE_CUDA

#include "stationary-methods.hh"

using namespace noa::TNL;

torch::Tensor solve(
    torch::Tensor crow_indices,
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor b)
{
    TORCH_CHECK(values.dtype() == torch::kFloat32, "values must be float");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "b must be float");

    TORCH_CHECK(crow_indices.is_cuda(), "crow_indices CUDA tensor required");
    TORCH_CHECK(col_indices.is_cuda(), "col_indices CUDA tensor required");
    TORCH_CHECK(values.is_cuda(), "values CUDA tensor required");
    TORCH_CHECK(b.is_cuda(), "b CUDA tensor required");


    return jacobi_solve<float, Devices::Cuda>(
        crow_indices, col_indices, values, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("solve", &solve,
          "Jacobi iterative method");
}