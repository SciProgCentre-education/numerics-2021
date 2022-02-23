#define HAVE_OPENMP

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
    
    TORCH_CHECK(crow_indices.is_cpu(), "crow_indices CPU tensor required");
    TORCH_CHECK(col_indices.is_cpu(), "col_indices CPU tensor required");
    TORCH_CHECK(values.is_cpu(), "values CPU tensor required");
    TORCH_CHECK(b.is_cpu(), "b CPU tensor required");

    return jacobi_solve<float, Devices::Host>(
        crow_indices, col_indices, values, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("solve", &solve,
          "Jacobi iterative method");
}
