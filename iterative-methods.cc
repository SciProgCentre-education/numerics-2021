#define HAVE_OPENMP

#include "iterative-methods.hh"

using namespace noa::TNL;

torch::Tensor jacobi_solve(
    torch::Tensor crow_indices,
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor b,
    float omega)
{
    check_format(crow_indices, col_indices, values, b);
    check_cpu(crow_indices, col_indices, values, b);

    auto solver = Jacobi<SparseMatrixView<float, Devices::Host>>{};
    solver.setOmega(omega);

    return solve<float, Devices::Host>(
        crow_indices, col_indices, values, b, solver);
}

torch::Tensor sor_solve(
    torch::Tensor crow_indices,
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor b,
    float omega)
{
    check_format(crow_indices, col_indices, values, b);
    check_cpu(crow_indices, col_indices, values, b);

    auto solver = SOR<SparseMatrixView<float, Devices::Host>>{};
    solver.setOmega(omega);

    return solve<float, Devices::Host>(
        crow_indices, col_indices, values, b, solver);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("jacobi_solve", &jacobi_solve, py::call_guard<py::gil_scoped_release>(), "Jacobi iterative method");
    m.def("sor_solve", &sor_solve, py::call_guard<py::gil_scoped_release>(), "SOR iterative method");
}

