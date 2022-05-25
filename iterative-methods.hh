#pragma once

#include <torch/extension.h>

#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/MatrixWrapping.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/SparseMatrix.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/VectorView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/Linear/Jacobi.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/Linear/SOR.h>

using namespace noa::TNL::Containers;
using namespace noa::TNL::Matrices;
using namespace noa::TNL::Solvers::Linear;

inline void check_format(const torch::Tensor &crow_indices,
                         const torch::Tensor &col_indices,
                         const torch::Tensor &values,
                         const torch::Tensor &tensor_b)
{
    TORCH_CHECK(crow_indices.dtype() == torch::kInt32, "crow_indices must be int");
    TORCH_CHECK(col_indices.dtype() == torch::kInt32, "col_indices must be int");
    TORCH_CHECK(values.dtype() == torch::kFloat32, "values must be float");
    TORCH_CHECK(tensor_b.dtype() == torch::kFloat32, "b must be float");
}

inline void check_cpu(const torch::Tensor &crow_indices,
                        const torch::Tensor &col_indices,
                        const torch::Tensor &values,
                        const torch::Tensor &tensor_b)
{
    TORCH_CHECK(crow_indices.is_cpu(), "crow_indices CPU tensor required");
    TORCH_CHECK(col_indices.is_cpu(), "col_indices CPU tensor required");
    TORCH_CHECK(values.is_cpu(), "values CPU tensor required");
    TORCH_CHECK(tensor_b.is_cpu(), "b CPU tensor required");
}

inline void check_cuda(const torch::Tensor &crow_indices,
                       const torch::Tensor &col_indices,
                       const torch::Tensor &values,
                       const torch::Tensor &tensor_b)
{
    TORCH_CHECK(crow_indices.is_cuda(), "crow_indices CUDA tensor required");
    TORCH_CHECK(col_indices.is_cuda(), "col_indices CUDA tensor required");
    TORCH_CHECK(values.is_cuda(), "values CUDA tensor required");
    TORCH_CHECK(tensor_b.is_cuda(), "b CUDA tensor required");
}


template <typename Dtype, typename Device, typename Solver>
torch::Tensor solve(
    const torch::Tensor &crow_indices,
    const torch::Tensor &col_indices,
    const torch::Tensor &values,
    const torch::Tensor &tensor_b,
    Solver &solver)
{

    int size = tensor_b.numel();
    const auto tensor_x = torch::zeros_like(tensor_b);

    auto matrix = std::make_shared<SparseMatrixView<Dtype, Device>>(
        wrapCSRMatrix<Device, Dtype>(size, size,
                                     crow_indices.data_ptr<int>(),
                                     values.data_ptr<Dtype>(),
                                     col_indices.data_ptr<int>()));

    const auto x = VectorView<Dtype, Device>{tensor_x.data_ptr<Dtype>(), size};
    auto b = VectorView<Dtype, Device>{tensor_b.data_ptr<Dtype>(), size};

    solver.setMatrix(matrix);
    solver.solve(b, x);

    return tensor_x;
}