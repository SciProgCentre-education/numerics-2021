#pragma once

#include <torch/extension.h>

#include <noa/3rdparty/TNL/Matrices/MatrixWrapping.h>
#include <noa/3rdparty/TNL/Matrices/SparseMatrix.h>
#include <noa/3rdparty/TNL/Containers/VectorView.h>
#include <noa/3rdparty/TNL/Solvers/Linear/Jacobi.h>


using namespace noa::TNL::Containers;
using namespace noa::TNL::Matrices;
using namespace noa::TNL::Solvers::Linear;

template<typename Dtype, typename Device>
torch::Tensor jacobi_solve(
    const torch::Tensor &crow_indices,
    const torch::Tensor &col_indices,
    const torch::Tensor &values,
    const torch::Tensor &tensor_b,
    const float &omega) {

    TORCH_CHECK(crow_indices.dtype() == torch::kInt32, "crow_indices must be int");
    TORCH_CHECK(col_indices.dtype() == torch::kInt32, "col_indices must be int");

    int size = tensor_b.numel();
    const auto tensor_x = torch::zeros_like(tensor_b);

    auto matrix = std::make_shared<SparseMatrixView<Dtype, Device>>(
            wrapCSRMatrix<Device, Dtype>(size, size,
                                  crow_indices.data_ptr<int>(),
                                  values.data_ptr<Dtype>(),
                                  col_indices.data_ptr<int>()));

   
    const auto x = VectorView<Dtype, Device>{tensor_x.data_ptr<Dtype>(), size};
    auto b = VectorView<Dtype, Device>{tensor_b.data_ptr<Dtype>(), size};

    auto solver = Jacobi<SparseMatrixView<Dtype, Device>>{};
    solver.setOmega(omega);
    solver.setMatrix(matrix);
    solver.solve(b, x);

    return tensor_x;
}