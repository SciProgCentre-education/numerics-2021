#include <torch/extension.h>

torch::Tensor get_rotations(const torch::Tensor &thetas) {
    const auto f = thetas.flatten();
    const auto n = f.numel();
    const auto c = torch::cos(f);
    const auto s = torch::sin(f);
    return torch::stack({c, -s, s, c}).t().view({n, 2, 2});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_rotations", &get_rotations, py::call_guard<py::gil_scoped_release>(),
          "Generate 2D rotations given angles thetas");
}