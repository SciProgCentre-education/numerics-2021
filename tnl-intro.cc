#include "tnl-intro.hh"
#include "utils.hh"
#include <torch/extension.h>

using namespace noa::TNL;

float map_reduce(torch::Tensor tensor) {
    const auto tensor_ = check_float_vec(tensor);
    TORCH_CHECK(tensor_.is_cpu(), "CPU tensor required");
    return map_reduce_tnl<float, Devices::Host>(tensor_);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("map_reduce", &map_reduce, py::call_guard<py::gil_scoped_release>(),
          "Parallel map-reduce TNL example on CPU");
}