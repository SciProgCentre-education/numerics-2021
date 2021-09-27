#include <torch/extension.h>

#include <noa/pms/kernels.cuh>

using namespace noa::pms;


torch::Tensor bremsstrahlung_dcs(torch::Tensor kinetic_energies, torch::Tensor recoil_energies)
{
    const auto result = torch::zeros_like(kinetic_energies);
    dcs::pumas::cuda::vmap_bremsstrahlung(result, kinetic_energies, recoil_energies, STANDARD_ROCK, MUON_MASS);
    return result;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("bremsstrahlung_dcs", &bremsstrahlung_dcs, py::call_guard<py::gil_scoped_release>(),
          "Standard Rock Bremsstrahlung DCS for Muons on CUDA");
}