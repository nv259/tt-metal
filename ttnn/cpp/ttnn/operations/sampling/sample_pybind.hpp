#pragma once

#include "pybind11/pybind_fwd.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace sample {

void py_module(py::module& module);

}  // namespace matmul
}  // namespace operations
}  // namespace ttnn