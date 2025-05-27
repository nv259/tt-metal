#include "ttnn/operations/sampling/grid_sample_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <utility>

#include "pybind11/decorators.hpp"
#include <tt-metalium/core_coord.hpp>
#include "cpp/pybind11/json_class.hpp"
#include "ttnn/operations/sampling/grid_sample.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::grid_sample {
    void py_module(py::module& module) {
        bind_registered_operation(
            module,
            ttnn::grid_sample,
            R"doc( dcm)doc",
            ttnn::pybind_overload_t{
                [](decltype(ttnn::grid_sample)& self,
                   const ttnn::Tensor& input,
                   const ttnn::Tensor& coords,
                    const uint32_t in_channels,
                    const uint32_t in_height,
                    const uint32_t in_width,
                    const uint32_t out_height,
                    const uint32_t out_width,
                    const bool align_corners,
                   const std::optional<const MemoryConfig>& memory_config,
                   const std::optional<Tensor> optional_output_tensor
                ) -> ttnn::Tensor {
                    return self(
                        input,
                        coords,
                        in_channels,
                        in_height,
                        in_width,
                        out_height,
                        out_width,
                        align_corners,
                        memory_config,
                        optional_output_tensor
                    );
                },
                py::arg("input"),
                py::arg("coords"),
                py::arg("in_channels"),
                py::arg("in_height"),
                py::arg("in_width"),
                py::arg("out_height"),
                py::arg("out_width"),
                py::kw_only(),
                py::arg("align_corners") = false,
                py::arg("memory_config") = std::nullopt,
                py::arg("output_tensor") = std::nullopt,
            }
        );
    }
}
