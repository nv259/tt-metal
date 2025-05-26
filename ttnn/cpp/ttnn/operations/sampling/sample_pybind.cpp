#include "ttnn/operations/sampling/sample_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <utility>

#include "pybind11/decorators.hpp"
#include <tt-metalium/core_coord.hpp>
#include "cpp/pybind11/json_class.hpp"
#include "ttnn/operations/sampling/sample.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::sample {
    void py_module(py::module& module) {
        bind_registered_operation(
            module,
            ttnn::sample,
            R"doc( dcm)doc",
            ttnn::pybind_overload_t{
                [](decltype(ttnn::sample)& self,
                   const ttnn::Tensor& input,
                   const ttnn::Tensor& coords,
                   const ttnn::Tensor& mask,
                    const uint32_t in_height,
                    const uint32_t in_width,
                    const uint32_t out_height,
                    const uint32_t out_width,
                   std::array<uint32_t, 2> stride,
                   std::array<uint32_t, 2> padding,
                   std::array<uint32_t, 2> dilation,
                   const std::optional<const MemoryConfig>& memory_config,
                   const std::optional<Tensor> optional_output_tensor
                ) -> ttnn::Tensor {
                    return self(
                        input,
                        coords,
                        mask,
                        in_height,
                        in_width,
                        out_height,
                        out_width,
                        stride,
                        padding,
                        dilation,
                        memory_config,
                        optional_output_tensor
                    );
                },
                py::arg("input"),
                py::arg("coords"),
                py::arg("mask"),
                py::arg("in_height"),
                py::arg("in_width"),
                py::arg("out_height"),
                py::arg("out_width"),
                py::kw_only(),
                py::arg("stride") = std::array<uint32_t, 2>{1, 1}, // Fixed default value
                py::arg("padding") = std::array<uint32_t, 2>{0, 0}, // Fixed default value
                py::arg("dilation") = std::array<uint32_t, 2>{1, 1}, // Fixed default value
                py::arg("memory_config") = std::nullopt,
                py::arg("output_tensor") = std::nullopt,
            }
        );
    }
}