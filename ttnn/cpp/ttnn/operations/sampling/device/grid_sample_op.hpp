#pragma once
#include <optional>

#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/types.hpp"

#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/global_cb_utils.hpp"


namespace ttnn::operations::grid_sample {
    struct Sample{
        const DataType output_dtype;
        const MemoryConfig output_mem_config;
        const uint32_t in_channels;
        const uint32_t in_height;
        const uint32_t in_width;
        const uint32_t out_height;
        const uint32_t out_width;
        const bool align_corners;

        void validate_with_output_tensors(
            const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
        std::vector<TensorSpec> compute_output_specs(
            const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
        std::vector<Tensor> create_output_tensors(
            const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
        operation::ProgramWithCallbacks create_program(
            const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
        };
}
