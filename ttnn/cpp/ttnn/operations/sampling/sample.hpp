#pragma once

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/command_queue.hpp>
#include "ttnn/operations/data_movement/bcast/bcast.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
// #include "ttnn/operations/sampling/device/sample_op.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::sample {
    struct SampleOperation {
        static Tensor invoke(
            uint8_t queue_id,
            const Tensor& input,
            const Tensor& coords,
            const Tensor& mask,
            const uint32_t in_height,
            const uint32_t in_width,
            const uint32_t out_height,
            const uint32_t out_width,
            std::array<uint32_t, 2> stride = {1, 1},
            std::array<uint32_t, 2> padding = {0, 0},
            std::array<uint32_t, 2> dilation = {1, 1},
            const std::optional<MemoryConfig>& memory_config = std::nullopt,
            const std::optional<Tensor> optional_output_tensor = std::nullopt);

        static Tensor invoke(
            const Tensor& input,
            const Tensor& coords,
            const Tensor& mask,
            const uint32_t in_height,
            const uint32_t in_width,
            const uint32_t out_height,
            const uint32_t out_width,
            std::array<uint32_t, 2> stride = {1, 1},
            std::array<uint32_t, 2> padding = {0, 0},
            std::array<uint32_t, 2> dilation = {1, 1},
            const std::optional<const MemoryConfig>& memory_config = std::nullopt,
            const std::optional<Tensor> optional_output_tensor = std::nullopt
        );
    };
}

namespace ttnn{
    constexpr auto sample = ttnn::register_operation<"ttnn::sample", operations::sample::SampleOperation>();
}