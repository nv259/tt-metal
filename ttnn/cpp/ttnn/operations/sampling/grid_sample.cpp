#include "grid_sample.hpp"
#include "device/grid_sample_op.hpp"
#include <cmath>

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::grid_sample {
    Tensor SampleOperation::invoke(
        uint8_t queue_id,
        const Tensor& input,
        const Tensor& grid,
        const uint32_t in_channels,
        const uint32_t in_height,
        const uint32_t in_width,
        const uint32_t out_height,
        const uint32_t out_width,
        const bool align_corners,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<Tensor> optional_output_tensor
    ) {
        return operation::run(
                Sample{
                    tt::tt_metal::DataType::BFLOAT16,
                    memory_config.value_or(input.memory_config()),
                    in_channels,
                    in_height,
                    in_width,
                    out_height,
                    out_width,
                    align_corners
                },
                {input, grid},
                {},
                {optional_output_tensor},
                queue_id)
            .at(0);
    }

    Tensor SampleOperation::invoke(
        const Tensor& input,
        const Tensor& grid,
        const uint32_t in_channels,
        const uint32_t in_height,
        const uint32_t in_width,
        const uint32_t out_height,
        const uint32_t out_width,
        const bool align_corners,
        const std::optional<const MemoryConfig>& memory_config,
        const std::optional<Tensor> optional_output_tensor) {
        return invoke(0, input, grid,
                        in_channels, in_height, in_width, out_height, out_width, align_corners,
                        memory_config, std::move(optional_output_tensor));
    }
};
