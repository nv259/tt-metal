#include "sample.hpp"
#include "device/sample_op.hpp"
#include <cmath>

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::sample {
    Tensor SampleOperation::invoke(
        uint8_t queue_id,
        const Tensor& input,
        const Tensor& coords,
        const Tensor& mask,
        const uint32_t in_height,
        const uint32_t in_width,
        const uint32_t out_height,
        const uint32_t out_width,
        std::array<uint32_t, 2> stride,
        std::array<uint32_t, 2> padding,
        std::array<uint32_t, 2> dilation,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<Tensor> optional_output_tensor
    ) {
        return operation::run(
                Sample{
                    tt::tt_metal::DataType::BFLOAT16,
                    memory_config.value_or(input.memory_config()),
                    in_height,
                    in_width,
                    out_height,
                    out_width,
                    stride,
                    padding,
                    dilation
                },
                {input, coords, mask},
                {},
                {optional_output_tensor},
                queue_id)
            .at(0);
    }

    Tensor SampleOperation::invoke(
        const Tensor& input,
        const Tensor& coords,
        const Tensor& mask,
        const uint32_t in_height,
        const uint32_t in_width,
        const uint32_t out_height,
        const uint32_t out_width,
        std::array<uint32_t, 2> stride,
        std::array<uint32_t, 2> padding,
        std::array<uint32_t, 2> dilation,
        const std::optional<const MemoryConfig>& memory_config,
        const std::optional<Tensor> optional_output_tensor) {
        return invoke(0, input, coords, mask, in_height, in_width, out_height, out_width,
                        stride, padding, dilation, memory_config, std::move(optional_output_tensor));
    }
};