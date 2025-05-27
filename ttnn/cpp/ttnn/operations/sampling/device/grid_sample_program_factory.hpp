#include "ttnn/run_operation.hpp"

namespace ttnn::operations::grid_sample {

using namespace tt::constants;

operation::ProgramWithCallbacks sample_multi_core(
    const Tensor& input,
    const Tensor& coords,
    const uint32_t in_channels,
    const uint32_t in_height,
    const uint32_t in_width,
    const uint32_t out_height,
    const uint32_t out_width,
    const bool align_corners,
    const Tensor& output);
}
