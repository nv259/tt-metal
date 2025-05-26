#include "ttnn/run_operation.hpp"

namespace ttnn::operations::sample {

using namespace tt::constants;

operation::ProgramWithCallbacks sample_multi_core(
    const Tensor& input,
    const Tensor& coords,
    const Tensor& mask,
    const uint32_t in_height,
    const uint32_t in_width,
    const uint32_t out_height,
    const uint32_t out_width,
    const Tensor& output);
}
