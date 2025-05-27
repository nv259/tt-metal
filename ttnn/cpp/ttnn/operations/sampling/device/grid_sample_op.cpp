#include "ttnn/operations/sampling/device/grid_sample_op.hpp"
#include "grid_sample_program_factory.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/run_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/tensor/tensor.hpp"


using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;


namespace ttnn::operations::grid_sample{
    void Sample::validate_with_output_tensors(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
        TT_FATAL(input_tensors.size() == 2, "Must have 2 input tensors");
        TT_FATAL(output_tensors.size() == 1, "Must have 1 output tensors");

        const auto& input_tensor = input_tensors[0];
        const auto& grid_tensor = input_tensors[1];
        const auto& mask_tensor = input_tensors[2];
        const auto& optional_output_tensor = output_tensors.at(0);

        TT_FATAL(input_tensor.get_dtype() == DataType::BFLOAT16, "Only BFLOAT16 is supported for inputs!");
        TT_FATAL(input_tensor.get_layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for inputs!");
        TT_FATAL(grid_tensor.get_dtype() == DataType::BFLOAT16, "Only BFLOAT16 is supported! Do not worry about data loss, we will cast it back to float32 later.");
        TT_FATAL(grid_tensor.get_layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for grid!");

        if (optional_output_tensor.has_value()) {
            TT_FATAL(
                optional_output_tensor.value().get_dtype() == DataType::BFLOAT16,
                "Only BFLOAT16 is supported for outputs!");
            TT_FATAL(
                optional_output_tensor.value().memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
                "Only INTERLEAVED memory layout is supported for outputs!");
        }
    }

    std::vector<TensorSpec> Sample::compute_output_specs(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
        if (output_tensors.at(0).has_value()) {
            return {output_tensors.at(0)->get_tensor_spec()};
        }

        const auto& input_tensor = input_tensors[0];
        const auto& grid_tensor = input_tensors[1];

        auto grid_shape = grid_tensor.get_padded_shape();    //1, 1, BHoWo, 2
        TT_FATAL(grid_shape[0] == 1, "dim 0 must be 1. In general, shape format for the grid should be (1, 1, B*H*W, 2)");
        TT_FATAL(grid_shape[1] == 1, "dim 1 must be 1. In general, shape format for the grid should be (1, 1, B*H*W, 2)");
        TT_FATAL(grid_shape[3] == 2, "The last dimension should contain only pixel coordinations (x, y)");

        auto input_shape = input_tensor.get_padded_shape(); //1, 1, BHiWi, Ci
        TT_FATAL(input_shape[0] == 1, "dim 0 must be 1. In general, shape format for the input should be (1, 1, B*H*W, C)");
        TT_FATAL(input_shape[1] == 1, "dim 1 must be 1. In general, shape format for the input should be (1, 1, B*H*W, C)");

        ttnn::SimpleShape output_shape({1, 1, grid_shape[2], in_channels});
        // std::cout << "output shape: " << output_shape[0] << " " << output_shape[1] << " " << output_shape[2] << " " << output_shape[3] << std::endl;
        return {TensorSpec(output_shape, TensorLayout(output_dtype, PageConfig(input_tensor.get_layout()), output_mem_config))};
    }

    std::vector<Tensor> Sample::create_output_tensors(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
        if (output_tensors.at(0).has_value()) {
            return {output_tensors.at(0).value()};
        }

        return {create_device_tensor(compute_output_specs(input_tensors, output_tensors)[0], input_tensors[0].device())};
    }

    operation::ProgramWithCallbacks Sample::create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
        const auto& input = input_tensors.at(0);
        const auto& coords = input_tensors.at(1);
        const auto& output_tensor = output_tensors.at(0);
        return sample_multi_core(input, coords,
            in_channels, in_height, in_width, out_height, out_width, align_corners,
            output_tensor);
    }
}
