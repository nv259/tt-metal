#include "ttnn/operations/sampling/device/sample_op.hpp"
#include "sample_program_factory.hpp"

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


namespace ttnn::operations::sample{
    void Sample::validate_with_output_tensors(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
        TT_FATAL(input_tensors.size() == 3, "Must have 3 input tensors");
        TT_FATAL(output_tensors.size() == 1, "Must have 1 output tensors");

        const auto& input_tensor = input_tensors[0];
        const auto& coords_tensor = input_tensors[1];
        const auto& mask_tensor = input_tensors[2];
        const auto& optional_output_tensor = output_tensors.at(0);

        TT_FATAL(input_tensor.get_dtype() == DataType::BFLOAT16, "Only BFLOAT16 is supported for inputs!");
        TT_FATAL(input_tensor.get_layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for inputs!");
        TT_FATAL(coords_tensor.get_dtype() == DataType::FLOAT32, "Only FLOAT32 is supported for grid because we can preserve accuracy!");
        TT_FATAL(coords_tensor.get_layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for grid!");
     
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
        const auto& coords_tensor = input_tensors[1];
        const auto& mask_tensor = input_tensors[2];

        auto coord_shape = coords_tensor.get_padded_shape();    //1, 1, BHoWo, 2*weight_shape*weight_shape
        TT_FATAL(coord_shape[0] == 1, "dim 0 must be 1");
        TT_FATAL(coord_shape[1] == 1, "dim 1 must be 1");

        auto input_shape = input_tensor.get_padded_shape(); //1, 1, Ci, BHiWi
        TT_FATAL(input_shape[0] == 1, "dim 0 must be 1");
        TT_FATAL(input_shape[1] == 1, "dim 1 must be 1");

        uint32_t n_coord = coord_shape[3];
        uint32_t weight_shape = (int)(std::sqrt(n_coord/2));

        // ttnn::SimpleShape output_shape({1, 1, input_shape[2]*weight_shape*weight_shape, coord_shape[2]});
        ttnn::SimpleShape output_shape({1, 1, input_shape[2], coord_shape[3]/2});
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
        const auto& mask = input_tensors.at(2);
        const auto& output_tensor = output_tensors.at(0);
        return sample_multi_core(input, coords, mask, in_height, in_width, out_height, out_width, output_tensor);
    }
}


