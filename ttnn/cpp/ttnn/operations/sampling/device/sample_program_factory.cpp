#include <algorithm>
#include <cmath>

#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/permute/device/permute_device_operation.hpp"

using namespace tt::tt_metal;

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
        const Tensor& output){

            tt::tt_metal::Program program{};
            tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.get_dtype());
            uint32_t input_unit_size = input.element_size();
            tt::DataFormat grid_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(coords.get_dtype());
            uint32_t grid_unit_size = coords.element_size();

            tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
            uint32_t output_unit_size = output.element_size();

            tt::tt_metal::IDevice* device = output.device();

            auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
            uint32_t num_cores_x = compute_with_storage_grid_size.x;
            uint32_t num_cores_y = compute_with_storage_grid_size.y;
            auto core_grid = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});
            uint32_t num_units = num_cores_x * num_cores_y;

            //all_cores: corerange
            //core_group: similar to all_cores but divided to groups
            //num_units_per_core_group_1
            auto [num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] =
                tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_units);

            const auto& input_shape = input.get_padded_shape();

            const auto& grid_shape = coords.get_padded_shape();
            const uint32_t num_coords = grid_shape[3];

            const auto& output_shape = output.get_padded_shape();

            const uint32_t weight_shape = std::sqrt(num_coords/2);
            const uint32_t stride_h = 1, stride_w = 1;
            const uint32_t pad_h = 1, pad_w = 1;

            uint32_t src0_cb_index = tt::CBIndex::c_0;
            uint32_t num_src0_units = input_shape[3];   // B*Hi*Wi
            uint32_t aligned_src0_unit_size = num_src0_units * input_unit_size;
            // total size and page size of a circular buffer is set to the same
            tt::tt_metal::CircularBufferConfig cb_src0_config =
                tt::tt_metal::CircularBufferConfig(aligned_src0_unit_size, {{src0_cb_index, input_cb_data_format}})
                    .set_page_size(src0_cb_index, aligned_src0_unit_size);
            auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

            uint32_t src1_cb_index = tt::CBIndex::c_1;
            uint32_t num_src1_units = coords.volume();  // BHoWo*2
            uint32_t aligned_src1_unit_size = num_src1_units * grid_unit_size;
            tt::tt_metal::CircularBufferConfig cb_src1_config =
                tt::tt_metal::CircularBufferConfig(aligned_src1_unit_size, {{src1_cb_index, grid_cb_data_format}})
                    .set_page_size(src1_cb_index, aligned_src1_unit_size);
            auto cb_src1 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

            uint32_t out_cb_index = tt::CBIndex::c_16;
            uint32_t num_out_units = output_shape[3];  // BHoWo
            uint32_t aligned_out_unit_size = num_out_units * output_unit_size;
            tt::tt_metal::CircularBufferConfig cb_out_config =
                tt::tt_metal::CircularBufferConfig(aligned_out_unit_size, {{out_cb_index, output_cb_data_format}})
                    .set_page_size(out_cb_index, aligned_out_unit_size);
            auto cb_out = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);

            auto src0_buffer = input.buffer();
            auto src1_buffer = coords.buffer();
            auto dst_buffer = output.buffer();
            bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
            bool src1_is_dram = src1_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
            bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

            std::vector<uint32_t> reader_compile_time_args = {
                src0_cb_index,
                src1_cb_index,
                out_cb_index,
                src0_is_dram,
                src1_is_dram,
                dst_is_dram,
                aligned_src0_unit_size,
                aligned_src1_unit_size,
                aligned_out_unit_size,
                output_unit_size,
                in_height,
                in_width,
                out_height,
                out_width,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                weight_shape,
            };

            tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/sampling/device/kernels/dataflow/reader_input_sample_interleaved.cpp",
                all_cores,
                tt::tt_metal::DataMovementConfig{
                    .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt::tt_metal::NOC::RISCV_1_default,
                    .compile_args = reader_compile_time_args});

            auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, false);
            uint32_t num_sticks = input_shape[2];
            uint32_t num_sticks_per_core = num_sticks/num_cores;
            uint32_t num_sticks_per_core_remainder = num_sticks%num_cores;
            uint32_t start_id = 0;
            for (uint32_t i = 0; i < cores.size(); ++i) {
                const CoreCoord& core = cores.at(i);
                uint32_t n_sticks_per_core = num_sticks_per_core;
                if (i < num_sticks_per_core_remainder) {
                    n_sticks_per_core++;
                }
                tt::tt_metal::SetRuntimeArgs(
                    program, reader_kernel_id, core, 
                    {src0_buffer->address(), src1_buffer->address(), dst_buffer->address(), start_id, n_sticks_per_core});
                start_id += n_sticks_per_core;
            }

            auto override_runtime_args_callback = [reader_kernel_id, cores, num_sticks, num_sticks_per_core, num_sticks_per_core_remainder](
                                                    const Program& program,
                                                    const std::vector<Buffer*>& input_buffers,
                                                    const std::vector<Buffer*>& output_buffers) {
                auto src0_buffer = input_buffers.at(0);
                auto src1_buffer = input_buffers.at(1);
                auto dst_buffer = output_buffers.at(0);
                uint32_t start_id = 0;
                for (uint32_t i = 0; i < cores.size(); i++){
                    auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id, cores[i]);
                    reader_runtime_args[0] = src0_buffer->address();
                    reader_runtime_args[1] = src1_buffer->address();
                    reader_runtime_args[2] = dst_buffer->address();
                    uint32_t n_sticks_per_core = num_sticks_per_core;
                    if (i < num_sticks_per_core_remainder) {
                        n_sticks_per_core++;
                    }
                    reader_runtime_args[3] = start_id;
                    reader_runtime_args[4] = n_sticks_per_core;
                    start_id += n_sticks_per_core;
                }
            };

            return {std::move(program), override_runtime_args_callback};
        }
}
