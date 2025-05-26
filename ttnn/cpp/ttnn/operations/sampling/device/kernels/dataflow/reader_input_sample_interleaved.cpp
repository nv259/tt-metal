#include "dataflow_api.h"
#include "debug/dprint.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>

#define ALWI inline __attribute__((always_inline))

ALWI float uint32_to_float(uint32_t f) {
    float ret;
    std::memcpy(&ret, &f, sizeof(float));
    return ret;
}

ALWI float bfloat16_to_float(uint16_t bfp16_bits) {
    uint32_t float_bits = static_cast<uint32_t>(bfp16_bits) << 16;
    float result;
    std::memcpy(&result, &float_bits, sizeof(result));
    return result;
}

ALWI uint16_t float_to_bfloat16(float value) {
    uint32_t float_bits;
    std::memcpy(&float_bits, &value, sizeof(float));
    return static_cast<uint16_t>(float_bits >> 16);
}

uint16_t bilinear(volatile tt_l1_ptr uint16_t* inp, const uint32_t b, const uint32_t w, const uint32_t h, const uint32_t height, const uint32_t width){
    float x = uint32_to_float(w);
    float y = uint32_to_float(h);

    if (x <= -1 || y <= -1 || height <= y || width <= x) return 0.0f;

    int16_t x_high = std::ceil(x);
    int16_t x_low = x_high - 1;
    int16_t y_high = std::ceil(y);
    int16_t y_low = y_high - 1;
    DPRINT << x_high << " " << x_low << " " << y_high << " " << y_low << ENDL();

    float ly = y - (float)y_low;
    float lx = x - (float)x_low;

    float v1 = 0.0f;   // x_low, y_low
    if (y_low >= 0 && x_low >= 0) v1 = bfloat16_to_float(inp[b * height * width + y_low * width + x_low]);

    float v2 = 0.0f;   // x_low, y_high
    if (y_high >= 0 && y_high <= (int16_t)(height - 1) && x_low >= 0) v2 = bfloat16_to_float(inp[b * height * width + y_high * width + x_low]);

    float v3 = 0.0f;   // x_high, y_low
    if (x_high >= 0 && x_high <= (int16_t)(width - 1) && y_low >= 0) v3 = bfloat16_to_float(inp[b * height * width + y_low * width + x_high]);

    float v4 = 0.0f;   // x_high, y_high
    if (x_high >= 0 && x_high <= (int16_t)(width - 1) && y_high >= 0 && y_high <= (int16_t)(height - 1)) v4 = bfloat16_to_float(inp[b * height * width + y_high * width + x_high]);

    float w1 = (1 - ly) * (1 - lx), w2 = (1 - lx) * ly, w3 = lx * (1 - ly), w4 = lx * ly;
    float result = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
    return float_to_bfloat16(result);
};


void kernel_main() {
    // matrix A (src0) has shape of BHiWi, C => should be WIDTH_SHARDED or INTERLEAVED with row stick (BHiWi, 1)
    // matrix B (src1) has shape of BHoWo, 2K^2 => should be HEIGHT_SHARDED or INTERLEAVED with col stick (1, 2K^2).
    // tensor layout of matrix A should be BC, HiWi
    // dst has shape of 1 core: BHoWo, BC//cores * K^2
    // In this file, we do INTERLEAVED reading
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t dst_addr = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);
    uint32_t num_sticks = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(2);
    constexpr bool src0_is_dram = (bool)get_compile_time_arg_val(3);
    constexpr bool src1_is_dram = (bool)get_compile_time_arg_val(4);
    constexpr bool dst_is_dram = (bool)get_compile_time_arg_val(5);
    constexpr uint32_t src0_stick_nbytes = get_compile_time_arg_val(6); //per core
    constexpr uint32_t src1_stick_nbytes = get_compile_time_arg_val(7); //all sticks at once
    constexpr uint32_t out_stick_nbytes = get_compile_time_arg_val(8);
    constexpr uint32_t output_unit_size = get_compile_time_arg_val(9);
    constexpr uint32_t in_height = get_compile_time_arg_val(10);
    constexpr uint32_t in_width = get_compile_time_arg_val(11);
    constexpr uint32_t out_height = get_compile_time_arg_val(12);
    constexpr uint32_t out_width = get_compile_time_arg_val(13);
    constexpr uint32_t stride_h = get_compile_time_arg_val(14);
    constexpr uint32_t stride_w = get_compile_time_arg_val(15);
    constexpr uint32_t pad_h = get_compile_time_arg_val(16);
    constexpr uint32_t pad_w = get_compile_time_arg_val(17);
    constexpr uint32_t weight_shape = get_compile_time_arg_val(18);


    const InterleavedAddrGen<src0_is_dram> s0 = {.bank_base_address = src0_addr, .page_size = src0_stick_nbytes};

    const InterleavedAddrGen<src1_is_dram> s1 = {.bank_base_address = src1_addr, .page_size = src1_stick_nbytes};
    uint64_t src1_noc_addr = get_noc_addr(0, s1);

    const InterleavedAddrGen<dst_is_dram> s_out = {.bank_base_address = dst_addr, .page_size = out_stick_nbytes};

    uint32_t cb0_addr = get_write_ptr(cb_id_in0);
    volatile tt_l1_ptr uint16_t* src0_stick = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb0_addr);

    uint32_t cb1_addr = get_write_ptr(cb_id_in1);
    volatile tt_l1_ptr uint32_t* src1_stick = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb1_addr);

    uint32_t cb_out_addr = get_write_ptr(cb_id_out);
    volatile tt_l1_ptr uint16_t* out_stick = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_out_addr);

    noc_async_read(src1_noc_addr, cb1_addr, src1_stick_nbytes);

    for (uint32_t num_stick = 0; num_stick < num_sticks; num_stick++){
        uint64_t src0_noc_addr = get_noc_addr(start_id, s0);
        noc_async_read(src0_noc_addr, cb0_addr, src0_stick_nbytes);
        noc_async_read_barrier();

        uint32_t b, h, w, x, y;
        for (uint32_t idx = 0; idx < out_stick_nbytes/output_unit_size; idx++){
            b = idx/(out_height*out_width);
            h = (idx/(out_width))%out_height;
            w = idx%out_width;
            x = src1_stick[b*out_height*out_width*2 + h*out_width*2 + w*2];
            y = src1_stick[b*out_height*out_width*2 + h*out_width*2 + w*2+1];
            out_stick[idx] = bilinear(src0_stick, b, x, y, in_height, in_width);
        }


        uint64_t dst_noc_addr = get_noc_addr(start_id, s_out);
        noc_async_write(cb_out_addr, dst_noc_addr, out_stick_nbytes);
        noc_async_write_barrier();
        start_id++;
    }
}
