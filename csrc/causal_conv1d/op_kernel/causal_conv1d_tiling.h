#pragma once

#include <cstdint>

namespace sglang {
namespace npu_kernel {

struct CausalConv1dTilingData {
    int32_t batch;
    int32_t dim;
    int32_t seqlen;
    int32_t width;
};

}  // namespace npu_kernel
}  // namespace sglang
