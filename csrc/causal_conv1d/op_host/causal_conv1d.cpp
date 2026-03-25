#include "defines.h"
#include "torch_helper.h"

#include "aclrtlaunch_causal_conv1d_prefill_fp32.h"
#include "causal_conv1d_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace sglang {
namespace npu_kernel {

namespace {
constexpr int32_t kBlockDims = 20;
constexpr int64_t kUserWorkspaceSize = 1024;
constexpr int64_t kSupportedKernelWidth = 4;
constexpr uint32_t kTilingPaddingBytes = 32U;

at::Tensor make_tiling_tensor(int32_t batch, int32_t dim, int32_t seqlen, int32_t width)
{
    int32_t tiling_size =
        (sizeof(CausalConv1dTilingData) + kTilingPaddingBytes - 1) / kTilingPaddingBytes * kTilingPaddingBytes;
    auto tiling_buffer = at::zeros({tiling_size}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    auto *tiling_data = reinterpret_cast<CausalConv1dTilingData *>(tiling_buffer.data_ptr());
    tiling_data->batch = batch;
    tiling_data->dim = dim;
    tiling_data->seqlen = seqlen;
    tiling_data->width = width;
    return TorchNpuHelper::CopyTensorHostToDevice(tiling_buffer);
}
}

HOST_API at::Tensor causal_conv1d(const at::Tensor &x, const at::Tensor &weight, const at::Tensor &bias)
{
    TORCH_CHECK(x.dim() == 3, "x must be a 3D tensor [B, D, S], got shape ", x.sizes());
    TORCH_CHECK(weight.dim() == 3, "weight must be a 3D tensor [D, 1, W], got shape ", weight.sizes());
    TORCH_CHECK(weight.size(1) == 1, "weight second dimension must be 1, got ", weight.size(1));
    TORCH_CHECK(weight.size(2) == kSupportedKernelWidth, "only kernel width 4 is supported, got ", weight.size(2));
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor [D], got shape ", bias.sizes());
    TORCH_CHECK(x.size(1) == weight.size(0), "x dim and weight dim mismatch: ", x.size(1), " vs ", weight.size(0));
    TORCH_CHECK(bias.size(0) == weight.size(0), "bias dim and weight dim mismatch: ", bias.size(0), " vs ", weight.size(0));
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    auto x_fp32 = x.to(at::kFloat);
    auto weight_fp32 = weight.to(at::kFloat);
    auto bias_fp16 = bias.to(at::kHalf);

    auto output = at::empty(x_fp32.sizes(), x_fp32.options());

    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t workspace_size = kUserWorkspaceSize + static_cast<int64_t>(ascendc_platform->GetLibApiWorkSpaceSize());
    auto workspace_tensor =
        at::empty({workspace_size}, at::TensorOptions().dtype(at::kByte).device(x.options().device()));

    int32_t batch = static_cast<int32_t>(x.size(0));
    int32_t dim = static_cast<int32_t>(x.size(1));
    int32_t seqlen = static_cast<int32_t>(x.size(2));
    int32_t width = static_cast<int32_t>(weight.size(2));
    auto tiling_tensor = make_tiling_tensor(batch, dim, seqlen, width);

    EXEC_KERNEL_CMD(causal_conv1d_prefill_fp32, kBlockDims, x_fp32, weight_fp32, bias_fp16, output, workspace_tensor, tiling_tensor);
    return output;
}

}  // namespace npu_kernel
}  // namespace sglang
