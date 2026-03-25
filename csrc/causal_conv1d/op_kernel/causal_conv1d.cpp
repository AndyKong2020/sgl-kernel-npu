#include "CustVec.h"

namespace npu_ops_transformer_ext {
namespace Mambav2CausalConv1d {

class CubeHandler {
public:
    __aicore__ inline CubeHandler() {}

    __aicore__ inline void Init(
        GM_ADDR xmtx, GM_ADDR wmtx, GM_ADDR bias, GM_ADDR outmtx, GM_ADDR workspace, int B, int D, int S, int W)
    {
    }

    __aicore__ inline void Compute() {}
};

class VecHandler {
public:
    __aicore__ inline VecHandler() {}

    __aicore__ inline void Init(
        GM_ADDR xmtx, GM_ADDR wmtx, GM_ADDR bias, GM_ADDR outmtx, GM_ADDR workspace, int B, int D, int S, int W)
    {
        tilingShapeCustVec(B, D, S, W, vec0shape);
        vec0.Init(xmtx, wmtx, bias, outmtx, vec0shape);
    }

    __aicore__ inline void Compute()
    {
        vec0.Compute();
    }

private:
    TPipe pipe;
    CustVec vec0;
    CustVecShapeInfo vec0shape;
};

}  // namespace Mambav2CausalConv1d
}  // namespace npu_ops_transformer_ext

extern "C" __global__ __aicore__ void causal_conv1d_fp32(
    GM_ADDR xmtx, GM_ADDR wmtx, GM_ADDR bias, GM_ADDR outmtx, GM_ADDR workspace, int B, int D, int S, int W)
{
    if ASCEND_IS_AIC {
        npu_ops_transformer_ext::Mambav2CausalConv1d::CubeHandler cube;
        cube.Init(xmtx, wmtx, bias, outmtx, workspace, B, D, S, W);
        cube.Compute();
        mad((__cc__ float *)0, (__ca__ float *)0, (__cb__ float *)0, 32, 32, 32, 0x0, false, false, true);
    }
    if ASCEND_IS_AIV {
        npu_ops_transformer_ext::Mambav2CausalConv1d::VecHandler vec;
        vec.Init(xmtx, wmtx, bias, outmtx, workspace, B, D, S, W);
        vec.Compute();
        copy_ubuf_to_ubuf((__ubuf__ float *)0, (__ubuf__ float *)0, 0, 1, 1, 0, 0);
    }
}
