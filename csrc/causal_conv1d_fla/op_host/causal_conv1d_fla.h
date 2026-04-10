/*!
 * \file causal_conv1d_fla.h
 * \brief causal_conv1d_fla host-side function declarations
 *        FLA-optimized CausalConv1D supporting both prefill and decode modes.
 */

#ifndef CAUSAL_CONV1D_FLA_HOST_H_
#define CAUSAL_CONV1D_FLA_HOST_H_

#include <ATen/ATen.h>

#include "defines.h"

namespace sglang {
namespace npu_kernel {

HOST_API at::Tensor causal_conv1d_fla_prefill_impl(const at::Tensor &x, const at::Tensor &weight,
                                                   const at::Tensor &conv_states, const at::Tensor &query_start_loc,
                                                   const at::Tensor &cache_indices,
                                                   const at::Tensor &initial_state_mode, const at::Tensor &bias,
                                                   bool activation_mode, int64_t pad_slot_id);

HOST_API at::Tensor causal_conv1d_fla_update_impl(const at::Tensor &x, const at::Tensor &weight,
                                                  const at::Tensor &conv_state, const at::Tensor &conv_state_indices,
                                                  const at::Tensor &bias, const at::Tensor &num_accepted_tokens,
                                                  const at::Tensor &query_start_loc, bool activation_mode,
                                                  int64_t pad_slot_id);

}  // namespace npu_kernel
}  // namespace sglang

#endif  // CAUSAL_CONV1D_FLA_HOST_H_
