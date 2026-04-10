#ifndef HEADER_ACLRTLAUNCH_CAUSAL_CONV1D_FLA_UPDATE_HALF_H
#define HEADER_ACLRTLAUNCH_CAUSAL_CONV1D_FLA_UPDATE_HALF_H

#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_causal_conv1d_fla_update_half(
    uint32_t numBlocks, aclrtStream stream, void *x, void *weight, void *bias, void *convStates,
    void *queryStartLoc, void *cacheIndices, void *initialStateMode, void *numAcceptedTokens, void *y,
    void *workspace, void *tiling);

#endif
