/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#ifndef ASU_HBM_INDEX_MAINTAIN_TORCH_ADPT_H
#define ASU_HBM_INDEX_MAINTAIN_TORCH_ADPT_H

namespace vllm_ascend {

void asu_hbm_index_maintain(at::Tensor& index,
                            at::Tensor& slot_to_index,
                            at::Tensor& free_slots,
                            at::Tensor& free_head,
                            const at::Tensor& last_query_slots,
                            int64_t req_num,
                            int64_t seed)
{
    TORCH_CHECK(index.scalar_type() == at::kInt, "index must be int32");
    TORCH_CHECK(slot_to_index.scalar_type() == at::kInt, "slot_to_index must be int32");
    TORCH_CHECK(free_slots.scalar_type() == at::kInt, "free_slots must be int32");
    TORCH_CHECK(free_head.scalar_type() == at::kInt, "free_head must be int32");
    TORCH_CHECK(last_query_slots.scalar_type() == at::kInt, "last_query_slots must be int32");
    TORCH_CHECK(req_num > 0, "req_num must be greater than 0");

    EXEC_NPU_CMD(aclnnAsuHbmIndexMaintain,
                 index,
                 slot_to_index,
                 free_slots,
                 free_head,
                 last_query_slots,
                 req_num,
                 seed,
                 index,
                 slot_to_index,
                 free_slots,
                 free_head);
}

}  // namespace vllm_ascend

#endif  // ASU_HBM_INDEX_MAINTAIN_TORCH_ADPT_H
