#include <torch/library.h>

#include "aclnn_torch_adapter/op_api_common.h"
#include "asu_hbm_index_lookup/asu_hbm_index_lookup_torch_adpt.h"
#include "asu_hbm_index_maintain/asu_hbm_index_maintain_torch_adpt.h"

TORCH_LIBRARY_FRAGMENT(_C_ascend, ops)
{
    ops.def("asu_hbm_index_lookup(Tensor(a!) index, "
            "Tensor(b!) slot_to_index, "
            "Tensor(c!) free_slots, "
            "Tensor(d!) free_head, "
            "Tensor query_index, "
            "int req_num) -> Tensor");
    ops.impl("asu_hbm_index_lookup", torch::kPrivateUse1, &vllm_ascend::asu_hbm_index_lookup);

    ops.def("asu_hbm_index_maintain(Tensor(a!) index, "
            "Tensor(b!) slot_to_index, "
            "Tensor(c!) free_slots, "
            "Tensor(d!) free_head, "
            "Tensor last_query_slots, "
            "int req_num, "
            "int seed) -> ()");
    ops.impl("asu_hbm_index_maintain", torch::kPrivateUse1, &vllm_ascend::asu_hbm_index_maintain);
}
