#include "register/op_def_registry.h"

namespace ops {

class AsuResolveKvSlots : public OpDef {
public:
    explicit AsuResolveKvSlots(const char* name) : OpDef(name)
    {
        this->Input("original_topk_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_len")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("managed_prefix_len")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("token_state")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("asu_record_addr")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("hbm_slot_of_token")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("slot_owner_token")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("free_slot_stack")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("free_slot_count")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("original_block_table")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("kv_cache0")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_UINT8})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("kv_cache1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_UINT8})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("asu_kv_cache0")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_UINT8})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("asu_kv_cache1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_UINT8})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("resolved_kv_slots")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});

        this->Attr("block_size").AttrType(OPTIONAL).Int(128);
        this->Attr("kv0_bytes_per_slot").Int();
        this->Attr("kv1_bytes_per_slot").Int();

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");

        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);
    }
};

OP_ADD(AsuResolveKvSlots);

}  // namespace ops
