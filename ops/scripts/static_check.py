#!/usr/bin/env python3
"""Static checks for the ASU custom-op source layout."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_SNIPPETS = {
    "asu_resolve_kv_slots/op_host/CMakeLists.txt": [
        "OP_NAME AsuResolveKvSlots",
        "target_sources(op_host_aclnn PRIVATE",
        "target_sources(optiling PRIVATE",
        "target_sources(opsproto PRIVATE",
    ],
    "asu_resolve_kv_slots/op_host/asu_resolve_kv_slots_def.cpp": [
        "class AsuResolveKvSlots : public OpDef",
        'this->Input("original_topk_indices")',
        'this->Output("resolved_kv_slots")',
        'this->Attr("block_size")',
    ],
    "asu_resolve_kv_slots/op_host/asu_resolve_kv_slots_proto.cpp": [
        "IMPL_OP_INFERSHAPE(AsuResolveKvSlots)",
        "InferShapeAsuResolveKvSlots",
        "InferDataTypeAsuResolveKvSlots",
    ],
    "asu_resolve_kv_slots/op_host/asu_resolve_kv_slots_tiling.h": [
        "BEGIN_TILING_DATA_DEF(AsuResolveKvSlotsTilingData)",
        "REGISTER_TILING_DATA_CLASS(AsuResolveKvSlots",
    ],
    "asu_resolve_kv_slots/op_host/asu_resolve_kv_slots_tiling.cpp": [
        "IMPL_OP_OPTILING(AsuResolveKvSlots)",
        "TilingFunc",
        "SetBlockDim(1)",
    ],
    "asu_resolve_kv_slots/op_kernel/asu_resolve_kv_slots.cpp": [
        'extern "C" __global__ __aicore__ void asu_resolve_kv_slots',
        "GET_TILING_DATA",
        "resolvedSlotsGm_",
        "CopyBytes",
    ],
}


def main() -> int:
    missing: list[str] = []
    bad_snippets: list[str] = []

    for rel_path, snippets in REQUIRED_SNIPPETS.items():
        path = ROOT / rel_path
        if not path.is_file():
            missing.append(rel_path)
            continue
        text = path.read_text(encoding="utf-8")
        for snippet in snippets:
            if snippet not in text:
                bad_snippets.append(f"{rel_path}: missing {snippet!r}")

    if missing or bad_snippets:
        if missing:
            print("Missing files:")
            for rel_path in missing:
                print(f"  - {rel_path}")
        if bad_snippets:
            print("Missing snippets:")
            for item in bad_snippets:
                print(f"  - {item}")
        return 1

    print("ASU ops static layout check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
