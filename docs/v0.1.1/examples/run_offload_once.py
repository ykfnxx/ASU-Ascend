"""Offline single-request runner for the v0.1.1 KV offload path.

Launches the vLLM engine in-process (no HTTP server), sends one fixed prompt, and
prints the generated text. Exercises the same model runner / SFA forward / offload
hook as ``vllm serve``, so it is the simplest way to trigger the offload path with a
fixed input during bring-up.

This script is self-contained: it makes the ``microkv`` client importable and sets
sensible defaults for the offload env vars *before* importing vllm (they must be in
place before the engine builds the model runner). Anything already exported in the
shell wins, so you can still switch modes without editing the file.

You still need the MicroKV server running:
    cd MicroKV && make && ./build/kv_stored --socket /tmp/microkv.sock

Env overrides (all optional):
    MICROKV_SOCKET                          default /tmp/microkv.sock
    MICROKV_PYTHON_PATH                     path added to sys.path for `import microkv`
    VLLM_ASCEND_KV_OFFLOAD_V0_COMPACT_SFA   set to 1 for the compact path
    VLLM_ASCEND_KV_OFFLOAD_V0_MAX_PINNED_REQS
    VLLM_ASCEND_KV_OFFLOAD_V0_REF_HBM_OPS   set to 1 to use the pure-Python ref ops
    OFFLOAD_PROMPT / OFFLOAD_MAX_TOKENS     override the fixed input

Usage:
    python run_offload_once.py [MODEL_PATH]      # default weights/tiny-random-glm-moe-dsa
"""

import os
import sys

# --- Configure the environment BEFORE importing vllm ------------------------------
# PYTHONPATH cannot be injected via os.environ after the interpreter has started, so
# make the microkv client importable by editing sys.path directly.
_DEFAULT_MICROKV_PYTHON = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "MicroKV", "python")
)
_MICROKV_PYTHON = os.environ.get("MICROKV_PYTHON_PATH", _DEFAULT_MICROKV_PYTHON)
if os.path.isdir(_MICROKV_PYTHON) and _MICROKV_PYTHON not in sys.path:
    sys.path.insert(0, _MICROKV_PYTHON)

# These are read lazily when the offload manager is constructed inside LLM(), so
# setting them here (before that call) is sufficient. setdefault lets shell exports win.
os.environ.setdefault("MICROKV_SOCKET", "/tmp/microkv.sock")
# Engage an offload path. Default to the validate path (tolerates out-of-range topk);
# the caller can instead export VLLM_ASCEND_KV_OFFLOAD_V0_COMPACT_SFA=1.
if os.environ.get("VLLM_ASCEND_KV_OFFLOAD_V0_COMPACT_SFA") != "1":
    os.environ.setdefault("VLLM_ASCEND_KV_OFFLOAD_V0_VALIDATE", "1")

from vllm import LLM, SamplingParams  # noqa: E402  (must follow the env setup above)

DEFAULT_MODEL = "weights/tiny-random-glm-moe-dsa"
DEFAULT_PROMPT = "你好，请介绍一下自己。"

_OFFLOAD_ENV_KEYS = (
    "MICROKV_SOCKET",
    "VLLM_ASCEND_KV_OFFLOAD_V0_VALIDATE",
    "VLLM_ASCEND_KV_OFFLOAD_V0_COMPACT_SFA",
    "VLLM_ASCEND_KV_OFFLOAD_V0_MAX_PINNED_REQS",
    "VLLM_ASCEND_KV_OFFLOAD_V0_REF_HBM_OPS",
)


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    prompt = os.environ.get("OFFLOAD_PROMPT", DEFAULT_PROMPT)
    max_tokens = int(os.environ.get("OFFLOAD_MAX_TOKENS", "32"))

    print("=== offload env ===")
    print(f"microkv python path: {_MICROKV_PYTHON} (on sys.path: {_MICROKV_PYTHON in sys.path})")
    for key in _OFFLOAD_ENV_KEYS:
        print(f"{key}={os.environ.get(key, '<unset>')}")

    # enforce_eager is required by the offload path; max_num_seqs=1 keeps a single
    # request so lookup/maintain stay on the currently supported single-request path.
    llm = LLM(
        model=model,
        enforce_eager=True,
        max_num_seqs=1,
        gpu_memory_utilization=0.9,
    )

    # max_tokens > 0 produces decode steps, which is what triggers the offload
    # decode lookup (prefill only populates MicroKV / the resident window).
    outputs = llm.generate([prompt], SamplingParams(max_tokens=max_tokens))

    completion = outputs[0].outputs[0]
    print("=== prompt ===")
    print(prompt)
    print("=== output ===")
    print(completion.text)


if __name__ == "__main__":
    main()
