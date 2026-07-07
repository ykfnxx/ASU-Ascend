"""Offline single-request runner for the v0.1.1 KV offload path.

Launches the vLLM engine in-process (no HTTP server), sends one fixed prompt, and
prints the generated text. Exercises the same model runner / SFA forward / offload
hook as ``vllm serve``, so it is the simplest way to trigger the offload path with a
fixed input during bring-up.

Prerequisites (see kv-cache-offload-v0.1.1-run-guide.md):
  - MicroKV server running at $MICROKV_SOCKET
  - microkv Python client importable (PYTHONPATH=.../MicroKV/python)
  - offload env vars set, e.g. VLLM_ASCEND_KV_OFFLOAD_V0_VALIDATE=1
    (or VLLM_ASCEND_KV_OFFLOAD_V0_COMPACT_SFA=1 + MAX_PINNED_REQS=1)

Usage:
  python run_offload_once.py [MODEL_PATH]
  MODEL_PATH defaults to weights/tiny-random-glm-moe-dsa.
  Prompt and max_tokens can be overridden via OFFLOAD_PROMPT / OFFLOAD_MAX_TOKENS.
"""

import os
import sys

from vllm import LLM, SamplingParams

DEFAULT_MODEL = "weights/tiny-random-glm-moe-dsa"
DEFAULT_PROMPT = "你好，请介绍一下自己。"


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    prompt = os.environ.get("OFFLOAD_PROMPT", DEFAULT_PROMPT)
    max_tokens = int(os.environ.get("OFFLOAD_MAX_TOKENS", "32"))

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
