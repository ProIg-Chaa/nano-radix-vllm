#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/logs/experiments"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/prefix_cache_test_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

eval "$(micromamba shell hook -s bash)"
micromamba activate nano_vllm

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR"

echo "[run] log_file=$LOG_FILE" | tee "$LOG_FILE"
python - <<'INNER_PY' 2>&1 | tee -a "$LOG_FILE"
from nanovllm import LLM, SamplingParams

MODEL = "/share/home/wangzixu/liudinghao/gushuo/proj/transfoemer-llm/model/Qwen3-0.6B"

common_prefix = [123] * 768
prompts = [
    common_prefix + [1000 + i, 2000 + i, 3000 + i]
    for i in range(8)
]

llm = LLM(MODEL, enforce_eager=True, max_model_len=4096)
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=8,
    ignore_eos=True,
)

outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

print("model:", MODEL)
print("num_prompts:", len(prompts))
print("prompt_len:", len(prompts[0]))
print("first_output_token_ids:", outputs[0]["token_ids"])
print("first_output_text:", outputs[0]["text"])
INNER_PY

echo "[run] finished log_file=$LOG_FILE" | tee -a "$LOG_FILE"
