#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="/share/home/wangzixu/liudinghao/gushuo/proj/exp/run_nano_radix_system_benchmarks.py"

eval "$(micromamba shell hook -s bash)"
micromamba activate nano_vllm

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR"

/share/home/wangzixu/.local/share/mamba/envs/nano_vllm/bin/python "$SCRIPT" "$@"
