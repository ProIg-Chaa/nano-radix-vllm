#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

eval "$(micromamba shell hook -s bash)"
micromamba activate nano_vllm

cd "$ROOT_DIR"
exec "${SHELL:-/bin/bash}" -i
