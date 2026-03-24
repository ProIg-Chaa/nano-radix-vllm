# Nano-vLLM Fix Log - 2026-03-23

## Summary
Patched nano-vllm to improve compatibility with local Qwen model configs.

## Change
1. Added a compatibility fix in `nanovllm/layers/rotary_embedding.py`.
   - Problem: `get_rope()` was cached with `lru_cache`, but some Qwen configs pass `rope_scaling` as a `dict`.
   - Failure: `TypeError: unhashable type: 'dict'` during model initialization.
   - Fix: normalize `rope_scaling` before it reaches the cached helper.
   - Compatibility rule: treat `{"rope_type": "default", ...}` as standard RoPE and pass `None` into the cached implementation.

## Files Touched
- `nanovllm/layers/rotary_embedding.py`

## Result
- Qwen-style configs that provide `rope_scaling` as a dictionary no longer fail at cache lookup time.
- The current implementation still only supports the default RoPE path; non-default rope scaling modes are intentionally not enabled by this fix.
