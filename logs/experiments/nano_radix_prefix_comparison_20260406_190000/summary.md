# nano-vllm vs nano-vllm-radix Prefix Comparison

- Timestamp: `20260406_190000`
- Model: `/share/home/wangzixu/liudinghao/gushuo/proj/transfoemer-llm/model/Qwen3-0.6B`
- Original repo: `/share/home/wangzixu/liudinghao/gushuo/proj/nano-vllm`
- Radix repo: `/share/home/wangzixu/liudinghao/gushuo/proj/nano-vllm-radix`

## Result Summary

| Workload | Original status | Radix status | Original latency (s) | Radix latency (s) | Original cached tokens | Radix cached tokens |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| unique_long_cold | success | failed | 27.9401 | N/A | 0 | N/A |
| aligned_shared_prefix_warm | success | success | 17.9579 | 17.8116 | 768 | 768 |
| nonaligned_shared_prefix_warm | success | success | 17.8564 | 17.6198 | 512 | 600 |

## Interpretation

- `aligned_shared_prefix_warm`: both backends reuse the 3 full 256-token blocks, so cached-token totals are the same at `768`.
- `nonaligned_shared_prefix_warm`: original `nano-vllm` stops at full-block reuse and caches `512` tokens, while `nano-vllm-radix` reuses `600` tokens by taking the partial tail prefix path.
- `unique_long_cold`: the radix backend currently has an independent bug on the no-prefix baseline path and fails in `store_kvcache` with a device-side assert, so this workload is recorded as a failure rather than a valid latency datapoint.

## Caveat

This experiment demonstrates the intended shared-prefix contrast successfully. It does not yet provide a complete apples-to-apples baseline on `unique_long_cold` because the current radix backend still has a separate prefill-store bug on that workload.
