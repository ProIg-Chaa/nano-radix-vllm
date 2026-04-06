# nano-vllm vs nano-vllm-radix Prefix Comparison

- Timestamp: `20260406_202424`
- Model: `/share/home/wangzixu/liudinghao/gushuo/proj/transfoemer-llm/model/Qwen3-0.6B`
- Original repo: `/share/home/wangzixu/liudinghao/gushuo/proj/nano-vllm`
- Radix repo: `/share/home/wangzixu/liudinghao/gushuo/proj/nano-vllm-radix`

## Workloads

| Workload | Original latency (s) | Radix latency (s) | Original cached tokens | Radix cached tokens | Delta cached tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| unique_long_cold | 27.4772 | 27.5612 | 0 | 0 | 0 |
| aligned_shared_prefix_warm | 19.2428 | 17.4658 | 768 | 768 | 0 |
| nonaligned_shared_prefix_warm | 18.5552 | 18.7836 | 512 | 600 | 88 |

## Notes

- `aligned_shared_prefix_warm` should show similar cached-token totals on both repos because both can reuse full 256-token blocks.
- `nonaligned_shared_prefix_warm` is the differentiator: original nano-vllm is limited by block alignment, while current nano-vllm-radix can reuse the nonaligned partial-tail prefix.
- `cached_tokens_total` is collected from the request allocation path, not inferred from generated text.
