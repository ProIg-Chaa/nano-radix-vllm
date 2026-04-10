# nano-vllm vs nano-radix-vllm System Benchmarks

- Timestamp: `20260409_201746`
- Model: `/share/home/wangzixu/liudinghao/gushuo/proj/transfoemer-llm/model/Qwen3-0.6B`
- GPU id: `1`

## Throughput vs Concurrency

| Concurrency | Backend | tokens/s | avg latency (s) | p95 latency (s) | cache hit rate | cached tokens |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | original | 34.5128 | 17.8482 | 17.8482 | 0.8421 | 512 |
| 1 | radix | 34.2062 | 18.0082 | 18.0082 | 0.9868 | 600 |
| 2 | original | 963.4194 | 1.2786 | 1.2786 | 0.8421 | 1024 |
| 2 | radix | 1555.0364 | 0.7920 | 0.7920 | 0.9145 | 1112 |
| 4 | original | 3110.8111 | 0.7918 | 0.7918 | 0.8421 | 2048 |
| 4 | radix | 3119.5075 | 0.7896 | 0.7896 | 0.8783 | 2136 |
| 8 | original | 4294.4678 | 1.1472 | 1.1472 | 0.8421 | 4096 |
| 8 | radix | 6144.6284 | 0.8016 | 0.8016 | 0.8602 | 4184 |
| 16 | original | 10260.5833 | 0.9601 | 0.9601 | 0.8421 | 8192 |
| 16 | radix | 12100.2731 | 0.8140 | 0.8140 | 0.8512 | 8280 |

## Prefix Overlap Ratio

| Overlap | Backend | cache hit rate | tokens/s | avg latency (s) | p95 latency (s) | cached tokens |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0% | original | 0.0000 | 12426.0279 | 0.7826 | 0.7826 | 0 |
| 0% | radix | 0.0000 | 12094.5047 | 0.8040 | 0.8040 | 0 |
| 25% | original | 0.2119 | 12250.9321 | 0.7937 | 0.7937 | 2048 |
| 25% | radix | 0.2119 | 11916.0465 | 0.8160 | 0.8160 | 2048 |
| 50% | original | 0.4238 | 12282.3046 | 0.7917 | 0.7917 | 4096 |
| 50% | radix | 0.4238 | 11892.4403 | 0.8177 | 0.8177 | 4096 |
| 75% | original | 0.6358 | 12341.9391 | 0.7879 | 0.7879 | 6144 |
| 75% | radix | 0.6358 | 11933.9512 | 0.8148 | 0.8148 | 6144 |
| 100% | original | 0.8477 | 12322.3602 | 0.7891 | 0.7891 | 8192 |
| 100% | radix | 0.8659 | 11804.8000 | 0.8237 | 0.8237 | 8368 |

## Long Context Stress

| Context | Backend | latency (s) | tokens/s | kv blocks used | GPU after init (MB) | GPU after run (MB) | GPU peak reserved (MB) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | original | 26.9220 | 38.0728 | 4 | 76705.0 | 76973.0 | 76456.0 |
| 1024 | radix | 26.8862 | 38.1234 | 4 | 76929.0 | 76949.0 | 76434.0 |
| 2048 | original | 26.6190 | 76.9746 | 8 | 76679.0 | 77005.0 | 76488.0 |
| 2048 | radix | 33.8702 | 60.4951 | 8 | 76901.0 | 76981.0 | 76466.0 |
| 4096 | original | 26.7728 | 153.0272 | 16 | 76621.0 | 77199.0 | 76682.0 |
| 4096 | radix | 26.9565 | 151.9846 | 16 | 76845.0 | 76997.0 | 76482.0 |
| 8192 | original | 27.1071 | 302.2435 | 32 | 76537.0 | 77207.0 | 76690.0 |
| 8192 | radix | 26.8545 | 305.0855 | 32 | 76733.0 | 77021.0 | 76506.0 |

## Notes

- `tokens/s` here means `(prompt_tokens + completion_tokens) / end_to_end_batch_wall_time`.
- `cache hit rate` here means `cached_tokens_total / prompt_tokens_total`, so it is comparable across both backends.
- Throughput-vs-concurrency uses a warmed 600-token nonaligned shared prefix to highlight the radix branch's finer-grained reuse path.
- Prefix-overlap experiment uses a 1200-token reusable region plus an 8-token unique tail to avoid the degenerate fully-identical prompt case.
