# nano-vllm vs nano-radix-vllm System Benchmarks

- Timestamp: `20260410_132551`
- Model: `/share/home/wangzixu/liudinghao/gushuo/proj/transfoemer-llm/model/Qwen3-0.6B`
- GPU id: `0,1`

## Throughput vs Concurrency

| Concurrency | Backend | tokens/s | avg latency (s) | p95 latency (s) | cache hit rate | cached tokens |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | original | 25.7815 | 23.8929 | 23.8929 | 0.8421 | 512 |
| 1 | radix | 33.4428 | 18.4193 | 18.4193 | 0.9868 | 600 |
| 2 | original | 856.2157 | 1.4386 | 1.4386 | 0.8421 | 1024 |
| 2 | radix | 1420.4871 | 0.8671 | 0.8671 | 0.9145 | 1112 |
| 4 | original | 2905.1697 | 0.8478 | 0.8478 | 0.8421 | 2048 |
| 4 | radix | 2869.4971 | 0.8584 | 0.8584 | 0.8783 | 2136 |
| 8 | original | 4065.6286 | 1.2118 | 1.2118 | 0.8421 | 4096 |
| 8 | radix | 5684.3809 | 0.8666 | 0.8666 | 0.8602 | 4184 |
| 16 | original | 9449.9830 | 1.0425 | 1.0425 | 0.8421 | 8192 |
| 16 | radix | 11267.6553 | 0.8742 | 0.8742 | 0.8512 | 8280 |

## Prefix Overlap Ratio

| Overlap | Backend | cache hit rate | tokens/s | avg latency (s) | p95 latency (s) | cached tokens |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0% | original | 0.0000 | 11452.4000 | 0.8491 | 0.8491 | 0 |
| 0% | radix | 0.0000 | 10872.6836 | 0.8944 | 0.8944 | 0 |
| 25% | original | 0.2119 | 11170.6434 | 0.8705 | 0.8705 | 2048 |
| 25% | radix | 0.2119 | 10908.0851 | 0.8915 | 0.8915 | 2048 |
| 50% | original | 0.4238 | 11160.2047 | 0.8713 | 0.8713 | 4096 |
| 50% | radix | 0.4238 | 10984.6243 | 0.8853 | 0.8853 | 4096 |
| 75% | original | 0.6358 | 11475.6467 | 0.8474 | 0.8474 | 6144 |
| 75% | radix | 0.6358 | 10488.3084 | 0.9272 | 0.9272 | 6144 |
| 100% | original | 0.8477 | 11459.0745 | 0.8486 | 0.8486 | 8192 |
| 100% | radix | 0.8659 | 11218.6416 | 0.8668 | 0.8668 | 8368 |

## Long Context Stress

| Context | Backend | latency (s) | tokens/s | kv blocks used | GPU after init (MB) | GPU after run (MB) | GPU peak reserved (MB) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | original | 27.2633 | 37.5961 | 4 | 76713.0 | 76961.0 | 76242.0 |
| 1024 | radix | 26.9106 | 38.0888 | 4 | 76951.0 | 76953.0 | 76234.0 |
| 2048 | original | 27.0663 | 75.7026 | 8 | 76699.0 | 76965.0 | 76246.0 |
| 2048 | radix | 26.8872 | 76.2066 | 8 | 76937.0 | 76969.0 | 76250.0 |
| 4096 | original | 27.1617 | 150.8365 | 16 | 76679.0 | 77227.0 | 76508.0 |
| 4096 | radix | 26.9373 | 152.0929 | 16 | 76903.0 | 77027.0 | 76308.0 |
| 8192 | original | 27.4555 | 298.4085 | 32 | 76609.0 | 77217.0 | 76498.0 |
| 8192 | radix | 26.7594 | 306.1706 | 32 | 76819.0 | 77047.0 | 76328.0 |

## Notes

- `tokens/s` here means `(prompt_tokens + completion_tokens) / end_to_end_batch_wall_time`.
- `cache hit rate` here means `cached_tokens_total / prompt_tokens_total`, so it is comparable across both backends.
- Throughput-vs-concurrency uses a warmed 600-token nonaligned shared prefix to highlight the radix branch's finer-grained reuse path.
- Prefix-overlap experiment uses a 1200-token reusable region plus an 8-token unique tail to avoid the degenerate fully-identical prompt case.
