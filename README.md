# nano-vllm-radix

Language: English | [中文](./README.zh-CN.md)

`nano-vllm-radix` is an independent repository derived from `nano-vllm` that
incrementally integrates SGLang-style radix/prefix reuse ideas into the original
lightweight inference framework.

The project goal is:

- preserve the simplicity of `nano-vllm`
- add cross-request KV cache reuse for shared prefixes
- gradually move from block-level prefix reuse toward finer-grained reuse
- keep every step measurable and debuggable

This repository is not a verbatim port of SGLang. It is a staged migration that reuses
the `nano-vllm` codebase and evolves it toward a more prefix-aware cache manager.

## Attribution and License

This project is derived from:

- `nano-vllm`
- upstream copyright: `Copyright (c) 2025 Xingkai Yu`
- upstream license: MIT

This repository keeps the upstream MIT license in [LICENSE](./LICENSE)
and adds project-level attribution in [NOTICE](./NOTICE).

The current repository contains substantial modifications and new components for:

- radix/prefix reuse integration
- prefix-aware scheduling
- logical-page-aware prefill preparation
- partial-tail reuse
- active partial-tail copy-on-write reuse
- benchmarking and experiment tooling

## Current Status

The current branch already supports:

- block-level shared-prefix KV reuse
- prefix-aware prefill scheduling
- tree-based prefix cache management
- request-side logical page metadata
- page-aware prefill preparation
- finer-grained partial-tail prefix reuse
- active partial-tail copy-on-write reuse

The most important practical result is that the current branch can reuse a non-block-
aligned shared prefix that the original `nano-vllm` cannot fully reuse.

Example:

- shared prefix length: `600` tokens
- original `nano-vllm`: reuses `512`
- `nano-vllm-radix`: reuses `600`

## What Changed vs Original nano-vllm

Compared with the original project in:

- `/share/home/wangzixu/liudinghao/gushuo/proj/nano-vllm`

this branch adds several major pieces.

### 1. Prefix-cache instrumentation

The engine now records and prints:

- `hit_blocks`
- `miss_blocks`
- `reused_tokens`
- `partial_reused_tokens`
- `reused_logical_pages`
- `prefill_cached_pages`
- `prefill_uncached_pages`
- eviction-related counters

This makes every cache-related change observable instead of speculative.

### 2. Cache-manager refactor

`BlockManager` was split conceptually into:

- physical KV block allocation
- prefix cache / prefix tree indexing
- scheduling-facing orchestration

This makes later radix-style changes safer than editing one monolithic allocator class.

### 3. Prefix-aware scheduling

Prefill scheduling is no longer based only on raw request length. The scheduler now:

1. builds a prefill plan
2. estimates cached vs uncached prefix/tail cost
3. estimates required free blocks
4. decides admission using the real uncached cost

### 4. Tree-based prefix cache

The original flat block-hash lookup evolved into a prefix tree with:

- parent/child structure
- ownership tracking
- subtree pruning
- leaf eviction / retention metadata

### 5. Logical page layer

The runtime now distinguishes:

- physical KV block size
- logical page size

Current defaults:

- `kvcache_block_size = 256`
- `logical_page_size = 32`

The request-side page layer includes:

- `LogicalPageRef`
- `LogicalPageSpan`
- `RequestPrefillLayout`
- physical address spans
- copy spans for partial-tail reuse

### 6. Finer-grained partial reuse

The branch now supports:

- retained free partial-tail reuse
- active partial-tail reuse through copy-on-write

This is the first real step beyond pure full-block reuse.

## Architecture Snapshot

At a high level, the current system has four layers:

### Physical KV layer

- block allocator
- `block_table`
- actual KV cache storage on GPU

### Prefix cache layer

- prefix tree
- block ownership
- retained leaf management
- partial-tail prefix index

### Request-side logical layer

- logical pages
- cached/uncached page spans
- request prefill layout
- physical slot spans derived from logical pages

### Runner preparation layer

- page-aware prefill planning
- physical slot mapping
- copy-on-write application before prefill

The attention backend is still block-based. That is intentional: the current branch has
already made the planner and request metadata page-aware, but it has not replaced the
underlying physical KV storage model with a true page allocator.

## Repository Layout

Important files and directories:

- `nanovllm/engine/block_manager.py`
  Cache manager, prefix tree, partial-tail reuse, planning
- `nanovllm/engine/scheduler.py`
  Prefix-aware scheduling
- `nanovllm/engine/sequence.py`
  Request-side page metadata and prefill layout
- `nanovllm/engine/model_runner.py`
  Page-aware prefill preparation and copy-span application
- `run_prefix_cache_test.sh`
  Basic shared-prefix synthetic benchmark
- `run_prefix_cache_test_small.sh`
  Smaller shared-prefix synthetic benchmark
- `run_page_aware_prefill_checks.sh`
  Directed validation for page-aware and partial-tail paths
- `run_nano_radix_prefix_comparison.sh`
  Original vs radix comparison experiment
- `logs/`
  Migration logs, bug logs, and experiment outputs

## Environment

The current `nano_vllm` micromamba environment was exported to:

- `environment.nano_vllm.yml`

To recreate:

```bash
micromamba env create -f environment.nano_vllm.yml
```

## Quick Start

Activate environment:

```bash
cd .
./enter_nano_vllm_env.sh
```

Run the small synthetic prefix-cache test:

```bash
./run_prefix_cache_test_small.sh
```

Run the larger synthetic prefix-cache test:

```bash
./run_prefix_cache_test.sh
```

Run directed page-aware checks:

```bash
./run_page_aware_prefill_checks.sh
```

Run the original-vs-radix comparison:

```bash
./run_nano_radix_prefix_comparison.sh --enforce-eager --gpu-id 1 --max-model-len 800 --max-num-batched-tokens 800 --max-num-seqs 8 --gpu-memory-utilization 0.95
```

Run the system benchmark extension (single GPU):

```bash
./run_nano_radix_system_benchmarks.sh --enforce-eager --gpu-id 1
```

Run the same system benchmark with multi-GPU TP=2:

```bash
./run_nano_radix_system_benchmarks.sh --enforce-eager --gpu-id 0,1 --tensor-parallel-size 2
```

## Comparison Results

Latest completed comparison:

- `exp/logs/nano_radix_prefix_comparison/20260406_202424/summary.md`
- `logs/experiments/nano_radix_prefix_comparison_20260406_202424/summary.md`

Summary:

| Workload | Original latency (s) | Radix latency (s) | Original cached tokens | Radix cached tokens |
| --- | ---: | ---: | ---: | ---: |
| `unique_long_cold` | `27.4772` | `27.5612` | `0` | `0` |
| `aligned_shared_prefix_warm` | `19.2428` | `17.4658` | `768` | `768` |
| `nonaligned_shared_prefix_warm` | `18.5552` | `18.7836` | `512` | `600` |

Interpretation:

- On block-aligned shared prefixes, both branches reuse the same amount.
- On nonaligned shared prefixes, `nano-vllm-radix` shows the intended advantage by
  reusing the partial tail prefix.

## System Benchmark Results (New)

### Single GPU run

Artifacts:

- `exp/logs/nano_radix_system_benchmarks/20260409_201746/summary.md`
- `logs/experiments/nano_radix_system_benchmarks_20260409_201746/summary.md`

Key findings:

- Throughput-vs-concurrency on warmed nonaligned shared prefixes shows clear radix gains
  at several concurrency levels, especially `2/8/16`.
- Prefix-overlap sweep shows the expected differentiation at `100%` overlap:
  original cached tokens `8192` vs radix cached tokens `8368`.
- Long-context stress (`1k/2k/4k/8k`) remains stable on both branches, with similar KV
  block growth and memory behavior.

### Multi-GPU run (`GPU 0,1`, `tensor_parallel_size=2`)

Artifacts:

- `exp/logs/nano_radix_system_benchmarks/20260410_132551/summary.md`
- `logs/experiments/nano_radix_system_benchmarks_20260410_132551/summary.md`

Representative points from throughput-vs-concurrency:

- `concurrency=2`: original `856.2157 tok/s` vs radix `1420.4871 tok/s`
- `concurrency=8`: original `4065.6286 tok/s` vs radix `5684.3809 tok/s`
- `concurrency=16`: original `9449.9830 tok/s` vs radix `11267.6553 tok/s`

Corresponding cache behavior remains consistent with the project goal:

- original nonaligned shared-prefix reuse is still bounded by full-block alignment
- radix branch continues to reuse the partial tail (`cached tokens` advantage remains visible)

That last row is the key result of this branch.

## Logs and Detailed Records

The root `README.md` is a high-level entry point. Full implementation history lives in:

- `logs/20260325_radix_migration_log.md`
- `logs/20260323_nano_vllm_radix_qwen_fix.md`

These files contain:

- phased migration notes
- design rationale
- bug investigations
- validation results
- benchmark debugging history

## Current Limitations

This branch is ahead of the original project, but it is not the final form of a full
SGLang-style radix cache.

Still missing or incomplete:

- more general token/page-level tree indexing
- broader non-tail partial matching
- a true physical page allocator
- deeper attention/backend integration beyond the current block-based storage model
- more policy work on copy-on-write overhead and cache retention under load

## Takeaway

This project already demonstrates a concrete and measurable benefit over original
`nano-vllm`:

- original branch: full-block shared-prefix reuse
- current branch: full-block reuse plus finer-grained partial-tail reuse

The branch therefore already validates the core migration hypothesis:

SGLang-style radix/prefix reuse ideas can be integrated into the `nano-vllm` framework
incrementally, and they produce a real cross-request KV reuse advantage before a full
page/token-level cache redesign is complete.
