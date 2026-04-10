# nano-vllm-radix（中文版）

语言: [English](./README.md) | 中文

`nano-vllm-radix` 是基于 `nano-vllm` 演进出来的独立仓库，目标是在保留
`nano-vllm` 轻量框架的前提下，逐步融合 SGLang 风格的 radix/prefix 复用思想。

项目核心目标：

- 保持 `nano-vllm` 的简洁性
- 增加跨请求共享前缀的 KV Cache 复用能力
- 从 block 级复用逐步推进到更细粒度复用
- 每一步都可观测、可验证、可回归

## 许可证与归属

本项目来源于：

- `nano-vllm`
- upstream copyright: `Copyright (c) 2025 Xingkai Yu`
- upstream license: MIT

当前仓库保留了 upstream MIT 许可证：

- [LICENSE](./LICENSE)

并补充了归属说明：

- [NOTICE](./NOTICE)

## 当前能力

当前分支已经具备：

- block 级共享前缀 KV 复用
- prefix-aware 的 prefill 调度
- tree-based prefix cache 管理
- request 侧 logical page 元数据
- page-aware 的 prefill 准备
- partial-tail 更细粒度复用
- active partial-tail copy-on-write 复用

关键能力差异（对比原版）：

- 共享前缀长度 `600` token
- 原版 `nano-vllm` 可复用 `512`
- `nano-vllm-radix` 可复用 `600`

## 与原版 nano-vllm 的主要差异

1. 可观测性增强（命中率、复用 token、logical page、eviction 等统计）
2. `BlockManager` 职责重构（物理分配 + 前缀索引 + 调度侧编排）
3. prefill 调度前置规划（先 plan，再 admission，再 allocate）
4. flat hash 向 prefix tree 演进（含 ownership、subtree prune、leaf eviction）
5. logical page 层引入（`kvcache_block_size=256`，`logical_page_size=32`）
6. partial-tail 与 copy-on-write 路径落地

## 关键目录与文件

- `nanovllm/engine/block_manager.py`
- `nanovllm/engine/scheduler.py`
- `nanovllm/engine/sequence.py`
- `nanovllm/engine/model_runner.py`
- `run_prefix_cache_test.sh`
- `run_prefix_cache_test_small.sh`
- `run_page_aware_prefill_checks.sh`
- `run_nano_radix_prefix_comparison.sh`
- `run_nano_radix_system_benchmarks.sh`
- `logs/`

## 环境

已导出当前环境：

- `environment.nano_vllm.yml`

重建命令：

```bash
micromamba env create -f environment.nano_vllm.yml
```

## 快速开始

激活环境：

```bash
cd .
./enter_nano_vllm_env.sh
```

基础前缀复用测试：

```bash
./run_prefix_cache_test_small.sh
./run_prefix_cache_test.sh
```

page-aware 定向校验：

```bash
./run_page_aware_prefill_checks.sh
```

原版 vs radix 对比：

```bash
./run_nano_radix_prefix_comparison.sh --enforce-eager --gpu-id 1 --max-model-len 800 --max-num-batched-tokens 800 --max-num-seqs 8 --gpu-memory-utilization 0.95
```

系统实验（单卡）：

```bash
./run_nano_radix_system_benchmarks.sh --enforce-eager --gpu-id 1
```

系统实验（双卡 TP=2）：

```bash
./run_nano_radix_system_benchmarks.sh --enforce-eager --gpu-id 0,1 --tensor-parallel-size 2
```

## 实验结果

### 第一轮对比实验

目录：

- `exp/logs/nano_radix_prefix_comparison/20260406_202424/summary.md`
- `logs/experiments/nano_radix_prefix_comparison_20260406_202424/summary.md`

结论：

- block 对齐共享前缀：两边复用量一致
- 非对齐共享前缀：radix 分支可复用 partial tail（`512 -> 600`）

### 系统实验扩展（单卡）

目录：

- `exp/logs/nano_radix_system_benchmarks/20260409_201746/summary.md`
- `logs/experiments/nano_radix_system_benchmarks_20260409_201746/summary.md`

结论：

- 并发曲线在 `2/8/16` 并发点上，radix 的吞吐优势明显
- 前缀重叠比实验在 `100%` overlap 时出现预期差异：
  original cached `8192` vs radix cached `8368`
- 长上下文压力测试两边都稳定，KV block 增长一致

### 系统实验扩展（双卡 TP=2）

目录：

- `exp/logs/nano_radix_system_benchmarks/20260410_132551/summary.md`
- `logs/experiments/nano_radix_system_benchmarks_20260410_132551/summary.md`

代表性吞吐点：

- `concurrency=2`：original `856.2157 tok/s` vs radix `1420.4871 tok/s`
- `concurrency=8`：original `4065.6286 tok/s` vs radix `5684.3809 tok/s`
- `concurrency=16`：original `9449.9830 tok/s` vs radix `11267.6553 tok/s`

说明当前设计在多卡 TP 场景下仍能保持 nonaligned prefix 复用优势。
