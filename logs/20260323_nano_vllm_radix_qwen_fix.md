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

## 2026-04-06 Experiment Issues

### Context
While building a direct comparison experiment between:
- original `nano-vllm`
- current `nano-vllm-radix`

using Qwen3-0.6B and shared-prefix workloads, two separate classes of issues surfaced.

### Issue A - Original nano-vLLM KV Cache Initialization Is Sensitive To Warmup Configuration

#### Symptom
The original backend repeatedly failed during initialization with:

```text
assert config.num_kvcache_blocks > 0
```

inside `nanovllm/engine/model_runner.py`.

#### Root Cause
The original project computes KV cache capacity only after a warmup prefill run and uses:

```python
num_kvcache_blocks =
    int(total * gpu_memory_utilization - used - peak + current) // block_bytes
```

This means the available KV blocks are affected by:
- selected GPU
- current background memory usage
- warmup peak memory
- `max_model_len`
- `max_num_batched_tokens`
- `gpu_memory_utilization`

Even on a machine with large total free memory, the original backend can still produce
`num_kvcache_blocks <= 0` if the warmup configuration is unnecessarily large relative to
the actual benchmark prompt lengths.

#### Mitigation Used
The comparison experiment was adjusted to:
- use the almost-empty GPU 1 rather than GPU 0
- reduce warmup-related config to values close to the real prompt length
  - `max_model_len=800`
  - `max_num_batched_tokens=800`
  - `max_num_seqs=8`
- raise `gpu_memory_utilization` to `0.95`

This made the original backend runnable.

### Issue B - Radix Nonaligned Partial-Reuse Path Has A Request-Layout Consistency Bug

#### Symptom
The current `nano-vllm-radix` backend failed during the comparison experiment on the
nonaligned shared-prefix workload with assertions such as:

```text
assert layout.uncached_start_page == plan.uncached_start_page
```

and later:

```text
assert total_span_pages == seq.num_logical_pages
```

inside the page-aware prefill path.

#### Observed Runtime Pattern
The failure appears only after combining:
- prompt-only warmup materialization
- retained partial tail reuse
- nonaligned shared prefix (`600` shared tokens)
- full `LLM.generate()` execution path

The following simpler paths were already known to work:
- aligned block-level prefix reuse
- isolated retained free partial-tail reuse checks
- isolated active partial-tail copy-on-write checks

#### Root Cause Hypothesis
The new finer-grained reuse path added several coupled structures:
- `num_materialized_tokens`
- `page_cached_tokens`
- logical page spans
- physical address spans
- partial-tail retained-block index
- active-source copy spans

Each individual piece worked in targeted checks, but the benchmark exercised a more
realistic sequence:

1. materialize a prompt-only warm request
2. retain its partial tail as reusable cache state
3. submit a new request with a non-block-aligned shared prefix
4. let the normal prefill planner, request layout builder, and runner consume that state

At that point, the request-side prefill layout and the runner-side validation no longer
agreed on the full logical-page coverage of the sequence. In other words:

- the planner's view of the cached/uncached boundary
- the `Sequence.prefill_layout` rebuilt from request metadata
- and the runner's logical/physical span validation

were no longer perfectly consistent for this nonaligned partial-reuse path.

The bug is therefore not in the high-level experiment design. It is a real runtime
consistency problem in the current finer-grained reuse implementation.

#### Why Earlier Scripts Did Not Expose It
Earlier validation mostly covered:
- block-aligned reuse
- direct single-case partial-tail checks
- controlled planner-level or metadata-level assertions

The comparison benchmark was the first test that exercised:
- warmed retained prompt state
- nonaligned partial prefix reuse
- and the full end-to-end `generate()` lifecycle

so it exposed a bug that the narrower checks did not cover.

### Current Status
- Original `nano-vllm` side: runnable after tuning initialization parameters.
- Current `nano-vllm-radix` side: still has a nonaligned partial-reuse consistency bug
  that must be fixed before the final comparison summary can be trusted.

### 2026-04-06 Fix Applied
The runner-side logical-page validation was updated so that nonaligned partial reuse is
validated by **page-interval union coverage** rather than by naively summing cached and
uncached span page counts.

Why this is correct:
- for a nonaligned cached boundary such as `cached_tokens = 600` with `page_size = 32`
- the cached span ends at page `19`
- the uncached span starts at page `18`
- so the boundary page is intentionally shared by both cached and uncached views

The previous validation incorrectly assumed those spans must be disjoint and therefore
reported an artificial coverage mismatch.

The new validation:
- computes covered logical pages by interval union
- still checks token-level coverage exactly
- explicitly allows the cached and uncached spans to overlap by exactly one boundary page
  in the partial-page case

### 2026-04-06 Additional Benchmark Driver Bug
After the nonaligned partial-reuse layout bug was fixed, the comparison benchmark still
showed a device-side assert on the radix-side `unique_long_cold` workload.

This turned out to be a benchmark-script bug rather than a radix runtime bug.

#### Trigger
The comparison driver was using a `seed_offset` to ensure the warmed prompts and
benchmark prompts were not identical. That offset is correct for the shared-prefix
workloads, but it was also being applied to the `unique` workload.

Because `make_unique_prompt(i)` already spaces prompts by `i * 10000`, applying
`seed_offset=100` changed the benchmark prompt token IDs into values far outside the
model vocabulary range. That produced embedding/indexing failures such as:
- `indexSelectLargeIndex`
- `device-side assert triggered`

#### Fix
The benchmark driver was updated so that:
- shared-prefix workloads still use different tail seeds between warm and bench
- `unique` prompts ignore the benchmark `seed_offset`

This restored valid token IDs for the unique baseline and allowed the full comparison
benchmark to complete successfully.

### Final Benchmark Outcome
Final completed experiment:
- `/share/home/wangzixu/liudinghao/gushuo/proj/exp/logs/nano_radix_prefix_comparison/20260406_202424`
- `/share/home/wangzixu/liudinghao/gushuo/proj/nano-vllm-radix/logs/experiments/nano_radix_prefix_comparison_20260406_202424`

Most important differentiator:
- original `nano-vllm` reused `512` tokens on the `600`-token nonaligned shared-prefix case
- current `nano-vllm-radix` reused `600` tokens on the same case

This confirms that the current radix branch now exposes a real finer-grained reuse
advantage in the intended comparison benchmark.
