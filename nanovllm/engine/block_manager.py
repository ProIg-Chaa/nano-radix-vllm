from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class KVBlockAllocator:

    def __init__(self, num_blocks: int):
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    def has_free_blocks(self, num_blocks: int) -> bool:
        return self.num_free_blocks() >= num_blocks

    def is_used(self, block_id: int) -> bool:
        return block_id in self.used_block_ids

    def allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def allocate_next_free_block(self) -> tuple[int, Block]:
        block_id = self.free_block_ids[0]
        return block_id, self.allocate_block(block_id)

    def incref(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count > 0
        block.ref_count += 1
        return block

    def decref(self, block_id: int):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self.used_block_ids.remove(block_id)
            self.free_block_ids.append(block_id)


class PrefixCache:

    def __init__(self, block_size: int):
        self.block_size = block_size
        self.hash_to_block_id: dict[int, int] = dict()

    @staticmethod
    def compute_hash(token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def compute_block_hash(self, token_ids: list[int], prefix: int = -1) -> int:
        return self.compute_hash(token_ids, prefix) if len(token_ids) == self.block_size else -1

    def lookup(self, token_ids: list[int], prefix: int = -1) -> tuple[int, int]:
        block_hash = self.compute_block_hash(token_ids, prefix)
        return block_hash, self.hash_to_block_id.get(block_hash, -1)

    def commit(self, block: Block, block_hash: int, token_ids: list[int]):
        if block_hash == -1:
            return
        block.update(block_hash, token_ids)
        self.hash_to_block_id[block_hash] = block.block_id


class PlanStepKind(Enum):
    HIT_USED = auto()
    HIT_FREE = auto()
    MISS = auto()


@dataclass
class PlanStep:
    kind: PlanStepKind
    block_id: int
    block_hash: int
    token_ids: list[int]


@dataclass
class PrefillPlan:
    steps: list[PlanStep]
    cached_tokens: int
    required_free_blocks: int


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.allocator = KVBlockAllocator(num_blocks)
        self.prefix_cache = PrefixCache(block_size)
        self.blocks = self.allocator.blocks
        self.free_block_ids = self.allocator.free_block_ids
        self.used_block_ids = self.allocator.used_block_ids
        self.hash_to_block_id = self.prefix_cache.hash_to_block_id
        self.stats = {
            "alloc_requests": 0,
            "dealloc_requests": 0,
            "queried_blocks": 0,
            "hit_blocks": 0,
            "miss_blocks": 0,
            "reused_tokens": 0,
            "new_blocks": 0,
        }

    def reset_stats(self):
        for key in self.stats:
            self.stats[key] = 0

    def get_stats(self):
        stats = dict(self.stats)
        queried_blocks = stats["queried_blocks"]
        stats["hit_rate"] = stats["hit_blocks"] / queried_blocks if queried_blocks else 0.0
        return stats

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        return PrefixCache.compute_hash(token_ids, prefix)

    def make_prefill_plan(self, seq: Sequence) -> PrefillPlan:
        steps = []
        prefix_hash = -1
        cache_miss = False
        cached_tokens = 0
        required_free_blocks = 0

        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            block_hash, block_id = self.prefix_cache.lookup(token_ids, prefix_hash)
            is_hit = block_id != -1 and not cache_miss and self.blocks[block_id].token_ids == token_ids
            if is_hit:
                cached_tokens += self.block_size
                if self.allocator.is_used(block_id):
                    steps.append(PlanStep(PlanStepKind.HIT_USED, block_id, block_hash, token_ids))
                else:
                    required_free_blocks += 1
                    steps.append(PlanStep(PlanStepKind.HIT_FREE, block_id, block_hash, token_ids))
            else:
                cache_miss = True
                required_free_blocks += 1
                steps.append(PlanStep(PlanStepKind.MISS, -1, block_hash, token_ids))
            prefix_hash = block_hash

        return PrefillPlan(steps, cached_tokens, required_free_blocks)

    def can_allocate(self, seq: Sequence, plan: PrefillPlan | None = None) -> bool:
        if plan is None:
            plan = self.make_prefill_plan(seq)
        return self.allocator.has_free_blocks(plan.required_free_blocks)

    def allocate(self, seq: Sequence, plan: PrefillPlan | None = None):
        assert not seq.block_table
        if plan is None:
            plan = self.make_prefill_plan(seq)

        self.stats["alloc_requests"] += 1
        seq.num_cached_tokens = plan.cached_tokens

        for step in plan.steps:
            self.stats["queried_blocks"] += 1
            token_ids = step.token_ids
            if step.kind == PlanStepKind.MISS:
                self.stats["miss_blocks"] += 1
                block_id, block = self.allocator.allocate_next_free_block()
                self.stats["new_blocks"] += 1
            elif step.kind == PlanStepKind.HIT_USED:
                self.stats["hit_blocks"] += 1
                self.stats["reused_tokens"] += self.block_size
                block_id = step.block_id
                block = self.allocator.incref(block_id)
            else:
                self.stats["hit_blocks"] += 1
                self.stats["reused_tokens"] += self.block_size
                block_id = step.block_id
                block = self.allocator.allocate_block(block_id)

            self.prefix_cache.commit(block, step.block_hash, token_ids)
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        self.stats["dealloc_requests"] += 1
        for block_id in reversed(seq.block_table):
            self.allocator.decref(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return self.allocator.has_free_blocks(len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id, _ = self.allocator.allocate_next_free_block()
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            block_hash = self.prefix_cache.compute_block_hash(token_ids, prefix)
            self.prefix_cache.commit(last_block, block_hash, token_ids)
        else:
            assert last_block.hash == -1
