#!/usr/bin/env python3
"""Mechanism-level HBM KVCache hit-rate simulator.

This demo models the current ASU-backed managed-token design:

- CPU updates LRU state and prepares free HBM slots at decode step boundaries.
- NPU resolves one step of topK token queries.
- HBM hits touch resident managed tokens.
- ASU_ONLY misses install into pre-prepared free slots.
- If the free-slot buffer is exhausted, the NPU does not pick emergency victims.
"""

from __future__ import annotations

import argparse
import bisect
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Sequence


TOPK_PER_REQ = 2048
DEFAULT_MANAGED_TOKENS_PER_REQ = 128 * 1024
DEFAULT_HBM_SLOTS_PER_REQ = 8192

TokenKey = tuple[int, int]


@dataclass(frozen=True)
class SimulationConfig:
    req_num: int = 4
    steps: int = 32
    managed_tokens_per_req: int = DEFAULT_MANAGED_TOKENS_PER_REQ
    hbm_slots_per_req: int = DEFAULT_HBM_SLOTS_PER_REQ
    free_slots_target_per_req: int | None = TOPK_PER_REQ
    workload: str = "hotset"
    hotset_size: int = 8192
    hotset_prob: float = 0.9
    zipf_exponent: float = 1.1
    seed: int = 0

    @property
    def total_slots(self) -> int:
        return self.req_num * self.hbm_slots_per_req

    @property
    def free_slot_target(self) -> int:
        if self.free_slots_target_per_req is None:
            target_per_req = TOPK_PER_REQ
        else:
            target_per_req = self.free_slots_target_per_req
        return min(self.total_slots, self.req_num * target_per_req)


@dataclass
class QueryStats:
    hits: int = 0
    misses: int = 0
    installs: int = 0
    shortages: int = 0

    @property
    def queries(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.queries == 0:
            return 0.0
        return self.hits / self.queries

    def add(self, other: QueryStats) -> None:
        self.hits += other.hits
        self.misses += other.misses
        self.installs += other.installs
        self.shortages += other.shortages


@dataclass
class StepStats(QueryStats):
    step: int = 0
    cpu_evictions: int = 0
    resident_count: int = 0
    free_slots: int = 0


@dataclass
class SimulationResult:
    config: SimulationConfig
    steps: list[StepStats]
    total: StepStats

    @property
    def total_hit_rate(self) -> float:
        return self.total.hit_rate


class LRUManagedCache:
    """Global managed-token HBM slot pool with LRU step-boundary eviction."""

    def __init__(self, total_slots: int, free_slot_target: int) -> None:
        if total_slots < 0:
            raise ValueError("total_slots must be non-negative")
        if free_slot_target < 0:
            raise ValueError("free_slot_target must be non-negative")

        self._free_slot_target = min(free_slot_target, total_slots)
        self._free_slots = list(range(total_slots - 1, -1, -1))
        self._residents: OrderedDict[TokenKey, int] = OrderedDict()

    @property
    def resident_count(self) -> int:
        return len(self._residents)

    @property
    def free_slots(self) -> int:
        return len(self._free_slots)

    @property
    def resident_tokens(self) -> set[TokenKey]:
        return set(self._residents)

    def prepare_free_slots(self) -> int:
        """Evict clean managed tokens until the next-step free target is met."""
        evicted = 0
        while self.free_slots < self._free_slot_target and self._residents:
            _, slot = self._residents.popitem(last=False)
            self._free_slots.append(slot)
            evicted += 1
        return evicted

    def resolve_query(self, tokens: Iterable[TokenKey]) -> QueryStats:
        """Resolve one NPU-visible query list without emergency victim choice."""
        stats = QueryStats()

        for token in tokens:
            if token in self._residents:
                stats.hits += 1
                self._residents.move_to_end(token)
                continue

            stats.misses += 1
            if self._free_slots:
                self._residents[token] = self._free_slots.pop()
                stats.installs += 1
            else:
                stats.shortages += 1

        return stats


class TopKWorkload:
    """Deterministic topK token generator for each request and step."""

    def __init__(self, config: SimulationConfig) -> None:
        self._config = config
        self._rng = random.Random(config.seed)
        self._hotsets = self._make_hotsets()
        self._zipf_cdf = self._make_zipf_cdf()

    def query(self, req_id: int, step: int) -> list[TokenKey]:
        if self._config.workload == "uniform":
            token_ids = self._uniform_tokens()
        elif self._config.workload == "hotset":
            token_ids = self._hotset_tokens(req_id)
        elif self._config.workload == "zipf":
            token_ids = self._zipf_tokens()
        elif self._config.workload == "sliding":
            token_ids = self._sliding_tokens(req_id, step)
        else:
            raise ValueError(f"unsupported workload: {self._config.workload}")

        return [(req_id, token_id) for token_id in token_ids]

    def _make_hotsets(self) -> list[list[int]]:
        if self._config.workload != "hotset":
            return []

        size = min(self._config.hotset_size, self._config.managed_tokens_per_req)
        return [
            self._rng.sample(range(self._config.managed_tokens_per_req), size)
            for _ in range(self._config.req_num)
        ]

    def _make_zipf_cdf(self) -> list[float]:
        if self._config.workload != "zipf":
            return []

        cumulative = 0.0
        cdf: list[float] = []
        for rank in range(1, self._config.managed_tokens_per_req + 1):
            cumulative += 1.0 / (rank**self._config.zipf_exponent)
            cdf.append(cumulative)
        return cdf

    def _uniform_tokens(self) -> list[int]:
        return self._rng.sample(
            range(self._config.managed_tokens_per_req), TOPK_PER_REQ
        )

    def _hotset_tokens(self, req_id: int) -> list[int]:
        hotset = self._hotsets[req_id]
        if self._config.hotset_prob >= 1.0 and len(hotset) >= TOPK_PER_REQ:
            return self._rng.sample(hotset, TOPK_PER_REQ)

        tokens: set[int] = set()
        max_attempts = TOPK_PER_REQ * 100
        for _ in range(max_attempts):
            if len(tokens) == TOPK_PER_REQ:
                break
            if self._rng.random() < self._config.hotset_prob:
                tokens.add(self._rng.choice(hotset))
            else:
                tokens.add(self._rng.randrange(self._config.managed_tokens_per_req))

        self._fill_unique_tokens(tokens)
        return list(tokens)

    def _zipf_tokens(self) -> list[int]:
        if not self._zipf_cdf:
            return self._uniform_tokens()

        tokens: set[int] = set()
        total_weight = self._zipf_cdf[-1]
        max_attempts = TOPK_PER_REQ * 200
        for _ in range(max_attempts):
            if len(tokens) == TOPK_PER_REQ:
                break
            sample = self._rng.random() * total_weight
            token_id = bisect.bisect_left(self._zipf_cdf, sample)
            tokens.add(token_id)

        self._fill_unique_tokens(tokens)
        return list(tokens)

    def _sliding_tokens(self, req_id: int, step: int) -> list[int]:
        window_size = max(TOPK_PER_REQ, min(self._config.hotset_size,
                                           self._config.managed_tokens_per_req))
        domain = self._config.managed_tokens_per_req
        base = (req_id * window_size + step * TOPK_PER_REQ) % domain
        window = [(base + offset) % domain for offset in range(window_size)]
        return self._rng.sample(window, TOPK_PER_REQ)

    def _fill_unique_tokens(self, tokens: set[int]) -> None:
        while len(tokens) < TOPK_PER_REQ:
            tokens.add(self._rng.randrange(self._config.managed_tokens_per_req))


def run_simulation(config: SimulationConfig) -> SimulationResult:
    validate_config(config)

    cache = LRUManagedCache(
        total_slots=config.total_slots,
        free_slot_target=config.free_slot_target,
    )
    workload = TopKWorkload(config)
    steps: list[StepStats] = []
    total = StepStats(step=-1)

    for step_id in range(config.steps):
        step_stats = StepStats(
            step=step_id,
            cpu_evictions=cache.prepare_free_slots(),
        )

        for req_id in range(config.req_num):
            step_stats.add(cache.resolve_query(workload.query(req_id, step_id)))

        step_stats.resident_count = cache.resident_count
        step_stats.free_slots = cache.free_slots
        total.add(step_stats)
        total.cpu_evictions += step_stats.cpu_evictions
        steps.append(step_stats)

    total.resident_count = cache.resident_count
    total.free_slots = cache.free_slots
    return SimulationResult(config=config, steps=steps, total=total)


def validate_config(config: SimulationConfig) -> None:
    positive_fields = {
        "req_num": config.req_num,
        "steps": config.steps,
        "managed_tokens_per_req": config.managed_tokens_per_req,
        "hbm_slots_per_req": config.hbm_slots_per_req,
        "hotset_size": config.hotset_size,
    }
    for name, value in positive_fields.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    if config.managed_tokens_per_req < TOPK_PER_REQ:
        raise ValueError(
            f"managed_tokens_per_req must be at least fixed topK {TOPK_PER_REQ}"
        )
    if config.free_slots_target_per_req is not None:
        if config.free_slots_target_per_req < 0:
            raise ValueError("free_slots_target_per_req must be non-negative")
    if not 0.0 <= config.hotset_prob <= 1.0:
        raise ValueError("hotset_prob must be in [0, 1]")
    if config.zipf_exponent <= 0:
        raise ValueError("zipf_exponent must be positive")
    if config.workload not in {"uniform", "hotset", "zipf", "sliding"}:
        raise ValueError("workload must be one of: uniform, hotset, zipf, sliding")


def print_result(result: SimulationResult, csv: bool = False) -> None:
    if csv:
        _print_csv(result)
        return

    config = result.config
    print("HBM KVCache eviction simulation")
    print("--------------------------------")
    print(f"req_num                 : {config.req_num}")
    print(f"steps                   : {config.steps}")
    print(f"topk_per_req            : {TOPK_PER_REQ}")
    print(f"managed_tokens_per_req  : {config.managed_tokens_per_req}")
    print(f"hbm_slots_total         : {config.total_slots}")
    print(f"hbm_slots_per_req       : {config.hbm_slots_per_req}")
    print(f"free_slot_target_total  : {config.free_slot_target}")
    print(f"workload                : {config.workload}")
    print(f"seed                    : {config.seed}")
    print()
    print(
        "step  queries    hits  misses  hit_rate  installs  shortages  "
        "cpu_evicts  resident  free"
    )
    for step in result.steps:
        print(
            f"{step.step:>4}  {step.queries:>7}  {step.hits:>6}  "
            f"{step.misses:>6}  {step.hit_rate:>8.2%}  "
            f"{step.installs:>8}  {step.shortages:>9}  "
            f"{step.cpu_evictions:>10}  {step.resident_count:>8}  "
            f"{step.free_slots:>4}"
        )

    print()
    print(
        f"total: queries={result.total.queries} hits={result.total.hits} "
        f"misses={result.total.misses} hit_rate={result.total_hit_rate:.2%} "
        f"installs={result.total.installs} shortages={result.total.shortages} "
        f"cpu_evictions={result.total.cpu_evictions}"
    )


def _print_csv(result: SimulationResult) -> None:
    print(
        "step,queries,hits,misses,hit_rate,installs,shortages,"
        "cpu_evictions,resident_count,free_slots"
    )
    for step in result.steps:
        print(
            f"{step.step},{step.queries},{step.hits},{step.misses},"
            f"{step.hit_rate:.6f},{step.installs},{step.shortages},"
            f"{step.cpu_evictions},{step.resident_count},{step.free_slots}"
        )
    total = result.total
    print(
        f"total,{total.queries},{total.hits},{total.misses},"
        f"{total.hit_rate:.6f},{total.installs},{total.shortages},"
        f"{total.cpu_evictions},{total.resident_count},{total.free_slots}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate HBM KVCache hit rate for fixed 2048-token topK queries "
            "under CPU step-boundary LRU managed eviction."
        )
    )
    parser.add_argument("--req-num", type=int, default=4,
                        help="Number of requests in one simulated decode step.")
    parser.add_argument("--steps", type=int, default=32,
                        help="Number of decode steps to simulate.")
    parser.add_argument("--managed-tokens-per-req", type=int,
                        default=DEFAULT_MANAGED_TOKENS_PER_REQ,
                        help="Managed historical token domain size per req.")
    parser.add_argument("--hbm-slots-per-req", type=int,
                        default=DEFAULT_HBM_SLOTS_PER_REQ,
                        help="Managed HBM token pair slots budget per req.")
    parser.add_argument(
        "--free-slots-target-per-req",
        type=int,
        default=TOPK_PER_REQ,
        help=(
            "CPU step-boundary free slot target per req. Set 0 to keep all "
            "resident tokens until capacity is full."
        ),
    )
    parser.add_argument("--workload", choices=("uniform", "hotset", "zipf", "sliding"),
                        default="hotset", help="TopK token distribution.")
    parser.add_argument("--hotset-size", type=int, default=8192,
                        help="Hot token set size per req for hotset/sliding workloads.")
    parser.add_argument("--hotset-prob", type=float, default=0.9,
                        help="Probability of drawing from the hotset workload.")
    parser.add_argument("--zipf-exponent", type=float, default=1.1,
                        help="Zipf exponent for the zipf workload.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for deterministic simulation.")
    parser.add_argument("--csv", action="store_true",
                        help="Print per-step stats as CSV.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = SimulationConfig(
        req_num=args.req_num,
        steps=args.steps,
        managed_tokens_per_req=args.managed_tokens_per_req,
        hbm_slots_per_req=args.hbm_slots_per_req,
        free_slots_target_per_req=args.free_slots_target_per_req,
        workload=args.workload,
        hotset_size=args.hotset_size,
        hotset_prob=args.hotset_prob,
        zipf_exponent=args.zipf_exponent,
        seed=args.seed,
    )

    try:
        result = run_simulation(config)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    print_result(result, csv=args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
