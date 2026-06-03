#!/usr/bin/env python3
"""Unit tests for the HBM KVCache eviction simulator."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

import kv_cache_eviction_sim as sim  # noqa: E402


class LRUManagedCacheTest(unittest.TestCase):
    def test_repeated_query_hits_after_first_install(self) -> None:
        cache = sim.LRUManagedCache(total_slots=8, free_slot_target=0)

        first = cache.resolve_query([(0, token) for token in range(4)])
        second = cache.resolve_query([(0, token) for token in range(4)])

        self.assertEqual(first.hits, 0)
        self.assertEqual(first.misses, 4)
        self.assertEqual(first.installs, 4)
        self.assertEqual(second.hits, 4)
        self.assertEqual(second.misses, 0)
        self.assertEqual(second.installs, 0)

    def test_prepare_free_slots_evicts_least_recently_used_tokens(self) -> None:
        cache = sim.LRUManagedCache(total_slots=4, free_slot_target=2)
        cache.resolve_query([(0, token) for token in (1, 2, 3, 4)])
        cache.resolve_query([(0, 2)])

        evicted = cache.prepare_free_slots()

        self.assertEqual(evicted, 2)
        self.assertEqual(cache.resident_count, 2)
        self.assertEqual(cache.free_slots, 2)
        self.assertEqual(cache.resident_tokens, {(0, 2), (0, 4)})

    def test_no_emergency_victim_selection_when_free_slots_run_out(self) -> None:
        cache = sim.LRUManagedCache(total_slots=1, free_slot_target=0)

        stats = cache.resolve_query([(0, 10), (0, 11)])

        self.assertEqual(stats.hits, 0)
        self.assertEqual(stats.misses, 2)
        self.assertEqual(stats.installs, 1)
        self.assertEqual(stats.shortages, 1)
        self.assertEqual(cache.resident_tokens, {(0, 10)})


class SimulationTest(unittest.TestCase):
    def test_simulation_uses_fixed_2048_topk_per_request(self) -> None:
        config = sim.SimulationConfig(
            req_num=3,
            steps=2,
            managed_tokens_per_req=4096,
            hbm_slots_per_req=4096,
            free_slots_target_per_req=0,
            workload="uniform",
            seed=7,
        )

        result = sim.run_simulation(config)

        self.assertEqual(result.total.queries, 3 * 2 * sim.TOPK_PER_REQ)
        for step in result.steps:
            self.assertEqual(step.queries, 3 * sim.TOPK_PER_REQ)

    def test_hotset_workload_reaches_steady_state_hits_after_warmup(self) -> None:
        config = sim.SimulationConfig(
            req_num=1,
            steps=3,
            managed_tokens_per_req=4096,
            hbm_slots_per_req=4096,
            free_slots_target_per_req=0,
            workload="hotset",
            hotset_size=sim.TOPK_PER_REQ,
            hotset_prob=1.0,
            seed=11,
        )

        result = sim.run_simulation(config)

        self.assertEqual(result.steps[0].hits, 0)
        self.assertEqual(result.steps[0].misses, sim.TOPK_PER_REQ)
        self.assertEqual(result.steps[1].hits, sim.TOPK_PER_REQ)
        self.assertEqual(result.steps[2].hits, sim.TOPK_PER_REQ)
        self.assertAlmostEqual(result.total_hit_rate, 2 / 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
