from __future__ import annotations

import sys
import unittest
from pathlib import Path


PKG_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_DIR))

from python.lookup_lru_reference import (  # noqa: E402
    NOT_FOUND,
    LookupState,
    lookup_allocate_evict,
)
from python.random_workload import (  # noqa: E402
    QUERY_COUNT,
    SLOT_COUNT,
    make_random_query_row,
    validate_hit_count,
)


def make_state(slot_tokens: list[int], token_capacity: int = 32) -> LookupState:
    token_to_slot = [NOT_FOUND] * token_capacity
    for slot, token in enumerate(slot_tokens):
        if token >= 0:
            token_to_slot[token] = slot
    return LookupState(
        token_to_slot=token_to_slot,
        slot_to_token=list(slot_tokens),
        lru_slots=list(range(len(slot_tokens))),
    )


class LookupLruReferenceTest(unittest.TestCase):
    def test_hits_move_to_mru_in_prior_lru_order(self):
        state = make_state([0, 1, 2, 3])

        slot_ids, miss_mask = lookup_allocate_evict([2, 0], state)

        self.assertEqual(slot_ids, [2, 0])
        self.assertEqual(miss_mask, [False, False])
        self.assertEqual(state.lru_slots, [1, 3, 0, 2])
        self.assertEqual(state.slot_to_token, [0, 1, 2, 3])

    def test_misses_evict_oldest_non_hit_slots_and_clear_reverse_state(self):
        state = make_state([1, 4, 2, 5])

        slot_ids, miss_mask = lookup_allocate_evict([4, 6, 7], state)

        self.assertEqual(slot_ids, [1, 0, 2])
        self.assertEqual(miss_mask, [False, True, True])
        self.assertEqual(state.slot_to_token, [6, 4, 7, 5])
        self.assertEqual(state.token_to_slot[1], NOT_FOUND)
        self.assertEqual(state.token_to_slot[2], NOT_FOUND)
        self.assertEqual(state.token_to_slot[6], 0)
        self.assertEqual(state.token_to_slot[7], 2)
        self.assertEqual(state.lru_slots, [3, 0, 2, 1])

    def test_duplicate_miss_allocates_once_and_invalid_token_is_ignored(self):
        state = make_state([0, 1, 2, 3])

        slot_ids, miss_mask = lookup_allocate_evict([4, 4, -1, 1], state)

        self.assertEqual(slot_ids, [0, 0, NOT_FOUND, 1])
        self.assertEqual(miss_mask, [True, False, False, False])
        self.assertEqual(state.slot_to_token, [4, 1, 2, 3])
        self.assertEqual(state.lru_slots, [2, 3, 0, 1])

    def test_empty_slots_are_used_before_residents_when_lru_places_them_first(self):
        state = make_state([NOT_FOUND, NOT_FOUND, 2, 3])

        slot_ids, miss_mask = lookup_allocate_evict([4, 3], state)

        self.assertEqual(slot_ids, [0, 3])
        self.assertEqual(miss_mask, [True, False])
        self.assertEqual(state.slot_to_token, [4, NOT_FOUND, 2, 3])
        self.assertEqual(state.token_to_slot[2], 2)
        self.assertEqual(state.lru_slots, [1, 2, 0, 3])

    def test_multiple_steps_keep_bidirectional_mapping_consistent(self):
        state = make_state([0, 1, 2, 3])
        lookup_allocate_evict([4, 1], state)
        slot_ids, miss_mask = lookup_allocate_evict([5, 4], state)

        self.assertEqual(slot_ids, [2, 0])
        self.assertEqual(miss_mask, [True, False])
        for slot, token in enumerate(state.slot_to_token):
            if token >= 0:
                self.assertEqual(state.token_to_slot[token], slot)


class RandomWorkloadTest(unittest.TestCase):
    def test_query_has_exact_hit_count_and_randomly_interleaved_misses(self):
        hit_count = 1536
        query = make_random_query_row(
            hit_count, seed=17, case_id=3, req_id=1
        )

        self.assertEqual(len(query), QUERY_COUNT)
        self.assertEqual(len(set(query)), QUERY_COUNT)
        self.assertEqual(sum(token < SLOT_COUNT for token in query), hit_count)
        miss_positions = [
            pos for pos, token in enumerate(query) if token >= SLOT_COUNT
        ]
        self.assertNotEqual(
            miss_positions,
            list(range(hit_count, QUERY_COUNT)),
        )

    def test_query_generation_is_reproducible_and_changes_by_case(self):
        first = make_random_query_row(
            1024, seed=23, case_id=5, req_id=0
        )
        repeated = make_random_query_row(
            1024, seed=23, case_id=5, req_id=0
        )
        next_case = make_random_query_row(
            1024, seed=23, case_id=6, req_id=0
        )

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, next_case)

    def test_all_hit_and_all_miss_boundaries(self):
        all_hits = make_random_query_row(
            QUERY_COUNT, seed=1, case_id=0, req_id=0
        )
        all_misses = make_random_query_row(
            0, seed=1, case_id=0, req_id=0
        )

        self.assertTrue(all(token < SLOT_COUNT for token in all_hits))
        self.assertTrue(all(token >= SLOT_COUNT for token in all_misses))

    def test_hit_count_outside_query_width_is_rejected(self):
        for hit_count in (-1, QUERY_COUNT + 1):
            with self.subTest(hit_count=hit_count):
                with self.assertRaises(ValueError):
                    validate_hit_count(hit_count)


if __name__ == "__main__":
    unittest.main()
