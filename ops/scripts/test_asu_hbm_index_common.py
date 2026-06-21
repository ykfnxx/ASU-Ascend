from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from asu_hbm_index_common import (  # noqa: E402
    FREE_SLOT_COUNT,
    QUERY_COUNT,
    RESIDENT_SLOT_COUNT,
    make_chained_maintain_case,
    make_maintain_case,
)


class MaintainCaseTest(unittest.TestCase):
    def test_evict_ratio_controls_free_head_per_req(self) -> None:
        case = make_maintain_case(req_num=2, evict_ratio=0.25)
        evict_slots = FREE_SLOT_COUNT // 4

        self.assertEqual(case.evict_slots, evict_slots)
        self.assertEqual(case.last_query_slots.shape, (2, QUERY_COUNT))
        self.assertTrue((case.free_head == evict_slots).all())

        allocated_slots = range(RESIDENT_SLOT_COUNT, RESIDENT_SLOT_COUNT + evict_slots)
        for req_id in range(2):
            for offset, slot in enumerate(allocated_slots):
                index_id = 20_000 + offset
                self.assertEqual(case.index[req_id, index_id], slot)
                self.assertEqual(case.slot_to_index[req_id, slot], index_id)

    def test_chained_maintain_case_expands_req_dimension(self) -> None:
        case = make_chained_maintain_case(req_num=2, evict_ratio=0.5, chain_iters=3)

        self.assertEqual(case.index.shape[0], 6)
        self.assertEqual(case.slot_to_index.shape[0], 6)
        self.assertEqual(case.free_slots.shape[0], 6)
        self.assertEqual(case.free_head.shape[0], 6)
        self.assertEqual(case.last_query_slots.shape[0], 6)
        self.assertEqual(case.evict_slots, FREE_SLOT_COUNT // 2)


if __name__ == "__main__":
    unittest.main()
