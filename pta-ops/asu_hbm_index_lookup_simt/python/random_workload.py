"""Reproducible random workloads for the SIMT lookup operator."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


INDEX_SIZE = 128 * 1024
SLOT_COUNT = 10 * 1024
QUERY_COUNT = 2 * 1024
NOT_FOUND = -1


@dataclass
class RandomLookupCase:
    token_to_slot: Any
    slot_to_token: Any
    lru_slots: Any
    query_token_ids: Any
    hit_count: int
    miss_count: int
    seed: int
    case_id: int


def validate_hit_count(hit_count: int) -> None:
    if hit_count < 0 or hit_count > QUERY_COUNT:
        raise ValueError(
            f"hit_count must be in [0, {QUERY_COUNT}]; got {hit_count}"
        )


def row_seed(seed: int, case_id: int, req_id: int) -> int:
    """Mix user seed, round id, and request id into a stable Python RNG seed."""

    return (
        (seed & 0xFFFFFFFFFFFFFFFF)
        ^ ((case_id + 1) * 0x9E3779B185EBCA87)
        ^ ((req_id + 1) * 0xC2B2AE3D27D4EB4F)
    ) & 0xFFFFFFFFFFFFFFFF


def make_random_query_row(
    hit_count: int,
    *,
    seed: int,
    case_id: int,
    req_id: int,
) -> list[int]:
    """Build one unique 2K query with exact hits and randomly placed misses."""

    validate_hit_count(hit_count)
    rng = random.Random(row_seed(seed, case_id, req_id))
    miss_count = QUERY_COUNT - hit_count
    hits = rng.sample(range(SLOT_COUNT), hit_count)
    misses = rng.sample(range(SLOT_COUNT, INDEX_SIZE), miss_count)
    query = hits + misses
    rng.shuffle(query)
    return query


def make_random_case(
    np: Any,
    req_num: int,
    hit_count: int,
    *,
    seed: int,
    case_id: int,
) -> RandomLookupCase:
    """Create a full-resident state and randomized exact-hit-count queries."""

    if req_num <= 0:
        raise ValueError("req_num must be positive")
    validate_hit_count(hit_count)

    token_to_slot = np.full(
        (req_num, INDEX_SIZE), NOT_FOUND, dtype=np.int32
    )
    slot_to_token = np.empty((req_num, SLOT_COUNT), dtype=np.int32)
    lru_slots = np.empty((req_num, SLOT_COUNT), dtype=np.int16)
    query_token_ids = np.empty((req_num, QUERY_COUNT), dtype=np.int32)
    resident_tokens = np.arange(SLOT_COUNT, dtype=np.int32)
    resident_slots_i32 = np.arange(SLOT_COUNT, dtype=np.int32)
    resident_slots_i16 = np.arange(SLOT_COUNT, dtype=np.int16)

    for req_id in range(req_num):
        token_to_slot[req_id, resident_tokens] = resident_slots_i32
        slot_to_token[req_id] = resident_tokens
        lru_slots[req_id] = resident_slots_i16
        query_token_ids[req_id] = make_random_query_row(
            hit_count,
            seed=seed,
            case_id=case_id,
            req_id=req_id,
        )

    return RandomLookupCase(
        token_to_slot=token_to_slot,
        slot_to_token=slot_to_token,
        lru_slots=lru_slots,
        query_token_ids=query_token_ids,
        hit_count=hit_count,
        miss_count=QUERY_COUNT - hit_count,
        seed=seed,
        case_id=case_id,
    )
