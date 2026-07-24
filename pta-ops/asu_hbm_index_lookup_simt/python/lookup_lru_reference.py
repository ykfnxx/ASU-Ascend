"""Dependency-free oracle for the Ascend 950 SIMT lookup/LRU state machine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import MutableSequence, Sequence


NOT_FOUND = -1


@dataclass
class LookupState:
    """Mutable token/slot mappings and an LRU-to-MRU slot permutation."""

    token_to_slot: MutableSequence[int]
    slot_to_token: MutableSequence[int]
    lru_slots: MutableSequence[int]


def lookup_allocate_evict(
    query_token_ids: Sequence[int],
    state: LookupState,
) -> tuple[list[int], list[bool]]:
    """Resolve queries and apply the HiSparse-style batch LRU approximation.

    Invalid query ids return ``slot_id=-1`` with ``miss_mask=False``. For a
    duplicated missing token, only its first occurrence has ``miss_mask=True``;
    every occurrence returns the same newly allocated slot.
    """

    slot_count = len(state.slot_to_token)
    if len(state.lru_slots) != slot_count:
        raise ValueError("lru_slots length must match slot_to_token")
    if sorted(state.lru_slots) != list(range(slot_count)):
        raise ValueError("lru_slots must be a permutation of all slot ids")

    slot_ids = [NOT_FOUND] * len(query_token_ids)
    miss_mask = [False] * len(query_token_ids)
    hit_slot_set: set[int] = set()
    unique_misses: list[tuple[int, int]] = []
    pending_misses: set[int] = set()

    for pos, token in enumerate(query_token_ids):
        if token < 0 or token >= len(state.token_to_slot):
            continue
        slot = state.token_to_slot[token]
        if slot >= 0:
            if slot >= slot_count:
                raise ValueError(f"token {token} maps outside the slot table")
            slot_ids[pos] = slot
            hit_slot_set.add(slot)
        elif slot == NOT_FOUND and token not in pending_misses:
            pending_misses.add(token)
            unique_misses.append((pos, token))
            miss_mask[pos] = True
        elif slot != NOT_FOUND:
            raise ValueError(f"token {token} has transient/invalid slot {slot}")

    hit_slots = [slot for slot in state.lru_slots if slot in hit_slot_set]
    evictable_slots = [
        slot for slot in state.lru_slots if slot not in hit_slot_set
    ]
    if len(unique_misses) > len(evictable_slots):
        raise ValueError("unique misses exceed evictable slot count")

    for miss_rank, (_, token) in enumerate(unique_misses):
        victim_slot = evictable_slots[miss_rank]
        victim_token = state.slot_to_token[victim_slot]
        if victim_token >= 0:
            if victim_token >= len(state.token_to_slot):
                raise ValueError(
                    f"slot {victim_slot} contains out-of-range token {victim_token}"
                )
            state.token_to_slot[victim_token] = NOT_FOUND
        state.slot_to_token[victim_slot] = token
        state.token_to_slot[token] = victim_slot

    for pos, token in enumerate(query_token_ids):
        if 0 <= token < len(state.token_to_slot):
            slot_ids[pos] = state.token_to_slot[token]

    miss_count = len(unique_misses)
    state.lru_slots[:] = (
        evictable_slots[miss_count:]
        + evictable_slots[:miss_count]
        + hit_slots
    )
    return slot_ids, miss_mask
