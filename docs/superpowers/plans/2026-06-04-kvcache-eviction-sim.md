# KVCache Eviction Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a simple Python demo that simulates HBM KVCache hit rate under the current managed-token LRU eviction mechanism.

**Architecture:** Add one CLI script with a small LRU slot-pool simulator and deterministic topK workload generators. Add standard-library unit tests that validate hit/miss accounting, step-boundary LRU eviction, fixed `2048` topK queries per request, and free-slot shortage behavior.

**Tech Stack:** Python standard library only: `argparse`, `dataclasses`, `collections.OrderedDict`, `random`, `unittest`.

---

### Task 1: Unit Tests

**Files:**
- Create: `simu/test_kv_cache_eviction_sim.py`

- [ ] **Step 1: Write failing tests**

Create tests that import `kv_cache_eviction_sim` and verify:

```python
def test_repeated_query_hits_after_first_install(self):
    cache = sim.LRUManagedCache(total_slots=8, free_slot_target=0)
    first = cache.resolve_query([(0, token) for token in range(4)])
    second = cache.resolve_query([(0, token) for token in range(4)])
    self.assertEqual(first.hits, 0)
    self.assertEqual(first.installs, 4)
    self.assertEqual(second.hits, 4)
    self.assertEqual(second.misses, 0)
```

Also test least-recently-used victim order, shortage accounting when no free slots exist, fixed `2048` topK entries per request, and hotset steady-state hit rate.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 simu/test_kv_cache_eviction_sim.py
```

Expected: fails because `kv_cache_eviction_sim` does not exist yet.

### Task 2: Simulation CLI

**Files:**
- Create: `simu/kv_cache_eviction_sim.py`

- [ ] **Step 1: Implement cache and workload model**

Create:

```python
TOPK_PER_REQ = 2048
@dataclass(frozen=True)
class SimulationConfig: ...
@dataclass
class StepStats: ...
class LRUManagedCache: ...
class TopKWorkload: ...
def run_simulation(config: SimulationConfig) -> SimulationResult: ...
```

Model current mechanism:

```text
CPU step boundary:
  evict HBM_CLEAN managed tokens by LRU until free_slot_buffer reaches target.
NPU step:
  HBM hit -> touch token and return resident slot.
  ASU_ONLY miss -> install into a free slot if available.
  no free slot -> count shortage; no emergency victim selection.
```

- [ ] **Step 2: Implement CLI**

Expose:

```bash
python3 simu/kv_cache_eviction_sim.py --req-num 50 --steps 32 --workload hotset
```

Options include `--req-num`, `--steps`, `--managed-tokens-per-req`, `--hbm-slots-per-req`, `--free-slots-target-per-req`, `--workload`, `--hotset-size`, `--hotset-prob`, `--zipf-exponent`, `--seed`, and `--csv`.

### Task 3: Verification

**Files:**
- Verify: `simu/test_kv_cache_eviction_sim.py`
- Verify: `simu/kv_cache_eviction_sim.py`

- [ ] **Step 1: Run unit tests**

Run:

```bash
python3 simu/test_kv_cache_eviction_sim.py
```

Expected: all tests pass.

- [ ] **Step 2: Run CLI smoke test**

Run:

```bash
python3 simu/kv_cache_eviction_sim.py --req-num 2 --steps 3 --workload hotset --managed-tokens-per-req 8192 --hbm-slots-per-req 4096
```

Expected: command prints a config summary, per-step stats, and total HBM hit rate.
