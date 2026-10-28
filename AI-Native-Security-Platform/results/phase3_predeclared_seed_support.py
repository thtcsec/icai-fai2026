"""Hash-only fixed-list P3 support census; no repeated model-performance experiment. Historical filename retained."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# Frozen from metric_integrity.json / Phase 3 assignment (seed 42).
ATTACK_GROUPS = [
    "1/1/2020|192.168.20.134",
    "10/1/2020|192.168.3.130",
    "10/1/2020|200.175.2.130",
    "12/1/2020|172.17.0.2",
    "12/1/2020|200.175.2.130",
    "26/12/2019|172.17.0.1",
    "26/12/2019|172.17.0.2",
    "26/12/2019|200.175.2.130",
    "27/12/2019|192.168.20.134",
    "31/12/2019|200.175.2.130",
    "9/1/2020|200.175.2.130",
    "1/1/2020|200.175.2.130",
    "25/12/2019|200.175.2.130",
    "27/12/2019|200.175.2.130",
    "31/12/2019|192.168.20.131",
    "9/1/2020|192.168.3.130",
    "31/12/2019|192.168.20.132",
]
assert len(ATTACK_GROUPS) == 17
assert len(set(ATTACK_GROUPS)) == 17

# Fixed list includes the already evaluated seed 42; advance declaration is unverified.
SEEDS = [42, 0, 1, 2, 3, 7, 13, 17, 99, 2026]
MIN_TEST_ATTACK_GROUPS = 3  # "multiple independent attack groups" floor


def assign(gid: str, seed: int) -> str:
    h = hashlib.sha256(f"phase3-gd|{seed}|{gid}".encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def main() -> None:
    rows = []
    for seed in SEEDS:
        buckets = {"train": [], "val": [], "test": []}
        for g in ATTACK_GROUPS:
            buckets[assign(g, seed)].append(g)
        n_tr, n_va, n_te = (len(buckets[k]) for k in ("train", "val", "test"))
        valid = n_te >= MIN_TEST_ATTACK_GROUPS and n_va >= 1 and n_tr >= 1
        rows.append(
            {
                "seed": seed,
                "n_attack_train": n_tr,
                "n_attack_val": n_va,
                "n_attack_test": n_te,
                "test_groups": buckets["test"],
                "meets_multi_attack_test_floor": valid,
                "models_trained": seed == 42,
            }
        )
    n_valid = sum(1 for r in rows if r["meets_multi_attack_test_floor"])
    out = {
        "retrained": False,
        "advance_declaration_verified": False,
        "repeated_model_performance_experiment": False,
        "seeds_fixed_list": SEEDS,
        "min_test_attack_groups": MIN_TEST_ATTACK_GROUPS,
        "n_seeds": len(SEEDS),
        "n_meeting_floor": n_valid,
        "executed_seed": 42,
        "executed_test_attack_groups": 1,
        "decision": (
            "single_split_case_study"
            if rows[0]["n_attack_test"] < MIN_TEST_ATTACK_GROUPS
            else "multi_seed_ok"
        ),
        "rows": rows,
        "note": (
            "70/15/15 hashing of 17 attack host-days often leaves 0–2 test groups. "
            "This census does not evaluate F1 or retrain; prior seed-selection history is unverified. Seed 42 (already trained) has 1 test attack group."
        ),
    }
    dest = Path(__file__).resolve().parents[2] / "revisions" / "icai2026_20261028" / "predeclared_seed_support.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n_seeds", "n_meeting_floor", "decision", "executed_test_attack_groups")}, indent=2))
    for r in rows:
        print(f"seed={r['seed']:4d}  tr/va/te={r['n_attack_train']}/{r['n_attack_val']}/{r['n_attack_test']}  floor={r['meets_multi_attack_test_floor']}")
    print("wrote", dest)


if __name__ == "__main__":
    main()
