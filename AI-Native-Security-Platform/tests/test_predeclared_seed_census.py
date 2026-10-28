"""Fixed-list census does not establish advance declaration; split stays one attack group."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
J = REPO / "revisions" / "icai2026_20261028" / "predeclared_seed_support.json"


def test_census_exists_and_limits_chronology_claims():
    assert J.is_file()
    d = json.loads(J.read_text(encoding="utf-8"))
    assert d["retrained"] is False
    assert d["advance_declaration_verified"] is False
    assert d["repeated_model_performance_experiment"] is False
    assert "cherry_picked" not in d
    assert d["seeds_fixed_list"][0] == 42
    assert d["executed_test_attack_groups"] == 1
    assert d["decision"] == "single_split_case_study"
    assert d["n_meeting_floor"] == 3
    assert d["n_seeds"] == 10
