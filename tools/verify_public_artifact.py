"""Limited stored-evidence consistency and checksum checks; no model/data replay."""
from pathlib import Path
import hashlib
import json
import math
import re

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "revisions/icai2026_20261028/evidence"


def read(name):
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def main():
    frozen = read("scientific_freeze.json")
    title = frozen["title"]
    assert frozen["scientific_freeze"]["sha256"] == "ccba97053f3c93ba27e7a8cf99ce422246ea63f698b835092f7d10a547f2b83d"
    assert frozen["scientific_freeze"]["included_in_public_git"] is False
    assert frozen["authors"] == ["Hoang Tu Trinh", "Tien Thanh Cao"]
    assert frozen["closure"]["remaining_confirmed_p0_p1"] == 0
    assert title in (ROOT / "README.md").read_text(encoding="utf-8")
    assert title in (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert title in json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))["title"]
    assert (ROOT / "CITATION.cff").read_text(encoding="utf-8") in (ROOT / "AI-Native-Security-Platform/CITATION.cff").read_text(encoding="utf-8")
    integ = read("metric_integrity.json")
    op = integ["int8_test"]["operating_point"]
    assert op["cm"] == dict(tn=457, fp=466, fn=0, tp=32)
    assert math.isclose(op["f1"], 64 / 530, rel_tol=1e-12)
    assert integ["int8_test"]["pr_auc"] == 1.0
    for name in ["p1_model_comparison.json", "p3_model_comparison.json"]:
        for row in read(name)["models"].values():
            c = row.get("confusion_matrix")
            if c:
                denominator = 2 * c["tp"] + c["fp"] + c["fn"]
                assert math.isclose(row["f1"], 2 * c["tp"] / denominator if denominator else 0, abs_tol=1e-12)
    support = read("consumed_support.json")["supports"]
    for split, expected in [("train", (14000, 66, 9)), ("val", (3000, 16, 4)), ("test", (955, 22, 1))]:
        v = support[split]["consumed"]
        assert (v["windows"], v["groups"], v["attack_groups"]) == expected
    provenance = read("experiment_manifest.json")["dataset_provenance"]
    assert provenance["repository"] == "Sharukesh/INSDN"
    assert provenance["sha256"] == "a93d51e1c2debc7246728578f22d84a442f156628be64b8c9bae5ccc5dd88a42"
    census = read("fixed_seed_support.json")
    assert census["advance_declaration_verified"] is False
    assert census["repeated_model_performance_experiment"] is False
    checked = 0
    for rel, expected in read("checksums.json")["files"].items():
        path = ROOT / rel
        assert path.is_file(), f"Missing release payload: {rel}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"Checksum mismatch: {rel}"
        checked += 1
    # GitHub links resolve against the repository, not the local withheld packet.
    for rel in ["README.md", "ARTIFACT.md", "paper/README.md", "revisions/icai2026_20261028/PUBLIC_ARTIFACT.md"]:
        path = ROOT / rel
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", text), f"Machine-specific instruction: {rel}"
        links = re.findall(r"\]\(([^)]+)\)", text) + re.findall(r'src="([^"]+)"', text)
        for link in links:
            if not link.startswith(("https://", "http://", "#", "mailto:")):
                assert (path.parent / link.split("#")[0]).exists(), f"Broken link in {rel}: {link}"
    print(f"LIMITED PUBLIC CONSISTENCY CHECK OK; {checked} payload hashes checked.")
    print("No prediction replay, training-history authentication or independent scientific audit performed by this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
