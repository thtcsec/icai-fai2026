"""Verify frozen processed window archives used by the paper tables."""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(HERE, "processed")

EXPECTED = {
    "insdn_windows.npz": {
        "sha256": "0f69729684c2150bec93b4b96d9b1396bb860e54a9e9006cc05455273117e273",
        "X_shape": [49991, 10, 10],
        "attack": 40043,
        "normal": 9948,
    },
    "cic_windows.npz": {
        "sha256": "92dc7b8181e1caa8b75aa0d235158869c4ff854bfe5e81ac434bf3a633546b47",
        "X_shape": [79991, 10, 10],
        "attack": 24465,
        "normal": 55526,
    },
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ok = True
    report = {}
    for name, exp in EXPECTED.items():
        path = os.path.join(PROCESSED, name)
        if not os.path.isfile(path):
            print(f"[FAIL] missing {path}")
            ok = False
            continue
        digest = sha256_file(path)
        data = np.load(path)
        X, y = data["X"], data["y"]
        attack = int((y > 0).sum())
        normal = int((y == 0).sum())
        entry = {
            "sha256": digest,
            "X_shape": list(X.shape),
            "attack": attack,
            "normal": normal,
        }
        report[name] = entry
        checks = [
            ("sha256", digest == exp["sha256"]),
            ("shape", list(X.shape) == exp["X_shape"]),
            ("attack", attack == exp["attack"]),
            ("normal", normal == exp["normal"]),
        ]
        for label, passed in checks:
            status = "OK" if passed else "FAIL"
            if not passed:
                ok = False
            print(f"[{status}] {name} {label}: got={entry.get(label, (attack if label=='attack' else normal if label=='normal' else digest if label=='sha256' else list(X.shape)))} expected={exp[label if label!='shape' else 'X_shape']}")
    out = os.path.join(PROCESSED, "archive_manifest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"expected": EXPECTED, "observed": report}, f, indent=2)
    print(f"[+] Wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
