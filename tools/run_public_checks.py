"""Run the public fixture subset; data-dependent checks may explicitly skip."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SUITE = [
    "test_temporal_windows.py", "test_threshold_leakage.py",
    "test_chrono_insdn_builder.py", "test_phase3_group_disjoint.py",
    "test_identity_soar.py", "test_dqn_checkpoint.py",
    "test_dqn_train_eval_split.py", "test_detector_policy_contract.py",
    "test_readme_assets.py",
]


def main():
    os.chdir(ROOT)
    subprocess.run([sys.executable, "-B", "tools/verify_public_artifact.py"], check=True)
    args = [sys.executable, "-B", "-m", "pytest", "--import-mode=importlib", "-o", "addopts=", "-q", "-p", "no:cacheprovider", "-k", "not test_root_originals_preserved"]
    args += [f"AI-Native-Security-Platform/tests/{name}" for name in SUITE]
    return subprocess.call(args)


if __name__ == "__main__":
    raise SystemExit(main())
