"""Package only a verified Git tree; never walk the private local workspace."""
import argparse
from pathlib import Path
import subprocess


def package_project(output_zip="icai-fai2026-bundle.zip", ref="icai2026-revision-20261028"):
    root = Path(__file__).resolve().parent
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", ref], cwd=root, text=True).splitlines()
    for name in names:
        p = Path(name)
        if p.suffix.lower() in {".npz", ".rar", ".7z", ".ipynb"} or p.name.startswith(".env"):
            raise RuntimeError(f"Refusing source tree containing excluded payload: {name}")
        if name.startswith("revisions/icai2026_20261028/paper/") or "insdn_telemetry_sample.csv" in name:
            raise RuntimeError(f"Refusing withheld paper/data payload: {name}")
    output = (root / output_zip).resolve()
    if output.exists():
        raise FileExistsError(output)
    subprocess.run(["git", "archive", "--format=zip", "-o", str(output), ref], cwd=root, check=True)
    print(f"Packaged {len(names)} tracked files from {ref}; no untracked files read.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="icai2026-revision-20261028")
    parser.add_argument("--output", default="icai-fai2026-bundle.zip")
    args = parser.parse_args()
    package_project(args.output, args.ref)
