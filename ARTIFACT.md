# Artifact evaluation guide

## Scope and status

ICAI-2026 Paper 26, Submitted / Under Review, not accepted. Current title and authors are in [CITATION.cff](CITATION.cff). The scientific PDF is frozen at SHA-256 `ccba97053f3c93ba27e7a8cf99ce422246ea63f698b835092f7d10a547f2b83d`, five A4 pages. It stays in the local author packet pending CMT replacement and public-sharing verification. No revised PDF/TeX is bundled in public Git.

[Public revision evidence](revisions/icai2026_20261028/PUBLIC_ARTIFACT.md) contains portable copies of aggregate result JSONs. Only machine-specific paths are normalized; scientific values are unchanged. Owned model checkpoints are listed in the evidence checksums. Dataset payloads and row-level samples are excluded from the new release tree. The historical tag and all local protected files remain intact.

## Level A — portable consistency and fixture tests

From the repository root:

```bash
python tools/verify_public_artifact.py
python -m venv .venv
# Windows: .venv\Scripts\activate
python -m pip install -r AI-Native-Security-Platform/requirements-lock.txt
python -m pip install -r requirements-test.txt
python tools/run_public_checks.py
```

The standard-library verifier checks selected stored numbers, metadata and file hashes. It does not replay predictions, validate training history or replace the independent scientific audit. Fixture tests do not require raw data, Docker or Redis. Dataset-dependent tests skip when local archives are missing. The original 62-test local closure suite additionally needs the withheld paper packet and local datasets; it is not advertised as a complete clean-checkout replay.

The dependency lock describes detector verification on Windows/Python 3.11.9, Torch 2.13.0+cpu, NumPy 2.4.6, sklearn 1.9.0 and LightGBM 4.7.0. Historical repaired no-Redis timing instead records Python 3.13.5/Torch 2.8.0+cpu. Past training execution environments lack immutable traces.

## Level B — optional dataset-dependent reproduction

Acquire the exact third-party derivative described in [dataset documentation](AI-Native-Security-Platform/datasets/README.md), check its CSV hash, and keep it outside Git. Existing builders accept explicit source/output arguments through their Python functions, avoiding machine-specific default locations:

```python
from pathlib import Path
import sys
app = Path('AI-Native-Security-Platform').resolve()
sys.path[:0] = [str(app), str(app / 'results')]
from build_chrono_insdn import main as build_p1
from build_phase3_splits import build_group_disjoint
source = str(Path('external-data/Dataset.csv').resolve())
build_p1(src=source, out_dir=str(app / 'results/corrected/chrono_insdn'))
build_group_disjoint(src=source, out_dir=str(app / 'results/corrected/phase3_group_disjoint'))
```

These are optional output-writing rebuilds, not commands run during release preparation. Do not overwrite local frozen arrays. Put exploratory reproductions in a separate checkout/output location. With appropriately prepared arrays, the original runners are `results/train_eval_chrono.py` and `results/train_eval_phase3.py` inside the nested platform. They retrain and can be expensive; they are not quick checks. Saved detector files are FP32 state dictionaries, dynamically quantized at inference. Whole-tensor validation/test sizes are P1 3,000/6,000 and P3 3,000/955.

To use the existing locally retained full regression packet, run pytest over the nested tests and `revisions/icai2026_20261028/test_p1_remediation.py` with `--import-mode=importlib`. Those private/local files are not silently fabricated for a public clone.

## Interpretation

P1 and P3 differ in test population/prevalence and trained weights. Their F1 difference is not a causal leakage effect. P3 test has one attack host-day; 32 positives are overlapping windows. Scores are uncalibrated and batch-sensitive. The ten-seed census reports support only. Policy results are simulator-only one-step Q-regression. Corrected Redis-inclusive timing, received-event-to-policy integration and enforcement are unmeasured/unverified.

## Packaging and history

The release manifest lists included and excluded files; checksums identify the owned models and aggregate evidence. Local dataset archives and protected historical tables are preserved. GitHub's automatic source archives use the tagged tree, so historical dataset payloads and a row-level telemetry sample are removed from the new tree's index without deleting their local copies or rewriting the original submission tag. Historical notebooks with embedded outputs are also retained locally rather than repackaged.

[Release notes](release/RELEASE_NOTES.md) describe the revision. The repository owner authorized the [code-only pre-release](https://github.com/thtcsec/icai-fai2026/releases/tag/icai2026-revision-20261028-v2). Manuscript distribution remains pending CMT/public-sharing verification. Zenodo JSON is unpublished preparation; no DOI or acceptance is claimed.
