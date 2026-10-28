# Dataset provenance (ICAI-2026 artifact)

**New release:** dataset payloads and row-level samples are not bundled. Historical NPZs below remain in the local author packet and frozen submission tag; hashes identify historical evidence, not bundled files. See [ARTIFACT.md](../../ARTIFACT.md) for explicit-path optional reproduction.

Historical original-submission tables load **frozen** window archives:

| Archive | Shape `X` | Attack / Normal | SHA-256 |
| :--- | :--- | :--- | :--- |
| `processed/insdn_windows.npz` | `(49991, 10, 10)` | 40043 / 9948 | `0f69729684c2150bec93b4b96d9b1396bb860e54a9e9006cc05455273117e273` |
| `processed/cic_windows.npz` | `(79991, 10, 10)` | 24465 / 55526 | `92dc7b8181e1caa8b75aa0d235158869c4ff854bfe5e81ac434bf3a633546b47` |

Verify locally:

```bash
python datasets/verify_processed_archives.py
```

## Current P1/P3 derivative and consumed support

The hybrid uses the **third-party InSDN derivative** `Sharukesh/INSDN`, `Dataset.csv`, revision `f59330c8ff0f1b231b382e05fc41b1e8fb0df6b4`, 343,889 rows. SHA-256: `a93d51e1c2debc7246728578f22d84a442f156628be64b8c9bae5ccc5dd88a42`. Source: https://huggingface.co/datasets/Sharukesh/INSDN/tree/f59330c8ff0f1b231b382e05fc41b1e8fb0df6b4 . This is not an authenticated original publisher export. All host-day findings apply only to this extract and binary `target > 0` mapping. Original InSDN citation: Elsayed et al., IEEE Access 2020, DOI 10.1109/ACCESS.2020.3022633.

| P3 partition | Full windows / groups / attack groups | Consumed windows / groups / attack groups |
|---|---|---|
| Train | 127,979 / 108 / 11 | 14,000 / 66 / 9 (7,504 attack windows) |
| Validation | 89,053 / 30 / 5 | 3,000 / 16 / 4 (2,855 attack, 145 normal; 12 normal groups) |
| Test | 955 / 22 / 1 | 955 / 22 / 1 (32 overlapping attack windows, 923 normal) |

Models consume stratified window caps, not all archive groups. These are host-days, not independent incident counts. See `revisions/icai2026_20261028/experiment_manifest.json` and `remediation_supports.json`. Ten actual features are Flow Duration, Flow Pkts/s, Flow Byts/s, Flow IAT Mean, Flow IAT Std, Pkt Len Mean, Tot Fwd Pkts, SYN Flag Cnt, Active Mean, Idle Mean. The following frozen-archive sections describe the **original submission only**, not current P1/P3 inputs or authoritative hybrid tables.

## Historical sources

* **InSDN** — Elsayed et al., *IEEE Access* 2020 (public SDN intrusion flows).
* **CIC-IDS2017** — Sharafaldin et al., ICISSP 2018 (public intrusion flows).

Raw CSVs are **not** redistributed here (size / license). Reviewers who already hold the public releases can rebuild windows with:

```bash
python datasets/build_windows.py --csv /path/to/standardized_flows.csv \
  --out datasets/processed/rebuild.npz --seq-len 10 --stride 1
```

## How the 50,000 InSDN source rows were chosen (frozen archive)

The current derivative has **343,889** rows; this historical archive recipe does not authenticate publisher-file equivalence. The frozen
`insdn_windows.npz` is a byte-identical copy of the offline prepare output
(SHA-256 above). Verified recipe:

1. Load the public InSDN flow CSV(s).
2. **Label-stratified subsample** of `max_rows=50_000` with `seed=42`
   (class proportions preserved; the stratified helper itself shuffles with
   that seed).
3. Map columns onto the 10 `FEATURE_COLUMNS` aggregates.
4. **Shuffle the canonical frame again** with `seed=42`, then
   `head(min(n, 50_000))` → exactly **50,000** source rows.
5. Contiguous sliding windows: `T=10`, stride `1` → **49,991** windows
   (`50000 - 10 + 1`).
6. **Window label = label of the last flow in the window**
   (`y[i] = flow_label[i + T - 1]`), *not* any-attack-in-window.

### Why ~9,948 normal windows is consistent

Under last-flow labeling, window class rates match flow class rates.
The ~80% attack / ~20% normal window split therefore tracks the stratified
InSDN subsample (~80% attack flows), not an any-attack inflation. An
any-attack rule on a shuffled 80%-attack stream would yield almost zero
all-normal windows; the frozen archive does **not** use that rule.

### What this means for “temporal” claims

Because steps 2 and 4 shuffle before windowing, adjacent timesteps inside a
window are **not** guaranteed to be consecutive capture-time flows. The
sequence model sees length-`T` contexts over a shuffled index order. The
paper’s blocked+purged split therefore controls **stride-1 index overlap
leakage** (shared window timesteps across train/test), not wall-clock
session leakage over released CSV order.

`datasets/build_windows.py` implements steps 5–6 only. Bit-identical
reproduction of the frozen NPZ also needs the same 50k stratified+shuffled
subsample and feature map; the SHA-256 digest is ground truth.

## Window construction (matches paper §V-B)

* Sequence length `T = 10`, stride `1` (adjacent windows share 9 timesteps).
* Feature dimension `F = 10` non-DPI aggregates (see `FEATURE_COLUMNS` in `prototype/data/data_loader.py`):
  1. `flow_duration`
  2. `packet_rate`
  3. `byte_rate`
  4. `pkt_iat_mean`
  5. `pkt_iat_std`
  6. `packet_len_mean`
  7. `packet_in_count`
  8. `tcp_flags_syn_count`
  9. `flow_active_ratio`
  10. `rsu_channel_occupancy` (legacy name; maps from Idle Mean / similar)
* Binary labels: last-flow label; any positive class → `1`, else `0`.
* Source-row ordering in the frozen archive is **seed-42 shuffled**, not raw
  CSV order.

Paper experiments then apply `train_utils.blocked_split_npz` (20 blocks, purge 9, stratified 14k/6k, train-only scaler). Cross-dataset uses a 20k stratified CIC holdout drawn from `cic_windows.npz`.

## What is *not* claimed

* `evaluation/datasets/generator.py` is a **synthetic** telemetry helper for demos — it is **not** the InSDN/CIC preprocessor.
* Exact vendor column maps from every CIC daily CSV variant into the 10 features are approximate aliases in `build_windows.py`; bit-identical reproduction of the frozen NPZ requires the same intermediate feature export used when the archives were frozen. The SHA-256 table above is the ground truth for this artifact.
* The frozen windows are **not** authentic capture-time trajectories; rebuilding with preserve-order windowing would be a different corpus and would require re-running the accuracy tables.

## Corrected windowing recipe (post-submission repair)

`datasets/build_windows.py` no longer shuffles. The corrected path:

1. Order flows by timestamp (if present) and optional session/group column. If no timestamp exists, `--assume-csv-order` is required; the script **will not invent timestamps**.
2. Split **flows** into contiguous train/val/test blocks (or whole groups) *before* windowing.
3. Build stride-1 windows of length `T=10` inside each partition. Windows never cross groups or partition cuts.
4. Window label = last-flow label.
5. Only completed **training** windows may be shuffled.

```bash
python datasets/build_windows.py --csv /path/to/standardized_flows.csv \
  --out-dir datasets/processed/corrected --seq-len 10 --stride 1 --assume-csv-order
```

The public InSDN/CIC CSVs are **not** in this artifact, so a capture-order rebuild cannot be executed here. Frozen `processed/*.npz` archives are immutable historical evidence (`temporal_validity=index_order_after_shuffle`).
