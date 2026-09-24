# Dataset provenance (ICAI-2026 artifact)

Authoritative paper tables load **frozen** window archives:

| Archive | Shape `X` | Attack / Normal | SHA-256 |
| :--- | :--- | :--- | :--- |
| `processed/insdn_windows.npz` | `(49991, 10, 10)` | 40043 / 9948 | `0f69729684c2150bec93b4b96d9b1396bb860e54a9e9006cc05455273117e273` |
| `processed/cic_windows.npz` | `(79991, 10, 10)` | 24465 / 55526 | `92dc7b8181e1caa8b75aa0d235158869c4ff854bfe5e81ac434bf3a633546b47` |

Verify locally:

```bash
python datasets/verify_processed_archives.py
```

## Sources

* **InSDN** — Elsayed et al., *IEEE Access* 2020 (public SDN intrusion flows).
* **CIC-IDS2017** — Sharafaldin et al., ICISSP 2018 (public intrusion flows).

Raw CSVs are **not** redistributed here (size / license). Reviewers who already hold the public releases can rebuild windows with:

```bash
python datasets/build_windows.py --csv /path/to/standardized_flows.csv \
  --out datasets/processed/rebuild.npz --seq-len 10 --stride 1
```

## How the 50,000 InSDN source rows were chosen (frozen archive)

Public InSDN has on the order of **343,939** flow instances. The frozen
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
