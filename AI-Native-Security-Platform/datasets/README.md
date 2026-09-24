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

## How the 50,000 InSDN source rows were chosen

Public InSDN has on the order of **343,939** flow instances. The frozen
`insdn_windows.npz` was produced by the offline prepare path that originally
built this archive (byte-identical copy; SHA-256 above), with defaults:

1. Load the public InSDN flow CSV(s).
2. If raw rows exceed the cap, take a **label-stratified subsample of
   `max_rows=50_000` with `seed=42`** (proportions preserved across label
   values; final class is residual-filled, then the subsample is shuffled with
   the same seed).
3. Map columns onto the 10 `FEATURE_COLUMNS` aggregates.
4. Shuffle the canonical frame again with `seed=42`, then keep
   `head(min(n, 50_000))` → exactly **50,000** source rows.
5. Contiguous sliding windows: `T=10`, stride `1` → **49,991** windows
   (`50000 - 10 + 1`), labeled attack if any flow in the window is attack.

`datasets/build_windows.py` documents **step 5 only** (CSV → windows). Bit-
identical reconstruction of the frozen NPZ additionally requires the same
50k stratified subsample and feature map used when the archive was frozen;
the SHA-256 digest is the ground truth for this artifact.

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
* Binary labels: any attack class → `1`, else `0`.
* Source-row ordering after the stratified subsample is seed-42 shuffled
  (not raw CSV order); the leakage protocol still treats the resulting window
  index order as a temporal proxy for blocking/purging.

Paper experiments then apply `train_utils.blocked_split_npz` (20 blocks, purge 9, stratified 14k/6k, train-only scaler). Cross-dataset uses a 20k stratified CIC holdout drawn from `cic_windows.npz`.

## What is *not* claimed

* `evaluation/datasets/generator.py` is a **synthetic** telemetry helper for demos — it is **not** the InSDN/CIC preprocessor.
* Exact vendor column maps from every CIC daily CSV variant into the 10 features are approximate aliases in `build_windows.py`; bit-identical reproduction of the frozen NPZ requires the same intermediate feature export used when the archives were frozen. The SHA-256 table above is the ground truth for this artifact.
