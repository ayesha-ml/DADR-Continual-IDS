# DADR-Continual-IDS

**Drift-Aware Dual-Pool Replay (DADR)** for continual network intrusion detection.

## Overview
Standard continual learning intrusion detection models store past network traffic in a single memory buffer. When normal user behavior evolves over time (concept/virtual drift), outdated normal traffic in memory causes high false alarms. DADR solves this by splitting fixed replay memory into two pools:
- **Protected Attack Pool:** Retains historical attack examples without resetting on normal traffic shifts.
- **Drift-Aware Normal Pool:** Monitors incoming normal traffic via Population Stability Index (PSI) and evicts outdated normal samples using Mahalanobis distance.

## Project Structure
- `src/` — Core modules (`config.py`, `data_loader.py`, `dadr_buffer.py`, `baselines.py`, `model.py`, `metrics.py`)
- `scripts/` — Experiment runners, ablations, and plotting scripts
- `data/` — Raw and processed dataset directory (ignored by Git)
- `results/` — Experiment logs and generated figures

## Team
- **Ayesha Amer** (23I-2503) & **Eesha Emaan** (23I-2501) — Section DS-7C
