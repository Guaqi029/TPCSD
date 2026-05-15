# Repository Guidelines

## Project Structure & Module Organization
Core training entry points live at the repository root: `train_stage1.py`, `train_stage2.py`, `eval_last_stage1.py`, and analysis tools such as `analyze_last_stage2.py` and `visualize_embeddings.py`.

Reusable code is organized by responsibility:
- `models/`: backbones, heads, and model utilities.
- `data/`: dataset loading and transforms.
- `utils/`: losses, metrics, checkpointing, and CSV helpers.
- `config/configs.yaml`: default hyperparameters and data/log/checkpoint paths.
- `scripts/`: recommended run scripts for ISIC2019LT and ISIC_Archive workflows.
- `prepare_datasets/`: dataset download/split/build helpers.

## Build, Test, and Development Commands
Use the shell scripts as the primary interface (they encode stable experiment settings).

- `bash scripts/run_stage1_isic2019lt.sh`: train Stage 1 on ISIC2019LT.
- `bash scripts/run_stage2_isic2019lt.sh`: run Stage 2 using saved Stage 1 checkpoints.
- `bash scripts/run_stage1_isic_archive.sh` / `run_stage2_isic_archive.sh`: same flow for ISIC_Archive.
- `bash scripts/eval_last_stage1_isic2019lt.sh`: evaluate the latest Stage 1 checkpoint.
- `bash scripts/analyze_last_stage2_isic2019lt.sh`: analyze Stage 2 outputs.

## Coding Style & Naming Conventions
Follow existing Python style in this repo:
- 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes.
- Keep modules focused (data logic in `data/`, losses in `utils/losses.py`, model pieces in `models/`).
- Prefer explicit argparse flags in root scripts; keep defaults aligned with `config/configs.yaml`.
- Keep comments short and only where logic is non-obvious.

## Testing Guidelines
There is no dedicated `pytest` suite yet. Treat evaluation scripts as regression checks:
- Run Stage 1/Stage 2 eval scripts after model or loss changes.
- Compare key metrics (ACC/F1/AUC/BACC) against prior runs before merging.
- If you add non-trivial logic, add a small reproducible check script under `scripts/`.

## Commit & Pull Request Guidelines
Current history uses short messages; prefer clear, imperative subjects with scope.
- Good format: `stage2: fix covariance jitter for virtual sampling`.
- Keep commits focused (one concern per commit).

For PRs, include:
- What changed and why.
- Exact commands run (train/eval/analyze).
- Metric deltas and affected dataset split.
- Paths to produced artifacts (for example `checkpoints/<run_name>` and `log/tpcsd`).

## Security & Configuration Tips
`config/configs.yaml` contains absolute dataset paths that may be machine-specific. Do not commit private/local credentials or sensitive filesystem paths; prefer environment-specific overrides in run scripts.
