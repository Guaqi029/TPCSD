# TPCSD Stage1 (ResNet + Prototype Contrastive Distillation)

This folder is a standalone Stage1 pipeline (no VAVAE, no Stage2) with:
- ResNet backbone (50/34)
- EMA teacher
- Logit KD + Prototype Contrastive Distillation (PCD)
- Variance-preserving tail regularizer

## Run (ISIC2019LT)
```
bash scripts/run_stage1_isic2019lt.sh
```

## Run (ISIC_Archive)
```
bash scripts/run_stage1_isic_archive.sh
```

## Stage 2 (PG-AVFC)
ISIC2019LT:
```
bash scripts/run_stage2_isic2019lt.sh
```
ISIC_Archive:
```
bash scripts/run_stage2_isic_archive.sh
```

## Offline Analysis
Stage1 last checkpoint:
```
bash scripts/eval_last_stage1_isic2019lt.sh
```
```
bash scripts/eval_last_stage1_isic_archive.sh
```

Stage2 last checkpoint:
```
bash scripts/analyze_last_stage2_isic2019lt.sh
```
```
bash scripts/analyze_last_stage2_isic_archive.sh
```

## Outputs
Checkpoints are saved under `./checkpoints/<run_name>`:
- `resnet_encoder_latest.pth`
- `classifier_latest.pth`
- `projector_latest.pth` (if projector enabled)
- `prototype_memory_latest.pth`

Logs under `./log/tpcsd`.

## Config
Defaults are in `config/configs.yaml`. The run scripts are the recommended entry points.
