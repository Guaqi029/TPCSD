# TPCSD Stage1 (ResNet + Prototype Contrastive Distillation)

This folder is a standalone Stage1 pipeline (no VAVAE, no Stage2) with:
- ResNet backbone (50/34)
- EMA teacher
- Prototype Contrastive Distillation (PCD)
- Relaxed PCD margin to preserve intra-class variance
- Similarity-preserving KD
- Prototype uniformity regularizer
- Prototype uniformity uses linear warm-up before full strength

## Run (ISIC2019LT)
```
bash scripts/run_stage1_isic2019lt.sh
```

## Run (ISIC_Archive)
```
bash scripts/run_stage1_isic_archive.sh
```

## Stage 2 (PG-AVFC)
Stage 2 freezes the Stage 1 encoder, works directly in the encoder feature space, fits class-wise Gaussian statistics, samples virtual features around the fused class centers, and retrains only a cosine classifier. The recommended scripts run the no-`class_weight` baseline and require the saved Stage 1 encoder and prototype checkpoints. Older projector-space Stage 1 checkpoints are not compatible with the current Stage 2.

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
- `prototype_memory_latest.pth`

Logs under `./log/tpcsd`.

## Config
Defaults are in `config/configs.yaml`. The run scripts are the recommended entry points.
