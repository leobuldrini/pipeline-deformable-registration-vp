# TCC Pipeline — How It Works & How To Run It

Deformable brain-MRI registration with tumor-volume preservation (VP), compared across **NODEO, VoxelMorph, TransMorph** (each ±VP) on the Yale Brain Mets dataset.

The pipeline follows the **4 etapas** of the thesis (`main.tex §Metodologia`). Run them **in order** — each etapa's output feeds the next. A 5th block produces the results/figures.

---

## 0. Prerequisites

Everything runs from the repo root:

```bash
cd /path/to/TCC_refactored
REPO=$PWD
source ~/miniconda3/etc/profile.d/conda.sh
```

**Conda envs** (already configured):
- `transmorph` — the 3 registration models + ANTs/atlas prep. torch 2.10 +cu128 (Blackwell). **Must** have the voxelmorph patch applied into its site-packages: `third_party_patches/voxelmorph_py311_blackwell.patch`.
- `fastsurfer` — FastSurfer, FastSurfer-LIT, nnU-Net (BraTS-METS) tumor seg.

**Inputs the repo expects at its root** (all gitignored; supply your own — a small test sample can be staged here):
- `Dataset/MRI/<patient>/<exam>/*_{PRE,POST,T2,FLAIR}.nii.gz` — raw exams.
- `Atlas/` — MNI ICBM152 template (etapa 2 fills the padded/cropped/seg variants).
- `FastSurfer/` — FastSurfer clone (incl. LIT); `FastSurfer/run_fastsurfer.sh` must exist. A FreeSurfer license (`FS_LICENSE`) must be reachable for FastSurfer/LIT.
- `brats_local/results` + `brats_local/raw` — nnU-Net BraTS-METS-2025 weights.

**Working paths** used below (created by the run):

```bash
FS_OUT=$REPO/fastsurfer_output/phase2          # FastSurfer brain-seg output (in-repo)
DATA=$REPO/data/yale_phase2_mni                # preprocessed npz (224³) + transforms/
DATA160=$REPO/data/yale_phase2_mni_160         # cropped npz (160×192×224)
ATLAS_PAD=$REPO/Atlas/mni_icbm152_t1_padded.nii.gz
ATLAS160=$REPO/Atlas/mni_icbm152_t1_padded_160x192x224.nii.gz
ATLAS_SEG=$REPO/Atlas/fastsurfer_seg_160x192x224.nii.gz
TUMOR=$REPO/tumor_masks_conformed
CSV=$REPO/phase1_filter/ranked_4mod.csv
```

> **Test-scale note.** On a tiny sample (e.g. ~12 exams) reduce `--top-k`, set `--steps-per-epoch` to ≈ number of training samples, and use small `--epochs`. The goal of a smoke run is "the pipeline executes end-to-end," not good metrics.

---

## Etapa 1 — Filtragem dos dados  (env: any → use `transmorph`)

Keep exams with all 4 modalities (PRE/POST/T2/FLAIR) closest to 1 mm isotropy.

```bash
conda activate transmorph

# rank PRE scans by isotropy, keep top-k complete exams → CSV
python phase1_filter/rank_raw_scans.py \
    --mri-dir Dataset/MRI --top-k 1000 --output-dir phase1_filter
```
**Out:** `phase1_filter/ranked_4mod.csv`.

---

## Etapa 2 — Pré-processamento e segmentação tumoral

### 2.1 Atlas — normalize + pad (224,256,224), then FastSurfer-seg it
```bash
conda activate transmorph
python phase2_preprocess/prepare_mni_template.py            # → Atlas/mni_icbm152_t1_padded.nii.gz

conda activate fastsurfer
python phase2_preprocess/prepare_atlas_fastsurfer_seg.py    # → Atlas/fastsurfer_seg_160x192x224.nii.gz
#   add --skip-fastsurfer if the atlas seg is already present
```

### 2.2 FastSurfer on PRE (T1) — skull-strip, conform 256³, N4ITK  *(slow: ~hrs/scan)*
```bash
conda activate fastsurfer
python phase2_preprocess/generate_fastsurfer_jobs.py \
    --input-csv "$CSV" --output-tsv phase2_preprocess/fastsurfer_jobs_v2.tsv
bash phase2_preprocess/run_fastsurfer_batch.sh --parallel 4   # → $FS_OUT/<sid>/mri/*.mgz
#   --limit N / --dry-run to test first
```

### 2.3 ANTs affine → ICBM152, label-merge → MNI, normalize [0,1], center-crop → npz
```bash
conda activate transmorph
python phase2_preprocess/preprocess_fastsurfer.py \
    --data-dir "$FS_OUT" --template "$ATLAS_PAD" \
    --output-dir "$DATA" --workers 2                          # → *.npz(vol,seg) + transforms/*.mat

python phase2_preprocess/crop_to_transmorph.py \
    --src-dir "$DATA" --dst-dir "$DATA160" --target-shape 160 192 224 \
    --atlas "$ATLAS_PAD" --atlas-seg "$ATLAS_SEG" --workers 8 # → 160×192×224 npz
```

### 2.4 nnU-Net tumor mask (POST/T2/FLAIR conformed) → reuse affine → into npz
```bash
conda activate fastsurfer
python phase2_preprocess/segment_tumors.py \
    --csv "$CSV" --fastsurfer-dir "$FS_OUT" \
    --npz-dir "$DATA160" --transforms-dir "$DATA/transforms" \
    --template "$ATLAS_PAD" --tumor-masks-dir "$TUMOR"
```
**Out:** each `.npz` now holds `vol` + `seg` + `tumor_mask`.

### 2.5 LIT priority list — which exams have a tumor (need inpainting)
```bash
conda activate transmorph
python phase2_preprocess/make_lit_priority.py \
    --npz-dir "$DATA160" --output phase1_filter/lit_priority.txt
```
**Out:** `phase1_filter/lit_priority.txt` — tumor-bearing exams only (empty-mask exams skip LIT). Derived from the `tumor_mask` written in 2.4; feeds Etapa 3 and the patient split.

---

## Etapa 3 — Preenchimento cerebral e re-segmentação  (env: fastsurfer)

FastSurfer-LIT inpaints the lesion with synthetic healthy tissue, FastSurfer re-segments the inpainted volume, the affine is reapplied, and the corrected labels replace `seg` — kept with the **original** (non-inpainted) `vol`.

```bash
conda activate fastsurfer
python phase3_inpaint/run_fastsurfer_lit.py \
    --tumor-masks-dir "$TUMOR" --fastsurfer-dir "$FS_OUT" \
    --fastsurfer-bin FastSurfer/run_fastsurfer.sh \
    --template "$ATLAS_PAD" --npz-dir "$DATA160" \
    --priority-list phase1_filter/lit_priority.txt
```
**Out:** final `.npz` = `vol` (original) + `seg` (LIT-corrected) + `tumor_mask`.

---

## Patient-level split  (env: any → `transmorph`)

Build `data_split.json` (80/10/10 **by patient**, no leakage) over the preprocessed `.npz`. Needs the npz from Etapa 2.3 to exist. Lands in `$DATA160/data_split.json` — consumed by Etapa 4 (training val/test + NODEO `--split-json`).

```bash
conda activate transmorph
python phase1_filter/create_patient_split.py \
    --data-dir "$DATA160" --ranking "$CSV" \
    --lit-priority /dev/null --lit-output-dir "$REPO/lit_output" \
    --top-k 1000 --force
```
- `--lit-priority /dev/null` includes **all** preprocessed exams (use for the smoke test, or whenever you don't gate on LIT status).
- **Real run:** create the split after Etapa 3 and pass the generated list, `--lit-priority phase1_filter/lit_priority.txt` (from step 2.5) — exams whose LIT inpaint isn't finished are then excluded.

---

## Etapa 4 — Treinamento e avaliação  (env: transmorph)

Two regimes: `atlas-to-scan` and `scan-to-scan-intra`. Baseline shown; **+VP** adds the VP loss (flags noted per model). `--eval` writes the metrics JSON the results block reads.

```bash
conda activate transmorph
```

**TransMorph** — run from its own dir (local module imports). `+VP` = `--vp --vol-pres-weight 0.1`.
```bash
cd "$REPO/phase4_train_eval/models/transmorph"
PYTORCH_ALLOC_CONF=expandable_segments:True python run_transmorph.py --train --amp \
    --mode atlas-to-scan --config TransMorph --model transmorph \
    --data-dir "$DATA160" --atlas "$ATLAS160" \
    --epochs 500 --steps-per-epoch 247 \
    --sim-weight 1.0 --reg-weight 0.1 --dice-weight 1.0 \
    --save-dir checkpoints_baseline
    # +VP: append  --vp --vol-pres-weight 0.1   and use  --save-dir checkpoints_vp
PYTORCH_ALLOC_CONF=expandable_segments:True python run_transmorph.py --eval --amp \
    --mode atlas-to-scan --config TransMorph --model transmorph \
    --data-dir "$DATA160" --atlas "$ATLAS160" --atlas-seg "$ATLAS_SEG" \
    --save-dir checkpoints_baseline                          # → checkpoints_baseline/eval_results*.json
cd "$REPO"
```

**VoxelMorph** — `VXM_BACKEND=pytorch`, `--int-steps 0`. `+VP` = `--vp --vol-pres-weight 0.1`.
```bash
VXM_BACKEND=pytorch python phase4_train_eval/models/voxelmorph/run_voxelmorph.py --train \
    --mode atlas-to-scan --int-steps 0 \
    --data-dir "$DATA160" --atlas "$ATLAS160" \
    --epochs 500 --steps-per-epoch 247 --reg-param 0.1 --dice-weight 1.0 \
    --save-dir "$REPO/checkpoints_vxm"
VXM_BACKEND=pytorch python phase4_train_eval/models/voxelmorph/run_voxelmorph.py --eval \
    --mode atlas-to-scan --int-steps 0 \
    --data-dir "$DATA160" --atlas "$ATLAS160" --atlas-seg "$ATLAS_SEG" \
    --save-dir "$REPO/checkpoints_vxm" --weights "$REPO/checkpoints_vxm/checkpoint.pt"
```

**NODEO** — no training; per-pair optimization. `+VP` = `--vol-pres-weight 0.1` (baseline `0.0`).
```bash
python phase4_train_eval/models/nodeo/run_nodeo_batch.py --mode atlas-to-scan \
    --data-dir "$DATA160" \
    --atlas-brain "$ATLAS160" --atlas-seg "$ATLAS_SEG" \
    --split-json "$DATA160/data_split.json" \
    --device cuda --vol-pres-weight 0.0 \
    --output-dir "$REPO/result/nodeo_baseline"
```

For the longitudinal regime, repeat all three with `--mode scan-to-scan-intra` (NODEO still needs `--atlas-brain/--atlas-seg`; it just pairs a patient's own exams). Full matrix = 3 models × {baseline, VP} × {atlas-to-scan, scan-to-scan-intra}.

**Out:** `checkpoints_*/eval_results*.json`, `result/nodeo_*/results.json`, `power_log.csv`.

---

## Resultados experimentais — tables & figures  (env: transmorph)

```bash
python phase5_results/update_results_csv.py                  # eval JSONs → results_all_models*.csv
PYTORCH_ALLOC_CONF=expandable_segments:True \
    python phase5_results/compare_vp_models.py --montage-only  # → imgs/vp_compare_montage.png
python phase5_results/sum_energy.py                          # power_log.csv → Wh
python common/plot_metrics.py "$REPO/checkpoints_baseline/train_log.csv"   # training curves
```

---

## Metrics
**Dice** (anatomical labels) · **NDJ** (% voxels with det(J)<0) · **STSR / aSTSR** (tumor-volume preservation; aSTSR = atlas-to-patient adaptation, this work's contribution) · **TVCF** (longitudinal tumor-change fidelity) · **energy** (Wh via NVML).
