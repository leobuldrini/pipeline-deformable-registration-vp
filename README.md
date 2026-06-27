# Pipeline baseado em aprendizado profundo para registro de imagens médicas com preservação de volumes tumorais

Registro deformável de RM cerebral com preservação de volume tumoral (NODEO, VoxelMorph, TransMorph, cada um ±VP) no Yale Brain Mets.

![Fluxograma do pipeline](imgs/pipeline_flowchart.png)

## Estrutura

- `common/`: perdas/métricas/energia compartilhadas
- `phase1_filter/` → `phase2_preprocess/` → `phase3_inpaint/` → `phase4_train_eval/models/{transmorph,voxelmorph,nodeo}/`.
- `phase5_results/`: pós-pipeline. Plota resultados, gera tabelas/figuras. Não é uma etapa do pipeline — o pipeline em si são as quatro etapas acima.

Ordem de execução + comandos: ver **RUNBOOK.md**.

## Ambientes

Na pesquisa, a ferramenta miniconda foi usada para gerenciar os ambientes Python.

- `transmorph` — os 3 modelos de registro + preparo do atlas/ANTs. `torch==2.10.0+cu128`.
  Instalação: `pip install -r requirements.txt` (+ etapa do voxelmorph abaixo).
- `fastsurfer` — FastSurfer, FastSurfer-LIT, nnU-Net (BraTS-METS). `nnunetv2`, `monai`.
  Instalação: `pip install -r requirements-fastsurfer.txt` (+ clone do FastSurfer abaixo).
Os pip freezes completos da máquina de referência estão em `env_snapshots/`.

## Ferramentas externas (NÃO incluídas no repositório) — fixadas nas versões em que este trabalho rodou

- **FastSurfer** — clone no commit `0b6c508` (`v2.4.2-270-g0b6c508`):

  ```
  git clone https://github.com/Deep-MI/FastSurfer.git
  git -C FastSurfer checkout 0b6c508d36d3ab74c42b4ab3ae9941a5c668508f
  ```

  Coloque o clone na raiz do repositório (`./FastSurfer/`); precisa de uma licença FreeSurfer
  (`FS_LICENSE`). Passada aos scripts via `--fastsurfer-bin FastSurfer/run_fastsurfer.sh`.
- **FastSurfer-LIT** (`neurolit`) — commit fixado `d23f6d0`, instalado por
  `requirements-fastsurfer.txt`:
  `git+https://github.com/Deep-MI/LIT.git@d23f6d0ca54426e151970133f257eab827961747`.
- **voxelmorph** (Blackwell/Py3.11):

  ```
  pip install "git+https://github.com/voxelmorph/voxelmorph.git@9bde7a270edfc19ad1c61115cb5ebd82124ee3af"
  d=$(python -c "import voxelmorph,os;print(os.path.dirname(os.path.dirname(voxelmorph.__file__)))")
  patch -p1 -d "$d" < third_party_patches/voxelmorph_py311_blackwell.patch
  ```

  Sempre defina `VXM_BACKEND=pytorch`.
- **Pesos nnU-Net** — vencedor do BraTS-METS-2025; baixe para `brats_local/results` + `brats_local/raw`.

## Ressalva Blackwell (RTX 50 / sm_120)

Requer CUDA-12.8 / PyTorch `+cu128`. Os pesos pré-treinados dos autores originais são incompatíveis — os modelos foram treinados do zero.

## Dados (não incluídos — forneça na raiz do repositório)

- **Dataset/** — Yale Brain Mets Longitudinal. Layout: `Dataset/MRI/<patient>/<exam>/*_{PRE,POST,T2,FLAIR}.nii.gz`.
- **Atlas/** — template MNI ICBM152 2009c não-linear simétrico (`mni_icbm152_nlin_sym_09c_nifti`, T1 + máscara), descompactado em `Atlas/`.
  - A etapa 2.1 (`prepare_mni_template.py` → T1 com padding, depois `prepare_atlas_fastsurfer_seg.py` → seg do atlas pelo FastSurfer) deriva os produtos que o pipeline lê: `mni_icbm152_t1_padded[_160x192x224].nii.gz`, `fastsurfer_seg_160x192x224.nii.gz`.
- **brats_local/{results,raw}** — pesos nnU-Net BraTS-METS-2025 (ver acima).

Todos os intermediários produzidos ficam no repositório: brain-seg do FastSurfer → `./fastsurfer_output/`, npz pré-processado → `./data/`, máscaras tumorais → `./tumor_masks_conformed/`, checkpoints/resultados → `./checkpoints*/`, `./result/`.
Tudo no gitignore.
