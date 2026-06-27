"""
Local BraTS25_1 MetastasesSegmenter inference using extracted nnU-Net weights.
Runs on GPU with local PyTorch (sm_120 compatible), bypassing Docker.

Usage:
    python code/brats_local_infer.py \
        --t1n path/to/PRE.nii.gz \
        --t1c path/to/POST.nii.gz \
        --t2w path/to/T2.nii.gz \
        --t2f path/to/FLAIR.nii.gz \
        --output path/to/output_seg.nii.gz

Images are co-registered to t1n space before inference.
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import ants
import torch

BRATS_LOCAL_DIR = Path(__file__).resolve().parent.parent / "brats_local"
RESULTS_DIR = BRATS_LOCAL_DIR / "results"
RAW_DIR = BRATS_LOCAL_DIR / "raw"

# nnU-Net environment variables
os.environ["nnUNet_results"] = str(RESULTS_DIR)
os.environ["nnUNet_raw"] = str(RAW_DIR)
os.environ["nnUNet_preprocessed"] = str(BRATS_LOCAL_DIR / "preprocessed")


def coregister_to_ref(ref_path: str, mov_path: str) -> "ants.ANTsImage":
    """Rigid-register moving image to reference space."""
    ref = ants.image_read(ref_path)
    mov = ants.image_read(mov_path)
    reg = ants.registration(fixed=ref, moving=mov, type_of_transform="Rigid")
    return reg["warpedmovout"]


def prepare_input(t1n: str, t1c: str, t2w: str, t2f: str, tmpdir: str) -> Path:
    """Co-register modalities and prepare nnU-Net input folder."""
    input_dir = Path(tmpdir) / "input"
    case_name = "BraTS-MET-00000-000"
    input_dir.mkdir(parents=True)

    ref = ants.image_read(t1n)

    # t1n is the reference — save directly
    ants.image_write(ref, str(input_dir / f"{case_name}_0000.nii.gz"))

    # Co-register other modalities
    for path, suffix in [(t1c, "_0001"), (t2w, "_0002"), (t2f, "_0003")]:
        print(f"  Registering {Path(path).name} -> t1n...")
        mov = ants.image_read(path)
        reg = ants.registration(fixed=ref, moving=mov, type_of_transform="Rigid")
        ants.image_write(
            reg["warpedmovout"],
            str(input_dir / f"{case_name}{suffix}.nii.gz"),
        )

    return input_dir


def run_inference(input_dir: Path, output_dir: Path):
    """Run nnU-Net inference using extracted BraTS25_1 weights."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
    )

    predictor.initialize_from_trained_model_folder(
        str(
            RESULTS_DIR
            / "Dataset101_submission"
            / "nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres"
        ),
        use_folds=("all",),
        checkpoint_name="checkpoint_final.pth",
    )

    predictor.predict_from_files(
        list_of_lists_or_source_folder=str(input_dir),
        output_folder_or_list_of_truncated_output_files=str(output_dir),
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Local BraTS25_1 tumor segmentation (GPU, no Docker)"
    )
    parser.add_argument("--t1n", required=True, help="T1 native (PRE)")
    parser.add_argument("--t1c", required=True, help="T1 contrast (POST)")
    parser.add_argument("--t2w", required=True, help="T2 weighted")
    parser.add_argument("--t2f", required=True, help="T2 FLAIR")
    parser.add_argument("--output", required=True, help="Output segmentation path")
    args = parser.parse_args()

    tmpdir = tempfile.mkdtemp(prefix="brats_local_")

    try:
        print("Step 1: Co-registering modalities...")
        input_dir = prepare_input(args.t1n, args.t1c, args.t2w, args.t2f, tmpdir)

        print("Step 2: Running nnU-Net inference...")
        output_dir = Path(tmpdir) / "output"
        run_inference(input_dir, output_dir)

        # Copy result
        result_file = output_dir / "BraTS-MET-00000-000.nii.gz"
        if result_file.exists():
            shutil.copy2(str(result_file), args.output)
            print(f"Done! Segmentation saved to {args.output}")
        else:
            print(f"Error: Expected output not found at {result_file}")
            print(f"Output dir contents: {list(output_dir.iterdir())}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
