import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable, List

filename_mapping = {
    "-t1n.nii.gz": "_0000.nii.gz",
    "-t1c.nii.gz": "_0001.nii.gz",
    "-t2w.nii.gz": "_0002.nii.gz",
    "-t2f.nii.gz": "_0003.nii.gz",
}


def set_tmp_symlinks(
    files: str | Path | List[str | Path],
    filename_mapping: dict,
    output_dir: str | Path,
    condition: Callable = lambda x: True,
) -> Path:
    """Creates a temporary directory with symlinks to files from files_dir with names compatible with nnUNet, as pointed by mapping dictionary.

    Args:
        files (str, Path, List[str, path]): path to folder with files or a list of paths to files
        filename_mapping (dict): a dictionary with filename mapping used by nnUNet
        output_dir (str, Path): path to output folder to create temporary directory in
        condition (Callable): condition on which files are copied, used e.g. for filtering full GT dataset to prediction files only
    """
    output_dir = Path(output_dir)
    if not isinstance(files, list):
        files = Path(files)

    for fragment, replacement in filename_mapping.items():
        if not isinstance(files, list):
            files_list = files.rglob("*" + fragment)
        else:
            files_list = [f for f in files if fragment in str(f)]

        for f in files_list:
            target_dir = output_dir / Path(f).name.replace(fragment, replacement)
            if not target_dir.exists() and condition(target_dir):
                os.symlink(f, target_dir)


def run_subprocess(command, check=True):
    # run in subprocess with "live" (modulo buffering at the subprocess' side) output
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in iter(process.stdout.readline, b""):
        sys.stdout.write(line.decode(sys.stdout.encoding))

    returncode = process.poll()
    if returncode != 0 and check:
        raise subprocess.CalledProcessError(returncode, command)

    return returncode


def main():
    input_path = Path("/input")
    tmp_input_path = Path("/input_tmp")
    output_path = Path("/output")

    # create symplinks for input data
    os.mkdir(tmp_input_path)
    set_tmp_symlinks(input_path, filename_mapping, tmp_input_path)

    # inference
    predict_cmd = f"nnUNetv2_predict -i {tmp_input_path} -o {output_path} -d Dataset101_submission -tr nnUNetTrainer -p nnUNetResEncUNetXLPlans -c 3d_fullres -f all -chk checkpoint_final.pth"
    run_subprocess(predict_cmd)

    # cleanup
    shutil.rmtree(tmp_input_path)
    for filepath in output_path.rglob("*"):
        if not filepath.name.endswith(".nii.gz"):
            os.remove(filepath)


if __name__ == "__main__":
    main()
