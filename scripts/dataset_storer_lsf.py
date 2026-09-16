from pathlib import Path
from tqdm import tqdm
import argparse
import subprocess
import numpy as np
from blosc2_containerization.image_container import ContainerWriter
import random
from typing import List


def store_dataset(json_filepath, save_dir, num_threads, patch_size, num_jobs, is_medium, is_long, is_verylong):
    sub_container_indices = get_container_indices(save_dir, patch_size)
    sub_container_indices = split_list(sub_container_indices, num_jobs)
    for job_index in tqdm(range(num_jobs)):
        command = create_command(json_filepath, save_dir, num_threads, patch_size, sub_container_indices[job_index])
        lsf_command = create_lsf_command(command, processes=num_threads, is_medium=is_medium, is_long=is_long, is_verylong=is_verylong)
        submit_lsf_job(lsf_command)


def create_command(json_filepath: Path, save_dir: Path, num_threads: int, patch_size, sub_container_indices: List[int]) -> str:
    """
    Create the command for converting an image.

    Args:
        input_dir (Path): Path to the input directory.
        output_dir (Path): Path to the output directory.
        dataset_image_index_path (Path): Path to the dataset image index file.
        index (int): Index of the image to convert.
        mitk_dir (str, optional): Path to the MITK directory required for image conversion.
        overwrite (bool): Whether to overwrite existing images in the output directory.

    Returns:
        str: The command as a string to be executed for job submission.
    """
    command = [
                "python", 
                "blosc2_containerization/dataset_storer.py",
                "-i",
                f"{json_filepath}",
                "-o",
                f"{save_dir}",
                "-t",
                f"{num_threads}",
                "-p",
                f"{patch_size[0]} {patch_size[1]} {patch_size[2]} {patch_size[3]}",
                "--indices",
                f"{' '.join(map(str, sub_container_indices))}",
            ]
    command = " ".join(command)
    return command


def create_lsf_command(command, queue: str = "short", processes: int = 1, mem: int = 20, is_medium: bool = False, is_long: bool = False, is_verylong: bool = False):
    if processes == 0:
        processes = 1
    if is_medium:
        queue = "medium"
    if is_long:
        queue = "long"
    if is_verylong:
        queue = "verylong"
    lsf_command = f'bsub -q "{queue}" -n {processes} -R "rusage[mem={mem}GB]" /bin/bash -l -c ". ~/start_nnunetv2.sh; {command}"'
    return lsf_command


def submit_lsf_job(command: str, shell: bool = True) -> subprocess.CompletedProcess:
    """
    Submit a job to the LSF cluster using the specified command.

    Args:
        command (str): The command to be executed for job submission.
        shell (bool): Whether to execute the command through the shell.

    Returns:
        subprocess.CompletedProcess: The completed submission process.

    Raises:
        RuntimeError: If bsub fails to submit the job (non-zero return code).
    """
    process = subprocess.run(
        command,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"bsub submission failed (returncode {process.returncode}):\n"
            f"stdout: {process.stdout}\nstderr: {process.stderr}"
        )
    if process.stdout.strip():
        print(f"LSF submission: {process.stdout.strip()}")
    return process


def get_container_indices(save_dir, patch_size, shuffled=True):
    container_writer = ContainerWriter(save_dir, patch_size, np.float32)
    container_writer.load_storage()
    sub_containers = container_writer.get_sub_containers()
    sub_container_indices = list(range(len(sub_containers)))
    if shuffled:
        random.shuffle(sub_container_indices)
    return sub_container_indices


def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True, type=str, help="Path to a json file containing a list of dict(image_id: image_path) pairs.")
    parser.add_argument('-o', "--output", required=True, type=str, help="Absolute output path to the directory used for saving the containerized dataset.")
    parser.add_argument('-t', "--threads", required=False, type=int, default=1, help="Number of threads.")
    parser.add_argument('-p', '--patch_size', default=(1, 160, 160, 160), required=False, type=int, nargs=4, help="The image patch size.")
    parser.add_argument("--num_jobs", required=False, type=int, default=1, help="Number of jobs for parallelization across jobs.")
    parser.add_argument('--medium', required=False, default=False, action="store_true", help="Whether the medium queue is required.")
    parser.add_argument('--long', required=False, default=False, action="store_true", help="Whether the long queue is required.")
    parser.add_argument('--verylong', required=False, default=False, action="store_true", help="Whether the long queue is required.")
    args = parser.parse_args()

    store_dataset(args.input, args.output, args.threads, args.patch_size, args.num_jobs, args.medium, args.long, args.verylong)
    