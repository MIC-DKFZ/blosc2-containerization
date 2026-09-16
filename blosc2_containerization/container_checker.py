import blosc2
from enum import Enum
import argparse
import subprocess
import sys
from pathlib import Path


class Status(Enum):
    NOT_EXISTING = 1
    UNFINISHED = 2
    FINISHED = 3
    PYTHON_ERROR = 4
    SEG_FAULT = 5


def check_image_container(path):
    if (path is None) or (not Path(path).is_file()):
        return Status.NOT_EXISTING

    result = subprocess.run([sys.executable, __file__, "-i", f"{path}", "--process"],
                            capture_output=True, text=True)

    # A segfaulted child never writes a status name, so detect it from the return
    # code before falling back to parsing stdout.
    if result.returncode == -11:
        return Status.SEG_FAULT

    try:
        return Status[result.stdout.strip()]
    except Exception:
        return Status.PYTHON_ERROR


def check_image_container_process(path, num_threads=1):
    blosc2.set_nthreads(num_threads)
    dparams = {'nthreads': num_threads}
    try:
        array = blosc2.open(path, mode="r", mmap_mode="r", dparams=dparams)
        is_container_stored = array.vlmeta[b"is_container_stored"]
        if is_container_stored:
            status = Status.FINISHED
        else:
            status = Status.UNFINISHED
    except Exception as e:
        print(e, file=sys.stderr)
        status = Status.PYTHON_ERROR

    print(status.name)  # Send enum name to parent process
    sys.exit(status.value)  # Optional: return code matches enum value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True, type=str, help="Absolute input path.")
    parser.add_argument('--process', required=False, default=False, action="store_true", help="Disable W&B.")
    args = parser.parse_args()

    if args.process:
        check_image_container_process(args.input)
    else:
        check_image_container(args.input)
