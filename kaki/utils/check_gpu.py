import os
import logging
import subprocess
from dotenv import load_dotenv
from .check_root_base import find_and_add_project_root
logging.basicConfig(level=logging.INFO)
base_dir = os.path.join(find_and_add_project_root(), f"config/gpu_conf.env")
print(base_dir)
load_dotenv(base_dir)

use_gpu = os.environ["use_gpu"]
logging.info(f"setting to use gpu: {use_gpu}")

def is_nvidia_gpu_present():
    try:
        output = subprocess.check_output("nvidia-smi", shell=True, text=True)
        # Check if the output contains a known error message
        if "fail to communicate" in output.lower():
            return False
        return True
    except subprocess.CalledProcessError:
        # nvidia-smi command failed which means NVIDIA GPU is not present or not configured properly
        return False


def enable_cudf_acceleration():
    if is_nvidia_gpu_present() and use_gpu == str(True):
        logging.info("Trying with CUDF")
        try:
            import cudf.pandas
            cudf.pandas.install()
            print("cuDF pandas accelerator enabled.")
        except ImportError:
            raise ImportError("cuDF is not installed. Please install cuDF to use GPU acceleration.")
    else:
        logging.info(f"Is NVIDIA GPU present? : {is_nvidia_gpu_present()}")
        logging.info("cuDF pandas accelerator not enabled. Using standard pandas.")

if __name__ == "__main__":
    enable_cudf_acceleration()  
    import pandas as pd
    print(pd)
