import subprocess
import os
import sys
import shutil

env_name = "vediorag"
main_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), "main.py")

def run(command, live=False):
    print(f"Running: {command}")
    if live:
        # For live output, just run and don't try to capture stdout
        result = subprocess.run(command, shell=True)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {command}\nReturn code: {result.returncode}")
        return ""  # Nothing to decode
    else:
        # Capture output
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {command}\nReturn code: {result.returncode}\nstdout: {result.stdout.decode(errors='ignore')}\nstderr: {result.stderr.decode(errors='ignore')}")
        return result.stdout.decode(errors="ignore")

def conda_env_exists(name):
    output = run("conda env list")
    return any(line.startswith(name) for line in output.splitlines())

def prepare_conda_env():
    if not conda_env_exists(env_name):
        print(f"Creating Conda environment '{env_name}' with Python 3.10...")
        run(f"conda create -y -n {env_name} python=3.10", live=True)
    else:
        print(f"Conda environment '{env_name}' already exists, skipping creation.")
    print(f"activating environment {env_name}")
    run(f"conda activate {env_name} ", live=True)

def install_packages():
    # Install PyTorch + CUDA
    # run(f'conda run -n {env_name} pip install torch==1.10.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113', live=True)
    run(f'pip install pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 --extra-index-url https://download.pytorch.org/whl/cu113', live=True)
    # Whisper
    run(f'pip install openai-whisper', live=True)
    # FFmpeg via imageio
    run(f'conda install imageio[ffmpeg]', live=True)
    # Other packages
    # extras = ["transformers", "faiss-cpu", "numpy", "scipy", "tqdm"]
    # for pkg in extras:
    #     run(f'conda run -n {env_name} pip install {pkg}', live=True)
    # requirements.txt if exists
    req_file = "requirements.txt"
    if os.path.exists(req_file):
        # run(f'conda run -n {env_name} pip install -r {req_file}', live=True)
        run(f'pip install -r {req_file}', live=True)

def start_main():
    if not os.path.exists(main_script):
        raise RuntimeError("main.py not found!")
    print(f"Running main.py inside Conda environment '{env_name}'...")
    run(f'python "{main_script}"', live=True)

if __name__ == "__main__":
    prepare_conda_env()
    install_packages()
    start_main()
