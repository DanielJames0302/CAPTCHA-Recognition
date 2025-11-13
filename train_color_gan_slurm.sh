#!/bin/bash
#SBATCH --output=logs/slurm-gan-improved-%j.out
#SBATCH --error=logs/slurm_gan_improved_error_%j.err
#SBATCH --job-name=nus-cs4243-gan
#SBATCH --mail-user=e1129786@comp.nus.edu.sg
#SBATCH --partition=gpu-long
#SBATCH --gpus=1     # or nv:1 -C cuda75
#SBATCH --time=24:00:00

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "/home/m/man0302/CS4243/project/CS4243-CAPTCHA" # change to your project directory


# Activate virtual environment (adjust path as needed)
source cs4243-captcha/bin/activate

echo "Node: $(hostname)"
echo "=== SLURM GAN Training Script ==="
echo "Starting training at $(date)"
echo "Running on host: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits || echo "No GPU detected"

if [ ! -f "train_color_gan.py" ]; then
    echo "Error: train_color_gan.py not found. Please run this script from the assignment directory."
    exit 1
fi

if [ ! -f "train_color_gan.py" ]; then
  echo "ERROR: train_color_gan.py not found in current directory"
  exit 1
fi

# PyTorch sanity check
echo "=== PyTorch Sanity Check ==="
python - <<'PY'
try:
    import torch
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("Number of GPUs:", torch.cuda.device_count())
        print("GPU name:", torch.cuda.get_device_name(0))
        print("GPU memory allocated:", torch.cuda.memory_allocated(0) / 1024**3, "GB")
except Exception as e:
    print("PyTorch import failed:", e)
PY
echo

echo "=== Starting GAN training ==="
date
srun --exclusive -N1 -n1 python train_color_gan.py

TRAIN_EXIT=$?
date
echo "=== Training finished with code $TRAIN_EXIT ==="
if [ $TRAIN_EXIT -ne 0 ]; then
  echo "Training failed."
  exit $TRAIN_EXIT
fi

echo
echo "=== Artifacts found (Captcha_DCGAN_results/) ==="
if [ -d "Captcha_DCGAN_results" ]; then
  echo "GAN results directory: Captcha_DCGAN_results"
  ls -lh Captcha_DCGAN_results/ || true
  echo
  echo "Model files:"
  ls -lh Captcha_DCGAN_results/*.pkl 2>/dev/null || echo "No .pkl model files found"
  echo
  echo "Training history:"
  ls -lh Captcha_DCGAN_results/*.pkl 2>/dev/null || echo "No training history file found"
  echo
  echo "Generated images:"
  if [ -d "Captcha_DCGAN_results/Fixed_results" ]; then
    echo "Fixed results (latest):"
    ls -lt Captcha_DCGAN_results/Fixed_results/*.png 2>/dev/null | head -5 || echo "No images found"
  fi
  if [ -d "Captcha_DCGAN_results/Random_results" ]; then
    echo "Random results (latest):"
    ls -lt Captcha_DCGAN_results/Random_results/*.png 2>/dev/null | head -5 || echo "No images found"
  fi
  if [ -f "Captcha_DCGAN_results/generation_animation.gif" ]; then
    echo "Animation GIF: Captcha_DCGAN_results/generation_animation.gif"
    ls -lh Captcha_DCGAN_results/generation_animation.gif
  fi
else
  echo "Catpcha_DCGAN_results directory not found."
fi

echo
echo "Done at $(date)"
