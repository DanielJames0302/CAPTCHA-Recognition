#!/bin/bash
#SBATCH --output=logs/slurm-generate-color-captcha-%j.out
#SBATCH --error=logs/slurm-generate-color-captcha-%j.err
#SBATCH --job-name=captcha-gen
#SBATCH --mail-user=e1129786@comp.nus.edu.sg
#SBATCH --partition=gpu-long
#SBATCH --gpus=1
#SBATCH --time=04:00:00

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "/home/m/man0302/CS4243/project/CS4243-CAPTCHA"   # <-- your project directory

# Activate venv
source cs4243-captcha/bin/activate

echo "Node: $(hostname)"
echo "=== CAPTCHA Generation Job ==="
echo "Start time: $(date)"
echo "GPU information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits || echo "No GPU detected"
echo

SCRIPT_NAME="generate_color_captcha_gan.py"

if [ ! -f "$SCRIPT_NAME" ]; then
    echo "ERROR: $SCRIPT_NAME not found in $(pwd)"
    exit 1
fi

GEN_WEIGHTS="/home/m/man0302/CS4243/project/CS4243-CAPTCHA/Captcha_DCGAN_results_20251113172531/generator_param.pkl"

if [ -z "$GEN_WEIGHTS" ]; then
    echo "ERROR: No generator weights (generator_param.pkl) found under Captcha_DCGAN_results_*"
    exit 1
fi

echo "Found generator weights: $GEN_WEIGHTS"
echo

echo "=== PyTorch Environment ==="
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU memory allocated:", torch.cuda.memory_allocated(0)/1024**3, "GB")
PY
echo

echo "=== Starting CAPTCHA Generation ==="
date

srun --exclusive -N1 -n1 python $SCRIPT_NAME --weights "$GEN_WEIGHTS"

EXIT_CODE=$?
date
echo "=== CAPTCHA Generation finished with exit code $EXIT_CODE ==="

if [ $EXIT_CODE -ne 0 ]; then
    echo "Generation process failed."
    exit $EXIT_CODE
fi

echo
echo "=== Generated Captcha Artifacts ==="

if ls generated_color_captchas/*.png 1>/dev/null 2>&1; then
    echo "Generated captcha images:"
    ls -lt generated_color_captchas/*.png | head -10
else
    echo "No generated captcha images found in generated_color_captchas/"
fi

echo
echo "Done at: $(date)"
