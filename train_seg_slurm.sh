#!/bin/bash
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm_error_%j.err
#SBATCH --job-name=nus-cs4243-project
#SBATCH --mail-user=e1536075@comp.nus.edu.sg
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1     # or nv:1 -C cuda75
#SBATCH --time=48:00:00

export TF_XLA_FLAGS="--tf_xla_auto_jit=0"     # disable XLA JIT (layout transposes)
export TF_ENABLE_ONEDNN_OPTS=0               # avoid aggressive layout rewrites
export TF_CUDNN_USE_FRONTEND=0               # bypass cuDNN frontend heuristics
export TF_DETERMINISTIC_OPS=1                # (optional) stable kernels
export TF_CUDNN_DETERMINISTIC=1              # (optional) stable kernels
export TF_FORCE_GPU_ALLOW_GROWTH=true        # avoid full preallocation
cd ~/CS4243-CAPTCHA # change to your project directory

# Activate virtual environment (adjust path as needed)
source cs4243-captcha/bin/activate

echo "Node: $(hostname)"
echo "=== Slurm Training Script ==="
echo "Starting training at $(date)"
echo "Running on host: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits || echo "No GPU detected"

# TensorFlow-friendly environment (safe defaults)
export TF_CPP_MIN_LOG_LEVEL=1            # reduce TF verbosity
export TF_FORCE_GPU_ALLOW_GROWTH=true    # avoid full GPU memory pre-allocation
# export XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false  # uncomment if you hit rare cuDNN alg issues

# Check if we're in the right directory
if [ ! -f "train_seg.py" ]; then
    echo "Error: train_seg.py not found. Please run this script from the assignment directory."
    exit 1
fi

# Quick repo sanity check
if [ ! -f "train_seg.py" ]; then
  echo "ERROR: train_seg.py not found in $PROJECT_DIR"
  exit 1
fi

# Optional: print TF version to the log
echo "=== TensorFlow Sanity Check ==="
python - <<'PY'
try:
    import tensorflow as tf
    print("TF version:", tf.__version__)
    print("TF visible GPUs:", tf.config.list_physical_devices('GPU'))
except Exception as e:
    print("TensorFlow import failed:", e)
PY
echo

echo "=== Starting training ==="
date
# NOTE: training configuration is controlled in configs.py.
# If you want to change dataset_path/model_path, edit configs.py before submitting.
# (This script intentionally does not pass CLI flags because the repo doesn't use them.)
srun --exclusive -N1 -n1 python train_seg.py

TRAIN_EXIT=$?
date
echo "=== Training finished with code $TRAIN_EXIT ==="
if [ $TRAIN_EXIT -ne 0 ]; then
  echo "Training failed."
  exit $TRAIN_EXIT
fi

echo
echo "=== Artifacts found (top-level under Models/) ==="
# List newest model directory (many repos timestamp under Models/tutorial/<timestamp>/)
NEWEST_DIR=$(find Models -maxdepth 2 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1{print $2}')
if [ -n "${NEWEST_DIR:-}" ] && [ -d "$NEWEST_DIR" ]; then
  echo "Newest model dir: $NEWEST_DIR"
  ls -lh "$NEWEST_DIR" || true
else
  echo "No subdirectories found under Models/ (training may have used a different path from configs.py)."
fi

echo
echo "If TensorBoard logs were written, you should see a 'logs/' dir inside the model directory."
echo "You can later run (on a login or compute node with port forwarding):"
echo "  tensorboard --logdir \"$NEWEST_DIR/logs\" --port 6006"
echo

echo "Done at $(date)"