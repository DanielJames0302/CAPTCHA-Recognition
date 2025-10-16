#!/bin/bash
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm_error_%j.err
#SBATCH --job-name=nus-cs4243-project
#SBATCH --mail-user=e1129786@comp.nus.edu.sg
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1     # or nv:1 -C cuda75
#SBATCH --time=24:00:00

export TF_XLA_FLAGS="--tf_xla_auto_jit=0"     # disable XLA JIT (layout transposes)
export TF_ENABLE_ONEDNN_OPTS=0               # avoid aggressive layout rewrites
export TF_CUDNN_USE_FRONTEND=0               # bypass cuDNN frontend heuristics
export TF_DETERMINISTIC_OPS=1                # (optional) stable kernels
export TF_CUDNN_DETERMINISTIC=1              # (optional) stable kernels
export TF_FORCE_GPU_ALLOW_GROWTH=true        # avoid full preallocation
cd "/home/m/man0302/CS4243/CS4243-CAPTCHA-main"

# Activate virtual environment (adjust path as needed)
source cs4243-captcha/bin/activate

echo "Node: $(hostname)"
echo "=== Slurm Evaluation Script ==="
echo "Starting evaluation at $(date)"
echo "Running on host: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits || echo "No GPU detected"

# TensorFlow-friendly environment (safe defaults)
export TF_CPP_MIN_LOG_LEVEL=1            # reduce TF verbosity
export TF_FORCE_GPU_ALLOW_GROWTH=true    # avoid full GPU memory pre-allocation
# export XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false  # uncomment if you hit rare cuDNN alg issues


# Quick repo sanity check
if [ ! -f "eval_test.py" ]; then
  echo "ERROR: eval_test.py not found in $PROJECT_DIR"
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

echo "=== Starting evaluation ==="
date
# NOTE: training configuration is controlled in configs.py.
# If you want to change dataset_path/model_path, edit configs.py before submitting.
# (This script intentionally does not pass CLI flags because the repo doesn't use them.)
srun --exclusive -N1 -n1 python eval_test.py

EVAL_EXIT=$?
date
echo "=== Evaluation finished with code $EVAL_EXIT ==="
if [ $EVAL_EXIT -ne 0 ]; then
  echo "Evaluation failed."
  exit $EVAL_EXIT
fi

echo

echo "Done at $(date)"
