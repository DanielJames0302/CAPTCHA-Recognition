# eval_test.py
import os
import sys
from datetime import datetime
import tensorflow as tf

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.annotations.images import CVImage
from mltu.tensorflow.metrics import CWERMetric
from mltu.tensorflow.losses import CTCloss

import data_helper
from configs import ModelConfigs


def log(msg: str):
    print(f"[eval] {msg}", flush=True)


def find_configs_file(model_dir: str) -> str:
    """Return full path to saved configs file inside model_dir."""
    candidates = [
        "configs.yaml", "configs.yml", "configs.json",
        "config.yaml", "config.yml", "config.json",
    ]
    for name in candidates:
        path = os.path.join(model_dir, name)
        if os.path.isfile(path):
            log(f"Found configs file: {path}")
            return path
    raise FileNotFoundError(
        f"No configs file found in {model_dir}. Expected one of: {', '.join(candidates)}"
    )


def load_configs_from_file(cfg_path: str) -> ModelConfigs:
    """Read YAML/JSON manually and populate a ModelConfigs instance."""
    # Lazy import to avoid hard dependency if not needed
    import json
    try:
        import yaml  # PyYAML
    except Exception:
        yaml = None

    with open(cfg_path, "r") as f:
        text = f.read()

    data = None
    if cfg_path.endswith((".yaml", ".yml")):
        if yaml is None:
            raise RuntimeError("PyYAML not installed but a YAML config is provided.")
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text or "{}")

    cfg = ModelConfigs()
    # only set attributes that exist on ModelConfigs
    for k, v in (data.items() if isinstance(data, dict) else []):
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    # Hard guards: these must be valid
    if not isinstance(cfg.vocab, str) or len(cfg.vocab) == 0:
        raise RuntimeError(f"Invalid vocab loaded from {cfg_path!r} (length={len(getattr(cfg, 'vocab', ''))}).")
    if int(getattr(cfg, "max_text_length", 0)) <= 0:
        raise RuntimeError(f"Invalid max_text_length loaded from {cfg_path!r}: {getattr(cfg, 'max_text_length', None)}")

    return cfg


def build_test_provider(cfg: ModelConfigs, dataset):
    """Build a DataProvider that mirrors training-time preprocessing."""
    from mltu.tensorflow.dataProvider import DataProvider
    return DataProvider(
        dataset=dataset,
        skip_validation=True,
        batch_size=cfg.batch_size,
        shuffle=False,  # deterministic eval
        data_preprocessors=[ImageReader(CVImage)],
        transformers=[
            ImageResizer(cfg.width, cfg.height),
            LabelIndexer(cfg.vocab),
            LabelPadding(
                max_word_length=cfg.max_text_length,
                padding_value=len(cfg.vocab),
            ),
        ],
    )


def main() -> int:
    log("Starting eval_test.py")

    # --- Resolve model file (ENV can override) ---
    default_model_file = os.path.join(ModelConfigs().trained_model_path, "model.keras")
    model_file = os.environ.get("MODEL_FILE", default_model_file)
    log(f"MODEL_FILE={model_file}")
    if not os.path.isfile(model_file):
        log(f"[ERROR] Model file not found: {model_file}")
        return 2
    model_dir = os.path.dirname(model_file)
    log(f"model_dir={model_dir}")

    # --- Load the trained run's saved configs (critical: vocab, max_text_length) ---
    try:
        cfg_file = find_configs_file(model_dir)
        configs = load_configs_from_file(cfg_file)  # manual YAML/JSON load
    except Exception as e:
        log(f"[ERROR] Failed to load saved configs: {e}")
        return 3

    log(
        f"vocab size={len(configs.vocab)} max_text_length={configs.max_text_length} "
        f"image_size=(H={configs.height}, W={configs.width}) batch_size={configs.batch_size}"
    )

    # --- Resolve test dataset path (ENV can override) ---
    test_path = os.environ.get("DATASET_TEST_PATH", getattr(configs, "dataset_test_path", None))
    log(f"DATASET_TEST_PATH={test_path}")
    if not test_path or not os.path.isdir(test_path):
        log(f"[ERROR] Invalid test dataset path: {test_path}")
        return 4

    # --- Load test pairs and FILTER invalid labels for CTC ---
    # Keep only non-empty labels, all chars in training vocab, and length ≤ max_text_length.
    dataset_raw = data_helper.load_dataset_from_path(test_path)
    vocab_set = set(configs.vocab)

    bad_empty = bad_oov = bad_too_long = 0
    filtered = []
    for p, y in dataset_raw:
        if not isinstance(y, str) or len(y) == 0:
            bad_empty += 1
            continue
        if len(y) > configs.max_text_length:
            bad_too_long += 1
            continue
        if not all(ch in vocab_set for ch in y):
            bad_oov += 1
            continue
        filtered.append((p, y))

    log(
        f"Loaded {len(dataset_raw)} samples; kept={len(filtered)} "
        f"drop_empty={bad_empty} drop_oov={bad_oov} drop_too_long={bad_too_long}"
    )

    if not filtered:
        log("[ERROR] No test samples remain after filtering. "
            "Test set may contain OOV chars or labels longer than training max_text_length.")
        return 5

    # --- Build provider mirroring training pipeline ---
    test_dp = build_test_provider(configs, filtered)

    # --- Load and compile model with the same loss/metric as training ---
    try:
        model = tf.keras.models.load_model(model_file, compile=False, safe_mode=False)
    except Exception as e:
        log(f"[ERROR] Failed to load model: {e}")
        return 6

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate),
        loss=CTCloss(),
        metrics=[CWERMetric(padding_token=len(configs.vocab))],
        run_eagerly=False,
        steps_per_execution=64,  # throughput boost; does not change numerics
    )

    # --- Prepare output dir next to the model ---
    eval_dir = os.path.join(model_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    log(f"eval_dir={eval_dir}")

    # --- Evaluate ---
    results = model.evaluate(test_dp, return_dict=True, verbose=1)
    log(f"Results keys: {list(results.keys())}")

    # --- Save results to timestamped file ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_txt = os.path.join(eval_dir, f"test_results_{ts}.txt")
    with open(out_txt, "w") as f:
        f.write(f"=== Test Results ({ts}) ===\n")
        for k, v in results.items():
            try:
                f.write(f"{k}: {float(v):.6f}\n")
            except Exception:
                f.write(f"{k}: {v}\n")
    log(f"Saved: {out_txt}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

