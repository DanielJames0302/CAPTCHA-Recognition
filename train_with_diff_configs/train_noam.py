import os
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from torchvision import transforms
import data_helper_proj
from model_proj import train_model
from configs_noam import ModelConfigs

# Import from the new utils directory
from utils.data_provider import DataProvider, ImageReader, ImageResizer, LabelIndexer, LabelPadding
#from utils.losses import CTCloss
#from utils.callbacks import Model2onnx, TrainLogger
#from utils.metrics import CWERMetric


from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.metrics import CWERMetric

# GPU memory growth
try:
    [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except:
    pass

# Load configs
configs = ModelConfigs()

# Build vocabulary from the dataset
vocab, max_len = set(), 0
dataset = data_helper_proj.load_dataset_from_path(configs.dataset_path)

for _, label in dataset:
    vocab.update(list(label))
    max_len = max(max_len, len(label))

# Save vocab and max length to configs
configs.vocab = "".join(sorted(vocab))
configs.max_text_length = max_len
configs.save()

# Create data provider
data_provider = data_helper_proj.CreateDataProvider(dataset, configs)

# Split dataset
train_data_provider, val_data_provider = data_provider.split(split=0.9)

# Define augmentations using torchvision
augmentation_pipeline = transforms.Compose([
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    # Add other transforms as needed, e.g., transforms.GaussianBlur(kernel_size=3)
])

# Assign augmentation pipeline to the training data provider
train_data_provider.augmentors = augmentation_pipeline

# Create model path directory
os.makedirs(configs.model_path, exist_ok=True)

# Load or create model
model_file_path = os.path.join(configs.model_path, "model.keras")
if os.path.exists(model_file_path):
    print(f"Loading existing model from {model_file_path}")
    try:
        model = tf.keras.models.load_model(model_file_path, compile=False, safe_mode=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        model = None
else:
    model = None

if model is None:
    print("Creating a new model.")
    model = train_model(
        input_dim=(configs.height, configs.width, 3),
        output_dim=len(configs.vocab),
    )

# Compile and train the model
if model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate),
        loss=CTCloss(),
        metrics=[CWERMetric(padding_token=len(configs.vocab))],
        run_eagerly=False
    )
    model.summary(line_length=110)

    callbacks = [
        EarlyStopping(monitor="val_CER", patience=50, verbose=1, mode="min"),
        ModelCheckpoint(model_file_path, monitor="val_CER", verbose=1, save_best_only=True, mode="min"),
        TrainLogger(configs.model_path),
        TensorBoard(os.path.join(configs.model_path, "logs"), update_freq=1),
        ReduceLROnPlateau(monitor="val_CER", factor=0.9, min_delta=1e-10, patience=20, verbose=1, mode="min"),
        Model2onnx(model_file_path)
    ]

    model.fit(
        train_data_provider,
        validation_data=val_data_provider,
        epochs=configs.train_epochs,
        callbacks=callbacks
    )

    train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
    val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))