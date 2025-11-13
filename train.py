import os
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from torchvision import transforms
import data_helper
from model import train_model
from configs import ModelConfigs

# Import from the new utils directory
from utils.losses import CTCloss
from utils.callbacks import Model2onnx, TrainLogger
from utils.metrics import CWERMetric

# GPU memory growth
try:
    [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except:
    pass

# Load configs
configs = ModelConfigs()

# Build vocabulary from the dataset
vocab, max_len = set(), 0
train_data_provider, val_data_provider = None, None

if configs.use_test_set_for_val:
    # Use test set as validation set during training
    if os.path.exists(configs.dataset_path) and os.path.exists(configs.dataset_test_path):
        train_dataset = data_helper.load_dataset_from_path(configs.dataset_path)
        val_dataset = data_helper.load_dataset_from_path(configs.dataset_test_path)

        # Build vocab from both train and val datasets
        for file_path, label in train_dataset + val_dataset:
            vocab.update(list(label))
            max_len = max(max_len, len(label))

        # Save vocab and maximum text length to configs
        configs.vocab = "".join(sorted(vocab))
        configs.max_text_length = max_len
        configs.save()

        train_data_provider = data_helper.CreateDataProvider(train_dataset, configs)
        val_data_provider = data_helper.CreateDataProvider(val_dataset, configs)
    else:
        print("Train/Test directories not found. Please ensure the dataset directories exist.")
        exit(1)
elif os.path.exists(configs.dataset_path):
    # Use a single dataset and split into train/val sets
    dataset = data_helper.load_dataset_from_path(configs.dataset_path)

    # Build vocab from single dataset
    for file_path, label in dataset:
        vocab.update(list(label))
        max_len = max(max_len, len(label))
    
    # Save vocab and maximum text length to configs
    configs.vocab = "".join(sorted(vocab))
    configs.max_text_length = max_len
    configs.save()

    data_provider = data_helper.CreateDataProvider(dataset, configs)
    # Split the dataset into training and validation sets
    train_data_provider, val_data_provider = data_provider.split(split = 0.9)
else:
    print("Train directory not found. Please ensure the dataset directory exists.")
    exit(1)
    

# Define augmentations using torchvision
augmentation_pipeline = transforms.Compose([
    transforms.RandomRotation(degrees=5),
    # Add other transforms as needed, e.g., transforms.GaussianBlur(kernel_size=3)
])

# Assign augmentation pipeline to the training data provider
train_data_provider.augmentors = augmentation_pipeline

# Create model path directory
os.makedirs(configs.model_path, exist_ok=True)

# Save training and validation datasets as csv files
train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))

model = None

# Check if model already exists and load it, otherwise create new model
model_file_path = f"{configs.model_path}/model.keras"
if os.path.exists(model_file_path):
    print(f"Loading existing model from {model_file_path}")
    try:
        model = tf.keras.models.load_model(
            model_file_path,
            compile=False,
            safe_mode=False  # Allow loading Lambda layers
        )
        print("Model loaded successfully - resuming training")
    except Exception as e:
        print(f"Failed to load existing model: {e}")

else:
    # Creating TensorFlow model architecture
    model = train_model(
        input_dim = (configs.height, configs.width, 3),
        output_dim = len(configs.vocab),
    )
    
    print("New model created")    

if model:
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate), 
        loss=CTCloss(), 
        metrics=[CWERMetric(padding_token=len(configs.vocab))],
        run_eagerly=False,
        steps_per_execution=64,
    )
    model.summary(line_length=110)

    callbacks = [
        EarlyStopping(monitor="CER", patience=50, verbose=1, mode="min"),
        ModelCheckpoint(model_file_path, monitor="CER", verbose=1, save_best_only=True, mode="min"),
        TrainLogger(configs.model_path),
        TensorBoard(os.path.join(configs.model_path, "logs"), update_freq=1),
        ReduceLROnPlateau(monitor="CER", factor=0.9, min_delta=1e-10, patience=20, verbose=1, mode="min"),
        Model2onnx(model_file_path)
    ]

    model.fit(
        train_data_provider,
        validation_data=val_data_provider,
        epochs=configs.train_epochs,
        callbacks=callbacks,
        validation_freq=5,
    )