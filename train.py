import tensorflow as tf

import data_helper
try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass

# Optimize CPU performance by using all available cores
#tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores for operations between ops
#tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores for operations within ops
#print("CPU threading optimization enabled - using all available CPU cores")

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from mltu.tensorflow.dataProvider import DataProvider
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.metrics import CWERMetric

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate
from mltu.annotations.images import CVImage

from custom_transformers import GrayscaleTransformer

from model import train_model
from configs import ModelConfigs

import os
# Removed unused imports: urlopen, BytesIO, ZipFile

configs = ModelConfigs()

# Build vocabulary from dataset (regardless of train/val split method)
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
    
# Augment training data with random brightness, rotation and erode/dilate
train_data_provider.augmentors = [RandomBrightness(), RandomRotate(), RandomErodeDilate()]

# Define path to save the model
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

    # Print model summary
    model.summary(line_length=110)

    # Define callbacks
    earlystopper = EarlyStopping(monitor="val_CER", patience=50, verbose=1, mode="min")
    checkpoint = ModelCheckpoint(f"{configs.model_path}/model.keras", monitor="val_CER", verbose=1, save_best_only=True, mode="min")
    trainLogger = TrainLogger(configs.model_path)
    tb_callback = TensorBoard(f"{configs.model_path}/logs", update_freq=1)
    reduceLROnPlat = ReduceLROnPlateau(monitor="val_CER", factor=0.9, min_delta=1e-10, patience=20, verbose=1, mode="min")
    model2onnx = Model2onnx(f"{configs.model_path}/model.keras")

    # Train the model
    model.fit(
        train_data_provider,
        validation_data=val_data_provider,
        epochs=configs.train_epochs,
        callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback, model2onnx],
        validation_freq=5,
    )