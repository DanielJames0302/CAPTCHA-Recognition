# CS4243 CAPTCHA REPO

## Requirements
- Python 3;
- TensorFlow 2.19
- mltu
- tf2onnx

## Training

### Configuration
The training configuration is managed in `configs.py`. Key parameters include:
- `dataset_path`: Path to your image dataset (default: `Datasets/tutorial`). If train_path and val_path are not specified, data will be split in 9:1 ratio.
- `dataset_train_path`: Path to your training dataset, optional
- `dataset_val_path`: Path to your validation dataset, optional
- `model_path`: Where to save the trained model
- `height`, `width`: Input image dimensions (default: 50x200)
- `batch_size`: Training batch size (default: 128)
- `learning_rate`: Learning rate for training (default: 2e-3)
- `train_epochs`: Maximum training epochs (default: 1000)

### Running Training
To start training the model:

```bash
python train.py
```

The training script will:
1. Load images from the dataset directory
2. Extract labels from filenames
3. Build vocabulary from all unique characters
4. Split data into training (90%) and validation (10%) sets
5. Apply data augmentation (brightness, rotation, erosion/dilation)
6. Train the model with early stopping and learning rate reduction
7. Save the best model to the specified model path
8. Export training and validation datasets as CSV files

### Training Output
During training, the following files will be created in the model directory:
- `model.h5`: Best trained model weights
- `model.onnx`: ONNX format model for inference
- `configs.yaml`: Model configuration
- `train.csv`: Training dataset split
- `val.csv`: Validation dataset split
- `logs/`: TensorBoard logs for monitoring training

### Resuming Training
To resume a training, simply specify an existing model path.

## Testing with Pre-trained Models

### Loading a Pre-trained Model
If you have a previously trained model, you can test it without retraining:

1. **Locate your trained model directory** (e.g., `Models/tutorial/202510041220/`)
2. **Ensure the following files exist:**
   - `model.onnx`: ONNX model file for inference
   - `configs.yaml`: Model configuration file
   - `val.csv` or `test.csv`: Test dataset (optional, for evaluation)

### Running Inference on Test Data
To test a pre-trained model on your validation/test dataset:

```bash
python inferenceModel.py
```

**Note:** You may need to update the model path in `inferenceModel.py`:
```python
configs = BaseModelConfigs.load("Models/tutorial/YOUR_MODEL_TIMESTAMP/configs.yaml")
```

### Evaluation Metrics
The testing script provides two key metrics:
- **CER (Character Error Rate)**: Percentage of character-level errors
- **WER (Word Error Rate)**: Percentage of word-level errors

Lower values indicate better performance. The script will output:
- Individual predictions for each test image
- Average CER and WER across the entire test dataset