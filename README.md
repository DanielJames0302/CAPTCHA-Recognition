# CS4243 CAPTCHA REPO

## Requirements
- Python 3
- TensorFlow 2.19
- mltu
- tf2onnx
- PyTorch (for GAN training and generation)
- torchvision
- PIL/Pillow
- matplotlib
- imageio
- numpy

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

## GAN Training

This repository also includes a Deep Convolutional GAN (DCGAN) implementation for generating synthetic CAPTCHA character images.

### GAN Configuration

The GAN training script (`train_gan_improved.py`) uses the following default settings:
- **Image size**: 64x64 pixels (grayscale)
- **Dataset path**: `Datasets/full/train/segmented` (segmented character images)
- **Batch size**: 128
- **Training epochs**: 200
- **Latent dimension (z)**: 100
- **Generator learning rate**: 1e-4
- **Discriminator learning rate**: 4e-4
- **Loss function**: Hinge loss

### Running GAN Training

To train the GAN model locally:

```bash
python train_gan_improved.py
```

**Note:** Before running, ensure you have:
1. A directory containing segmented character images (PNG format)
2. Update the `dataset_path` variable in `train_gan_improved.py` if your dataset is in a different location

### Running GAN Training on SLURM Cluster

If you're using a SLURM cluster, you can submit the training job:

```bash
sbatch train_gan_improved_slurm.sh
```

The SLURM script will:
- Request 1 GPU for 24 hours
- Activate the virtual environment
- Run the training script
- Save all outputs to `logs/` directory

### GAN Training Output

After training completes, a timestamped results directory will be created (e.g., `Captcha_DCGAN_results_20251111154440/`) containing:

- **Model weights:**
  - `generator_param.pkl`: Trained generator weights
  - `discriminator_param.pkl`: Trained discriminator weights
  
- **Training history:**
  - `train_hist.pkl`: Training loss history (pickle file)
  - `Captcha_DCGAN_train_hist.png`: Loss curves visualization
  
- **Generated samples:**
  - `Random_results/`: Randomly sampled images at each epoch
  - `Fixed_results/`: Images generated from fixed noise vectors (for tracking training progress)
  - `generation_animation.gif`: Animated GIF showing training progress over epochs

### GAN Architecture

- **Generator**: Transposed convolutional network that upsamples from 100-dimensional noise to 64x64 grayscale images
- **Discriminator**: Convolutional network with spectral normalization that classifies real vs. fake images

## Generating CAPTCHA Images

Once you have a trained GAN generator, you can use it to generate synthetic CAPTCHA images.

### Configuration

The generation script (`generate_captcha_gan.py`) has the following parameters:
- **Generator weights path**: Path to `generator_param.pkl` (update line 48 in the script)
- **Number of characters**: Randomly sampled between 5-8 characters per CAPTCHA
- **Character size**: 64x64 pixels
- **Final CAPTCHA dimensions**: 500x80 pixels
- **Output directory**: `generated_captchas/`

### Running CAPTCHA Generation

1. **Update the generator weights path** in `generate_captcha_gan.py`:
   ```python
   save_root = "Captcha_DCGAN_results_20251111154440"  # Your results directory
   ```

2. **Run the generation script**:
   ```bash
   python generate_captcha_gan.py
   ```

The script will:
- Load the trained generator model
- Generate 100 CAPTCHA images (default)
- Save each image with a timestamp to `generated_captchas/` directory

### Running Generation on SLURM Cluster

To generate CAPTCHA images on a SLURM cluster:

```bash
sbatch generate_captcha_gan.sh
```

The script will automatically:
- Find the latest generator weights in `Captcha_DCGAN_results_*/` directories
- Run the generation process
- Save outputs to `generated_captchas/` directory

### Generation Process

Each generated CAPTCHA image is created by:
1. Randomly selecting the number of characters (5-8)
2. Generating individual character images using the trained generator
3. Concatenating characters horizontally
4. Resizing to final CAPTCHA dimensions (500x80)

The generated images are saved as grayscale PNG files with timestamps in their filenames.