import os
import pandas as pd
from typing import List

# Updated import paths
from utils.data_provider import DataProvider, ImageReader, ImageResizer, LabelIndexer, LabelPadding
from custom_transformers import GrayscaleTransformer

def load_dataset_from_path(path: str) -> List[List[str]]:
    """
    Load dataset from image directory where filenames contain labels.
    
    Args:
        path (str): Path to directory containing CAPTCHA images
    
    Returns:
        List of [file_path, label] pairs
        
    The function expects image filenames to follow the pattern: "label-something.extension"
    where the label is the part before the first hyphen.
    """
    if not os.path.exists(path):
        raise ValueError(f"Directory path does not exist: {path}")
    
    if not os.path.isdir(path):
        raise ValueError(f"Path is not a directory: {path}")
    
    dataset = []
    files = os.listdir(path)
    
    if not files:
        raise ValueError(f"No files found in directory: {path}")
    
    for file in files:
        file_path = os.path.join(path, file)
        
        # Skip if not a file
        if not os.path.isfile(file_path):
            continue
        
        # Get filename without extension
        filename_without_ext = os.path.splitext(file)[0]
        
        delims = ['-', '_']
        # Extract label (part before the first delimiter)
        label = None
        for delim in delims:
            if delim in filename_without_ext:
                label = filename_without_ext.split(delim)[0]
                break
        if not label:
            continue
        
        # Skip if label is empty
        if not label:
            continue
        
        dataset.append([file_path, label])
    
    if not dataset:
        raise ValueError("No valid CAPTCHA files found in the directory")
    
    return dataset


def load_dataset_from_csv(path: str) -> List[List[str]]:
    """
    Load dataset from CSV file.
    
    Args:
        path (str): Path to CSV file containing file_path and label columns
    
    Returns:
        List of [file_path, label] pairs
    """
    if not os.path.exists(path):
        raise ValueError(f"CSV file not found: {path}")
    
    df = pd.read_csv(path)
    if 'file_path' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV file must contain 'file_path' and 'label' columns")
    
    dataset = df[['file_path', 'label']].values.tolist()
    return dataset


def save_dataset_to_csv(data_provider, path: str) -> None:
    """
    Save a DataProvider dataset to CSV file.
    
    Args:
        data_provider: DataProvider object with dataset attribute
        path (str): Path where CSV file will be saved
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Extract dataset from data provider
    dataset = data_provider.dataset
    
    # Convert to DataFrame and save
    df = pd.DataFrame(dataset, columns=['file_path', 'label'])
    df.to_csv(path, index=False)


def CreateDataProvider(dataset: List[List[str]], configs) -> DataProvider:
    """
    Create a DataProvider with standardized transformers for CAPTCHA training.
    
    Args:
        dataset: List of [file_path, label] pairs
        configs: ModelConfigs object containing training configuration
    
    Returns:
        DataProvider object ready for training or validation
    """
    data_provider = DataProvider(
        dataset=dataset,
        skip_validation=True,
        batch_size=configs.batch_size,
        # ImageReader no longer needs CVImage
        data_preprocessors=[ImageReader()],
        transformers=[
            ImageResizer(configs.width, configs.height, keep_aspect_ratio=True, padding_color=(255,255,255)),
            GrayscaleTransformer(keep_3_channels=True), 
            LabelIndexer(configs.vocab),
            LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
        ],
    )
    return data_provider