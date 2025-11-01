import os
import cv2
import math
import typing
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from torchvision import transforms

class DataProvider(tf.keras.utils.Sequence):
    def __init__(
        self,
        dataset: list,
        batch_size: int,
        data_preprocessors: list = None,
        transformers: list = None,
        augmentors: list = None,
        skip_validation: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.data_preprocessors = data_preprocessors or []
        self.transformers = transformers or []
        self.augmentors = augmentors or []
        self.skip_validation = skip_validation
        self.on_epoch_end()

        if not self.skip_validation:
            self.validate_data()

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch_items = [self.dataset[k] for k in indexes]

        X, Y = [], []
        for item in batch_items:
            # Separate file_path and label
            file_path, label = item
            
            # Preprocess the data
            for preprocessor in self.data_preprocessors:
                file_path, label = preprocessor(file_path, label)
            
            # Augment the data if augmentors are provided
            if self.augmentors:
                img_pil = Image.fromarray(cv2.cvtColor(file_path, cv2.COLOR_BGR2RGB))
                file_path = self.augmentors(img_pil)
                file_path = cv2.cvtColor(np.array(file_path), cv2.COLOR_RGB2BGR)

            # Transform the data
            for transformer in self.transformers:
                file_path, label = transformer(file_path, label)
            
            X.append(file_path)
            Y.append(label)

        return np.array(X), np.array(Y)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.dataset))
        np.random.shuffle(self.indexes)

    def split(self, split: float = 0.9, shuffle: bool = True):
        if shuffle:
            np.random.shuffle(self.dataset)
        
        split_index = int(len(self.dataset) * split)
        train_split = self.dataset[:split_index]
        val_split = self.dataset[split_index:]

        train_provider = DataProvider(
            dataset=train_split,
            batch_size=self.batch_size,
            data_preprocessors=self.data_preprocessors,
            transformers=self.transformers,
        )
        val_provider = DataProvider(
            dataset=val_split,
            batch_size=self.batch_size,
            data_preprocessors=self.data_preprocessors,
            transformers=self.transformers,
        )
        return train_provider, val_provider

    def to_csv(self, path: str):
        df = pd.DataFrame(self.dataset, columns=['file_path', 'label'])
        df.to_csv(path, index=False)

    def validate_data(self):
        validated_data = []
        for i, (file_path, label) in enumerate(self.dataset):
            try:
                # Add validation logic here, e.g., check if file exists
                if os.path.exists(file_path):
                    validated_data.append([file_path, label])
            except Exception as e:
                print(f"Skipping invalid data point: {file_path}, error: {e}")
        self.dataset = validated_data

# Custom ImageReader preprocessor
class ImageReader:
    def __init__(self, image_format='CV2'):
        self.image_format = image_format

    def __call__(self, image_path, label):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found or unable to read: {image_path}")
        return image, label

# Custom ImageResizer transformer
class ImageResizer:
    def __init__(self, width, height, keep_aspect_ratio=False, padding_color=(0, 0, 0)):
        self.width = width
        self.height = height
        self.keep_aspect_ratio = keep_aspect_ratio
        self.padding_color = padding_color

    def __call__(self, image, label):
        if self.keep_aspect_ratio:
            h, w, _ = image.shape
            ratio = min(self.width / w, self.height / h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            resized = cv2.resize(image, (new_w, new_h))
            
            delta_w = self.width - new_w
            delta_h = self.height - new_h
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            
            padded_image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.padding_color)
            return padded_image, label
        else:
            return cv2.resize(image, (self.width, self.height)), label

# Custom LabelIndexer transformer
class LabelIndexer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, data, label):
        return data, np.array([self.vocab.index(l) for l in label if l in self.vocab])

# Custom LabelPadding transformer
class LabelPadding:
    def __init__(self, max_word_length, padding_value):
        self.max_word_length = max_word_length
        self.padding_value = padding_value

    def __call__(self, data, label):
        padded_label = np.pad(label, (0, self.max_word_length - len(label)), 'constant', constant_values=self.padding_value)
        return data, padded_label