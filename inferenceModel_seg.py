
import os
import logging
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from mltu.configs import BaseModelConfigs

# Import from the new utils directory
from custom_transformers import GrayscaleTransformer
from character_extraction.extract_char import preprocess_image, get_center_aligned_boxes, handle_connected_characters
from utils.data_provider import ImageResizer
from utils.text_utils import ctc_decoder, get_cer, get_wer

class ImageToWordModel:
    def __init__(self, model_path: str, char_list, width, height):
        self.char_list = char_list
        self.model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        self.width = width
        self.height = height

    def predict(self, image: np.ndarray):
        resizer = ImageResizer(self.width, self.height, keep_aspect_ratio=True, padding_color=(255,255,255))
        image, _ = resizer(image, None)

        grayscaler = GrayscaleTransformer(keep_3_channels=True)
        image, _ = grayscaler(image, None)

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        
        preds = self.model.predict(image_pred)
        if preds.ndim == 2:
            preds = np.expand_dims(preds, axis=0)
            
        return ctc_decoder(preds, self.char_list)[0]

# Run test dataset without training
if __name__ == "__main__":
    model_directory = "Models/20251101192227"
    model_path = f"{model_directory}/model.keras"
    configs = BaseModelConfigs.load(f"{model_directory}/configs.yaml")

    # Ensure model directory exists and configure logging to file
    os.makedirs(model_directory, exist_ok=True)
    log_path = os.path.join(model_directory, "inference.txt")

    logger = logging.getLogger("inference")
    logger.setLevel(logging.INFO)
    # avoid adding multiple handlers if this file is imported/run multiple times
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        # also log to console
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    model = ImageToWordModel(model_path=model_path, char_list=configs.vocab, width=configs.width, height=configs.height)

    df = pd.read_csv(f"{model_directory}/val.csv").values.tolist()

    accum_cer = []
    accum_wer = []
    for image_path, label in tqdm(df):
        image = cv2.imread(image_path.replace("\\", "/"))
        
        image_cleaned, gray, processed = preprocess_image(image, visualize=False)

        boxes, vis = get_center_aligned_boxes(processed, gray, visualize=False, merge_thresh=1)
        segmentations = handle_connected_characters(image_cleaned, boxes, visualize=False)

        prediction_text = ""
        for _, char_image in segmentations:
            prediction_char = model.predict(char_image)
            prediction_text += prediction_char

        cer = get_cer(prediction_text, label)
        wer = get_wer(prediction_text, label)
        logger.info(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer:.3f}, WER: {wer:.3f}")

        accum_cer.append(cer)
        accum_wer.append(wer)

    logger.info(f"Average CER: {np.average(accum_cer):.3f}")
    logger.info(f"Average WER: {np.average(accum_wer):.3f}")
