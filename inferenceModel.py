import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

# Import from the new utils directory
from custom_transformers import GrayscaleTransformer
from utils.data_provider import ImageResizer
from utils.text_utils import ctc_decoder, get_cer, get_wer
from utils.configs import BaseModelConfigs

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
        return ctc_decoder(preds, self.char_list)[0]

# Run test dataset without training
if __name__ == "__main__":
    model_directory = "Models/20251101165903"
    model_path = f"{model_directory}/model.keras"
    configs = BaseModelConfigs.load(f"{model_directory}/configs.yaml")
    model = ImageToWordModel(model_path=model_path, char_list=configs.vocab, width=configs.width, height=configs.height)

    df = pd.read_csv(f"{model_directory}/val.csv").values.tolist()

    accum_cer = []
    accum_wer = []
    for image_path, label in tqdm(df):
        image = cv2.imread(image_path.replace("\\", "/"))

        prediction_text = model.predict(image)

        cer = get_cer(prediction_text, label)
        wer = get_wer(prediction_text, label)
        print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer:.3f}, WER: {wer:.3f}")

        accum_cer.append(cer)
        accum_wer.append(wer)

    print(f"Average CER: {np.average(accum_cer):.3f}")
    print(f"Average WER: {np.average(accum_wer):.3f}")