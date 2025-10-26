import cv2
import numpy as np
import tensorflow as tf

from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer

class ImageToWordModel:
    def __init__(self, model_path: str, char_list, width, height):
        self.char_list = char_list
        self.model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        self.width = width
        self.height = height

    def predict(self, image: np.ndarray):
        # @todo: Apply proper preprocessing steps
        image = cv2.resize(image, (self.width, self.height))

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.predict(image_pred)
        return ctc_decoder(preds, self.char_list)[0]

# Run test dataset without training
if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    configs = BaseModelConfigs.load("Models/02_captcha_to_text/202510041220/configs.yaml")

    model_path = f"{configs.model_path}/model.keras"
    model = ImageToWordModel(model_path=model_path, char_list=configs.vocab, width=configs.width, height=configs.height)

    df = pd.read_csv("Models/02_captcha_to_text/202510041220/val.csv").values.tolist()

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