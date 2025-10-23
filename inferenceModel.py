import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer

from test_accuracy import test_accuracy

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text

# Run test dataset without training
# compute character recognition rate and word recognition rate for each epoch
if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    configs = BaseModelConfigs.load("Models/02_captcha_to_text/202510041220/configs.yaml")

    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    df = pd.read_csv("Models/02_captcha_to_text/202510041220/val.csv").values.tolist()



    accum_char_accuracy = []
    accum_word_accuracy = []
    for image_path, label in tqdm(df):
        image = cv2.imread(image_path.replace("\\", "/"))

        if image is None:
            print(f"Image not found: {image_path}")
            continue

        prediction_text = model.predict(image)

        char_accuracy, word_accuracy = test_accuracy(prediction_text, label)

        print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, Char Accuracy: {char_accuracy:.2f}%, Word Accuracy: {word_accuracy:.2f}%")

        accum_char_accuracy.append(char_accuracy)
        accum_word_accuracy.append(word_accuracy)

    print(f"Average Character Accuracy: {np.average(accum_char_accuracy):.3f}")
    print(f"Average Word Accuracy: {np.average(accum_word_accuracy):.3f}")