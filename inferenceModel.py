import cv2
import typing
import numpy as np
import os
import time
from collections import deque
import onnxruntime as ort
import pandas as pd
from tqdm import tqdm

# Import from the new utils directory
from utils.text_utils import ctc_decoder, get_cer, get_wer
from utils.configs import BaseModelConfigs

class FpsWrapper:
    """ Decorator to calculate the frames per second of a function
    """
    def __init__(self, func: typing.Callable):
        self.func = func
        self.fps_list = deque([], maxlen=100)

    def __call__(self, *args, **kwargs):
        start = time.time()
        results = self.func(self.instance, *args, **kwargs)
        self.fps_list.append(1 / (time.time() - start))
        self.instance.fps = np.mean(self.fps_list)
        return results

    def __get__(self, instance, owner):
        self.instance = instance
        return self.__call__.__get__(instance, owner)


class OnnxInferenceModel:
    """ Base class for all inference models that use onnxruntime 

    Attributes:
        model_path (str, optional): Path to the model folder. Defaults to "".
        force_cpu (bool, optional): Force the model to run on CPU or GPU. Defaults to GPU.
        default_model_name (str, optional): Default model name. Defaults to "model.onnx".
    """
    def __init__(
        self, 
        model_path: str = "",
        force_cpu: bool = False,
        default_model_name: str = "model.onnx",
        *args, **kwargs
        ):
        self.model_path = model_path.replace("\\", "/")
        self.force_cpu = force_cpu
        self.default_model_name = default_model_name

        # check if model path is a directory with os path
        if os.path.isdir(self.model_path):
            self.model_path = os.path.join(self.model_path, self.default_model_name)

        if not os.path.exists(self.model_path):
            raise Exception(f"Model path ({self.model_path}) does not exist")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" and not force_cpu else ["CPUExecutionProvider"]

        self.model = ort.InferenceSession(self.model_path, providers=providers)

        self.metadata = {}
        if self.model.get_modelmeta().custom_metadata_map:
            # add metadata to self object
            for key, value in self.model.get_modelmeta().custom_metadata_map.items():
                try:
                    new_value = eval(value) # in case the value is a list or dict
                except:
                    new_value = value
                self.metadata[key] = new_value
                
        # Update providers priority to only CPUExecutionProvider
        if self.force_cpu:
            self.model.set_providers(["CPUExecutionProvider"])

        self.input_shapes = [meta.shape for meta in self.model.get_inputs()]
        self.input_names = [meta.name for meta in self.model._inputs_meta]
        self.output_names = [meta.name for meta in self.model._outputs_meta]

    def predict(self, data: np.ndarray, *args, **kwargs):
        raise NotImplementedError

    @FpsWrapper
    def __call__(self, data: np.ndarray):
        results = self.predict(data)
        return results


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
if __name__ == "__main__":
    configs = BaseModelConfigs.load("Models/02_captcha_to_text/202510041220/configs.yaml")

    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

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