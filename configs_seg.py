import os
from datetime import datetime

from utils.configs import BaseModelConfigs


class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        #self.model_path = os.path.join("Models", datetime.strftime(datetime.now(), "%Y%m%d%H%M%S"))
        self.model_path = os.path.join("Models", "seg_1Aug_v2")
        self.trained_model_path = os.path.join("Models", "seg_1Aug_v2") # change this path for evaluating different trained models
        self.dataset_path = os.path.join("Datasets", "full", "train", "segmented") # path for training dataset
        self.dataset_val_path  = os.path.join("Datasets", "full", "val", "segmented") # path for validation dataset if needed
        self.dataset_test_path  = os.path.join("Datasets", "full", "val") # path for test dataset
        self.vocab = ""
        self.height = 50
        self.width = 50
        self.max_text_length = 1
        self.batch_size = 32
        self.learning_rate = 3e-4
        self.train_epochs = 500
        self.train_workers = 20

