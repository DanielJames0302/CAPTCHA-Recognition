import os
from datetime import datetime

from mltu.configs import BaseModelConfigs


class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = "Models/small_no_ambiguous/202510071838"
        #self.model_path = os.path.join("Models/small_no_ambiguous", datetime.strftime(datetime.now(), "%Y%m%d%H%M"))
        self.dataset_path = os.path.join("Datasets", "small_no_ambiguous")
        self.dataset_train_path = os.path.join(self.dataset_path, "train")
        self.dataset_val_path = os.path.join(self.dataset_path, "val")
        self.vocab = ""
        self.height = 50
        self.width = 300
        self.max_text_length = 0
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.train_epochs = 2000
        self.train_workers = 20

