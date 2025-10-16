import os
from datetime import datetime

# Updated import path
from utils.configs import BaseModelConfigs

class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        #self.model_path = os.path.join("Models", datetime.strftime(datetime.now(), "%Y%m%d%H%M%S"))
        self.model_path = os.path.join("Models", "202510111835")
        self.trained_model_path = os.path.join("Models", "202510112239") # change this path for evaluating different trained models
        self.dataset_path = os.path.join("Datasets", "full", "train",) # path for training dataset
        self.dataset_test_path  = os.path.join("Datasets", "full", "val") # path for test dataset
        self.vocab = ""
        self.height = 80
        self.width = 300
        self.max_text_length = 0
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.train_epochs = 2000
        self.train_workers = 20