import os
from datetime import datetime

from mltu.configs import BaseModelConfigs


class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join("Models/tutorial", datetime.strftime(datetime.now(), "%Y%m%d%H%M"))
        self.dataset_path = os.path.join("Datasets", "tutorial")
        self.vocab = ""
        self.height = 50
        self.width = 200
        self.max_text_length = 0
        self.batch_size = 128  # Increased from 64 for better CPU utilization
        self.learning_rate = 2e-3
        self.train_epochs = 1000
        self.train_workers = 20

