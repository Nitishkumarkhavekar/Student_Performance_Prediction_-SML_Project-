import os
import sys
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.custom_exception import CustomException

class TrainerPipeline:
    def __init__(self):
        pass
    def initiate_trainer_pipeline(self):
        try:
            logging.info("Model trainer pipeline is initiated")
            model_trainer = ModelTrainer()
            model_trainer.initiate_model_trainer()
            logging.info("Model trainer pipeline is completed successfully")
        except Exception as e:
            raise CustomException(e, sys.exc_info())