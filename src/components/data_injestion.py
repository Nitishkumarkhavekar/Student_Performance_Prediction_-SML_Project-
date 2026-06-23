import numpy as np
import pandas as pd
import os
from src.logger import logging
from src.custom_exception import CustomException
import sys
from dataclasses import dataclass

@dataclass
class DataIngestionConfig():
    raw_data_path: str = os.path.join('artifacts', 'raw_data.csv')
    train_data_path: str = os.path.join('artifacts', 'train_data.csv')
    test_data_path: str = os.path.join('artifacts', 'test_data.csv')

class DataIngestion():
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            source_data_path = os.path.join('artifacts', 'raw_data.csv')
            if not os.path.exists(source_data_path):
                source_data_path = os.path.join('notebook', 'data', 'stud.csv')

            df = pd.read_csv(source_data_path)
            logging.info("Read the dataset as dataframe from %s", source_data_path)

            df["total_score"] = df["math score"] + df["reading score"] + df["writing score"]

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Saved the raw data to path: %s", self.ingestion_config.raw_data_path)

            from sklearn.model_selection import train_test_split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Split the data into train and test sets and saved to paths: %s and %s", 
                         self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys.exc_info())