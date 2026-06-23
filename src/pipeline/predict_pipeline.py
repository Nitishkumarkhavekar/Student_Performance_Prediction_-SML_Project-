import os
import sys
from src.components.model_trainer import ModelTrainerconfig
from src.logger import logging
from src.custom_exception import CustomException
import pickle
import pandas as pd
import numpy as np

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        """
        features: dict with keys: ['gender', 'race/ethnicity', 'parental level of education', 
                                   'lunch', 'test preparation course', 'math score', 
                                   'reading score', 'writing score']
        """
        try:
            model_trainer_config = ModelTrainerconfig()
            with open(model_trainer_config.trained_model_file_path, 'rb') as f:
                model = pickle.load(f)
            logging.info("Trained model loaded successfully for prediction")
            
            preprocessor_path = model_trainer_config.trained_model_file_path.replace('model.pkl', 'preprocessor.pkl')
            with open(preprocessor_path, 'rb') as f:
                preprocessor_obj = pickle.load(f)
            
            # Create DataFrame with proper column names and order
            feature_order = ['gender', 'race/ethnicity', 'parental level of education', 
                           'lunch', 'test preparation course', 'math score', 
                           'reading score', 'writing score']
            
            if isinstance(features, dict):
                features_df = pd.DataFrame([features])
            else:
                # Assume it's a list/array with values in the correct order
                features_df = pd.DataFrame([[features]], columns=feature_order)
            
            # Ensure columns are in the correct order
            features_df = features_df[feature_order]
            
            data_scaled = preprocessor_obj.transform(features_df)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            raise CustomException(e, sys.exc_info())