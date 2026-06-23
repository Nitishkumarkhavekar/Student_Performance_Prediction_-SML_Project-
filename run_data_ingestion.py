import sys
import os

# Ensure project root is on sys.path so `src` package is importable
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.components.data_injestion import DataIngestion

if __name__ == '__main__':
    di = DataIngestion()
    print(di.initiate_data_ingestion())
