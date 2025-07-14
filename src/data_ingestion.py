import psycopg2
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
from sklearn.model_selection import train_test_split
from config.database_config import DB_CONFIG
from config.paths_config import RAW_DIR, DATA_PATH
import os
import sys

logger = get_logger(__name__)

class DataIngestion:
    
    def __init__(self, db_config, output_dir):
        self.db_config = db_config
        self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def connect_to_db(self):
        try:
            conn = psycopg2.connect(**self.db_config)
            logger.info("Connected to the database successfully")
            return conn
        except Exception as e:
            logger.error(f"Error connecting to the database: {e}")
            raise CustomException(str(e), sys)
        
    def get_data(self):
        try:
            conn = self.connect_to_db()
            query = "SELECT * FROM public.ecommerce_data"
            data = pd.read_sql(query, conn)
            logger.info(f"Data fetched successfully from the database")
            conn.close()
            return data
        except Exception as e:
            logger.error(f"Error fetching data from the database: {e}")
            raise CustomException(str(e), sys)
        
    def save_data(self, data):
        try:
            data.to_csv(DATA_PATH, index=False)
            logger.info(f"Data saved successfully to {self.output_dir}")
        except Exception as e:
            logger.error(f"Error saving data to {self.output_dir}: {e}")
            raise CustomException(str(e), sys)
    
    def run(self):
        try:
            logger.info("Starting data ingestion...")
            data = self.get_data()
            self.save_data(data)
            logger.info("Data ingestion completed successfully")
        except Exception as e:
            logger.error(f"Error in data ingestion: {e}")
            raise CustomException(str(e), sys)
        
    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()

if __name__ == "__main__":
    data_ingestion = DataIngestion(DB_CONFIG, RAW_DIR)
    data_ingestion.run()