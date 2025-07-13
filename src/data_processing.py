import pandas as pd
import sys
import nltk
from src.feature_store import RedisFeatureStore
from src.logger import get_logger
from src.custom_exception import CustomException
from utils.data_cleaning import data_cleaning
from utils.text_data_processing import text_data_processing
from utils.customers_clustering import customers_clustering
from utils.products_clustering import products_clustering
from utils.customers_radar import customers_radar
from config.paths_config import TRAIN_DATA_PATH

nltk.download('punkt_tab')
logger = get_logger(__name__)

class DataProcessing:
    def __init__(self, train_data_path, feature_store: RedisFeatureStore):
        self.train_data_path = train_data_path
        self.feature_store = feature_store
        self.train_data = None
        logger.info(f"Data Processing initialized")

    def load_data(self):
        try:
            self.train_data = pd.read_csv(self.train_data_path)
            logger.info(f"Data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException(e, sys)
        
    def preprocess_data(self):
        try:
            cleaned_data = data_cleaning(self.train_data)
            products_df, products_with_features = text_data_processing(cleaned_data)
            product_clusters_df = products_clustering(products_df, products_with_features)
            customer_clusters_df = customers_clustering(cleaned_data, product_clusters_df)
            cluster_morphology = customers_radar(customer_clusters_df)
            self.train_data = customer_clusters_df
            logger.info("Data Preprocessing completed")

        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise CustomException(e, sys)
        
    def store_feature_in_redis(self):
        try:
            batch_data = {}
            for _, row in self.train_data.iterrows():
                entity_id = row['CustomerID']
                features = {
                    'tsne_1': row['tsne_1'],
                    'tsne_2': row['tsne_2'],
                    'cat_0': row['cat_0'],
                    'cat_1': row['cat_1'],
                    'cat_2': row['cat_2'],
                    'cat_3': row['cat_3'],
                    'cat_4': row['cat_4'],
                    'order_count': row['order_count'],
                    'mean_order': row['mean_order'],
                    'total_spend': row['total_spend'],
                    'first_purchase_days': row['first_purchase_days'],
                    'last_purchase_days': row['last_purchase_days'],
                    'customer_clusters': row['customer_clusters'],
                }
                batch_data[entity_id] = features
            self.feature_store.store_batch_features(batch_data=batch_data)
            logger.info("Features stored in Redis successfully")
        except Exception as e:
            logger.error(f"Error storing features in Redis: {e}")
            raise CustomException(e, sys)
        
    def run(self):
        try:
            logger.info("Running Data Processing")
            self.load_data()
            self.preprocess_data()
            self.store_feature_in_redis()
            logger.info("Data Processing completed")
        except Exception as e:
            logger.error(f"Error running data processing: {e}")
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    feature_store = RedisFeatureStore()
    data_processor = DataProcessing(train_data_path=TRAIN_DATA_PATH, feature_store=feature_store)
    data_processor.run()