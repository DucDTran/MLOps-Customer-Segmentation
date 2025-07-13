from src.data_ingestion import DataIngestion
from src.model_training import ModelTraining
from config.database_config import DB_CONFIG
from config.paths_config import RAW_DIR, TRAIN_DATA_PATH
from src.feature_store import RedisFeatureStore
from src.data_processing import DataProcessing

if __name__ == "__main__":
    data_ingestion = DataIngestion(DB_CONFIG, RAW_DIR)
    data_ingestion.run()

    feature_store = RedisFeatureStore()
    data_processor = DataProcessing(train_data_path=TRAIN_DATA_PATH, feature_store=feature_store)
    data_processor.run()

    model_training = ModelTraining(feature_store)
    model_training.run()
