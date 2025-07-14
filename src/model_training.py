import mlflow.xgboost
from src.logger import get_logger
from src.custom_exception import CustomException
from src.feature_store import RedisFeatureStore
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.feature_engineering import feature_engineering
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from joblib import dump, load
from config.paths_config import MODELS_DIR, PROCESSED_DIR
import os
import mlflow

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, feature_store: RedisFeatureStore, model_save_path = MODELS_DIR, processed_data_path = PROCESSED_DIR):
        self.feature_store = feature_store
        self.model_save_path = model_save_path
        self.train_data = None
        self.processed_data_path = processed_data_path

        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.processed_data_path, exist_ok=True)
        logger.info(f"Model Training initialized")

    def load_data_from_redis(self, entity_ids):
        try:
            logger.info("Extracting data from Redis")
            data = []
            for entity_id in entity_ids:
                features = self.feature_store.get_features(entity_id)
                if features is not None:
                    data.append(features)
                else:
                    logger.warning(f"No features found for entity_id: {entity_id}")
            logger.info("Data extracted from Redis successfully")
            return data
        except Exception as e:
            logger.error(f"Error extracting data from Redis: {e}")
            raise CustomException(e, sys)
    
    def prepare_data(self):
        try:
            entity_ids = self.feature_store.get_all_entity_ids()
            train_entity_ids, val_entity_ids = train_test_split(entity_ids, test_size=0.2, random_state=84)
            train_data = self.load_data_from_redis(train_entity_ids)
            val_data = self.load_data_from_redis(val_entity_ids)
            
            train_df = pd.DataFrame(train_data)
            val_df = pd.DataFrame(val_data)
            X_train, y_train = feature_engineering(train_df)
            X_val, y_val = feature_engineering(val_df)

            return X_train, y_train, X_val, y_val
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise CustomException(e, sys)
        
    def hyperparameter_tuning(self, X_train, y_train):
        try:
            rf_param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30]}
            xgb_param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30]}
            lgbm_param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30], 'learning_rate': [0.01, 0.05, 0.1], 'subsample': [0.5, 0.75, 1.0], 'colsample_bytree': [0.5, 0.75, 1.0]}

            mlflow.log_param("rf_param_grid", rf_param_grid)
            mlflow.log_param("xgb_param_grid", xgb_param_grid)
            mlflow.log_param("lgbm_param_grid", lgbm_param_grid)

            rf_grid = GridSearchCV(RandomForestClassifier(random_state=84), rf_param_grid, cv=3)
            xgb_grid = GridSearchCV(XGBClassifier(random_state=84), xgb_param_grid, cv=3)
            lgbm_grid = GridSearchCV(LGBMClassifier(random_state=84), lgbm_param_grid, cv=3)

            rf_grid.fit(X_train, y_train)
            xgb_grid.fit(X_train, y_train)
            lgbm_grid.fit(X_train, y_train)

            logger.info("Hyperparameter tuning completed")
            logger.info(f"Random Forest Grid Search: {rf_grid.best_params_}")
            logger.info(f"XGBoost Grid Search: {xgb_grid.best_params_}")
            logger.info(f"LightGBM Grid Search: {lgbm_grid.best_params_}")

            return rf_grid.best_estimator_, xgb_grid.best_estimator_, lgbm_grid.best_estimator_

        except Exception as e:
            logger.error(f"Error hyperparameter tuning: {e}")
            raise CustomException(e, sys)
        
    def train_model(self, X_train, y_train, X_val, y_val):
        try:
            best_rf, best_xgb, best_lgbm = self.hyperparameter_tuning(X_train, y_train)
            best_rf.fit(X_train, y_train)
            best_xgb.fit(X_train, y_train)
            best_lgbm.fit(X_train, y_train)
            
            rf_preds = best_rf.predict(X_val)
            xgb_preds = best_xgb.predict(X_val)
            lgbm_preds = best_lgbm.predict(X_val)

            rf_accuracy = accuracy_score(y_val, rf_preds)
            xgb_accuracy = accuracy_score(y_val, xgb_preds)
            lgbm_accuracy = accuracy_score(y_val, lgbm_preds)

            mlflow.log_metric("rf_accuracy", rf_accuracy)
            mlflow.log_metric("xgb_accuracy", xgb_accuracy)
            mlflow.log_metric("lgbm_accuracy", lgbm_accuracy)

            logger.info("Model training completed")
            logger.info(f"Random Forest Accuracy: {rf_accuracy:.4f}")
            logger.info(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
            logger.info(f"LightGBM Accuracy: {lgbm_accuracy:.4f}")

            self.save_model(best_rf, "best_rf")
            self.save_model(best_xgb, "best_xgb")
            self.save_model(best_lgbm, "best_lgbm")

            mlflow.sklearn.log_model(best_rf, "best_rf")
            mlflow.xgboost.log_model(best_xgb, "best_xgb")
            mlflow.lightgbm.log_model(best_lgbm, "best_lgbm")

            return rf_accuracy, xgb_accuracy, lgbm_accuracy, best_rf, best_xgb, best_lgbm

        except Exception as e:
            logger.error(f"Error training model: {e}")

    def save_model(self, model, model_name):
        try:
            model_filename = f"{self.model_save_path}/{model_name}.joblib"
            with open(model_filename, 'wb') as f:
                dump(model, f)
            logger.info(f"Model saved to {model_filename}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise CustomException(e, sys)
    
    def load_model(self, model_name):
        try:
            model_filename = f"{self.model_save_path}/{model_name}.joblib"
            with open(model_filename, 'rb') as f:
                model = load(f)
            logger.info(f"Model loaded from {model_filename}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise CustomException(e, sys)

    def run(self):
        try:
            logger.info("Starting model training...")
            with mlflow.start_run():
                run_id = mlflow.active_run().info.run_id
                logger.info(f"Starting MLFlow run with ID: {run_id}")
                mlflow.set_tag("task", "customer_segmentation_training")
                X_train, y_train, X_val, y_val = self.prepare_data()

                (rf_accuracy, xgb_accuracy, lgbm_accuracy, best_rf, best_xgb, best_lgbm) = self.train_model(X_train, y_train, X_val, y_val)

                logger.info("Model training completed in MLFlow run")

                total_accuracy = rf_accuracy + xgb_accuracy + lgbm_accuracy
                rf_weight = rf_accuracy / total_accuracy
                xgb_weight = xgb_accuracy / total_accuracy 
                lgbm_weight = lgbm_accuracy / total_accuracy

                print("\nEnsemble Weights:")
                print(f"Random Forest Weight: {rf_weight:.4f}")
                print(f"XGBoost Weight: {xgb_weight:.4f}")
                print(f"LightGBM Weight: {lgbm_weight:.4f}")

                # Get probabilities for each model
                rf_proba = best_rf.predict_proba(X_val)
                xgb_proba = best_xgb.predict_proba(X_val)
                lgbm_proba = best_lgbm.predict_proba(X_val)

                # Calculate weighted ensemble probabilities
                ensemble_proba = (rf_weight * rf_proba + 
                                xgb_weight * xgb_proba + 
                                lgbm_weight * lgbm_proba)

                # Make ensemble predictions
                ensemble_predictions = np.argmax(ensemble_proba, axis=1)

                # Calculate ensemble accuracy
                ensemble_accuracy = accuracy_score(y_val, ensemble_predictions)
                mlflow.log_metric("ensemble_accuracy", ensemble_accuracy)
                logger.info(f"\nEnsemble Accuracy: {ensemble_accuracy:.4f}")

                logger.info("Saving results...")
                

        except Exception as e:
            logger.error(f"Error running model training: {e}")
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    feature_store = RedisFeatureStore()
    model_training = ModelTraining(feature_store)
    model_training.run()