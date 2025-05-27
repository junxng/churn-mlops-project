import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
import joblib as jb
import pandas as pd
from datetime import datetime
from src.Churn.utils.logging import logger
from src.Churn.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        # Generate datetime suffix for this run
        self.datetime_suffix = datetime.now().strftime('%Y%m%dT%H%M%S')
        
    def get_base_model(self):
        """Create and return a base Random Forest model for churn prediction."""
        logger.info("Creating base Random Forest model")
        
        model = RandomForestClassifier(
            n_estimators=self.config.n_estimators, 
            random_state=self.config.random_state,
            criterion=self.config.criterion,
            max_depth=self.config.max_depth,
            max_features=self.config.max_features,
            min_samples_leaf=self.config.min_samples_leaf
        )
        logger.info(f"Model params:{self.config.n_estimators}, {self.config.random_state}, {self.config.criterion}, {self.config.max_depth}, {self.config.max_features}, {self.config.min_samples_leaf}")
        return model
    
    def scaler(self, train_path, test_path, y_train_path, y_test_path):
        """Prepare and save scaler based on training data."""
        logger.info("Creating scaler from training data")
        
        os.makedirs(self.config.data_version_dir, exist_ok=True)
        os.makedirs(self.config.model_version_dir, exist_ok=True)
        
        scaled_test_data_path = os.path.join(self.config.data_version_dir, f"test_feature_scaled_version_{self.datetime_suffix}.csv")
        scaled_train_data_path = os.path.join(self.config.data_version_dir, f"train_feature_scaled_version_{self.datetime_suffix}.csv")
        scaler_path = os.path.join(self.config.model_version_dir, f"scaler_churn_version_{self.datetime_suffix}.pkl")
        
        try:
            X_train = pd.read_csv(train_path)
            X_test = pd.read_csv(test_path)
            y_train = pd.read_csv(y_train_path)
            y_test = pd.read_csv(y_test_path)
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            jb.dump(scaler, scaler_path)
            
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
            X_train_scaled_df.to_csv(scaled_train_data_path, index=False)
            X_test_scaled_df.to_csv(scaled_test_data_path, index=False)
            
            logger.info(f"Scalers and scaled data saved:")
            logger.info(f"  Scaler: {scaler_path}")
            logger.info(f"  X_train_scaled: {scaled_train_data_path}")
            logger.info(f"  X_test_scaled: {scaled_test_data_path}")
            
            return scaler, X_train, X_test, y_train, y_test, scaled_train_data_path, scaled_test_data_path, scaler_path
 
        except Exception as e:
            logger.error(f"Error in preparing scaler: {e}")
            raise e
    
    def full_model(self, train_path, test_path, y_train_path, y_test_path):
        """Create the base model and scaler."""
        logger.info("Creating base model and scaler")
        
        model = self.get_base_model()
        scaler, X_train, X_test, y_train, y_test, scaled_train_path, scaled_test_path, scaler_path = self.scaler(
            train_path, test_path, y_train_path, y_test_path
        )
        
        base_model_path = os.path.join(self.config.model_version_dir, f"base_model_churn_{self.datetime_suffix}.pkl")
        
        # Save base model
        jb.dump(model, base_model_path)
        
        logger.info(f"Base model saved: {base_model_path}")

        return model, scaler, X_train, X_test, y_train, y_test, base_model_path, scaled_train_path, scaled_test_path, scaler_path
        