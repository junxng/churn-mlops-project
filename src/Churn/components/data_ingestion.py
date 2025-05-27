import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.Churn.utils.logging import logger
from src.Churn.entity.config_entity import DataIngestionConfig
from pathlib import Path
from datetime import datetime, timezone
from .support import most_common,get_dummies
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from src.Churn.utils.cloud_storage import upload_many_blobs_with_transfer_manager

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.rows_processed = 0
        self.datetime_suffix = datetime.now().strftime('%Y%m%dT%H%M%S')

    def process_data_for_churn(self,df_input):
        df_input.columns = df_input.columns.map(lambda x: str(x).strip())

        cols_to_drop = {"Returns", "Age", "Total Purchase Amount"}
        df_input.drop(columns=[col for col in cols_to_drop if col in df_input.columns], inplace=True)
        df_input.dropna(inplace=True)

        if 'Price' not in df_input.columns:
            if 'Product Price' in df_input.columns:
                df_input['Price'] = df_input['Product Price']
            else:
                raise KeyError("Required column 'Product Price' is missing from the dataset.")
        else:
            print("Price column already exists, skipping.")

        df_input['TotalSpent'] = df_input['Quantity'] * df_input['Price']
        df_features = df_input.groupby("customer_id", as_index=False, sort=False).agg(
            LastPurchaseDate = ("Purchase Date","max"),
            Favoured_Product_Categories = ("Product Category", lambda x: most_common(list(x))),
            Frequency = ("Purchase Date", "count"),
            TotalSpent = ("TotalSpent", "sum"),
            Favoured_Payment_Methods = ("Payment Method", lambda x: most_common(list(x))),
            Customer_Name = ("Customer Name", "first"),
            Customer_Label = ("Customer_Labels", "first"),
            Churn = ("Churn", "first"),
            # Churn = ("Churn", lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else 0) 
        )

        df_features = df_features.drop_duplicates(subset=['Customer_Name'], keep='first')
        df_features['LastPurchaseDate'] = pd.to_datetime(df_features['LastPurchaseDate'])
        df_features['LastPurchaseDate'] = df_features['LastPurchaseDate'].dt.date
        df_features['LastPurchaseDate'] = pd.to_datetime(df_features['LastPurchaseDate'])
        max_LastBuyingDate = df_features["LastPurchaseDate"].max()
        df_features['Recency'] = (max_LastBuyingDate - df_features['LastPurchaseDate']).dt.days
        df_features['LastPurchaseDate'] = df_features['LastPurchaseDate'].dt.date
        df_features['Avg_Spend_Per_Purchase'] = df_features['TotalSpent']/df_features['Frequency'].replace(0,1)
        df_features['Purchase_Consistency'] = df_features['Recency'] / df_features['Frequency'].replace(0, 1)
        df_features.drop(columns=["customer_id","LastPurchaseDate",'Customer_Name'], inplace=True)
        
        return df_features
    def encode_churn(self,df_features: pd.DataFrame):
        df_copy = df_features.copy()
        df_features_encode = get_dummies(df_copy)
        return df_features_encode

    def load_data(self):
        """Load the dataset from CSV file and save versioned copy to cloud storage."""
        try:
            logger.info(f"Loading data from {self.config.local_data_file}")
            df = pd.read_csv(self.config.local_data_file)
            
            input_data_versioned_name = f"input_raw_data_version_{self.datetime_suffix}.csv"
            input_data_versioned_path = Path(self.config.data_version_dir) / input_data_versioned_name
            if not input_data_versioned_path.exists():
                df.to_csv(input_data_versioned_path, index=False)
                logger.info(f"Created versioned input data file: {input_data_versioned_path}")
            else:
                logger.info(f"Versioned file already exists: {input_data_versioned_path}, skipping save.")

            try:
                upload_many_blobs_with_transfer_manager(
                    bucket_name=self.config.bucket_name,
                    filenames=[input_data_versioned_name],  
                    source_directory=str(input_data_versioned_path.parent), 
                    workers=8
                )
                logger.info(f"Successfully uploaded {input_data_versioned_name} to Google Cloud Storage bucket: {self.config.bucket_name}")
            except Exception as cloud_error:
                logger.warning(f"Failed to upload to cloud storage: {cloud_error}")
                logger.info("Continuing with local processing...")

            logger.info(f"Loaded dataset with {len(df)} rows")
            logger.info(f"Columns found: {list(df.columns)}")

            return df

        except Exception as e:
            logger.error(f"Error while loading data: {e}")
            raise

    def preprocess_data(self, df_clean):
        self.rows_processed = 0
        logger.info(f"Starting preprocessing of {len(df_clean)} rows...")
        logger.info(f"Initial columns: {list(df_clean.columns)}")
        
        df_clean = self.process_data_for_churn(df_clean)
        logger.info(f"After feature engineering columns: {list(df_clean.columns)}")
        logger.info(f"Data shape after feature engineering: {df_clean.shape}")
        
        df_clean = self.encode_churn(df_clean)
        logger.info(f"After encoding columns: {list(df_clean.columns)}")
        logger.info(f"Data shape after encoding: {df_clean.shape}")
        
        df_clean = df_clean.dropna()
        logger.info(f"Data shape after removing NaN: {df_clean.shape}")
        
        if "Churn" not in df_clean.columns:
            logger.error(f"Churn column not found! Available columns: {list(df_clean.columns)}")
            raise KeyError("Churn column is missing after preprocessing")
        
        X = df_clean.drop("Churn", axis=1)
        y = df_clean["Churn"]
        
        logger.info(f"Target variable shape: {y.shape}")
        logger.info(f"NaN count in target: {y.isna().sum()}")
        logger.info(f"Target variable distribution: \n{y.value_counts(normalize=True)}")
        
        if y.isna().sum() > 0:
            logger.warning(f"Found {y.isna().sum()} NaN values in target variable, filling with 0")
            y = y.fillna(0)
        
        nan_cols = X.columns[X.isna().any()].tolist()
        if nan_cols:
            logger.warning(f"Found NaN values in feature columns: {nan_cols}")
            X = X.fillna(0)
        
        logger.info(f"Final feature matrix shape: {X.shape}")
        logger.info(f"Final target vector shape: {y.shape}")
        
        class_distribution = y.value_counts(normalize=True)
        logger.info(f"Target variable distribution (normalized): \n{class_distribution}")

        imbalance_threshold = 0.4

        if class_distribution.min() < imbalance_threshold:
            logger.info("Target variable is imbalanced. Applying SMOTEENN...")
            smote = SMOTEENN(random_state=42)
            X_res, y_res = smote.fit_resample(X, y)
            logger.info(f"Resampled feature matrix shape: {X_res.shape}")
            logger.info(f"Resampled target distribution: \n{y_res.value_counts(normalize=True)}")
        else:
            logger.info("Target variable is balanced. Skipping SMOTEENN.")
            X_res, y_res = X, y
        return X_res, y_res

    def split_data(self, X_res, y_res):
        """Split data into training and testing sets."""
        logger.info("Splitting data into train and test sets")
        X_train,X_test,y_train,y_test = train_test_split(X_res,y_res,test_size=self.config.test_size,
            random_state=self.config.random_state)
        logger.info(f"Train data: {X_train.shape}, Test data: {X_test.shape}")
        return X_test, X_train, y_test, y_train

    def save_data(self, X_train, X_test, y_train, y_test):
        """Save training and testing data to CSV files and optionally upload to cloud storage."""
        logger.info("Saving processed feature data (X_train and X_test) to versioned directory")
        
        # Create versioning directory if it doesn't exist
        os.makedirs(self.config.data_version_dir, exist_ok=True)
        
        # Datetime-based paths for versioning
        train_feature_path = os.path.join(self.config.data_version_dir, f"train_feature_version_{self.datetime_suffix}.csv")
        test_feature_path = os.path.join(self.config.data_version_dir, f"test_feature_version_{self.datetime_suffix}.csv")
        train_target_path = os.path.join(self.config.data_version_dir, f"train_target_version_{self.datetime_suffix}.csv")
        test_target_path = os.path.join(self.config.data_version_dir, f"test_target_version_{self.datetime_suffix}.csv")
        
        # Save versioned files
        X_train.to_csv(train_feature_path, index=False)
        X_test.to_csv(test_feature_path, index=False)
        y_train.to_csv(train_target_path, index=False, header=['Churn'])
        y_test.to_csv(test_target_path, index=False, header=['Churn'])
        
        try:
            versioned_files = [
                f"train_feature_version_{self.datetime_suffix}.csv",
                f"test_feature_version_{self.datetime_suffix}.csv", 
                f"train_target_version_{self.datetime_suffix}.csv",
                f"test_target_version_{self.datetime_suffix}.csv"
            ]
            
            upload_many_blobs_with_transfer_manager(
                bucket_name=self.config.bucket_name,
                filenames=versioned_files,
                source_directory=self.config.data_version_dir,
                workers=8
            )
            logger.info(f"Successfully uploaded versioned training data to Google Cloud Storage bucket: {self.config.bucket_name}")
            


        except Exception as cloud_error:
            logger.warning(f"Failed to upload training data to cloud storage: {cloud_error}")
            logger.info("Training data saved locally only...")
        
        logger.info(f"Processed features saved to:")
        logger.info(f"  X_train: {train_feature_path}")
        logger.info(f"  X_test: {test_feature_path}")
        logger.info(f"  y_train: {train_target_path}")
        logger.info(f"  y_test: {test_target_path}")
        logger.info(f"  Data shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
        
        return train_feature_path, test_feature_path, train_target_path, test_target_path

    def data_ingestion_pipeline(self):
        """Main method to perform data ingestion."""
        logger.info("Initiating data ingestion")
        df = self.load_data()
        X, y = self.preprocess_data(df)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        train_target, test_target, y_train_path, y_test_path = self.save_data(X_train, X_test, y_train, y_test)
        
        logger.info("Data ingestion completed successfully")
        logger.info(f"Data shape - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test, train_target, test_target, y_train_path, y_test_path
