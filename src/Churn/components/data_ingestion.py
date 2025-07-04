import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.Churn.utils.logging import logger
from src.Churn.entity.config_entity import DataIngestionConfig
from pathlib import Path
from datetime import datetime
from .support import most_common, get_dummies
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.rows_processed = 0
        self.datetime_suffix = datetime.now().strftime('%Y%m%dT%H%M%S')

    def process_data_for_churn(self, df_input: pd.DataFrame) -> pd.DataFrame:
        df = df_input.copy()
        df.columns = df.columns.map(lambda x: str(x).strip())
        
        # Drop columns specified in the config
        cols_to_drop = self.config.columns_to_drop
        df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
        
        df.dropna(inplace=True)
        
        if 'Price' not in df.columns and 'Product Price' in df.columns:
            df['Price'] = df['Product Price']
        
        if 'Price' not in df.columns:
            raise KeyError("Required column 'Price' or 'Product Price' is missing.")

        df['TotalSpent'] = df['Quantity'] * df['Price']
        
        df_features = df.groupby("customer_id", as_index=False, sort=False).agg(
            LastPurchaseDate=("Purchase Date", "max"),
            Favoured_Product_Categories=("Product Category", lambda x: most_common(list(x))),
            Frequency=("Purchase Date", "count"),
            TotalSpent=("TotalSpent", "sum"),
            Favoured_Payment_Methods=("Payment Method", lambda x: most_common(list(x))),
            Customer_Name=("Customer Name", "first"),
            Customer_Label=("Customer_Labels", "first"),
            Churn=("Churn", "first"),
        )

        df_features = df_features.drop_duplicates(subset=['Customer_Name'], keep='first')
        df_features['LastPurchaseDate'] = pd.to_datetime(df_features['LastPurchaseDate'])
        max_last_buying_date = df_features["LastPurchaseDate"].max()
        df_features['Recency'] = (max_last_buying_date - df_features['LastPurchaseDate']).dt.days
        
        df_features['Avg_Spend_Per_Purchase'] = df_features['TotalSpent'] / df_features['Frequency'].replace(0, 1)
        df_features['Purchase_Consistency'] = df_features['Recency'] / df_features['Frequency'].replace(0, 1)
        
        df_features.drop(columns=["customer_id", "LastPurchaseDate", 'Customer_Name'], inplace=True)
        
        return df_features

    def encode_churn(self, df_features: pd.DataFrame) -> pd.DataFrame:
        return get_dummies(df_features.copy())

    def load_data(self) -> pd.DataFrame:
        """Load the dataset from CSV file."""
        try:
            logger.info(f"Loading data from {self.config.local_data_file}")
            df = pd.read_csv(self.config.local_data_file)
            logger.info(f"Loaded dataset with {len(df)} rows and columns: {list(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Error while loading data: {e}")
            raise

    def save_data(self, df: pd.DataFrame, df_processed: pd.DataFrame):
        """Save versioned copies of the raw and processed data."""
        os.makedirs(self.config.data_version_dir, exist_ok=True)
        
        input_data_versioned_path = self.config.data_version_dir / f"input_raw_data_version_{self.datetime_suffix}.csv"
        processed_data_versioned_path = self.config.data_version_dir / f"processed_data_version_{self.datetime_suffix}.csv"
        
        df.to_csv(input_data_versioned_path, index=False)
        df_processed.to_csv(processed_data_versioned_path, index=False)
        
        logger.info(f"Created versioned input data file: {input_data_versioned_path}")
        logger.info(f"Created versioned processed data file: {processed_data_versioned_path}")

    def preprocess_data(self, df_clean: pd.DataFrame):
        """Preprocess the data, including feature engineering, encoding, and resampling."""
        logger.info(f"Starting preprocessing of {len(df_clean)} rows...")
        
        df_processed = self.process_data_for_churn(df_clean)
        df_processed = self.encode_churn(df_processed)
        df_processed = df_processed.dropna()
        
        if "Churn" not in df_processed.columns:
            raise KeyError("Churn column is missing after preprocessing")
        
        X = df_processed.drop("Churn", axis=1)
        y = df_processed["Churn"]
        
        y = y.fillna(0)
        X = X.fillna(0)
        
        if self.config.imbalance_handling['apply']:
            class_distribution = y.value_counts(normalize=True)
            if class_distribution.min() < self.config.imbalance_handling['imbalance_threshold']:
                logger.info("Target variable is imbalanced. Applying SMOTE and EditedNearestNeighbours...")
                smote = SMOTE(random_state=self.config.imbalance_handling['smote_random_state'])
                enn = EditedNearestNeighbours()
                X_res, y_res = smote.fit_resample(X, y)
                X_res, y_res = enn.fit_resample(X_res, y_res)
                logger.info(f"Resampled feature matrix shape: {X_res.shape}")
            else:
                logger.info("Target variable is balanced. Skipping resampling.")
                X_res, y_res = X, y
        else:
            X_res, y_res = X, y
            
        return X_res, y_res, df_processed

    def split_data(self, X_res: pd.DataFrame, y_res: pd.Series):
        """Split data into training and testing sets."""
        logger.info("Splitting data into train and test sets")
        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, 
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        logger.info(f"Train data: {len(X_train)} rows, Test data: {len(X_test)} rows")
        return X_train, X_test, y_train, y_test

    def data_ingestion_pipeline(self):
        """Main method to perform data ingestion."""
        logger.info("Initiating data ingestion")
        df = self.load_data()
        X, y, df_processed = self.preprocess_data(df)
        self.save_data(df, df_processed)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        logger.info("Data ingestion completed successfully")
        return X_train, X_test, y_train, y_test, df_processed, df
