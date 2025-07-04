from fastapi import UploadFile, HTTPException, Form, BackgroundTasks
from src.Churn.components.support import import_data, most_common, get_dummies
from typing import Optional
import pandas as pd
from dotenv import load_dotenv
from src.Churn.components.data_ingestion import DataIngestion
from src.Churn.config.configuration import ConfigurationManager, WebhookConfig
import joblib
import mlflow
from src.Churn.utils.logging import logger
from src.Churn.utils.notify_webhook import post_to_webhook
from src.Churn.utils.visualize_ouput import visualize_customer_churn
from datetime import datetime
import time
import os
import dagshub
import tempfile
import numpy as np

load_dotenv()
web_hook_url = WebhookConfig().url

async def send_webhook_payload(message: str, avg_confidence: Optional[float] = None):
    try:
        logger.info("Preparing webhook notification")
        payload = {"message": message, "avg_confidence": avg_confidence}
        await post_to_webhook(web_hook_url, payload)
    except Exception as e:
        logger.error(f"Webhook notification failed: {str(e)}")

class PredictionPipeline:
    def __init__(self, model_uri: str, scaler_uri: str):
        self.config_manager = ConfigurationManager()
        self.mlflow_config = self.config_manager.get_mlflow_config()
        self.prediction_config = self.config_manager.get_prediction_config()
        
        mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)
        try:
            self.model = mlflow.pyfunc.load_model(model_uri)
            scaler_path = mlflow.artifacts.download_artifacts(artifact_uri=scaler_uri)
            self.scaler = joblib.load(scaler_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model or scaler: {e}")

    def process_data_for_churn(self, df_input: pd.DataFrame) -> pd.DataFrame:
        df = df_input.copy()
        df.columns = df.columns.map(str.strip)
        cols_to_drop = {"Age"}
        df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
        df.dropna(inplace=True)
        if 'Price' not in df.columns and 'Product Price' in df.columns:
            df['Price'] = df['Product Price']
        df['TotalSpent'] = df['Quantity'] * df['Price']
        
        df_features = df.groupby("customer_id", as_index=False, sort=False).agg(
            LastPurchaseDate=("Purchase Date", "max"),
            Favoured_Product_Categories=("Product Category", lambda x: most_common(list(x))),
            Frequency=("Purchase Date", "count"),
            TotalSpent=("TotalSpent", "sum"),
            Favoured_Payment_Methods=("Payment Method", lambda x: most_common(list(x))),
            Customer_Name=("Customer Name", "first"),
            Customer_Label=("Customer_Labels", "first"),
        ).groupby('Customer_Name', as_index=False).first()
        
        df_features['LastPurchaseDate'] = pd.to_datetime(df_features['LastPurchaseDate'])
        max_last_buying_date = df_features["LastPurchaseDate"].max()
        df_features['Recency'] = (max_last_buying_date - df_features['LastPurchaseDate']).dt.days
        
        frequency_safe = np.where(df_features['Frequency'] == 0, 1, df_features['Frequency'])
        df_features['Avg_Spend_Per_Purchase'] = df_features['TotalSpent'] / frequency_safe
        df_features['Purchase_Consistency'] = df_features['Recency'] / frequency_safe
        df_features.drop(columns=["LastPurchaseDate"], inplace=True)
        return df_features

    def encode_churn(self, df_features: pd.DataFrame) -> pd.DataFrame:
        df_copy = df_features.copy()
        df_copy.drop(columns=["customer_id", "Customer_Name"], inplace=True)
        return get_dummies(df_copy)

    async def predict(self):
        start_time = time.time()
        start_datetime = datetime.now()
        time_str = start_datetime.strftime('%Y%m%dT%H%M%S')
        
        dagshub.init(repo_owner=self.mlflow_config.dagshub_username, repo_name=self.mlflow_config.dagshub_repo_name, mlflow=True)
        mlflow.set_experiment(self.mlflow_config.prediction_experiment_name)
        
        with mlflow.start_run(run_name=f"prediction_run_{time_str}"):
            data_ingestion = DataIngestion(config=self.config_manager.get_data_ingestion_config())
            df = data_ingestion.load_data()
            df_features = self.process_data_for_churn(df)
            df_encoded = self.encode_churn(df_features)
            X = self.scaler.transform(df_encoded)
            y_pred = self.model.predict(X)
            df_features['Churn_RATE'] = y_pred

            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
                prediction_csv_path = temp_file.name
                df_features.to_csv(prediction_csv_path, index=False)
                mlflow.log_artifact(prediction_csv_path, "predictions")
            os.remove(prediction_csv_path)

            try:
                sklearn_model = self.model._model_impl
                y_proba = sklearn_model.predict_proba(X)
                average_confidence = y_proba.max(axis=1).mean()
            except AttributeError:
                average_confidence = None

            end_time = time.time()
            processing_time = end_time - start_time
            
            plot_path = visualize_customer_churn(df_features)
            mlflow.log_artifact(plot_path, "visualization")
            os.remove(plot_path)

            mlflow.log_metrics({
                "processing_time_seconds": processing_time,
                "count_churn": int(np.sum(y_pred == 1)),
                "count_not_churn": int(np.sum(y_pred == 0)),
                "records_processed": len(df_encoded),
            })
            
            message = ""
            if average_confidence is not None:
                mlflow.log_metric("average_prediction_confidence", average_confidence)
                if average_confidence < self.prediction_config.retraining_confidence_threshold:
                    message = f"⚠️ Average prediction confidence ({average_confidence:.2%}) is below the threshold. Consider retraining."
                else:
                    message = f"Average prediction confidence ({average_confidence:.2%}) is acceptable."
                await send_webhook_payload(message=message, avg_confidence=average_confidence)
            
            mlflow.log_text(message, "prediction_summary.txt")
            return message

async def run_prediction_task(file_path: str, model_version: str, scaler_version: str, run_id: str, model_name: str):
    try:
        model_uri = f"models:/{model_name}/{model_version}"
        scaler_uri = f"runs:/{run_id}/{scaler_version}"
        pipeline = PredictionPipeline(model_uri, scaler_uri)
        message = await pipeline.predict()
        if os.path.exists(file_path):
            os.remove(file_path)
        return message
    except Exception as e:
        logger.error(f"Background prediction task error: {e}")
        return f"Prediction error: {e}"

class ChurnController:
    @staticmethod
    async def predict_churn(
        background_tasks: BackgroundTasks,
        file: UploadFile,
        model_version: str = Form(...),
        scaler_version: str = Form(...),
        run_id: str = Form(...),
        model_name: str = Form(...)
    ):
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded.")

        config_manager = ConfigurationManager()
        prediction_config = config_manager.get_prediction_config()
        
        # Use provided values or fall back to config defaults
        model_version = model_version or prediction_config.default_model_version
        scaler_version = scaler_version or prediction_config.default_scaler_version
        run_id = run_id or prediction_config.default_run_id
        model_name = model_name or prediction_config.default_model_name

        data_ingestion_config = config_manager.get_data_ingestion_config()
        input_file_path = data_ingestion_config.local_data_file

        try:
            await import_data(file)
            background_tasks.add_task(
                run_prediction_task,
                file_path=str(input_file_path),
                model_version=model_version,
                scaler_version=scaler_version,
                run_id=run_id,
                model_name=model_name
            )
            return {"message": "Prediction task started in background. Check MLflow for results."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")