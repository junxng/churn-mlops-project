from fastapi import UploadFile, HTTPException,Form,BackgroundTasks
from src.Churn.components.support import import_data,most_common,get_dummies
from typing_extensions import Optional
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
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
import os
web_hook_url = WebhookConfig().url

async def send_webhook_payload(
    message: str,
    avg_confidence: Optional[float] = None,
):
    try:
        logger.info("Preparing webhook notification")

        payload = {
            "message": message,
            "avg_confidence": avg_confidence,
        }

        await post_to_webhook(web_hook_url, payload)

    except Exception as e:
        logger.error(f"Webhook notification failed with unexpected error: {str(e)}")
class PredictionPipeline:
    def __init__(self, model_uri: str, scaler_uri: str):
        mlflow.set_tracking_uri("https://dagshub.com/junxng/churn-mlops-project.mlflow")

        try:
            self.model = mlflow.pyfunc.load_model(model_uri)
            scaler_path = mlflow.artifacts.download_artifacts(artifact_uri=scaler_uri)
            self.scaler = joblib.load(scaler_path)
        except Exception as e:
                raise RuntimeError(f"Failed to load model or scaler: {e}")
    def process_data_for_churn(self,df_input: pd.DataFrame):
        df_input.columns = df_input.columns.map(str.strip)
        cols_to_drop = {"Age"}
        df_input.drop(columns=[col for col in cols_to_drop if col in df_input.columns], inplace=True)    
        df_input.dropna(inplace=True)
        if 'Price' not in df_input.columns:
            df_input['Price'] = df_input['Product Price']
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
        df_features.drop(columns=["LastPurchaseDate"],axis=1,inplace=True)
        return df_features
    def encode_churn(self, df_features):
        df_copy = df_features.copy()
        df_copy.drop(columns=["customer_id","Customer_Name"],axis=1,inplace=True)
        df_features_encode = get_dummies(df_copy)
        return df_features_encode
    async def predict(self):
        try:
            start_time = time.time()
            start_datetime = datetime.now()
            time_str = start_datetime.strftime('%Y%m%dT%H%M%S')
            
            config_manager = ConfigurationManager()
            data_ingestion_config = config_manager.get_data_ingestion_config()
            dagshub.init(
            repo_owner="junxng",
            repo_name="churn-mlops-project",
            mlflow=True
        )
            mlflow.set_tracking_uri("https://dagshub.com/junxng/churn-mlops-project.mlflow")
            mlflow.set_experiment("Churn_model_prediction_cycle")  
            with mlflow.start_run(run_name=f"prediction_run_{time_str}"):
                data_ingestion = DataIngestion(config=data_ingestion_config)
                
                df = data_ingestion.load_data()
                df_features = self.process_data_for_churn(df)
                df_encoded = self.encode_churn(df_features)
                X = self.scaler.transform(df_encoded)

                y_pred = self.model.predict(X)
                df_features['Churn_RATE'] = y_pred
                counts = df_features['Churn_RATE'].value_counts()
                count_churn = counts.get(1, 0)
                count_not_churn = counts.get(0, 0)
                try:
                    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
                        prediction_csv_path = temp_file.name
                        df_features.to_csv(prediction_csv_path, index=False)
                        mlflow.log_artifact(prediction_csv_path, "predictions")
                        logger.info(f"Successfully saved prediction results to {prediction_csv_path} and logged as MLflow artifact")
                    
                    os.remove(prediction_csv_path)
                    logger.info(f"Deleted temporary prediction file: {prediction_csv_path}")

                except Exception as e:
                    logger.error(f"An error occurred during prediction saving or cleanup: {e}")
                
                try:    
                    sklearn_model = self.model._model_impl  
                    y_proba = sklearn_model.predict_proba(X)
                    max_confidence = y_proba.max(axis=1)
                    average_confidence = max_confidence.mean()
                except AttributeError:
                    average_confidence = None 
                end_time = time.time()
                end_datetime = datetime.now()
                processing_time = end_time - start_time

                logger.info(f"Prediction processing time: {processing_time:.2f} seconds")
                logger.info(f"Started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"Completed at: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                plot_path = visualize_customer_churn(df_features)
                mlflow.log_artifact(plot_path, "visualization")
                os.remove(plot_path)
                mlflow.log_metrics({
                    "processing_time_seconds": processing_time,
                    "count_churn": count_churn,
                    "count_not_churn": count_not_churn,
                    "records_processed": len(df_encoded),
                })
                mlflow.log_params({
                    "start_time": start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                    "end_time": end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                    "rawdata_records": len(df),
                })
                
                message = ""
                CONFIDENCE_THRESHOLD = 0.7
                if average_confidence is not None:
                    mlflow.log_metric("average_prediction_confidence", average_confidence)
                    
                    if average_confidence < CONFIDENCE_THRESHOLD:
                        message = (
                            f"⚠️ Average prediction confidence ({average_confidence:.2%}) is below the threshold "
                            f"of {CONFIDENCE_THRESHOLD:.2%}. Consider retraining the model."
                        )
                    else:
                        message = (
                            f"Average prediction confidence ({average_confidence:.2%}) is above the threshold "
                            f"of {CONFIDENCE_THRESHOLD:.2%}. No further action required."
                        )
                    
                    await send_webhook_payload(message=message, avg_confidence=average_confidence)
                
                mlflow.log_text(message, "prediction_summary.txt")
            return message

        except Exception as e:
            raise RuntimeError(f"Prediction error: {e}")
        
async def run_prediction_task(
    file_path: str,
    model_version: str,
    scaler_version: str,
    run_id: str,
    model_name: str = "RandomForestClassifier",
):
    """
    Background task to run prediction pipeline
    """
    try:
        model_uri = f"models:/{model_name}/{model_version}"
        scaler_uri = f"runs:/{run_id}/{scaler_version}"
        pipeline = PredictionPipeline(model_uri, scaler_uri)
        message = await pipeline.predict()

        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleanup: Deleted input file {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete input file during cleanup: {e}")

        return message

    except Exception as e:
        logger.error(f"Background prediction task error: {e}")
        return f"Prediction error: {e}"


class ChurnController:
    @staticmethod
    async def predict_churn(
        background_tasks: BackgroundTasks,
        file: UploadFile,
        model_version: str = Form(default="1"),
        scaler_version: str = Form(default="scaler_churn_version_20250701T105905.pkl"),
        run_id: str = Form(default="b523ba441ea0465085716dcebb916294"),
        model_name: str = Form(default="RandomForestClassifier"),
    ):
        """
        Predict churn using uploaded file and dynamic model/scaler versions.
        """
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded.")

        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        input_file_path = data_ingestion_config.local_data_file

        try:
            await import_data(file)

            background_tasks.add_task(
                run_prediction_task,
                file_path=input_file_path,
                model_version=model_version,
                scaler_version=scaler_version,
                run_id=run_id,
                model_name=model_name
            )
            
            message = "Prediction task started in background. Results will be saved to experiment 'Churn-model-prediction-cycle' in https://dagshub.com/junxng/churn-mlops-project.mlflow, check your webhook for status "
            
            return {
                "message": message
            }

        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")