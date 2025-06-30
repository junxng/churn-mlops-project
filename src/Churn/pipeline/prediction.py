from fastapi import UploadFile, HTTPException,Form
from src.Churn.components.support import import_data,df_to_records
from dotenv import load_dotenv
load_dotenv()
from src.Churn.components.data_ingestion import DataIngestion
import joblib 
import mlflow
from src.Churn.utils.logging import logger
from datetime import datetime
import time

version_model_path = f"models:/RandomForestClassifier/1"
scaler_uri_path = f"runs:/f3ab09385e414fd2abf29d80f74cd67a/scaler_churn_version_20250625T154746.pkl"

try:
    loaded_model = mlflow.pyfunc.load_model(version_model_path)
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=scaler_uri_path)
    loaded_scaler = joblib.load(local_path)
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    loaded_model = None
    scaler_churn = None



class PredictionPipeline:
    def __init__(self, model_uri: str, scaler_uri: str):
        mlflow.set_tracking_uri("https://dagshub.com/Teungtran/churn_mlops.mlflow")

        try:
            self.model = mlflow.pyfunc.load_model(model_uri)
            scaler_path = mlflow.artifacts.download_artifacts(artifact_uri=scaler_uri)
            self.scaler = joblib.load(scaler_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model or scaler: {e}")

    def predict(self):
        try:
            start_time = time.time()
            start_datetime = datetime.now()
            time_str = start_datetime.strftime('%Y%m%dT%H%M%S')
            df = DataIngestion.load_data()
            df_processed = DataIngestion.process_data_for_churn(df)
            X = self.scaler.transform(df_processed)

            y_pred = self.model.predict(X)
            df['Churn_RATE'] = y_pred

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

            with mlflow.start_run(run_name=f"prediction_run_{time_str}"):
                mlflow.log_metric("processing_time_seconds", processing_time)
                mlflow.log_param("start_time", start_datetime.strftime('%Y-%m-%d %H:%M:%S'))
                mlflow.log_param("end_time", end_datetime.strftime('%Y-%m-%d %H:%M:%S'))
                mlflow.log_param("rawdata_records", len(df))
                mlflow.log_metric("records_processed", len(df_processed))
                if average_confidence is not None:
                    mlflow.log_metric("average_prediction_confidence", average_confidence)

            return df, average_confidence, {
                "processing_time_seconds": processing_time,
                "start_time": start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                "end_time": end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                "records_processed": len(df),
                "average_confidence": average_confidence
            }

        except Exception as e:
            raise RuntimeError(f"Prediction error: {e}")


class ChurnController:
    @staticmethod
    async def predict_churn(
        file: UploadFile,
        model_version: str = Form(default="1"),
        scaler_version: str = Form(default="scaler_churn_version_20250625T154746.pkl"),
        run_id: str = Form(default="f3ab09385e414fd2abf29d80f74cd67a"),
    ):
        """
        Predict churn using uploaded file and dynamic model/scaler versions.
        """
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded.")

        try:
            await import_data(file)

            model_uri = f"models:/RandomForestClassifier/{model_version}"
            scaler_uri = f"runs:/{run_id}/{scaler_version}"

            pipeline = PredictionPipeline(model_uri, scaler_uri)
            df, avg_confidence, timing_info = pipeline.predict()

            response = {
                "results": df_to_records(df),
                "timing": timing_info
            }

            CONFIDENCE_THRESHOLD = 0.7 
            if avg_confidence is not None and avg_confidence < CONFIDENCE_THRESHOLD:
                warning_msg = (
                    f"⚠️ Average prediction confidence ({avg_confidence:.2%}) is below the threshold "
                    f"of {CONFIDENCE_THRESHOLD:.2%}. Consider retraining the model."
                )
                response["warning"] = warning_msg
                logger.warning(warning_msg)

            return response

        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")