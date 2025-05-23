from fastapi import HTTPException, UploadFile
import os
import logging
import pandas as pd
import json
from typing import Optional, Tuple, Dict, List, Any
from io import BytesIO

# Import existing components and utilities
from src.Sentiment_Analysis.config.configuration import ConfigurationManager
from src.Sentiment_Analysis.components.base_model import PrepareBaseModel
from src.Sentiment_Analysis.components.model_training import ModelTraining
from src.Sentiment_Analysis.components.evaluation_mlflows import ModelEvaluation
from src.Sentiment_Analysis.utils.common import read_yaml
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib as jb

# Initialize logger
logger = logging.getLogger(__name__)

# Global variables
model = None
tokenizer = None
max_len = 200
config_manager = ConfigurationManager()

def load_model_from_artifacts():
    """Load model and tokenizer from local artifacts directory using existing components."""
    global model, tokenizer, max_len
    
    try:
        # Get configuration from existing config manager
        training_config = config_manager.get_training_config()
        prepare_base_model_config = config_manager.get_prepare_base_model_config()
        model_params = config_manager.get_model_params()
        
        # Use paths from configuration
        model_path = training_config.trained_model_path
        tokenizer_path = prepare_base_model_config.tokenizer_path
        max_len = model_params.maxlen
        
        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            logger.info(f"Loading model from {model_path}")
            model = load_model(model_path)
            
            logger.info(f"Loading tokenizer from {tokenizer_path}")
            tokenizer = jb.load(tokenizer_path)
            
            logger.info("Model and tokenizer loaded successfully from artifacts")
            return True
        else:
            logger.warning("Model or tokenizer not found in artifacts directory")
            return False
    except Exception as e:
        logger.error(f"Error loading model from artifacts: {e}")
        return False

def load_model_from_mlflow():
    """Load model from MLflow if available using existing components."""
    global model, tokenizer, max_len
    
    try:
        # Get MLflow configuration
        mlflow_config = config_manager.get_mlflow_config()
        evaluation_config = config_manager.get_evaluation_config()
        model_params = config_manager.get_model_params()
        max_len = model_params.maxlen
        
        # Create evaluation component to leverage its MLflow functionality
        model_evaluation = ModelEvaluation(config=evaluation_config)
        
        # Use the same MLflow client setup
        import mlflow
        from mlflow.tracking import MlflowClient
        
        # MLflow server details from config
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        client = MlflowClient()
        
        # Get the latest experiment
        experiment = client.get_experiment_by_name(mlflow_config.experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
            
            # Get the latest run
            runs = client.search_runs(
                experiment_ids=[experiment_id],
                order_by=["attributes.start_time DESC"],
                max_results=1
            )
            
            if runs:
                run_id = runs[0].info.run_id
                logger.info(f"Loading model from MLflow run: {run_id}")
                
                # Load model
                model_uri = f"runs:/{run_id}/final_model"
                model = mlflow.tensorflow.load_model(model_uri)
                
                # Download tokenizer artifact
                client.download_artifacts(run_id, "tokenizer", ".")
                tokenizer_path = os.path.join("tokenizer", "tokenizer.pkl")
                tokenizer = jb.load(tokenizer_path)
                
                logger.info("Model and tokenizer loaded successfully from MLflow")
                return True
            else:
                logger.warning("No runs found in the experiment")
                return False
        else:
            logger.warning("Experiment not found")
            return False
    except Exception as e:
        logger.error(f"Error loading model from MLflow: {e}")
        return False

def initialize():
    """Initialize the model and tokenizer using existing components."""
    if load_model_from_artifacts():
        return True
    elif load_model_from_mlflow():
        return True
    else:
        logger.error("Failed to load model and tokenizer")
        return False

def analyze_single_text(text: str) -> Tuple[str, Any]:
    """Analyze a single text using the loaded model from the pipeline."""
    global model, tokenizer, max_len
    
    if model is None or tokenizer is None:
        if not initialize():
            raise Exception("Model not initialized")
    
    try:
        # Preprocess text
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=max_len)
        
        # Make prediction
        prediction = model.predict(padded_sequence)
        sentiment_score = float(prediction[0][0])
        sentiment = "positive" if sentiment_score >= 0.5 else "negative"
        
        return sentiment, prediction
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise e

def analyze_customer_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze multiple reviews in a DataFrame."""
    # Check if 'review' or 'text' column exists
    text_column = None
    for col in ['review', 'text', 'Review', 'Text', 'comment', 'Comment']:
        if col in df.columns:
            text_column = col
            break
    
    if text_column is None:
        raise ValueError("No text column found in the dataset. Please include a column named 'review', 'text', or similar.")
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Add sentiment columns
    result_df['sentiment'] = ''
    result_df['score'] = 0.0
    
    # Process each text
    for idx, row in df.iterrows():
        try:
            text = str(row[text_column])
            sentiment, score = analyze_single_text(text)
            result_df.at[idx, 'sentiment'] = sentiment
            result_df.at[idx, 'score'] = float(score[0])
        except Exception as e:
            logger.error(f"Error analyzing row {idx}: {e}")
            result_df.at[idx, 'sentiment'] = 'error'
            result_df.at[idx, 'score'] = 0.0
    
    return result_df

def df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert DataFrame to JSON records format."""
    return json.loads(df.to_json(orient='records', date_format='iso'))

async def import_data(uploaded_file: UploadFile) -> pd.DataFrame:
    """Process uploaded file and convert to DataFrame."""
    df = None
    if uploaded_file is not None:
        filename = uploaded_file.filename.lower()
        content = await uploaded_file.read()
        # Try utf-8 first for speed
        try:
            sample = content.decode('utf-8', errors='replace')[:4096]
            encoding = 'utf-8'
        except Exception:
            import chardet
            result = chardet.detect(content)
            encoding = result['encoding'] or 'utf-8'
            sample = content.decode(encoding, errors='replace')[:4096]
        
        if filename.endswith(('.csv', '.txt')):
            # Fast delimiter detection: check for most common delimiters
            delimiter = ','
            for delim in [',', ';', '\t', '|']:
                if sample.count(delim) > 0:
                    delimiter = delim
                    break
            try:
                df = pd.read_csv(BytesIO(content), encoding=encoding, delimiter=delimiter, 
                                 low_memory=False, on_bad_lines='skip')
            except Exception as e:
                raise ValueError(f"Error reading CSV: {str(e)}")
        elif filename.endswith(('.xlsx', '.xls')):
            if filename.endswith('.xlsx'):
                import openpyxl
                engine = 'openpyxl'
            else:
                import xlrd
                engine = 'xlrd'
            try:
                df = pd.read_excel(BytesIO(content), engine=engine)
            except Exception as e:
                raise ValueError(f"Error reading Excel: {str(e)}")
            df = convert_dates(df)
        else:
            raise ValueError("Unsupported file format. Please upload CSV or Excel files.")
        
        if df is None:
            raise ValueError("Unable to read file. Please check the file format.")
        if df.empty:
            raise ValueError("No data found in the file. Please check the file content.")
        return df
    else:
        raise ValueError("No file provided.")

def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert potential date columns to datetime."""
    # Only process columns likely to be dates
    date_columns = []
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            date_columns.append(col)
        elif df[col].dtype == 'object':
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else ''
            if isinstance(sample, str) and any(char in sample for char in ['-', '/', '.', ':']):
                date_columns.append(col)
    
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
    
    return df

class SentimentController:
    @staticmethod
    async def analyze_text(text: str):
        try:
            sentiment, rating = analyze_single_text(text)
            return {
                "sentiment": sentiment,
                "rating": float(rating[0])
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @staticmethod
    async def analyze_batch(file: Optional[UploadFile] = None):
        if not file:
            raise HTTPException(status_code=400, detail="File must be provided")
        try:
            df = await import_data(file)
            df_sentiment = analyze_customer_reviews(df)
            return {
                "results": df_to_records(df_sentiment)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) 