import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, SpatialDropout1D, Conv1D, 
    GlobalMaxPooling1D, Dense, Dropout
)
import joblib as jb
import pandas as pd
import mlflow
from src.Sentiment_Analysis.utils.logging import logger
from src.Sentiment_Analysis.entity.config_entity import PrepareBaseModelConfig
tf.keras.__version__ = tf.__version__

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        
    def get_base_model(self):
        """Create and return a base CNN model for sentiment analysis."""
        logger.info("Creating base CNN model")
        
        model = Sequential([
            Embedding(
                input_dim=self.config.params_num_words, 
                output_dim=self.config.params_embedding_dim, 
                input_length=self.config.params_maxlen
            ),
            SpatialDropout1D(self.config.dropout_rate),
            Conv1D(self.config.filters, self.config.kernel_size, padding='same', activation='relu'),
            GlobalMaxPooling1D(),
            Dropout(self.config.dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Base model summary:")
        model.summary(print_fn=logger.info)
        logger.info(f"Base model saved at: {self.config.base_model_path}")
        
        return model
    
    def prepare_tokenizer(self):
        """Prepare and save tokenizer based on training data."""
        logger.info("Creating tokenizer from training data")

        train_data_path = os.path.join(os.path.dirname(self.config.root_dir), "training", "train_data.csv")

        try:
            logger.info(f"Loading training data from: {train_data_path}")
            train_data = pd.read_csv(train_data_path)
            if "review" not in train_data.columns:
                raise ValueError("Column 'review' not found in training data.")
            original_count = len(train_data)
            train_data = train_data.dropna(subset=["review"])
            train_data["review"] = train_data["review"].astype(str)
            processed_count = len(train_data)
            dropped_count = original_count - processed_count
            logger.info(f"Tokenizer fitted on {processed_count} reviews (dropped {dropped_count} rows with missing 'review')")
            tokenizer = Tokenizer(num_words=self.config.params_num_words)
            tokenizer.fit_on_texts(train_data["review"])
            jb.dump(tokenizer, self.config.tokenizer_path)
            logger.info(f"Tokenizer saved at: {self.config.tokenizer_path}")
            return tokenizer

        except Exception as e:
            logger.error(f"Error in preparing tokenizer: {e}")
            raise e
    
    def full_model(self):
        """Update the base model with specific configurations."""
        logger.info("Updating base model configuration")
        
        model = self.get_base_model()
        tokenizer = self.prepare_tokenizer()
        
        model.save(self.config.updated_base_model_path)
        logger.info(f"Updated base model saved at: {self.config.updated_base_model_path}")
        
        mlflow.log_artifact(str(self.config.tokenizer_path), "tokenizer")
        mlflow.log_artifact(str(self.config.updated_base_model_path), "base_model")

        return model, tokenizer
        