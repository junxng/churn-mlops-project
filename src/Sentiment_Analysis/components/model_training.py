import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import mlflow
import joblib as jb
import numpy as np
import json
import seaborn as sns
from datetime import datetime
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from src.Sentiment_Analysis.utils.logging import logger
from src.Sentiment_Analysis.entity.config_entity import TrainingConfig, EvaluationConfig
from src.Sentiment_Analysis.utils.upload_data_to_s3 import upload_dataset_to_s3
from src.Sentiment_Analysis.utils.common import save_json

tf.keras.__version__ = tf.__version__

class MLflowCallback(tf.keras.callbacks.Callback):
    """Callback for logging metrics to MLflow."""
    def __init__(self, model_name):
        self.model_name = model_name
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for metric_name, metric_value in logs.items():
            mlflow.log_metric(f"{self.model_name}_{metric_name}", metric_value, step=epoch)


class TrainAndEvaluateModel:
    def __init__(self, config_train: TrainingConfig, config_eval: EvaluationConfig = None):
        self.train_config = config_train
        self.eval_config = config_eval
        self.model_name = f"model_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"

    def get_callbacks(self):
        """Get callbacks for model training."""
        return [
            MLflowCallback(self.model_name),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-5
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]
    
    def load_data(self):
        """Load training and test data."""
        logger.info(f"Loading training data from: {self.train_config.training_data}")
        train_data = pd.read_csv(self.train_config.training_data)
        logger.info(f"Loading test data from: {self.train_config.test_data}")
        test_data = pd.read_csv(self.train_config.test_data)
        upload_dataset_to_s3(
            train_file=self.train_config.training_data,
            test_file=self.train_config.test_data,
            bucket_name=self.train_config.bucket_name,
            prefix="sentiment-analysis/data"
        )
        return train_data, test_data
    
    def preprocess_data(self, train_data, test_data):
        """Preprocess data for model training."""
        logger.info("Loading tokenizer")
        tokenizer = jb.load(self.train_config.tokenizer_path)
        
        # Clean training data
        train_data = train_data.dropna(subset=["review"])
        train_data["review"] = train_data["review"].astype(str)
        
        # Clean test data
        test_data = test_data.dropna(subset=["review"])
        test_data["review"] = test_data["review"].astype(str)
        
        logger.info("Converting text to sequences")
        max_len = self.train_config.params_maxlen
        X_train = pad_sequences(
            tokenizer.texts_to_sequences(train_data["review"]), 
            maxlen=max_len
        )
        X_test = pad_sequences(
            tokenizer.texts_to_sequences(test_data["review"]), 
            maxlen=max_len
        )
        
        y_train = train_data["sentiment"]
        y_test = test_data["sentiment"]
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        
        return X_train, y_train, X_test, y_test

    def train(self, X_train, y_train, X_test, y_test):
        """Train the model."""
        logger.info(f"Loading model from: {self.train_config.updated_base_model_path}")
        model = load_model(self.train_config.updated_base_model_path)
        
        logger.info(f"Starting model training for {self.model_name}")
        history = model.fit(
            X_train, y_train,
            epochs=self.train_config.params_epochs,
            batch_size=self.train_config.params_batch_size,
            validation_split=0.2,
            callbacks=self.get_callbacks()
        )
        
        logger.info(f"Model training for {self.model_name} completed")
        
        logger.info(f"Basic evaluation on test data for {self.model_name}")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
        logger.info(f"Test loss: {test_loss:.4f}")
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        
        # Log metrics to MLflow with the model name
        mlflow.log_metric(f"{self.model_name}_test_loss", test_loss)
        mlflow.log_metric(f"{self.model_name}_test_accuracy", test_accuracy)
        
        # Save trained model
        model.save(self.train_config.trained_model_path)
        logger.info(f"Trained model saved at: {self.train_config.trained_model_path}")
        
        # Create a dictionary for basic metrics
        basic_metrics = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy)
        }
        
        return model, history, basic_metrics
    
    def plot_history(self, history):
        """Plot and save training history."""
        logger.info("Creating training history plots")
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        
        plots_dir = os.path.join(self.train_config.root_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        history_fig_path = os.path.join(plots_dir, f"{self.model_name}_training_history.png")
        plt.savefig(history_fig_path)
        plt.close()
        logger.info(f"Training history plot saved at: {history_fig_path}")
        
        # Log plot to MLflow
        mlflow.log_artifact(history_fig_path)
        
        return history_fig_path
    
    def perform_detailed_evaluation(self, model, X_test, y_test):
        """Evaluate the model in detail and log results to MLflow."""
        logger.info("Performing detailed evaluation on test data")
        
        # Get predictions
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Build metrics dictionary
        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics["classification_report"] = report
        
        # Save metrics
        save_json(self.eval_config.metrics_file, metrics)
        logger.info(f"Detailed metrics saved to: {self.eval_config.metrics_file}")
        
        # Log metrics to MLflow
        mlflow.log_metrics({
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        })
        
        mlflow.log_artifact(self.eval_config.metrics_file)
        
        return metrics, y_pred, y_pred_prob
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """Plot and save confusion matrix."""
        logger.info("Creating confusion matrix plot")
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save the plot
        cm_path = os.path.join(self.eval_config.plots_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        logger.info(f"Confusion matrix saved to: {cm_path}")
        
        # Log to MLflow
        mlflow.log_artifact(cm_path)
        
        return cm_path
    
    def plot_roc_curve(self, y_test, y_pred_prob):
        """Plot and save ROC curve."""
        logger.info("Creating ROC curve plot")
        
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Save the plot
        roc_path = os.path.join(self.eval_config.plots_dir, "roc_curve.png")
        plt.savefig(roc_path)
        plt.close()
        logger.info(f"ROC curve saved to: {roc_path}")
        
        # Log to MLflow
        mlflow.log_artifact(roc_path)
        mlflow.log_metric("roc_auc", roc_auc)
        
        return roc_path
    
    def log_model_params(self):
        """Log model parameters to MLflow."""
        logger.info("Logging model parameters to MLflow")
        
        for param_name, param_value in self.eval_config.all_params.items():
            mlflow.log_param(param_name, param_value)
    
    def log_model(self, model):
        """Log model to MLflow."""
        logger.info("Logging model to MLflow")
        mlflow.tensorflow.log_model(model, "final_cnn_model")
        mlflow.log_artifact(str(self.train_config.trained_model_path))
    
    def train_and_evaluate(self):
        """Main method to train and evaluate the model."""
        logger.info("Initiating model training and evaluation")
        
        # Load and preprocess data
        train_data, test_data = self.load_data()
        X_train, y_train, X_test, y_test = self.preprocess_data(train_data, test_data)
        
        # Train the model
        model, history, basic_metrics = self.train(X_train, y_train, X_test, y_test)
        
        # Plot training history
        history_fig_path = self.plot_history(history)
        
        # Skip detailed evaluation if eval_config is not provided
        if self.eval_config is None:
            logger.info("Evaluation config not provided, skipping detailed evaluation")
            return model, history, basic_metrics
        
        # Log all model parameters
        self.log_model_params()
        
        os.makedirs(self.eval_config.plots_dir, exist_ok=True)
        
        # Perform detailed evaluation and get predictions
        detailed_metrics, y_pred, y_pred_prob = self.perform_detailed_evaluation(model, X_test, y_test)
        
        # Create and save plots
        cm_path = self.plot_confusion_matrix(y_test, y_pred)
        roc_path = self.plot_roc_curve(y_test, y_pred_prob)
        
        # Log the model
        self.log_model(model)
        
        # Combine basic and detailed metrics
        combined_metrics = {**basic_metrics, **detailed_metrics}
        
        return model, history, combined_metrics
