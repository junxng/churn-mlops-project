import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib as jb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, matthews_corrcoef
)
import mlflow
import mlflow.sklearn
from src.Churn.utils.logging import logger
from src.Churn.entity.config_entity import TrainingConfig, EvaluationConfig, VisualizationConfig
from src.Churn.utils.common import save_json
from typing import Optional

class TrainAndEvaluateModel:
    def __init__(self, config_train: TrainingConfig, config_eval: EvaluationConfig, config_viz: VisualizationConfig):
        self.train_config = config_train
        self.eval_config = config_eval
        self.viz_config = config_viz
        self.datetime_suffix = datetime.now().strftime('%Y%m%dT%H%M%S')
        self.model_name = f"model_churn_{self.datetime_suffix}"
        self.fine_tuned_model_name = f"finetuned_churn_{self.datetime_suffix}"

    def log_model_to_mlflow(self, model, model_name: str):
        logger.info(f"Logging model to MLflow with artifact path: {model_name}")
        try:
            mlflow.sklearn.log_model(model, artifact_path=model_name)
            active_run = mlflow.active_run()
            if not active_run:
                raise RuntimeError("No active MLflow run found")
            
            artifact_uri = f"runs:/{active_run.info.run_id}/{model_name}"
            registered_model_name = "RandomForestClassifier"
            mlflow.register_model(model_uri=artifact_uri, name=registered_model_name)
            logger.info(f"Successfully registered model as '{registered_model_name}' from: {artifact_uri}")
        except Exception as e:
            logger.warning(f"Failed to log or register model to MLflow: {e}")

    def train(self, X_train_scaled, y_train, model):
        logger.info(f"Starting model training for {self.model_name}")
        trained_model = model.fit(X_train_scaled, y_train)
        logger.info(f"Model training for {self.model_name} completed")
        
        os.makedirs(self.train_config.model_version_dir, exist_ok=True)
        trained_model_path = self.train_config.model_version_dir / f"model_churn_version_{self.datetime_suffix}.pkl"
        jb.dump(trained_model, trained_model_path)
        logger.info(f"Trained model saved to: {trained_model_path}")
        return trained_model, trained_model_path

    def fine_tune(self, trained_model, X_train_scaled, y_train):
        logger.info("Starting fine-tuning of the trained model")
        
        params = self.train_config.fine_tuning_params
        random_search = RandomizedSearchCV(
            trained_model, 
            param_distributions=params.param_grid,
            n_iter=params.n_iter,
            cv=params.cv,
            verbose=params.verbose,
            random_state=self.train_config.random_state,
            n_jobs=params.n_jobs
        )
        random_search.fit(X_train_scaled, y_train)
        
        best_model = random_search.best_estimator_
        logger.info(f"Best parameters found: {random_search.best_params_}")
        
        os.makedirs(self.train_config.model_version_dir, exist_ok=True)
        fine_tuned_model_path = self.train_config.model_version_dir / f"finetuned_churn_{self.datetime_suffix}.pkl"
        jb.dump(best_model, fine_tuned_model_path)
        logger.info(f"Fine-tuned model saved to: {fine_tuned_model_path}")
        
        return best_model, fine_tuned_model_path

    def perform_detailed_evaluation(self, model, X_test_scaled, y_test):
        logger.info("Performing detailed evaluation on test data")
        y_pred = model.predict(X_test_scaled)
        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_prob),
            "mcc": matthews_corrcoef(y_test, y_pred),
            "avg_precision": average_precision_score(y_test, y_pred_prob)
        }
        
        os.makedirs(self.eval_config.evaluation_dir, exist_ok=True)
        metrics_file_path = self.eval_config.evaluation_dir / f"metrics_{self.datetime_suffix}.json"
        save_json(metrics_file_path, metrics)
        logger.info(f"Detailed metrics saved to: {metrics_file_path}")
        
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(metrics_file_path))
        
        return metrics, y_pred, y_pred_prob

    def plot_confusion_matrix(self, y_test, y_pred):
        logger.info("Creating confusion matrix plot")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        os.makedirs(self.eval_config.evaluation_dir, exist_ok=True)
        cm_path = self.eval_config.evaluation_dir / f"confusion_matrix_{self.datetime_suffix}.png"
        plt.savefig(cm_path)
        plt.close()
        logger.info(f"Confusion matrix saved to: {cm_path}")
        mlflow.log_artifact(str(cm_path))

    def plot_precision_recall_curve(self, y_test, y_pred_prob):
        logger.info("Creating Precision-Recall curve plot")
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        avg_precision = average_precision_score(y_test, y_pred_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='purple', label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='best')
        
        os.makedirs(self.eval_config.evaluation_dir, exist_ok=True)
        pr_path = self.eval_config.evaluation_dir / f"precision_recall_curve_{self.datetime_suffix}.png"
        plt.savefig(pr_path)
        plt.close()
        logger.info(f"Precision-Recall curve saved to: {pr_path}")
        mlflow.log_artifact(str(pr_path))

    def plot_roc_curve(self, y_test, y_pred_prob):
        logger.info("Creating ROC curve plot")
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        os.makedirs(self.eval_config.evaluation_dir, exist_ok=True)
        roc_path = self.eval_config.evaluation_dir / f"roc_curve_{self.datetime_suffix}.png"
        plt.savefig(roc_path)
        plt.close()
        logger.info(f"ROC curve saved to: {roc_path}")
        mlflow.log_artifact(str(roc_path))

    def train_and_evaluate(self, base_model, X_train_scaled, X_test_scaled, y_train, y_test):
        logger.info("Initiating model training and evaluation")
        
        y_train = y_train.fillna(0)
        y_test = y_test.fillna(0)

        model, trained_model_path = self.train(X_train_scaled, y_train, base_model)
        accuracy = model.score(X_test_scaled, y_test)
        mlflow.log_metric("accuracy_before_tuning", accuracy)
        logger.info(f"Model accuracy on test data: {accuracy}")

        if accuracy < self.train_config.accuracy_threshold:
            logger.info("Model accuracy is below threshold, fine-tuning needed.")
            final_model, final_model_path = self.fine_tune(model, X_train_scaled, y_train)
            self.log_model_to_mlflow(final_model, self.fine_tuned_model_name)
            mlflow.log_artifact(str(trained_model_path), "model_before_tuning")
        else:
            logger.info("Model accuracy is sufficient, no fine-tuning needed.")
            final_model, final_model_path = model, trained_model_path
            self.log_model_to_mlflow(final_model, self.model_name)

        metrics, y_pred, y_pred_prob = self.perform_detailed_evaluation(final_model, X_test_scaled, y_test)
        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_precision_recall_curve(y_test, y_pred_prob)
        self.plot_roc_curve(y_test, y_pred_prob)
        
        return final_model, metrics, final_model_path
