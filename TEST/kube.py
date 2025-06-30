from kfp import dsl
from kfp.dsl import Input, Output, Model, Dataset, Metrics, Artifact
from kfp import compiler
from kfp.client import Client
from typing import NamedTuple
import os
import time
import sys

@dsl.pipeline(
    name='churn-prediction-pipeline-gcs',
    description='A pipeline for churn prediction with GCS data source'
)
def churn_prediction_pipeline():

    @dsl.component(
        base_image='python:3.11.9',
        packages_to_install=[
            'pandas==2.0.3',
            'numpy==1.24.3',
            'scikit-learn==1.3.0',
            'imbalanced-learn==0.11.0',
            'joblib==1.3.2',
            'matplotlib==3.7.2',
        ]
    )
    def data_ingestion(processed_data: Output[Dataset]):
        import pandas as pd
        import os
        from datetime import datetime
        from pathlib import Path
        from collections import Counter

        def handle_error(error_msg, error_type=Exception):
            print(f"Error: {error_msg}", file=sys.stderr)
            raise error_type(error_msg)

        try:
            class VersionConfiguration:
                def __init__(self):
                    try:
                        self.root_folder = "/tmp/artifacts"
                        self.data_version_dir = os.path.join(self.root_folder, "data_version")
                        os.makedirs(self.data_version_dir, exist_ok=True)
                        self.datetime_suffix = datetime.now().strftime('%Y%m%dT%H%M%S')
                    except Exception as e:
                        handle_error(f"Failed to initialize version configuration: {str(e)}")

            class SupportModel:
                @staticmethod
                def most_common(lst):
                    counts = Counter(lst)
                    if not counts:
                        return None 
                    return counts.most_common(1)[0][0]
                
                @staticmethod
                def get_dummies(df):
                    categorical_cols = df.select_dtypes(include=['object']).columns
                    if len(categorical_cols) > 0:
                        for col in categorical_cols:
                            if df[col].isin(['yes', 'no', 'True', 'False']).any():
                                df[col] = df[col].map({'yes': 1, 'True': 1, 'no': 0, 'False': 0})
                            else:
                                df = pd.get_dummies(df, columns=[col], prefix=col)
                    return df

            def get_dataset():
                try:
                    version_config = VersionConfiguration()

                    try:
                        df_input = pd.read_csv("https://raw.githubusercontent.com/Teungtran/churn_mlops/main/artifacts/data_ingestion/input_raw.csv")
                        print(f"Loaded dataset with {len(df_input)} rows and columns {list(df_input.columns)}")
                    except Exception as e:
                        handle_error(f"Error loading data from github: {str(e)}")
                    
                    if df_input is None or df_input.empty:
                        handle_error("Failed to load data from github or data is empty", ValueError)
                    
                    input_data_versioned_name = f"input_raw_data_version_{version_config.datetime_suffix}.csv"
                    input_data_versioned_path = Path(version_config.data_version_dir) / input_data_versioned_name
                    df_input.to_csv(input_data_versioned_path, index=False)
                    print(f"Created versioned input data file: {input_data_versioned_path}")
                    
                    df_input.columns = df_input.columns.map(lambda x: str(x).strip())
                    
                    cols_to_drop = {"Returns", "Age", "Total Purchase Amount"}
                    df_input.drop(columns=[col for col in cols_to_drop if col in df_input.columns], inplace=True)
                    df_input.dropna(inplace=True)
                    
                    if 'Price' not in df_input.columns and 'Product Price' in df_input.columns:
                        df_input['Price'] = df_input['Product Price']
                    
                    required_columns = ['customer_id', 'Purchase Date', 'Product Category', 'Quantity', 'Price', 'Payment Method', 'Customer Name', 'Customer_Labels', 'Churn']
                    missing_columns = [col for col in required_columns if col not in df_input.columns]
                    
                    if missing_columns:
                        print(f"Warning: Missing required columns: {missing_columns}")
                        print(f"Available columns: {list(df_input.columns)}")
                        
                        # Try to map common column variations
                        column_mappings = {
                            'customer_id': ['Customer ID', 'CustomerId', 'ID'],
                            'Purchase Date': ['PurchaseDate', 'Date', 'purchase_date'],
                            'Product Category': ['ProductCategory', 'Category', 'product_category'],
                            'Customer Name': ['CustomerName', 'Name', 'customer_name'],
                            'Customer_Labels': ['CustomerLabels', 'Labels', 'customer_labels'],
                            'Payment Method': ['PaymentMethod', 'payment_method']
                        }
                        
                        for standard_col, alternatives in column_mappings.items():
                            if standard_col not in df_input.columns:
                                for alt in alternatives:
                                    if alt in df_input.columns:
                                        df_input[standard_col] = df_input[alt]
                                        print(f"Mapped {alt} -> {standard_col}")
                                        break
                    
                    if all(col in df_input.columns for col in ['Quantity', 'Price']):
                        df_input['TotalSpent'] = df_input['Quantity'] * df_input['Price']
                        
                        df_features = df_input.groupby("customer_id", as_index=False, sort=False).agg(
                            LastPurchaseDate=("Purchase Date", "max"),
                            Favoured_Product_Categories=("Product Category", lambda x: SupportModel.most_common(list(x))),
                            Frequency=("Purchase Date", "count"),
                            TotalSpent=("TotalSpent", "sum"),
                            Favoured_Payment_Methods=("Payment Method", lambda x: SupportModel.most_common(list(x))),
                            Customer_Name=("Customer Name", "first"),
                            Customer_Label=("Customer_Labels", "first"),
                            Churn=("Churn", "first"),
                        )

                        df_features = df_features.drop_duplicates(subset=['Customer_Name'], keep='first')
                        df_features['LastPurchaseDate'] = pd.to_datetime(df_features['LastPurchaseDate'])
                        max_LastBuyingDate = df_features["LastPurchaseDate"].max()
                        df_features['Recency'] = (max_LastBuyingDate - df_features['LastPurchaseDate']).dt.days
                        df_features['Avg_Spend_Per_Purchase'] = df_features['TotalSpent'] / df_features['Frequency'].replace(0, 1)
                        df_features['Purchase_Consistency'] = df_features['Recency'] / df_features['Frequency'].replace(0, 1)
                        
                        # Drop unnecessary columns
                        df_features.drop(columns=["customer_id", "LastPurchaseDate", 'Customer_Name'], inplace=True)
                        df_features.dropna(inplace=True)
                        
                        print(f"Processed dataset with {len(df_features)} rows")
                        print(f"After feature engineering columns: {list(df_features.columns)}")
                        
                        # Encode categorical variables
                        df_encode = SupportModel.get_dummies(df_features)
                    else:
                        print("Warning: Required columns for feature engineering not found. Using original data.")
                        df_encode = SupportModel.get_dummies(df_input)
                    
                    # Save processed data with error handling
                    try:
                        df_encode.to_csv(processed_data.path, index=False)
                        print(f"Processed data saved to: {processed_data.path}")
                        print(f"Final dataset shape: {df_encode.shape}")
                        print(f"Final columns: {list(df_encode.columns)}")
                    except Exception as e:
                        handle_error(f"Failed to save processed data: {str(e)}")
                    
                    return df_encode
                
                except Exception as e:
                    handle_error(f"Error in get_dataset: {str(e)}")

            get_dataset()

        except Exception as e:
            handle_error(f"Fatal error in data_ingestion: {str(e)}")

    @dsl.component(
        base_image='python:3.11.9',
        packages_to_install=[
            'pandas==2.0.3',
            'numpy==1.24.3',
            'scikit-learn==1.3.0',
            'imbalanced-learn==0.11.0'
        ]
    )
    def data_split(
        processed_data: Input[Dataset],
        X_train: Output[Dataset],
        X_test: Output[Dataset],
        y_train: Output[Dataset],
        y_test: Output[Dataset]
    ):
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from imblearn.combine import SMOTEENN
        from collections import Counter

        def split_data():
            # Load processed data
            df_model = pd.read_csv(processed_data.path)
            print(f"Loaded data with shape: {df_model.shape}")
            print(f"Columns: {list(df_model.columns)}")
            
            # Check if Churn column exists
            if 'Churn' not in df_model.columns:
                print("Warning: 'Churn' column not found. Available columns:")
                print(list(df_model.columns))
                potential_targets = [col for col in df_model.columns if 'churn' in col.lower() or 'target' in col.lower()]
                if potential_targets:
                    target_col = potential_targets[0]
                    print(f"Using '{target_col}' as target column")
                    df_model['Churn'] = df_model[target_col]
                else:
                    raise ValueError("No suitable target column found")
            
            # Separate features and target
            X = df_model.drop('Churn', axis=1)
            y = df_model['Churn']
            
            print(f"X shape (features): {X.shape}")
            print(f"y shape (target): {y.shape}")
            print(f"Target distribution: \n{y.value_counts(normalize=True)}")

            # Handle missing values
            if y.isna().sum() > 0:
                print(f"Found {y.isna().sum()} NaN values in target variable, filling with 0")
                y = y.fillna(0)

            if X.isna().any().any():
                print("Found NaN values in features, filling with 0")
                X = X.fillna(0)

            # Check for class imbalance
            class_distribution = y.value_counts(normalize=True)
            imbalance_threshold = 0.4

            if len(class_distribution) > 1 and class_distribution.min() < imbalance_threshold:
                print("Target variable is imbalanced. Applying SMOTEENN...")
                try:
                    smote = SMOTEENN(random_state=42)
                    X_res, y_res = smote.fit_resample(X, y)
                    print(f"Resampled feature matrix shape: {X_res.shape}")
                    print(f"Resampled target distribution: \n{Counter(y_res)}")
                except Exception as e:
                    print(f"SMOTEENN failed: {e}. Using original data.")
                    X_res, y_res = X, y
            else:
                print("Target variable is balanced or single class. Skipping SMOTEENN.")
                X_res, y_res = X, y
            
            # Split data
            X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(
                X_res, y_res, test_size=0.2, random_state=42, 
                stratify=y_res if len(Counter(y_res)) > 1 else None
            )
            
            print(f"Train data: {X_train_data.shape}, Test data: {X_test_data.shape}")
            
            # Convert to DataFrames if needed
            if hasattr(X_res, 'columns'):
                columns = X_res.columns
            else:
                columns = X.columns
                
            X_train_df = pd.DataFrame(X_train_data, columns=columns)
            X_test_df = pd.DataFrame(X_test_data, columns=columns)
            y_train_df = pd.DataFrame(y_train_data, columns=['Churn'])
            y_test_df = pd.DataFrame(y_test_data, columns=['Churn'])
            
            # Save split data
            X_train_df.to_csv(X_train.path, index=False)
            X_test_df.to_csv(X_test.path, index=False)
            y_train_df.to_csv(y_train.path, index=False)
            y_test_df.to_csv(y_test.path, index=False)
            
            print("Data split completed and saved")

        split_data()

    @dsl.component(
        base_image='python:3.11.9',
        packages_to_install=[
            'pandas==2.0.3',
            'numpy==1.24.3',
            'scikit-learn==1.3.0',
            'joblib==1.3.2'
        ]
    )
    def prepare_model(
        X_train: Input[Dataset],
        X_test: Input[Dataset],
        base_model: Output[Model],
        X_train_scaled: Output[Dataset],
        X_test_scaled: Output[Dataset],
        scaler_artifact: Output[Artifact]
    ):
        import pandas as pd
        import joblib as jb
        from sklearn.preprocessing import RobustScaler
        from sklearn.ensemble import RandomForestClassifier

        def prepare_base_model():
            # Create base model
            model = RandomForestClassifier(
                n_estimators=500, 
                random_state=42,
                criterion="entropy",
                max_depth=39,
                max_features="log2",
                min_samples_leaf=2
            )
            
            # Save base model
            jb.dump(model, base_model.path)
            print(f"Base model saved to: {base_model.path}")
            
            # Load training and test data
            X_train_data = pd.read_csv(X_train.path)
            X_test_data = pd.read_csv(X_test.path)
            
            # Scale data
            scaler = RobustScaler()
            X_train_scaled_data = scaler.fit_transform(X_train_data)
            X_test_scaled_data = scaler.transform(X_test_data)
            
            # Save scaler
            jb.dump(scaler, scaler_artifact.path)
            print(f"Scaler saved to: {scaler_artifact.path}")
            
            # Convert back to DataFrames and save
            X_train_scaled_df = pd.DataFrame(X_train_scaled_data, columns=X_train_data.columns)
            X_test_scaled_df = pd.DataFrame(X_test_scaled_data, columns=X_test_data.columns)
            
            X_train_scaled_df.to_csv(X_train_scaled.path, index=False)
            X_test_scaled_df.to_csv(X_test_scaled.path, index=False)
            
            print("Model preparation completed")

        prepare_base_model()

    @dsl.component(
        base_image='python:3.11.9',
        packages_to_install=[
            'pandas==2.0.3',
            'numpy==1.24.3',
            'scikit-learn==1.3.0',
            'joblib==1.3.2'
        ]
    )
    def model_training(
        base_model: Input[Model],
        X_train_scaled: Input[Dataset],
        X_test_scaled: Input[Dataset],
        y_train: Input[Dataset],
        y_test: Input[Dataset],
        trained_model: Output[Model]
    ):
        import pandas as pd
        import joblib as jb
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import RandomizedSearchCV

        def train_model():
            # Load model and data
            model = jb.load(base_model.path)
            X_train_data = pd.read_csv(X_train_scaled.path)
            X_test_data = pd.read_csv(X_test_scaled.path)
            y_train_data = pd.read_csv(y_train.path)['Churn'].astype('int32')
            y_test_data = pd.read_csv(y_test.path)['Churn'].astype('int32')
            
            print(f"Training data shape: {X_train_data.shape}")
            print(f"Test data shape: {X_test_data.shape}")
            
            # Initial training
            model.fit(X_train_data, y_train_data)
            prediction = model.predict(X_test_data)
            accuracy = accuracy_score(y_test_data, prediction)
            print(f"Initial model accuracy: {accuracy:.4f}")
            
            # Fine-tuning if accuracy is low
            if accuracy < 0.85:
                print("Triggering fine-tuning...")
                rf_params = {
                    'n_estimators': [100, 200, 300, 500],
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2'],
                    'bootstrap': [True, False]
                }
                
                random_search = RandomizedSearchCV(
                    model, rf_params, cv=3, n_jobs=-1, n_iter=10, 
                    random_state=42, scoring='accuracy'
                )
                random_search.fit(X_train_data, y_train_data)
                
                best_model = random_search.best_estimator_
                print(f"Best parameters: {random_search.best_params_}")
                print(f"Best CV score: {random_search.best_score_:.4f}")
                
                # Evaluate best model
                best_prediction = best_model.predict(X_test_data)
                best_accuracy = accuracy_score(y_test_data, best_prediction)
                print(f"Best model test accuracy: {best_accuracy:.4f}")
            else:
                print("No fine-tuning needed!")
                best_model = model
            
            # Save trained model
            jb.dump(best_model, trained_model.path)
            print(f"Trained model saved to: {trained_model.path}")

        train_model()

    @dsl.component(
        base_image='python:3.11.9',
        packages_to_install=[
            'pandas==2.0.3',
            'numpy==1.24.3',
            'scikit-learn==1.3.0',
            'matplotlib==3.7.2'
        ]
    )
    def model_evaluation(
        trained_model: Input[Model],
        X_test_scaled: Input[Dataset],
        y_test: Input[Dataset],
        metrics_output: Output[Metrics],
        pr_curve: Output[Artifact],
        roc_curve: Output[Artifact]
    ):
        import pandas as pd
        import joblib as jb
        import matplotlib.pyplot as plt
        import json
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            classification_report, roc_curve as roc_curve_func, auc,
            precision_recall_curve, average_precision_score,
            matthews_corrcoef, roc_auc_score
        )

        def evaluate_model():
            # Load model and data
            model = jb.load(trained_model.path)
            X_test_data = pd.read_csv(X_test_scaled.path)
            y_test_data = pd.read_csv(y_test.path)['Churn'].astype('int32')
            
            # Make predictions
            y_pred = model.predict(X_test_data)
            y_pred_prob = model.predict_proba(X_test_data)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_data, y_pred)
            precision = precision_score(y_test_data, y_pred, average='weighted')
            recall = recall_score(y_test_data, y_pred, average='weighted')
            f1 = f1_score(y_test_data, y_pred, average='weighted')
            roc_auc = roc_auc_score(y_test_data, y_pred_prob)
            mcc = matthews_corrcoef(y_test_data, y_pred)
            avg_precision = average_precision_score(y_test_data, y_pred_prob)
            
            metrics = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "roc_auc": float(roc_auc),
                "mcc": float(mcc),
                "avg_precision": float(avg_precision)
            }
            metrics['classification_report'] = classification_report(y_test_data, y_pred)
            print(f"Model Evaluation Results:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            # Save metrics
            with open(metrics_output.path, "w") as f:
                json.dump(metrics, f, indent=2)
            
            # Generate Precision-Recall curve
            precision_vals, recall_vals, _ = precision_recall_curve(y_test_data, y_pred_prob)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall_vals, precision_vals, color='purple', lw=2,
                    label=f'PR curve (AP = {avg_precision:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.grid(True)
            plt.savefig(pr_curve.path)
            plt.close()
            
            # Generate ROC curve
            fpr, tpr, _ = roc_curve_func(y_test_data, y_pred_prob)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True)
            plt.savefig(roc_curve.path)
            plt.close()
            
            print("Model evaluation completed")

        evaluate_model()

    # Create tasks
    data_ingestion_task = data_ingestion()
    

    data_split_task = data_split(processed_data=data_ingestion_task.outputs['processed_data'])
    
    model_prep_task = prepare_model(
        X_train=data_split_task.outputs['X_train'],
        X_test=data_split_task.outputs['X_test']
    )
    
    training_task = model_training(
        base_model=model_prep_task.outputs['base_model'],
        X_train_scaled=model_prep_task.outputs['X_train_scaled'],
        X_test_scaled=model_prep_task.outputs['X_test_scaled'],
        y_train=data_split_task.outputs['y_train'],
        y_test=data_split_task.outputs['y_test']
    )
    
    evaluation_task = model_evaluation(
        trained_model=training_task.outputs['trained_model'],
        X_test_scaled=model_prep_task.outputs['X_test_scaled'],
        y_test=data_split_task.outputs['y_test']
    )
    data_split_task.after(data_ingestion_task)
    model_prep_task.after(data_split_task)
    training_task.after(model_prep_task)
    evaluation_task.after(training_task)
    
    
    
def compile_pipeline():
    """Compile the Kubeflow pipeline to YAML"""
    try:
        print("Compiling Kubeflow pipeline with GCS support...")
        compiler_instance = compiler.Compiler()
        pipeline_file = "churn_pipeline_gcs.yaml"
        compiler_instance.compile(
            pipeline_func=churn_prediction_pipeline,
            package_path=pipeline_file
        )
        print(f"Pipeline compiled successfully: {pipeline_file}")
        return True, pipeline_file
    except Exception as e:
        print(f"Error compiling pipeline: {e}")
        return False, None

def deploy_pipeline():
    """Deploy pipeline to Kubeflow with GCS data source"""
    max_retries = 3
    retry_delay = 5

    success, pipeline_file = compile_pipeline()
    if not success:
        return False

    for attempt in range(max_retries):
        try:
            print(f"Deploying pipeline to Kubeflow (attempt {attempt + 1}/{max_retries})...")

            # Configure client with explicit artifact storage settings
            client = Client(
                host='http://localhost:8080',
                namespace='kubeflow'
            )
            
            run_name = f"churn-prediction-gcs-{int(time.time())}"

            try:
                experiment = client.get_experiment(experiment_name="churn-prediction-gcs-experiments")
            except:
                experiment = client.create_experiment(name="churn-prediction-gcs-experiments")

            # Submit pipeline run with explicit artifact storage configuration
            run = client.create_run_from_pipeline_package(
                pipeline_file=pipeline_file,
                run_name=run_name,
                experiment_name="churn-prediction-gcs-experiments",
                enable_caching=True  # Enable caching to improve performance
            )

            print(f"✅ Pipeline run created successfully: {run.run_id}")
            print(f"📊 Data source: gs://churn_data_version/input_raw.csv")
            print(f"🔗 Monitor progress at: http://localhost:8080/#/runs/details/{run.run_id}")
            
            return True

        except Exception as e:
            print(f"Error deploying pipeline (attempt {attempt + 1}): {e}")
            
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("All retry attempts failed")
                return False

    return False

def run_kubeflow_pipeline():
    """Main function to run the Kubeflow pipeline with GCS"""
    print("=" * 70)
    print("STARTING KUBEFLOW PIPELINE WITH GOOGLE CLOUD STORAGE")
    print("=" * 70)
    
    if not deploy_pipeline():
        print("Pipeline deployment failed. Exiting.")
        return False
    
    print("=" * 70)
    print("KUBEFLOW PIPELINE DEPLOYMENT COMPLETED")
    print("Check Kubeflow UI at http://localhost:8080 for execution status")
    print("=" * 70)
    
    return True

