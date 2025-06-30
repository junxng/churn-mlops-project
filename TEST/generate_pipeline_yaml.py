#!/usr/bin/env python3
"""
Script to generate Kubeflow pipeline YAML for churn prediction
"""

import os
from kfp.compiler import Compiler
from TEST.kubeflow_pipeline import churn_prediction_pipeline
from src.Churn.utils.logging import logger

def generate_pipeline_yaml():
    """Generate the Kubeflow pipeline YAML file"""
    try:
        logger.info("Starting pipeline YAML generation...")
        
        # Compile the pipeline
        compiler = Compiler()
        compiler.compile(
            pipeline_func=churn_prediction_pipeline,
            package_path="churn_pipeline.yaml"
        )
        
        logger.info("Pipeline YAML generated successfully: churn_pipeline.yaml")
        
        if os.path.exists("churn_pipeline.yaml"):
            file_size = os.path.getsize("churn_pipeline.yaml") / 1024  # Size in KB
            logger.info(f"Pipeline YAML file size: {file_size:.2f} KB")
            return True
        else:
            logger.error("Pipeline YAML file was not created")
            return False
            
    except Exception as e:
        logger.error(f"Error generating pipeline YAML: {e}")
        return False

if __name__ == "__main__":
    success = generate_pipeline_yaml()
    if success:
        print("churn_pipeline.yaml generated successfully!")
        print("You can now run: python main_kube.py")
    else:
        print("Failed to generate pipeline YAML")
        exit(1) 
        
        
# class VersionConfiguration:
#     def __init__(self):
#         self.root_folder = "artifacts"
#         self.data_version_dir = os.path.join(self.root_folder, "data_version")
#         self.data_ingestion_dir = os.path.join(self.root_folder, "data_ingestion")
#         self.model_version_dir = os.path.join(self.root_folder, "model_version")
#         self.evaluation_dir = os.path.join(self.root_folder, "evaluation")
#         self.datetime_suffix = datetime.now().strftime('%Y%m%dT%H%M%S')
#         self.credential = "C:/Users/Admin/Downloads/llmops-460406-f379299f4261.json"

# class gcpConfig:
#     def __init__(self):
#         self.credential = "C:/Users/Admin/Downloads/llmops-460406-f379299f4261.json"  
#         self.bucket = "churn_data_version"
#         self.workers = 8
# class SupportModel:
#     def most_common(lst):
#         counts = Counter(lst)
#         if not counts:
#             return None 
#         return counts.most_common(1)[0][0]
#     def get_dummies(df):
#         categorical_cols = df.select_dtypes(include=['object']).columns
#         if len(categorical_cols) > 0:
#             for col in categorical_cols:
#                 if df[col].isin(['yes', 'no', 'True', 'False']).any():
#                     df[col] = df[col].map({'yes': 1, 'True': 1, 'no': 0, 'False': 0})
#                 else:
#                     df = pd.get_dummies(df, columns=[col])
#         return df
# class ModelConfiguration:
#     def __init__(self):
#         self.random_state = 42
#         self.test_size = 0.2
#         self.n_estimators = 500
#         self.criterion="entropy"
#         self.max_depth=39
#         self.max_features="log2"
#         self.min_samples_leaf=2


# def get_dataset():
#     version_config = VersionConfiguration()
#     df_input = pd.read_csv(version_config.data_ingestion_dir + "/input_raw.csv")
#     print(f"Loaded dataset with {len(df_input)} rows")
#     print(f"Columns found: {list(df_input.columns)}")
#     input_data_versioned_name = f"input_raw_data_version_{version_config.datetime_suffix}.csv"
#     input_data_versioned_path = Path(version_config.data_version_dir) / input_data_versioned_name
#     df_input.to_csv(input_data_versioned_path, index=False)
#     print(f"Created versioned input data file: {input_data_versioned_path}")
#     df_input.columns = df_input.columns.map(lambda x: str(x).strip())
#     cols_to_drop = {"Returns", "Age", "Total Purchase Amount"}
#     df_input.drop(columns=[col for col in cols_to_drop if col in df_input.columns], inplace=True)
#     df_input.dropna(inplace=True)
#     if 'Price' not in df_input.columns:
#         df_input['Price'] = df_input['Product Price']
#     if 'Product Price' not in df_input.columns:
#         raise KeyError("Required column 'Product Price' is missing from the dataset.")
    
#     df_input['TotalSpent'] = df_input['Quantity'] * df_input['Price']
#     df_features = df_input.groupby("customer_id", as_index=False, sort=False).agg(
#         LastPurchaseDate = ("Purchase Date","max"),
#         Favoured_Product_Categories = ("Product Category", lambda x: SupportModel.most_common(list(x))),
#         Frequency = ("Purchase Date", "count"),
#         TotalSpent = ("TotalSpent", "sum"),
#         Favoured_Payment_Methods = ("Payment Method", lambda x: SupportModel.most_common(list(x))),
#         Customer_Name = ("Customer Name", "first"),
#         Customer_Label = ("Customer_Labels", "first"),
#         Churn = ("Churn", "first"),
#     )

#     df_features = df_features.drop_duplicates(subset=['Customer_Name'], keep='first')
#     df_features['LastPurchaseDate'] = pd.to_datetime(df_features['LastPurchaseDate'])
#     df_features['LastPurchaseDate'] = df_features['LastPurchaseDate'].dt.date
#     df_features['LastPurchaseDate'] = pd.to_datetime(df_features['LastPurchaseDate'])
#     max_LastBuyingDate = df_features["LastPurchaseDate"].max()
#     df_features['Recency'] = (max_LastBuyingDate - df_features['LastPurchaseDate']).dt.days
#     df_features['LastPurchaseDate'] = df_features['LastPurchaseDate'].dt.date
#     df_features['Avg_Spend_Per_Purchase'] = df_features['TotalSpent'] / df_features['Frequency'].replace(0, 1)
#     df_features['Purchase_Consistency'] = df_features['Recency'] / df_features['Frequency'].replace(0, 1)
#     df_features.drop(columns=["customer_id","LastPurchaseDate",'Customer_Name'], inplace=True)
#     df_features.dropna(inplace=True)
#     print(f"processed dataset with {len(df_features)} rows")
#     print(f"After feature engineering columns: {list(df_features.columns)}")
#     print(f"Data shape after feature engineering: {df_features.shape}")
#     print(f"First few rows of feature engineered data: \n{df_features.head()}")
#     model_data_versioned_name = f"model_data_version_{version_config.datetime_suffix}.csv"
#     model_data_versioned_path = Path(version_config.data_version_dir) / model_data_versioned_name
#     df_features.to_csv(model_data_versioned_path, index=False)
#     print(f"Created versioned model data file: {model_data_versioned_path}")
#     df_encode = SupportModel.get_dummies(df_features)
#     print("Successfully processed data")
#     return df_encode




# def split_data(df):
#     version_config = VersionConfiguration()
#     df_model = df.copy()
#     X = df_model.drop('Churn',axis=1)
#     y = df_model['Churn']
#     print(f"X shape (features): {X.shape}")
#     print(f"y shape (target): {y.shape}")

#     print(f"Target variable shape: {y.shape}")
#     print(f"NaN count in target: {y.isna().sum()}")
#     print(f"Target variable distribution: \n{y.value_counts(normalize=True)}")

#     if y.isna().sum() > 0:
#         print(f"Found {y.isna().sum()} NaN values in target variable, filling with 0")
#         y = y.fillna(0)

#     nan_cols = X.columns[X.isna().any()].tolist()
#     if nan_cols:
#         print(f"Found NaN values in feature columns: {nan_cols}")
#         X = X.fillna(0)

#     print(f"Final feature matrix shape: {X.shape}")
#     print(f"Final target vector shape: {y.shape}")

#     class_distribution = y.value_counts(normalize=True)
#     print(f"Target variable distribution (normalized): \n{class_distribution}")

#     imbalance_threshold = 0.4

#     if class_distribution.min() < imbalance_threshold:
#         print("Target variable is imbalanced. Applying SMOTEENN...")
#         smote = SMOTEENN(random_state=42)
#         X_res, y_res = smote.fit_resample(X, y)
#         print(f"Resampled feature matrix shape: {X_res.shape}")
#         print(f"Resampled target distribution: \n{y_res.value_counts()}")
#     else:
#         print("Target variable is balanced. Skipping SMOTEENN.")
#         X_res, y_res = X, y
    
#     print("Splitting data into train and test sets")
#     X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
#     print(f"Train data: {X_train.shape}, Test data: {X_test.shape}")
    
#     print("Converting variables to DataFrames...")
    
#     if not isinstance(X_train, pd.DataFrame):
#         if hasattr(X_res, 'columns'):
#             X_train = pd.DataFrame(X_train, columns=X_res.columns)
#         else:
#             X_train = pd.DataFrame(X_train, columns=X.columns)
    
#     if not isinstance(X_test, pd.DataFrame):
#         if hasattr(X_res, 'columns'):
#             X_test = pd.DataFrame(X_test, columns=X_res.columns)
#         else:
#             X_test = pd.DataFrame(X_test, columns=X.columns)
    
#     if not isinstance(y_train, pd.DataFrame):
#         y_train = pd.DataFrame(y_train, columns=['Churn'])
    
#     if not isinstance(y_test, pd.DataFrame):
#         y_test = pd.DataFrame(y_test, columns=['Churn'])
    
#     print(f"Final DataFrame shapes:")
#     print(f"X_train: {X_train.shape}, type: {type(X_train)}")
#     print(f"X_test: {X_test.shape}, type: {type(X_test)}")
#     print(f"y_train: {y_train.shape}, type: {type(y_train)}")
#     print(f"y_test: {y_test.shape}, type: {type(y_test)}")
    
#     train_feature_path = os.path.join(version_config.data_version_dir, f"train_feature_version_{version_config.datetime_suffix}.csv")
#     test_feature_path = os.path.join(version_config.data_version_dir, f"test_feature_version_{version_config.datetime_suffix}.csv")
#     train_target_path = os.path.join(version_config.data_version_dir, f"train_target_version_{version_config.datetime_suffix}.csv")
#     test_target_path = os.path.join(version_config.data_version_dir, f"test_target_version_{version_config.datetime_suffix}.csv")
    
#     X_train.to_csv(train_feature_path, index=False)
#     X_test.to_csv(test_feature_path, index=False)
#     y_train.to_csv(train_target_path, index=False)
#     y_test.to_csv(test_target_path, index=False)
    
#     print("Data successfully saved to CSV files")
    
#     return train_feature_path, test_feature_path, train_target_path, test_target_path



# def prepare_base_model(X_train_path, X_test_path):
#     model_config = ModelConfiguration()
#     version_config = VersionConfiguration()
#     model = RandomForestClassifier(
#     n_estimators=model_config.n_estimators, 
#     random_state=model_config.random_state,
#     criterion=model_config.criterion,
#     max_depth=model_config.max_depth,
#     max_features=model_config.max_features,
#     min_samples_leaf=model_config.min_samples_leaf
# )
#     base_model_version_name = f"base_model_version_{version_config.datetime_suffix}.pkl"
#     base_model_version_path = Path(version_config.model_version_dir) / base_model_version_name
#     jb.dump(model, base_model_version_path)
#     print("Created versioned base model file:", base_model_version_path)
#     X_train = pd.read_csv(X_train_path)
#     X_test = pd.read_csv(X_test_path)

#     scaled_test_data_path = os.path.join(version_config.data_version_dir, f"test_feature_scaled_version_{version_config.datetime_suffix}.csv")
#     scaled_train_data_path = os.path.join(version_config.data_version_dir, f"train_feature_scaled_version_{version_config.datetime_suffix}.csv")
#     scaler_path = os.path.join(version_config.model_version_dir, f"scaler_churn_version_{version_config.datetime_suffix}.pkl")
#     scaler = RobustScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#     jb.dump(scaler, scaler_path)
    
#     X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
#     X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

#     X_train_scaled_df.to_csv(scaled_train_data_path, index=False)
#     X_test_scaled_df.to_csv(scaled_test_data_path, index=False)
    
#     print(f"Scalers and scaled data saved:")
#     print(f"  Scaler: {scaler_path}")
#     print(f"  X_train_scaled: {scaled_train_data_path}")
#     print(f"  X_test_scaled: {scaled_test_data_path}")
#     return base_model_version_path, scaled_test_data_path, scaled_train_data_path


# def trainning(base_model_path, scaled_train_path, scaled_test_path, y_train_path, y_test_path):
#     version_config = VersionConfiguration()
#     model = jb.load(base_model_path)
#     X_train_scaled = pd.read_csv(scaled_train_path)
#     X_test_scaled = pd.read_csv(scaled_test_path)
#     # Read target variables and convert to categorical
#     y_train = pd.read_csv(y_train_path)['Churn'].astype('int32')
#     y_test = pd.read_csv(y_test_path)['Churn'].astype('int32')
    
#     model.fit(X_train_scaled, y_train)
#     prediction = model.predict(X_test_scaled)
#     accuracy = accuracy_score(y_test, prediction)
#     print(f"Initial model accuracy: {accuracy:.4f}")
    
#     if accuracy < 0.85:
#         print("Trigger fine-tuning!")
#         rf_params = {
#             'n_estimators': [100, 200, 300, 400, 500, 700, 1000],  
#             'criterion': ['gini', 'entropy', 'log_loss'],          
#             'max_depth': [None, 10, 20, 30, 50, 70],                
#             'min_samples_split': [2, 5, 10, 15],                    
#             'min_samples_leaf': [1, 2, 4, 6],                       
#             'max_features': ['sqrt', 'log2', None],                
#             'bootstrap': [True, False],                             
#             'class_weight': [None, 'balanced', 'balanced_subsample']  
#         }
#         random_search = RandomizedSearchCV(model, rf_params, cv=5, n_jobs=1, n_iter=20, random_state=42, scoring='accuracy')
#         random_search.fit(X_train_scaled, y_train)
#         best_model = RandomForestClassifier(**random_search.best_params_, random_state=42)
#         best_model.fit(X_train_scaled, y_train)  # Train the best model
#         print(f"Best parameters found: {random_search.best_params_}")
#         print(f"Best cross-validation score: {random_search.best_score_:.4f}")
        
#         # Evaluate best model
#         best_prediction = best_model.predict(X_test_scaled)
#         best_accuracy = accuracy_score(y_test, best_prediction)
#         print(f"Best model accuracy on test set: {best_accuracy:.4f}")
        
#         best_model_version_name = f"fine_tuned_model_version_{version_config.datetime_suffix}.pkl"
#         best_model_version_path = Path(version_config.model_version_dir) / best_model_version_name
#         jb.dump(best_model, best_model_version_path)
#         print(f"Best model saved to: {best_model_version_path}")
#     else:
#         print("No fine-tuning needed!")
#         best_model = model

#     return best_model, X_test_scaled, y_test



# def evaluating(model, X_test_scaled, y_test):
#     version_config = VersionConfiguration()
#     y_pred = model.predict(X_test_scaled)
#     y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     roc_auc = roc_auc_score(y_test, y_pred_prob)
#     mcc = matthews_corrcoef(y_test, y_pred)
#     avg_precision = average_precision_score(y_test, y_pred_prob)
#     metrics = {
#     "precision": float(precision),
#     "recall": float(recall),
#     "f1_score": float(f1),
#     "roc_auc": float(roc_auc),
#     "mcc": float(mcc),
#     "avg_precision": float(avg_precision)
# }
        
#         # Generate classification report
#     report = classification_report(y_test, y_pred, output_dict=True)
#     metrics["classification_report"] = report
    
#     metrics_file_versioned = Path(version_config.evaluation_dir) / f"metrics_{version_config.datetime_suffix}.json"
#     with open(metrics_file_versioned, "w") as f:
#         json.dump(metrics, f)
#     print(f"Metrics saved to: {metrics_file_versioned}")
    
#     precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_prob)
#     avg_precision = average_precision_score(y_test, y_pred_prob)

#     plt.figure(figsize=(8, 6))
#     plt.plot(recall_vals, precision_vals, color='purple', lw=2,
#             label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve')
#     plt.legend(loc='lower left')

#     pr_path = os.path.join(version_config.evaluation_dir, f"precision_recall_curve_{version_config.datetime_suffix}.png")
#     plt.savefig(pr_path)
#     plt.close()
#     print(f"Precision-Recall curve saved to: {pr_path}")
#     print("Creating ROC curve plot")
    
#     fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
#     roc_auc = auc(fpr, tpr)
    
#     plt.figure(figsize=(8, 6))
#     plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
#     plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')
#     plt.legend(loc="lower right")
    
#     # Save the plot with datetime naming
#     roc_path = os.path.join(version_config.evaluation_dir, f"roc_curve_{version_config.datetime_suffix}.png")
#     plt.savefig(roc_path)
#     plt.close()
#     print(f"ROC curve saved to: {roc_path}")
#     return pr_path, roc_path, metrics




# def upload_to_GCP(filenames=None):
#     """Upload multiple files to Google Cloud Storage with proper error handling."""
#     try:
#         version_config = VersionConfiguration()
#         gcp_config = gcpConfig()
#         bucket_name = gcp_config.bucket
#         workers = gcp_config.workers

#         os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_config.credential
        
#         files_to_upload = []
        
#         if filenames:
#             if isinstance(filenames, (Path, str)):
#                 filenames = [str(filenames)]
#             else:
#                 filenames = [str(f) for f in filenames]
            
#             for filename in filenames:
#                 for source_dir in [version_config.data_version_dir, version_config.evaluation_dir]:
#                     file_path = os.path.join(source_dir, filename)
#                     if os.path.exists(file_path):
#                         files_to_upload.append((filename, source_dir))
#                         break
#                 else:
#                     print(f"File does not exist in any source directory: {filename}")
#         else:
#             # Upload all files from both directories
#             target_dirs = [
#                 version_config.data_version_dir,
#                 version_config.evaluation_dir
#             ]
            
#             for source_dir in target_dirs:
#                 if os.path.exists(source_dir):
#                     print(f"Scanning directory: {source_dir}")
#                     data_files = glob.glob(os.path.join(source_dir, "**", "*"), recursive=True)
#                     for file_path in data_files:
#                         if os.path.isfile(file_path):
#                             rel_path = os.path.relpath(file_path, source_dir)
#                             files_to_upload.append((rel_path, source_dir))
#                 else:
#                     print(f"Directory does not exist: {source_dir}")
        
#         if not files_to_upload:
#             print("No files found to upload")
#             return 0, 0
        
#         print(f"Found {len(files_to_upload)} files to upload")
        
#         # Initialize storage client and bucket
#         storage_client = Client()
#         bucket = storage_client.bucket(bucket_name)

#         # Group files by source directory for batch upload
#         uploads_by_source = {}
#         for rel_path, source_dir in files_to_upload:
#             if source_dir not in uploads_by_source:
#                 uploads_by_source[source_dir] = []
#             uploads_by_source[source_dir].append(rel_path)

#         total_success = 0
#         total_errors = 0

#         for source_dir, file_list in uploads_by_source.items():
#             print(f"Uploading {len(file_list)} files from {source_dir}")
            
#             try:
#                 results = transfer_manager.upload_many_from_filenames(
#                     bucket, 
#                     file_list, 
#                     source_directory=source_dir,
#                     max_workers=workers, 
#                     blob_name_prefix="churn_data_store/"
#                 )

#                 success_count = 0
#                 error_count = 0
                
#                 for name, result in zip(file_list, results):
#                     if isinstance(result, Exception):
#                         print(f"Failed to upload {name}: {result}")
#                         error_count += 1
#                     else:
#                         print(f"Uploaded {name} to bucket {bucket.name}")
#                         success_count += 1
                
#                 total_success += success_count
#                 total_errors += error_count
                
#             except Exception as e:
#                 print(f"Failed to upload batch from {source_dir}: {e}")
#                 total_errors += len(file_list)
        
#         print(f"Upload completed: {total_success} successful, {total_errors} failed")
#         return total_success, total_errors
                
#     except Exception as e:
#         print(f"Cloud storage upload failed: {e}")
#         raise e
    
    
    
# def cleanup_temp_files():
#     """Clean up temporary versioned files after pipeline completion."""

#     version_config = VersionConfiguration()
#     data_version_dir = version_config.data_version_dir
#     evaluation_dir = version_config.evaluation_dir
#     try:
#         # Pattern: *_version_YYYYMMDDTHHMMSS.csv
#         data_version_files = glob.glob(os.path.join(data_version_dir, "*_version_????????T??????.csv"))
#         print(f"Found {len(data_version_files)} timestamp-versioned data files to clean")
        
#         for file_path in data_version_files:
#             try:
#                 os.remove(file_path)
#                 print(f"Deleted temporary file: {os.path.basename(file_path)}")
#             except Exception as e:
#                 print(f"Failed to delete file {file_path}: {e}")
        
#         # Pattern: *_YYYYMMDDTHHMMSS.json and *_YYYYMMDDTHHMMSS.png
#         eval_json_files = glob.glob(os.path.join(evaluation_dir, "*_????????T??????.json"))
#         eval_png_files = glob.glob(os.path.join(evaluation_dir, "*_????????T??????.png"))
#         eval_files = eval_json_files + eval_png_files
#         print(f"Found {len(eval_files)} timestamp-versioned evaluation files to clean")
        
#         for file_path in eval_files:
#             try:
#                 os.remove(file_path)
#                 print(f"Deleted temporary file: {os.path.basename(file_path)}")
#             except Exception as e:
#                 print(f"Failed to delete file {file_path}: {e}")
        
#         remaining_data_files = len(glob.glob(os.path.join(data_version_dir, "*.csv")))
#         remaining_eval_files = len(glob.glob(os.path.join(evaluation_dir, "*")))
#         print(f"Kept {remaining_data_files} essential data files for DVC tracking")
#         print(f"Kept {remaining_eval_files} essential evaluation files for DVC tracking")
                
#     except Exception as e:
#         print(f"Error during cleanup: {e}")
        