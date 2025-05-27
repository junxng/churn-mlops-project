# Fixes and Improvements Summary

## Issues Found and Fixed

### 1. **Datetime Naming Requirements Implementation**

**Issue**: The requirements specified that all artifacts should use datetime-based naming with format `%Y%m%dT%H%M%S`, but this was not consistently implemented.

**Fixes Applied**:

#### In `data_ingestion.py`:
- Added `self.datetime_suffix = datetime.now().strftime('%Y%m%dT%H%M%S')` in `__init__`
- Added `self.data_version = f"model_churn_data_{self.datetime_suffix}"`
- Updated file naming:
  - `X_train_{datetime_suffix}.csv`
  - `X_test_{datetime_suffix}.csv`
  - `y_train_{datetime_suffix}.csv`
  - `y_test_{datetime_suffix}.csv`

#### In `base_model.py`:
- Added datetime suffix generation in `__init__`
- Updated file naming:
  - `base_model_churn_{datetime_suffix}.pkl`
  - `scaler_churn_{datetime_suffix}.pkl`
  - `X_train_scaled_{datetime_suffix}.csv`
  - `X_test_scaled_{datetime_suffix}.csv`

#### In `model_training.py`:
- Added shared datetime suffix across all artifacts
- Updated file naming:
  - `model_churn_{datetime_suffix}.pkl`
  - `finetuned_churn_{datetime_suffix}.pkl`
  - `metrics_{datetime_suffix}.json`
  - `confusion_matrix_{datetime_suffix}.png`
  - `roc_curve_{datetime_suffix}.png`
  - `precision_recall_curve_{datetime_suffix}.png`

### 2. **Config Entity Fixes**

**Issue**: Duplicate `base_model_path` field in `PrepareBaseModelConfig`
```python
# Before (BROKEN)
base_model_path: Path
base_model_path: Path  # Duplicate
```

**Fix**: Removed duplicate field and cleaned up missing fields in `TrainingConfig`

### 3. **Critical Scaler Logic Bug**

**Issue**: Major data leakage bug in `base_model.py`:
```python
# Before (WRONG - Data Leakage)
X_train_scaled = scaler.fit(X_train)  # Only fit, no transform
X_test_scaled = scaler.fit_transform(X_test)  # Re-fitting on test data!
```

**Fix**: Proper scaler usage:
```python
# After (CORRECT)
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform train
X_test_scaled = scaler.transform(X_test)  # Only transform test
```

### 4. **Fine-tuning Logic Flaw**

**Issue**: In `model_training.py`, fine-tuning created a new unfitted model:
```python
# Before (WRONG)
best_model = RandomForestClassifier(**random_search.best_params_, random_state=42)
# Model was not fitted!
```

**Fix**: Use the already fitted best estimator:
```python
# After (CORRECT)
best_model = random_search.best_estimator_
```

### 5. **Multiprocessing Issues Fixed**

**Issue**: Windows multiprocessing errors from SMOTEENN and RandomizedSearchCV:
```
An attempt has been made to start a new process before the
current process has finished its bootstrapping phase.
```

**Fixes Applied**:
- **In `data_ingestion.py`**: `SMOTEENN(random_state=42, n_jobs=1)`
- **In `model_training.py`**: `RandomizedSearchCV(model, rf_params, cv=5, n_jobs=1)`
- **In `main_pipeline.py`**: Added `multiprocessing.freeze_support()` protection

### 6. **Configuration Path Issues**

**Issue**: Config still referenced static file names (`X_train.csv`) instead of dynamic datetime-based names.

**Fix**: Created proper pipeline orchestration in `main_pipeline.py` that:
- Passes dynamic paths between components
- Each component generates its own datetime-versioned files
- Components return the actual generated file paths
- Downstream components use the returned paths instead of config paths

### 7. **JSON Serialization Fix**

**Issue**: Trying to serialize numpy arrays in metrics:
```python
# Before (BROKEN)
"precision_vals": float(precision_vals),  # Can't convert array to float
```

**Fix**: Convert arrays to lists:
```python
# After (CORRECT)
"precision_vals": precision_vals.tolist(),
"recall_vals": recall_vals.tolist(),
"thresholds": thresholds.tolist()
```

### 8. **Method Signature Updates**

**Issue**: Methods didn't return the dynamically created paths

**Fixes**:
- Updated `data_ingestion_pipeline()` to return all file paths
- Updated `full_model()` to return all generated paths
- Updated `train_and_evaluate()` to accept dynamic paths as parameters
- Updated `train()` and `fine_tune()` to return model paths

### 9. **Proper Pipeline Orchestration**

**Created `main_pipeline.py`** that:
- Manages the entire workflow with proper path passing
- Includes multiprocessing protection
- Provides clear logging for each stage
- Ensures all artifacts share the same datetime suffix
- Handles errors gracefully

## Directory Structure Requirements Met

✅ **Input data**: `C:\Users\Admin\Desktop\Data projects\python\Decision-making-system\churn_mlops\artifacts\data_ingestion`

✅ **Training data**: `C:\Users\Admin\Desktop\Data projects\python\Decision-making-system\churn_mlops\artifacts\training`
- `X_train_{datetime}.csv`
- `X_test_{datetime}.csv`
- `y_train_{datetime}.csv`
- `y_test_{datetime}.csv`
- `X_train_scaled_{datetime}.csv`
- `X_test_scaled_{datetime}.csv`

✅ **Models and artifacts**: All now follow datetime naming:
- `base_model_churn_{datetime}.pkl`
- `model_churn_{datetime}.pkl`
- `finetuned_churn_{datetime}.pkl`
- `scaler_churn_{datetime}.pkl`

## Example Usage After Fixes

```python
# Run the complete pipeline
python main_pipeline.py

# Example datetime suffix: 20241210T143022
# All artifacts will use this same suffix:
# - model_churn_data_20241210T143022
# - base_model_churn_20241210T143022.pkl
# - model_churn_20241210T143022.pkl
# - finetuned_churn_20241210T143022.pkl
# - scaler_churn_20241210T143022.pkl
```

## Key Benefits of Fixes

1. **Consistent Versioning**: All artifacts from a single run share the same datetime
2. **No Data Leakage**: Proper scaler fit/transform usage
3. **Correct Fine-tuning**: Uses actually fitted models
4. **Multiprocessing Safe**: Works on Windows without process spawning issues
5. **Traceability**: Easy to track which files belong to which run
6. **No Overwriting**: Each run creates unique files
7. **Proper Dependencies**: Methods return correct paths for downstream usage
8. **Complete Pipeline**: Single command runs entire workflow
9. **Error Handling**: Proper exception handling and logging

## How to Run

1. Ensure your input data is in `artifacts/data_ingestion/input_raw.csv`
2. Run the complete pipeline:
   ```bash
   python main_pipeline.py
   ```
3. All versioned artifacts will be created with the same datetime suffix
4. Check logs for detailed progress and file locations 