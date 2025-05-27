# Multiprocessing Fixes and Workflow Updates

## Issues Fixed

### 1. **Main Multiprocessing Protection**

**Problem**: RuntimeError due to missing multiprocessing protection in `main.py`

**Solution**: Added proper multiprocessing guards to `main.py`:
```python
import multiprocessing

def main():
    # All execution logic here
    pass

if __name__ == '__main__':
    # Multiprocessing protection for Windows
    multiprocessing.freeze_support()
    main()
```

### 2. **Component-Level Multiprocessing Fixes**

**Applied to**:
- `SMOTEENN(random_state=42, n_jobs=1)` in `data_ingestion.py`
- `RandomizedSearchCV(model, rf_params, cv=5, n_jobs=1)` in `model_training.py`

### 3. **Dual File Naming System**

**Implementation**: All components now save files with both naming conventions:

#### Static Files (Config Compatible):
```
artifacts/training/X_train.csv
artifacts/training/X_test.csv
artifacts/training/y_train.csv
artifacts/training/y_test.csv
artifacts/training/X_train_scaled.csv
artifacts/training/X_test_scaled.csv
artifacts/prepare_base_model/base_model.pkl
artifacts/prepare_base_model/scaler_path.pkl
artifacts/training/model.pkl
artifacts/training/fine_tuned_model.pkl
artifacts/evaluation/metrics.json
```

#### Versioned Files (Datetime Stamped):
```
artifacts/training/X_train_20241210T143022.csv
artifacts/training/X_test_20241210T143022.csv
artifacts/training/y_train_20241210T143022.csv
artifacts/training/y_test_20241210T143022.csv
artifacts/training/X_train_scaled_20241210T143022.csv
artifacts/training/X_test_scaled_20241210T143022.csv
artifacts/prepare_base_model/base_model_churn_20241210T143022.pkl
artifacts/prepare_base_model/scaler_churn_20241210T143022.pkl
artifacts/training/model_churn_20241210T143022.pkl
artifacts/training/finetuned_churn_20241210T143022.pkl
artifacts/evaluation/metrics_20241210T143022.json
```

### 4. **Pipeline Updates**

#### Updated `main.py`:
- Added multiprocessing protection
- Fixed file path checks to use correct file names (`X_train.csv` instead of `train_data.csv`)
- Wrapped execution in proper function

#### Updated `prepare_data.py`:
- Now handles all 8 return values from `data_ingestion_pipeline()`
- Returns static paths for config compatibility

#### Updated `prepare_model.py`:
- Handles all 8 return values from `full_model()`
- Returns all necessary paths for downstream usage

#### Updated `train_evaluation.py`:
- Added flexible method signature to accept explicit paths
- Falls back to config paths if no explicit paths provided
- Properly handles the new 3-return signature

#### Updated `main_pipeline.py`:
- Added data preparation stage
- Proper path passing between all stages
- Comprehensive logging for each stage

### 5. **Workflow Execution Options**

**Option 1: Use existing main.py**
```bash
python main.py
```

**Option 2: Use new comprehensive pipeline**
```bash
python main_pipeline.py
```

Both will now work without multiprocessing errors!

## Key Benefits

1. ✅ **No Multiprocessing Errors**: Proper Windows compatibility
2. ✅ **Config Compatibility**: Static file names match existing config
3. ✅ **Version Control**: Datetime-stamped files for tracking
4. ✅ **Flexible Execution**: Multiple pipeline entry points
5. ✅ **Proper Path Passing**: Components communicate correctly
6. ✅ **Comprehensive Logging**: Clear progress tracking
7. ✅ **Error Handling**: Proper exception management

## File Structure After Execution

```
artifacts/
├── data_ingestion/
│   └── input_raw.csv
├── training/
│   ├── X_train.csv                    # Static (config compatible)
│   ├── X_test.csv                     # Static (config compatible)
│   ├── y_train.csv                    # Static (config compatible)
│   ├── y_test.csv                     # Static (config compatible)
│   ├── X_train_scaled.csv             # Static (config compatible)
│   ├── X_test_scaled.csv              # Static (config compatible)
│   ├── model.pkl                      # Static (config compatible)
│   ├── fine_tuned_model.pkl           # Static (config compatible)
│   ├── X_train_20241210T143022.csv    # Versioned
│   ├── X_test_20241210T143022.csv     # Versioned
│   ├── y_train_20241210T143022.csv    # Versioned
│   ├── y_test_20241210T143022.csv     # Versioned
│   ├── X_train_scaled_20241210T143022.csv  # Versioned
│   ├── X_test_scaled_20241210T143022.csv   # Versioned
│   ├── model_churn_20241210T143022.pkl     # Versioned
│   └── finetuned_churn_20241210T143022.pkl # Versioned
├── prepare_base_model/
│   ├── base_model.pkl                 # Static (config compatible)
│   ├── scaler_path.pkl                # Static (config compatible)
│   ├── base_model_churn_20241210T143022.pkl  # Versioned
│   └── scaler_churn_20241210T143022.pkl      # Versioned
└── evaluation/
    ├── metrics.json                   # Static (config compatible)
    ├── metrics_20241210T143022.json   # Versioned
    └── plots/
        ├── confusion_matrix_20241210T143022.png
        ├── roc_curve_20241210T143022.png
        └── precision_recall_curve_20241210T143022.png
```

## Running the Pipeline

Now you can safely run:
```bash
python main.py
```

The pipeline will execute without multiprocessing errors and create all the required files with both static and versioned naming! 