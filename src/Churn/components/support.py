from collections import Counter
import pandas as pd
import pandas as pd
import json
from src.Churn.config.configuration import ConfigurationManager
from pathlib import Path
from src.Churn.utils.logging import logger

def most_common(lst):
    counts = Counter(lst)
    if not counts:
        return None 
    return counts.most_common(1)[0][0]


def df_to_records(df):
    return json.loads(df.to_json(orient='records', date_format='iso'))



async def import_data(uploaded_file):
    # Create configuration manager and get data ingestion config
    config_manager = ConfigurationManager()
    config = config_manager.get_data_ingestion_config()
        
    path_save = Path(config.local_data_file)  
    df = None

    if uploaded_file is not None:
        path_save.parent.mkdir(parents=True, exist_ok=True)

        filename = uploaded_file.filename.lower()
        content = await uploaded_file.read()

        with open(path_save, "wb") as f:
            f.write(content)

        try:
            sample = content.decode('utf-8', errors='replace')[:4096]
            encoding = 'utf-8'
        except Exception:
            import chardet
            result = chardet.detect(content)
            encoding = result['encoding'] or 'utf-8'
            sample = content.decode(encoding, errors='replace')[:4096]

        # Read CSV or Excel
        if filename.endswith(('.csv', '.txt')):
            delimiter = ','
            for delim in [',', ';', '\t', '|']:
                if sample.count(delim) > 0:
                    delimiter = delim
                    break
            try:
                df = pd.read_csv(path_save, encoding=encoding, delimiter=delimiter, 
                                low_memory=False, on_bad_lines='skip')
            except Exception as e:
                raise ValueError(f"Error reading CSV: {str(e)}")

        elif filename.endswith(('.xlsx', '.xls')):
            engine = 'openpyxl' if filename.endswith('.xlsx') else 'xlrd'
            try:
                df = pd.read_excel(path_save, engine=engine)
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


def convert_dates(df):
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

def get_dummies(df):
    """
    One-hot encode all object-type columns in a DataFrame, leaving other columns unchanged.
    """
    # Separate the target variable 'Churn' before encoding
    if 'Churn' in df.columns:
        churn_series = df['Churn']
        df_features = df.drop(columns=['Churn'])
    else:
        churn_series = None
        df_features = df

    # Identify categorical columns for one-hot encoding
    categorical_cols = df_features.select_dtypes(include=['object']).columns
    
    # Apply one-hot encoding
    df_encoded = pd.get_dummies(df_features, columns=categorical_cols, drop_first=True)
    
    # Re-attach the 'Churn' column if it was present
    if churn_series is not None:
        df_encoded['Churn'] = churn_series
        
    return df_encoded




