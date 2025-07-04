import pandas as pd
import json
import csv
from io import BytesIO
from typing import Optional
from fastapi import UploadFile, HTTPException
from .pretrained_model import churn_prediction

def df_to_records(df):
    return json.loads(df.to_json(orient='records', date_format='iso'))

async def import_data(uploaded_file):
    df = None
    if uploaded_file is not None:
        filename = uploaded_file.filename.lower()
        content = await uploaded_file.read()
        try:
            sample = content.decode('utf-8', errors='replace')[:4096]
            encoding = 'utf-8'
        except Exception:
            import chardet
            result = chardet.detect(content)
            encoding = result['encoding'] or 'utf-8'
            sample = content.decode(encoding, errors='replace')[:4096]
        if filename.endswith(('.csv', '.txt')):
            delimiter = ','
            for delim in [',', ';', '\t', '|']:
                if sample.count(delim) > 0:
                    delimiter = delim
                    break
            try:
                df = pd.read_csv(BytesIO(content), encoding=encoding, delimiter=delimiter, low_memory=False, on_bad_lines='skip')
            except Exception as e:
                raise ValueError(f"Error reading CSV: {str(e)}")
        elif filename.endswith(('.xlsx', '.xls')):
            if filename.endswith('.xlsx'):
                try:
                    import openpyxl
                    engine = 'openpyxl'
                except ImportError:
                    raise ValueError("openpyxl is required to read Excel files. Please install it.")
            else:
                try:
                    import xlrd
                    engine = 'xlrd'
                except ImportError:
                    raise ValueError("xlrd is required to read Excel files. Please install it.")
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

def convert_dates(df):
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

def get_dummies(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            if bool(df[col].isin(['yes', 'no', 'True', 'False']).any()):
                df[col] = df[col].replace({'yes': 1, 'True': 1, 'no': 0, 'False': 0})
            else:
                df = pd.get_dummies(df, columns=[col])
    return df


class ChurnController:
    @staticmethod
    async def predict_churn(file: Optional[UploadFile] = None):
        if not file:
            raise HTTPException(status_code=400, detail="Either file or data must be provided")
        try:
            df = await import_data(file)
            df_churn = churn_prediction(df)
            return {
                "results": df_to_records(df_churn)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

