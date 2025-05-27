from collections import Counter
import pandas as pd
def get_dummies(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            if df[col].isin(['yes', 'no', 'True', 'False']).any():
                df[col] = df[col].map({'yes': 1, 'True': 1, 'no': 0, 'False': 0})
            else:
                df = pd.get_dummies(df, columns=[col])
    return df
def most_common(lst):
    counts = Counter(lst)
    if not counts:
        return None 
    return counts.most_common(1)[0][0]