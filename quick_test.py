#!/usr/bin/env python3
"""
Quick test to check data loading and structure
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from src.Churn.config.configuration import ConfigurationManager
from src.Churn.components.data_ingestion import DataIngestion

def quick_test():
    """Quick test of data loading"""
    try:
        print("Loading configuration...")
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        
        print("Creating data ingestion instance...")
        data_ingestion = DataIngestion(config=data_ingestion_config)
        
        print("Loading raw data...")
        df = data_ingestion.load_data()
        print(f"Raw data shape: {df.shape}")
        print(f"Raw columns: {list(df.columns)}")
        print(f"Churn column values: {df['Churn'].value_counts() if 'Churn' in df.columns else 'No Churn column'}")
        
        print("\nProcessing features...")
        df_processed = data_ingestion.process_data_for_churn(df)
        print(f"Processed data shape: {df_processed.shape}")
        print(f"Processed columns: {list(df_processed.columns)}")
        print(f"Churn in processed data: {'Churn' in df_processed.columns}")
        
        if 'Churn' in df_processed.columns:
            print(f"Churn values after processing: {df_processed['Churn'].value_counts()}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    print("✅ Test completed successfully!" if success else "❌ Test failed!") 