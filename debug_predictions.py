#!/usr/bin/env python3
"""
Debug script to understand why predictions are the same for different inputs.
"""

from ml_model import CropPriceRandomForestPredictor
import pandas as pd
import numpy as np

def debug_predictions():
    """Debug why predictions are the same"""
    
    predictor = CropPriceRandomForestPredictor()
    predictor.load_model('models/best_rf_model.pkl')
    
    print("=== DEBUGGING PREDICTIONS ===")
    
    # Check available values in label encoders
    print(f"\nAvailable APMC values: {len(predictor.label_encoders['APMC'].classes_)}")
    apmc_classes = list(predictor.label_encoders['APMC'].classes_)
    print(f"Sample APMCs: {apmc_classes[:5]}")
    
    print(f"\nAvailable Commodity values: {len(predictor.label_encoders['Commodity'].classes_)}")
    commodity_classes = list(predictor.label_encoders['Commodity'].classes_)
    print(f"Sample Commodities: {commodity_classes[:5]}")
    
    # Test with known values
    test_cases = [
        ("Guntur", "Chilli Teja", "2025-09-15"),
        (apmc_classes[0], commodity_classes[0], "2025-09-15"),
        (apmc_classes[1] if len(apmc_classes) > 1 else apmc_classes[0], 
         commodity_classes[1] if len(commodity_classes) > 1 else commodity_classes[0], "2025-09-15"),
    ]
    
    print("\n=== TESTING DIFFERENT INPUTS ===")
    for i, (apmc, commodity, date) in enumerate(test_cases):
        print(f"\nTest {i+1}: {apmc} + {commodity}")
        
        # Create input dataframe manually to debug
        input_df = pd.DataFrame({
            'APMC': [apmc],
            'Commodity': [commodity], 
            'Date': [pd.to_datetime(date)],
            'Min Price (Rs)': [0],
            'Modal Price (Rs)': [0],
            'Max Price (Rs)': [0]
        })
        
        # Check if values are in encoders
        try:
            apmc_encoded = predictor.label_encoders['APMC'].transform([apmc])[0]
            print(f"  APMC '{apmc}' encoded as: {apmc_encoded}")
        except ValueError:
            print(f"  APMC '{apmc}' NOT FOUND in encoder - will use 0")
            apmc_encoded = 0
            
        try:
            commodity_encoded = predictor.label_encoders['Commodity'].transform([commodity])[0]
            print(f"  Commodity '{commodity}' encoded as: {commodity_encoded}")
        except ValueError:
            print(f"  Commodity '{commodity}' NOT FOUND in encoder - will use 0")
            commodity_encoded = 0
        
        # Get prediction
        pred = predictor.predict(apmc, commodity, date)
        print(f"  Prediction: {pred}")
    
    print("\n=== CHECKING FEATURE ENGINEERING ===")
    # Test if the issue is in feature engineering
    apmc1, commodity1 = "Guntur", "Chilli Teja"
    input_df1 = pd.DataFrame({
        'APMC': [apmc1],
        'Commodity': [commodity1],
        'Date': [pd.to_datetime("2025-09-15")],
        'Min Price (Rs)': [0],
        'Modal Price (Rs)': [0],
        'Max Price (Rs)': [0]
    })
    
    # Apply feature engineering
    input_df1_processed = predictor.create_advanced_features(input_df1)
    print(f"Features after engineering: {input_df1_processed.shape[1]} columns")
    
    # Check if APMC and Commodity features are different for different inputs
    apmc2, commodity2 = apmc_classes[0], commodity_classes[0]
    input_df2 = pd.DataFrame({
        'APMC': [apmc2],
        'Commodity': [commodity2],
        'Date': [pd.to_datetime("2025-09-15")],
        'Min Price (Rs)': [0],
        'Modal Price (Rs)': [0],
        'Max Price (Rs)': [0]
    })
    
    input_df2_processed = predictor.create_advanced_features(input_df2)
    
    # Compare key features
    print(f"\nComparing features between inputs:")
    print(f"Input 1: {apmc1} + {commodity1}")
    print(f"Input 2: {apmc2} + {commodity2}")
    
    # Check if any features are different
    for col in ['APMC', 'Commodity']:
        if col in input_df1_processed.columns and col in input_df2_processed.columns:
            val1 = input_df1_processed[col].iloc[0]
            val2 = input_df2_processed[col].iloc[0] 
            print(f"  {col}: {val1} vs {val2} (different: {val1 != val2})")

if __name__ == "__main__":
    debug_predictions()