"""
Test script for enhanced analysis module
"""

import sys
import traceback

try:
    print("Testing enhanced analysis module...")
    
    # Test imports
    print("1. Testing imports...")
    from enhanced_ml_analysis_new import enhanced_analyzer
    print("✅ Enhanced analyzer imported successfully")
    
    # Test with sample data
    print("2. Testing with sample data...")
    import pandas as pd
    import numpy as np
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    data = {
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'target': np.random.randint(0, 2, n_samples)
    }
    
    df = pd.DataFrame(data)
    print(f"✅ Sample data created: {df.shape}")
    
    # Test analysis
    print("3. Running enhanced analysis...")
    result = enhanced_analyzer.run_complete_analysis(
        df=df,
        target_column='target',
        models_to_train=['Random Forest', 'Logistic Regression'],
        handle_imbalance=True,
        handle_outliers=True,
        feature_selection_method='rfe'
    )
    
    print("✅ Analysis completed successfully!")
    print(f"Success: {result.get('success', False)}")
    print(f"Problem type: {result.get('problem_type', 'unknown')}")
    print(f"Best model: {result.get('best_model', 'None')}")
    print(f"Best score: {result.get('best_score', 0)}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Full traceback:")
    traceback.print_exc()
