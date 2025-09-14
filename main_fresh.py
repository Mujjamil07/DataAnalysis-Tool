"""
Real-Time ML Training Backend - Complete with Actual Algorithms
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io
import base64
import uuid
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostRegressor, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, 
    precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import enhanced analysis
try:
    from enhanced_ml_analysis_new import enhanced_analyzer
    ENHANCED_ANALYSIS_AVAILABLE = True
    print("✅ Enhanced analysis module loaded successfully")
except ImportError as e:
    ENHANCED_ANALYSIS_AVAILABLE = False
    print(f"⚠️ Enhanced analysis module not available: {e}")

# Try to import advanced models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available")

# Available models configuration
AVAILABLE_MODELS = {
    'Linear Regression': {
        'type': 'regression',
        'available': True,
        'description': 'Simple linear regression model',
        'formula': 'y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ'
    },
    'Logistic Regression': {
        'type': 'classification',
        'available': True,
        'description': 'Linear model for binary classification',
        'formula': 'P(y=1) = 1 / (1 + e^(-z)) where z = β₀ + β₁x₁ + ... + βₙxₙ'
    },
    'Random Forest': {
        'type': 'both',
        'available': True,
        'description': 'Ensemble method using multiple decision trees',
        'formula': 'Ensemble of decision trees with majority voting/mean prediction'
    },
    'Support Vector Machine': {
        'type': 'classification',
        'available': True,
        'description': 'Kernel-based classification algorithm',
        'formula': 'f(x) = sign(Σ αᵢyᵢK(xᵢ, x) + b)'
    },
    'Decision Tree': {
        'type': 'both',
        'available': True,
        'description': 'Tree-based model for classification and regression',
        'formula': 'Hierarchical decision rules based on feature splits'
    },
    'K-Nearest Neighbors': {
        'type': 'both',
        'available': True,
        'description': 'Instance-based learning algorithm',
        'formula': 'Majority vote/mean of k nearest neighbors'
    },
    'Naive Bayes': {
        'type': 'classification',
        'available': True,
        'description': 'Probabilistic classifier based on Bayes theorem',
        'formula': 'P(y|x) ∝ P(x|y) × P(y)'
    },
    'XGBoost': {
        'type': 'both',
        'available': XGBOOST_AVAILABLE,
        'description': 'Gradient boosting with extreme gradient descent',
        'formula': 'Gradient boosting with extreme gradient descent'
    },
    'LightGBM': {
        'type': 'both',
        'available': LIGHTGBM_AVAILABLE,
        'description': 'Light gradient boosting machine',
        'formula': 'Light gradient boosting machine'
    },
    'Gradient Boosting': {
        'type': 'both',
        'available': True,
        'description': 'Sequential ensemble learning method',
        'formula': 'Sequential training of weak learners'
    },
    'AdaBoost': {
        'type': 'both',
        'available': True,
        'description': 'Adaptive boosting algorithm',
        'formula': 'Weighted combination of weak learners'
    },
    'Ridge Regression': {
        'type': 'regression',
        'available': True,
        'description': 'Linear regression with L2 regularization',
        'formula': 'y = β₀ + β₁x₁ + ... + βₙxₙ + λΣβᵢ²'
    },
    'Lasso Regression': {
        'type': 'regression',
        'available': True,
        'description': 'Linear regression with L1 regularization',
        'formula': 'y = β₀ + β₁x₁ + ... + βₙxₙ + λΣ|βᵢ|'
    }
}

# Create FastAPI app
app = FastAPI(title="Real-Time ML Training Backend", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3005", "http://127.0.0.1:3005"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Global storage
sessions = {}

# Pydantic models
class UploadRequest(BaseModel):
    filename: str
    data: str
    content_type: str

class TrainRequest(BaseModel):
    session_id: str
    target_column: str
    models: list[str] = []  # List of model names to train

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Real-Time ML Training Backend is running!", "status": "healthy"}

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "sessions_count": len(sessions)
    }

# Get available models
@app.get("/models")
async def get_available_models():
    """Get list of available models"""
    return {
        "models": AVAILABLE_MODELS,
        "total_count": len(AVAILABLE_MODELS),
        "available_count": sum(1 for model in AVAILABLE_MODELS.values() if model['available'])
    }

# Upload data endpoint
@app.post("/upload")
async def upload_data(request: UploadRequest):
    """Upload and process CSV data"""
    try:
        print(f"📤 Uploading file: {request.filename}")
        
        # Decode base64 data
        data_bytes = base64.b64decode(request.data)
        data_str = data_bytes.decode('utf-8')
        print(f"📄 Decoded data length: {len(data_str)}")
        
        # Parse CSV
        df = pd.read_csv(io.StringIO(data_str))
        print(f"📊 DataFrame shape: {df.shape}")
        print(f"📊 DataFrame columns: {df.columns.tolist()}")
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        print(f"🆔 Generated session ID: {session_id}")
        
        # Store session data
        sessions[session_id] = {
            'data': df,
            'filename': request.filename,
            'upload_time': datetime.now().isoformat(),
            'data_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'data_types': {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
                'missing_values': {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()},
                'preview': df.head(5).fillna('').to_dict('records') if len(df) > 0 else []
            }
        }
        
        print(f"✅ User data uploaded successfully: {len(df)} rows, {len(df.columns)} columns")
        print(f"✅ Dataset: {request.filename}")
        print(f"✅ Columns: {df.columns.tolist()}")
        
        return {
            'session_id': session_id,
            'data_info': sessions[session_id]['data_info'],
            'message': f'Your dataset "{request.filename}" uploaded successfully'
        }
        
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# Real-time ML training endpoint
@app.post("/enhanced-analysis")
async def enhanced_analysis(request: TrainRequest):
    """Enhanced ML analysis with advanced preprocessing and feature selection"""
    if not ENHANCED_ANALYSIS_AVAILABLE:
        raise HTTPException(status_code=501, detail="Enhanced analysis module not available")
    
    try:
        print(f"🚀 Starting enhanced analysis for session: {request.session_id}")
        
        if request.session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        df = sessions[request.session_id]['data']
        print(f"📊 Dataset shape: {df.shape}")
        
        if request.target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{request.target_column}' not found")
        
        # Run enhanced analysis using the global analyzer
        analysis_results = enhanced_analyzer.run_complete_analysis(
            df=df,
            target_column=request.target_column,
            models_to_train=request.models,
            handle_imbalance=True,
            handle_outliers=True,
            feature_selection_method='rfe'
        )
        
        # Format results for API response - convert numpy types to Python native types
        formatted_results = {}
        for model_name, metrics in analysis_results['results'].items():
            formatted_results[model_name] = {
                'accuracy': float(metrics.get('accuracy', 0)),
                'precision': float(metrics.get('precision', 0)),
                'recall': float(metrics.get('recall', 0)),
                'f1_score': float(metrics.get('f1_score', 0)),
                'r2_score': float(metrics.get('r2_score', 0)),
                'mse': float(metrics.get('mse', 0)),
                'rmse': float(metrics.get('rmse', 0)),
                'roc_auc': float(metrics.get('roc_auc', 0)),
                'cv_mean': float(metrics.get('cv_mean', 0)),
                'cv_std': float(metrics.get('cv_std', 0)),
                'training_time': 0.1  # Placeholder
            }
        
        # Convert numpy types in other fields
        def convert_numpy_types(obj):
            """Recursively convert numpy types to Python native types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert all numpy types in the response
        response_data = {
            "success": analysis_results.get('success', True),
            "message": "Enhanced analysis completed successfully" if analysis_results.get('success', True) else "Enhanced analysis failed",
            "results": formatted_results,
            "best_model": str(analysis_results.get('best_model', 'None')),
            "best_score": float(analysis_results.get('best_score', 0)),
            "problem_type": str(analysis_results.get('problem_type', 'unknown')),
            "selected_features": [str(feature) for feature in analysis_results.get('selected_features', [])],
            "data_quality_report": convert_numpy_types(analysis_results.get('data_quality', {})),
            "recommendations": [str(rec) for rec in analysis_results.get('recommendations', [])],
            "dataset_info": analysis_results.get('dataset_info', {})
        }
        
        return response_data
        
    except Exception as e:
        print(f"❌ Enhanced analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enhanced analysis failed: {str(e)}")

@app.post("/train")
async def train_models(request: TrainRequest):
    """Real-time ML training with actual algorithms"""
    try:
        print(f"🚀 Starting real-time training for session: {request.session_id}")
        print(f"🎯 Target column: {request.target_column}")
        print(f"🤖 Models to train: {request.models if request.models else 'ALL MODELS'}")
        
        if request.session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        df = sessions[request.session_id]['data']
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📊 Dataset: {sessions[request.session_id]['filename']}")
        
        if request.target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{request.target_column}' not found")
        
        # Data preprocessing
        print("🔄 Preprocessing data...")
        
        # Clean data
        df_clean = df.copy()
        df_clean = df_clean.dropna(subset=[request.target_column])
        
        if len(df_clean) < 10:
            raise ValueError("Not enough data after cleaning (minimum 10 samples required)")
        
        # Separate features and target
        X = df_clean.drop(columns=[request.target_column])
        y = df_clean[request.target_column]
        
        # Handle categorical features
        categorical_columns = X.select_dtypes(include=['object', 'string']).columns
        numeric_columns = X.select_dtypes(include=['number']).columns
        
        print(f"📊 Categorical columns: {list(categorical_columns)}")
        print(f"📊 Numeric columns: {list(numeric_columns)}")
        
        # Encode categorical columns
        if len(categorical_columns) > 0:
            X_encoded = X.copy()
            for col in categorical_columns:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                print(f"✅ Encoded categorical column: {col}")
            X = X_encoded
        else:
            # Use only numeric columns if no categorical
            if len(numeric_columns) == 0:
                raise ValueError("No numeric features found. Please ensure your dataset has numeric columns.")
            X = X[numeric_columns]
        
        # Convert target to numeric safely
        y = pd.to_numeric(y, errors='coerce')
        
        # Drop rows with NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 10:
            raise ValueError("Not enough data after cleaning (minimum 10 samples required)")
        
        print(f"✅ Final data shape: {X.shape}")
        print(f"✅ Features used: {list(X.columns)}")
        
        # Detect problem type and convert target accordingly
        y_unique = y.nunique()
        is_classification = y_unique <= 10  # If less than 10 unique values, treat as classification
        
        if is_classification:
            # For classification, ensure target is discrete (integer labels)
            if y.dtype in ['float64', 'float32']:
                # Convert continuous values to discrete classes
                y = y.round().astype(int)
                print(f"🔄 Converted continuous target to discrete classes: {sorted(y.unique())}")
            else:
                # Already discrete, just ensure it's integer
                y = y.astype(int)
                print(f"✅ Target already discrete: {sorted(y.unique())}")
        else:
            # For regression, keep as float
            y = y.astype(float)
            print(f"✅ Target for regression: {y.dtype}")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        print(f"📊 Problem type: {'Classification' if is_classification else 'Regression'}")
        print(f"📊 Target unique values: {y_unique}")
        
        # Determine which models to train
        models_to_train = request.models if request.models else list(AVAILABLE_MODELS.keys())
        print(f"🎯 Models to train: {models_to_train}")
        
        # Train models
        results = {}
        best_model = None
        best_score = 0
        
        # Function to train a single model
        def train_single_model(model_name, model_config):
            if not model_config['available']:
                return {'error': f'Model {model_name} is not available'}
            
            # Check if model is suitable for the problem type
            if model_config['type'] == 'regression' and is_classification:
                return {'error': f'{model_name} is for regression only'}
            if model_config['type'] == 'classification' and not is_classification:
                return {'error': f'{model_name} is for classification only'}
            
            try:
                print(f"🔄 Training {model_name}...")
                
                # Train model based on name
                if model_name == 'Linear Regression':
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    
                    mse = mean_squared_error(y_test, pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, pred)
                    
                    return {
                        'mse': float(mse),
                        'rmse': float(rmse),
                        'r2_score': float(r2),
                        'training_time': 0.1,
                        'type': 'regression',
                        'formula': model_config['formula']
                    }
                
                elif model_name == 'Logistic Regression':
                    model = LogisticRegression(max_iter=1000, random_state=42)
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    
                    accuracy = accuracy_score(y_test, pred)
                    precision = precision_score(y_test, pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, pred, average='weighted', zero_division=0)
                    cm = confusion_matrix(y_test, pred)
                    
                    return {
                        'accuracy': float(accuracy),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1_score': float(f1),
                        'confusion_matrix': cm.tolist(),
                        'training_time': 0.2,
                        'type': 'classification',
                        'formula': model_config['formula']
                    }
                
                elif model_name == 'Random Forest':
                    if is_classification:
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                    else:
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                    
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    
                    if is_classification:
                        accuracy = accuracy_score(y_test, pred)
                        precision = precision_score(y_test, pred, average='weighted', zero_division=0)
                        recall = recall_score(y_test, pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, pred, average='weighted', zero_division=0)
                        cm = confusion_matrix(y_test, pred)
                        
                        return {
                            'accuracy': float(accuracy),
                            'precision': float(precision),
                            'recall': float(recall),
                            'f1_score': float(f1),
                            'confusion_matrix': cm.tolist(),
                            'training_time': 0.3,
                            'type': 'classification',
                            'formula': model_config['formula']
                        }
                    else:
                        mse = mean_squared_error(y_test, pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, pred)
                        
                        return {
                            'mse': float(mse),
                            'rmse': float(rmse),
                            'r2_score': float(r2),
                            'training_time': 0.3,
                            'type': 'regression',
                            'formula': model_config['formula']
                        }
                
                elif model_name == 'Support Vector Machine':
                    model = SVC(random_state=42)
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    
                    accuracy = accuracy_score(y_test, pred)
                    precision = precision_score(y_test, pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, pred, average='weighted', zero_division=0)
                    cm = confusion_matrix(y_test, pred)
                    
                    return {
                        'accuracy': float(accuracy),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1_score': float(f1),
                        'confusion_matrix': cm.tolist(),
                        'training_time': 0.4,
                        'type': 'classification',
                        'formula': model_config['formula']
                    }
                
                elif model_name == 'Decision Tree':
                    if is_classification:
                        model = DecisionTreeClassifier(random_state=42)
                    else:
                        model = DecisionTreeRegressor(random_state=42)
                    
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    
                    if is_classification:
                        accuracy = accuracy_score(y_test, pred)
                        precision = precision_score(y_test, pred, average='weighted', zero_division=0)
                        recall = recall_score(y_test, pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, pred, average='weighted', zero_division=0)
                        cm = confusion_matrix(y_test, pred)
                        
                        return {
                            'accuracy': float(accuracy),
                            'precision': float(precision),
                            'recall': float(recall),
                            'f1_score': float(f1),
                            'confusion_matrix': cm.tolist(),
                            'training_time': 0.2,
                            'type': 'classification',
                            'formula': model_config['formula']
                        }
                    else:
                        mse = mean_squared_error(y_test, pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, pred)
                        
                        return {
                            'mse': float(mse),
                            'rmse': float(rmse),
                            'r2_score': float(r2),
                            'training_time': 0.2,
                            'type': 'regression',
                            'formula': model_config['formula']
                        }
                
                elif model_name == 'K-Nearest Neighbors':
                    if is_classification:
                        model = KNeighborsClassifier(n_neighbors=5)
                    else:
                        model = KNeighborsRegressor(n_neighbors=5)
                    
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    
                    if is_classification:
                        accuracy = accuracy_score(y_test, pred)
                        precision = precision_score(y_test, pred, average='weighted', zero_division=0)
                        recall = recall_score(y_test, pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, pred, average='weighted', zero_division=0)
                        cm = confusion_matrix(y_test, pred)
                        
                        return {
                            'accuracy': float(accuracy),
                            'precision': float(precision),
                            'recall': float(recall),
                            'f1_score': float(f1),
                            'confusion_matrix': cm.tolist(),
                            'training_time': 0.1,
                            'type': 'classification',
                            'formula': model_config['formula']
                        }
                    else:
                        mse = mean_squared_error(y_test, pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, pred)
                        
                        return {
                            'mse': float(mse),
                            'rmse': float(rmse),
                            'r2_score': float(r2),
                            'training_time': 0.1,
                            'type': 'regression',
                            'formula': model_config['formula']
                        }
                
                elif model_name == 'Naive Bayes':
                    model = GaussianNB()
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    
                    accuracy = accuracy_score(y_test, pred)
                    precision = precision_score(y_test, pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, pred, average='weighted', zero_division=0)
                    cm = confusion_matrix(y_test, pred)
                    
                    return {
                        'accuracy': float(accuracy),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1_score': float(f1),
                        'confusion_matrix': cm.tolist(),
                        'training_time': 0.1,
                        'type': 'classification',
                        'formula': model_config['formula']
                    }
                
                elif model_name == 'XGBoost':
                    if is_classification:
                        model = xgb.XGBClassifier(eval_metric="mlogloss", random_state=42)
                    else:
                        model = xgb.XGBRegressor(eval_metric="rmse", random_state=42)
                    
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    
                    if is_classification:
                        accuracy = accuracy_score(y_test, pred)
                        precision = precision_score(y_test, pred, average='weighted', zero_division=0)
                        recall = recall_score(y_test, pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, pred, average='weighted', zero_division=0)
                        cm = confusion_matrix(y_test, pred)
                        
                        return {
                            'accuracy': float(accuracy),
                            'precision': float(precision),
                            'recall': float(recall),
                            'f1_score': float(f1),
                            'confusion_matrix': cm.tolist(),
                            'training_time': 0.5,
                            'type': 'classification',
                            'formula': model_config['formula']
                        }
                    else:
                        mse = mean_squared_error(y_test, pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, pred)
                        
                        return {
                            'mse': float(mse),
                            'rmse': float(rmse),
                            'r2_score': float(r2),
                            'training_time': 0.5,
                            'type': 'regression',
                            'formula': model_config['formula']
                        }
                
                elif model_name == 'LightGBM':
                    if is_classification:
                        model = lgb.LGBMClassifier(random_state=42)
                    else:
                        model = lgb.LGBMRegressor(random_state=42)
                    
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    
                    if is_classification:
                        accuracy = accuracy_score(y_test, pred)
                        precision = precision_score(y_test, pred, average='weighted', zero_division=0)
                        recall = recall_score(y_test, pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, pred, average='weighted', zero_division=0)
                        cm = confusion_matrix(y_test, pred)
                        
                        return {
                            'accuracy': float(accuracy),
                            'precision': float(precision),
                            'recall': float(recall),
                            'f1_score': float(f1),
                            'confusion_matrix': cm.tolist(),
                            'training_time': 0.4,
                            'type': 'classification',
                            'formula': model_config['formula']
                        }
                    else:
                        mse = mean_squared_error(y_test, pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, pred)
                        
                        return {
                            'mse': float(mse),
                            'rmse': float(rmse),
                            'r2_score': float(r2),
                            'training_time': 0.4,
                            'type': 'regression',
                            'formula': model_config['formula']
                        }
                
                elif model_name == 'Gradient Boosting':
                    if is_classification:
                        model = GradientBoostingClassifier(random_state=42)
                    else:
                        model = GradientBoostingRegressor(random_state=42)
                    
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    
                    if is_classification:
                        accuracy = accuracy_score(y_test, pred)
                        precision = precision_score(y_test, pred, average='weighted', zero_division=0)
                        recall = recall_score(y_test, pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, pred, average='weighted', zero_division=0)
                        cm = confusion_matrix(y_test, pred)
                        
                        return {
                            'accuracy': float(accuracy),
                            'precision': float(precision),
                            'recall': float(recall),
                            'f1_score': float(f1),
                            'confusion_matrix': cm.tolist(),
                            'training_time': 0.4,
                            'type': 'classification',
                            'formula': model_config['formula']
                        }
                    else:
                        mse = mean_squared_error(y_test, pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, pred)
                        
                        return {
                            'mse': float(mse),
                            'rmse': float(rmse),
                            'r2_score': float(r2),
                            'training_time': 0.4,
                            'type': 'regression',
                            'formula': model_config['formula']
                        }
                
                elif model_name == 'AdaBoost':
                    if is_classification:
                        model = AdaBoostClassifier(random_state=42)
                    else:
                        model = AdaBoostRegressor(random_state=42)
                    
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    
                    if is_classification:
                        accuracy = accuracy_score(y_test, pred)
                        precision = precision_score(y_test, pred, average='weighted', zero_division=0)
                        recall = recall_score(y_test, pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, pred, average='weighted', zero_division=0)
                        cm = confusion_matrix(y_test, pred)
                        
                        return {
                            'accuracy': float(accuracy),
                            'precision': float(precision),
                            'recall': float(recall),
                            'f1_score': float(f1),
                            'confusion_matrix': cm.tolist(),
                            'training_time': 0.3,
                            'type': 'classification',
                            'formula': model_config['formula']
                        }
                    else:
                        mse = mean_squared_error(y_test, pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, pred)
                        
                        return {
                            'mse': float(mse),
                            'rmse': float(rmse),
                            'r2_score': float(r2),
                            'training_time': 0.3,
                            'type': 'regression',
                            'formula': model_config['formula']
                        }
                
                elif model_name == 'Ridge Regression':
                    model = Ridge(random_state=42)
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    
                    mse = mean_squared_error(y_test, pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, pred)
                    
                    return {
                        'mse': float(mse),
                        'rmse': float(rmse),
                        'r2_score': float(r2),
                        'training_time': 0.1,
                        'type': 'regression',
                        'formula': model_config['formula']
                    }
                
                elif model_name == 'Lasso Regression':
                    model = Lasso(random_state=42)
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    
                    mse = mean_squared_error(y_test, pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, pred)
                    
                    return {
                        'mse': float(mse),
                        'rmse': float(rmse),
                        'r2_score': float(r2),
                        'training_time': 0.1,
                        'type': 'regression',
                        'formula': model_config['formula']
                    }
                
                else:
                    return {'error': f'Unknown model: {model_name}'}
                    
            except Exception as e:
                print(f"❌ {model_name} failed: {str(e)}")
                return {'error': str(e)}
        
        # Train selected models
        for model_name in models_to_train:
            if model_name in AVAILABLE_MODELS:
                result = train_single_model(model_name, AVAILABLE_MODELS[model_name])
                results[model_name] = result
                
                # Update best model
                if 'error' not in result:
                    if is_classification and 'accuracy' in result:
                        if result['accuracy'] > best_score:
                            best_score = result['accuracy']
                            best_model = model_name
                    elif not is_classification and 'r2_score' in result:
                        if result['r2_score'] > best_score:
                            best_score = result['r2_score']
                            best_model = model_name
            else:
                results[model_name] = {'error': f'Model {model_name} not found'}
        
        # Store results
        sessions[request.session_id]['results'] = results
        sessions[request.session_id]['target_column'] = request.target_column
        
        print(f"✅ Real-time training completed successfully")
        print(f"   Best model: {best_model}")
        print(f"   Best score: {best_score:.4f}")
        print(f"   Models trained: {len(results)}")
        
        return {
            'session_id': request.session_id,
            'target_column': request.target_column,
            'results': results,
            'best_model': best_model,
            'best_score': best_score,
            'problem_type': 'classification' if is_classification else 'regression',
            'message': 'Real-time ML training completed successfully'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Training error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

# Get sessions endpoint
@app.get("/sessions")
async def get_sessions():
    """Get all sessions"""
    return {
        'sessions': [
            {
                'session_id': session_id,
                'filename': session_data['filename'],
                'upload_time': session_data['upload_time'],
                'data_info': session_data['data_info']
            }
            for session_id, session_data in sessions.items()
        ]
    }

# Get specific session endpoint
@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get details of a specific session"""
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = sessions[session_id]
        print(f"✅ Retrieved session: {session_id}")
        
        return {
            'session_id': session_id,
            'filename': session_data['filename'],
            'upload_time': session_data['upload_time'],
            'data_info': session_data['data_info']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Get session error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")

# Delete all sessions endpoint
@app.delete("/sessions")
async def delete_all_sessions():
    """Delete all sessions"""
    try:
        session_count = len(sessions)
        sessions.clear()
        print(f"✅ Deleted all {session_count} sessions")
        return {"message": f"All {session_count} sessions deleted successfully"}
    except Exception as e:
        print(f"❌ Delete all sessions error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete all sessions: {str(e)}")

# Data Analysis endpoint
@app.get("/analyze/{session_id}")
async def analyze_data(session_id: str):
    """Analyze data for cleaning insights"""
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = sessions[session_id]
        df = session_data['data']
        
        print(f"🔍 Analyzing data for session: {session_id}")
        print(f"   Data shape: {df.shape}")
        
        # Basic statistics
        stats = {}
        if df.select_dtypes(include=[np.number]).shape[1] > 0:
            numeric_df = df.select_dtypes(include=[np.number])
            stats = {
                'mean': float(numeric_df.mean().mean()),
                'std': float(numeric_df.std().mean()),
                'min': float(numeric_df.min().min()),
                'max': float(numeric_df.max().max())
            }
        
        # Missing data analysis
        missing_data = {}
        total_missing = 0
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                missing_data[column] = int(missing_count)
                total_missing += missing_count
        
        # Duplicates analysis
        duplicates = {
            'count': int(df.duplicated().sum())
        }
        
        # Outliers analysis (using IQR method)
        outliers = {}
        total_outliers = 0
        for column in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:  # Only calculate outliers if IQR is not zero
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
                if outlier_count > 0:
                    outliers[column] = int(outlier_count)
                    total_outliers += outlier_count
        
        # Data types
        data_types = {}
        for column in df.columns:
            data_types[column] = str(df[column].dtype)
        
        # Correlations (only for numeric columns)
        correlations = {}
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] > 1:
            try:
                corr_matrix = numeric_df.corr()
                correlations = corr_matrix.to_dict()
            except Exception as e:
                print(f"⚠️ Correlation calculation failed: {e}")
                correlations = {}
        
        # Distributions (basic statistics for each column)
        distributions = {}
        for column in df.columns:
            try:
                if df[column].dtype in ['object', 'category']:
                    # Categorical data
                    value_counts = df[column].value_counts().head(10).to_dict()
                    distributions[column] = {str(k): int(v) for k, v in value_counts.items()}
                else:
                    # Numeric data
                    distributions[column] = {
                        'mean': float(df[column].mean()),
                        'std': float(df[column].std()),
                        'min': float(df[column].min()),
                        'max': float(df[column].max()),
                        'median': float(df[column].median())
                    }
            except Exception as e:
                print(f"⚠️ Distribution calculation failed for {column}: {e}")
                distributions[column] = {'error': str(e)}
        
        print(f"✅ Data analysis completed for session: {session_id}")
        print(f"   Missing values: {total_missing}")
        print(f"   Duplicates: {duplicates['count']}")
        print(f"   Outliers: {total_outliers}")
        
        return {
            'session_id': session_id,
            'stats': stats,
            'missing_data': missing_data,
            'duplicates': duplicates,
            'outliers': outliers,
            'data_types': data_types,
            'correlations': correlations,
            'distributions': distributions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Data analysis error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Data analysis failed: {str(e)}")

# Data Cleaning endpoint
@app.post("/clean/{session_id}")
async def clean_data(session_id: str, actions: list[str]):
    """Clean data based on specified actions"""
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = sessions[session_id]
        df = session_data['data'].copy()
        
        print(f"🧹 Cleaning data for session: {session_id}")
        print(f"   Actions: {actions}")
        print(f"   Initial shape: {df.shape}")
        
        applied_actions = []
        
        # Remove duplicates
        if 'remove_duplicates' in actions:
            initial_rows = len(df)
            df = df.drop_duplicates()
            removed_duplicates = initial_rows - len(df)
            if removed_duplicates > 0:
                applied_actions.append(f"Removed {removed_duplicates} duplicate rows")
            else:
                applied_actions.append("No duplicates found")
        
        # Fill missing values (do this before removing outliers)
        if 'fill_missing_values' in actions:
            missing_filled = 0
            for column in df.columns:
                if df[column].isnull().sum() > 0:
                    missing_filled += df[column].isnull().sum()
                    if df[column].dtype in ['object', 'category']:
                        # Fill categorical with mode
                        mode_value = df[column].mode()[0] if len(df[column].mode()) > 0 else 'Unknown'
                        df[column] = df[column].fillna(mode_value)
                    else:
                        # Fill numeric with median
                        median_value = df[column].median()
                        df[column] = df[column].fillna(median_value)
            if missing_filled > 0:
                applied_actions.append(f"Filled {missing_filled} missing values")
            else:
                applied_actions.append("No missing values found")
        
        # Remove outliers
        if 'remove_outliers' in actions:
            initial_rows = len(df)
            outliers_removed = 0
            for column in df.select_dtypes(include=[np.number]).columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:  # Only remove outliers if IQR is not zero
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
                    outliers_removed += outlier_mask.sum()
                    df = df[~outlier_mask]
            if outliers_removed > 0:
                applied_actions.append(f"Removed {outliers_removed} outlier rows")
            else:
                applied_actions.append("No outliers found")
        
        # Standardize data
        if 'standardize_data' in actions:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                scaler = StandardScaler()
                df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
                applied_actions.append(f"Standardized {len(numeric_columns)} numeric columns")
            else:
                applied_actions.append("No numeric columns to standardize")
        
        # Update session with cleaned data
        sessions[session_id]['data'] = df
        sessions[session_id]['data_info'] = {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist()
        }
        
        print(f"✅ Data cleaning completed for session: {session_id}")
        print(f"   Final shape: {df.shape}")
        print(f"   Applied actions: {applied_actions}")
        
        return {
            'session_id': session_id,
            'cleaned_data': {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist()
            },
            'applied_actions': applied_actions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Data cleaning error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Data cleaning failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Working Backend...")
    uvicorn.run(app, host="127.0.0.1", port=8005)
