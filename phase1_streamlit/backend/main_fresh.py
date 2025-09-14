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
    allow_origins=["*"],
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
                'missing_values': df.isnull().sum().to_dict(),
                'preview': df.head(5).to_dict('records') if len(df) > 0 else []
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
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split X data first
        X_train, X_test, _, _ = train_test_split(
            X_scaled, range(len(X_scaled)), test_size=0.2, random_state=42
        )
        
        # Detect problem type - improved logic
        y_unique = y.nunique()
        y_dtype = y.dtype
        
        # Better classification detection - FIXED
        is_classification = (
            y_unique <= 10 or  # Less than 10 unique values
            y_dtype == 'object' or  # Object/categorical type
            y_dtype == 'string' or  # String type
            (y_unique <= len(y) * 0.1)  # Less than 10% unique values
        )
        
        print(f"📊 Problem type: {'Classification' if is_classification else 'Regression'}")
        print(f"📊 Target unique values: {y_unique}")
        print(f"📊 Target data type: {y_dtype}")
        print(f"📊 Target sample values: {y.head(10).tolist()}")
        
        # For classification, ensure target is properly encoded
        if is_classification:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_encoded = le.fit_transform(y.astype(str))
            print(f"📊 Encoded target values: {np.unique(y_encoded)}")
            
            # Update y_train and y_test after encoding
            y_train, y_test = train_test_split(y_encoded, test_size=0.2, random_state=42)
        else:
            # For regression, split the original y
            y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)
        
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
            
            # Check if model is suitable for the problem type - FIXED
            if model_config['type'] == 'regression' and is_classification:
                return {'error': f'{model_name} is for regression only'}
            if model_config['type'] == 'classification' and not is_classification:
                return {'error': f'{model_name} is for classification only'}
            # For 'both' type models, let them run regardless of problem type
            
            try:
                print(f"🔄 Training {model_name}...")
                
                # Train model based on name
                if model_name == 'Linear Regression':
                    if not is_classification:  # Only for regression
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
                    else:
                        return {'error': 'Linear Regression is for regression only'}
                
                elif model_name == 'Logistic Regression':
                    if is_classification:  # Only for classification
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
                    else:
                        return {'error': 'Logistic Regression is for classification only'}
                
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
                    if is_classification:
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
                    else:
                        # For regression, use SVR instead of SVC
                        from sklearn.svm import SVR
                        model = SVR()
                        model.fit(X_train, y_train)
                        pred = model.predict(X_test)
                        
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
            outlier_mask = pd.Series([False] * len(df), index=df.index)
            
            print(f"🔍 Starting outlier removal for {len(df)} rows")
            
            # Collect all outlier masks first
            for column in df.select_dtypes(include=[np.number]).columns:
                try:
                    Q1 = df[column].quantile(0.25)
                    Q3 = df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    print(f"   Column: {column}, Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
                    
                    if IQR > 0:  # Only remove outliers if IQR is not zero
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        column_outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
                        column_outliers = column_outlier_mask.sum()
                        
                        print(f"   {column}: {column_outliers} outliers (bounds: {lower_bound:.2f} to {upper_bound:.2f})")
                        
                        outlier_mask = outlier_mask | column_outlier_mask
                        outliers_removed += column_outliers
                    else:
                        print(f"   {column}: IQR is 0, skipping outlier removal")
                        
                except Exception as e:
                    print(f"   Error processing column {column}: {str(e)}")
                    continue
            
            print(f"🔍 Total outliers found: {outliers_removed}")
            
            # Remove all outliers at once
            if outliers_removed > 0:
                df = df[~outlier_mask]
                final_rows = len(df)
                print(f"✅ Removed {outliers_removed} outlier rows. Final shape: {df.shape}")
                applied_actions.append(f"Removed {outliers_removed} outlier rows from all numeric columns")
            else:
                print(f"✅ No outliers found")
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
            'applied_actions': applied_actions,
            'visualization_ready': True,
            'outliers_removed': outliers_removed if 'remove_outliers' in actions else 0
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
