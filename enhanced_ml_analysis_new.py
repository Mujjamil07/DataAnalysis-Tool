"""
Enhanced ML Analysis Module - Clean Implementation
Comprehensive machine learning analysis with advanced preprocessing and model comparison
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, roc_auc_score, roc_curve,
    precision_recall_curve, confusion_matrix, classification_report
)
from sklearn.utils.multiclass import type_of_target
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.over_sampling import RandomOverSampler, SMOTE
from scipy import stats
import time

# Suppress warnings
warnings.filterwarnings('ignore')

# Try to import advanced models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class EnhancedMLAnalyzer:
    """
    Enhanced ML Analysis class with comprehensive preprocessing and model comparison
    """
    
    def __init__(self):
        self.results = {}
        self.best_model = None
        self.best_score = 0
        self.problem_type = None
        self.feature_importance = None
        self.correlation_matrix = None
        self.data_quality_report = {}
        self.recommendations = []
        
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality analysis"""
        print("🔍 Analyzing data quality...")
        
        quality_report = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicates': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'outliers': {},
            'correlations': {},
            'distributions': {}
        }
        
        # Analyze outliers for numeric columns
        numeric_cols = quality_report['numeric_columns']
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                quality_report['outliers'][col] = int(outliers)
        
        # Calculate correlations for numeric columns
        if len(numeric_cols) > 1:
            try:
                corr_matrix = df[numeric_cols].corr()
                quality_report['correlations'] = corr_matrix.to_dict()
            except Exception as e:
                print(f"⚠️ Correlation calculation failed: {e}")
                quality_report['correlations'] = {}
        
        # Analyze distributions
        for col in df.columns:
            try:
                if col in numeric_cols:
                    quality_report['distributions'][col] = {
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'median': float(df[col].median()),
                        'skewness': float(df[col].skew()),
                        'kurtosis': float(df[col].kurtosis())
                    }
                else:
                    value_counts = df[col].value_counts().head(10)
                    quality_report['distributions'][col] = {
                        'unique_values': int(df[col].nunique()),
                        'top_values': value_counts.to_dict()
                    }
            except Exception as e:
                print(f"⚠️ Distribution analysis failed for {col}: {e}")
                quality_report['distributions'][col] = {'error': str(e)}
        
        self.data_quality_report = quality_report
        return quality_report
    
    def detect_problem_type(self, target_column: pd.Series) -> str:
        """Detect problem type using sklearn's robust detection"""
        try:
            target_type = type_of_target(target_column)
            
            if target_type in ['binary', 'multiclass']:
                return 'classification'
            elif target_type in ['continuous', 'continuous-multioutput']:
                return 'regression'
            else:
                # Fallback detection
                unique_values = target_column.nunique()
                if unique_values <= 2:
                    return 'classification'
                elif unique_values <= 10 and target_column.dtype in ['object', 'category']:
                    return 'classification'
                else:
                    return 'regression'
        except Exception as e:
            print(f"⚠️ Problem type detection failed: {e}")
            # Safe fallback
            unique_values = target_column.nunique()
            return 'classification' if unique_values <= 10 else 'regression'
    
    def preprocess_data(self, df: pd.DataFrame, target_column: str, 
                       handle_missing: bool = True, handle_outliers: bool = True,
                       handle_imbalance: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Comprehensive data preprocessing"""
        print("🔧 Preprocessing data...")
        
        df_processed = df.copy()
        target = df_processed[target_column]
        
        # Handle missing values
        if handle_missing:
            print("  📝 Handling missing values...")
            for col in df_processed.columns:
                if df_processed[col].isnull().sum() > 0:
                    if df_processed[col].dtype in ['object', 'category']:
                        # Fill categorical with mode
                        mode_value = df_processed[col].mode()
                        if len(mode_value) > 0:
                            df_processed[col] = df_processed[col].fillna(mode_value[0])
                        else:
                            df_processed[col] = df_processed[col].fillna('Unknown')
                    else:
                        # Fill numeric with median
                        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        # Handle outliers
        if handle_outliers:
            print("  📊 Handling outliers...")
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != target_column:
                    Q1 = df_processed[col].quantile(0.25)
                    Q3 = df_processed[col].quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
        
        # Encode categorical variables
        print("  🏷️ Encoding categorical variables...")
        label_encoders = {}
        for col in df_processed.columns:
            if col != target_column and df_processed[col].dtype in ['object', 'category']:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                label_encoders[col] = le
        
        # Handle target variable
        if target.dtype in ['object', 'category']:
            le_target = LabelEncoder()
            target = le_target.fit_transform(target.astype(str))
            print(f"  🎯 Target encoded: {len(le_target.classes_)} classes")
        
        # Handle class imbalance for classification
        if handle_imbalance and self.problem_type == 'classification':
            print("  ⚖️ Handling class imbalance...")
            try:
                # Check if imbalance exists
                class_counts = pd.Series(target).value_counts()
                imbalance_ratio = class_counts.max() / class_counts.min()
                
                if imbalance_ratio > 2:  # Significant imbalance
                    print(f"    Imbalance ratio: {imbalance_ratio:.2f}")
                    
                    # Use SMOTE for oversampling
                    if len(class_counts) == 2:  # Binary classification
                        smote = SMOTE(random_state=42)
                        X_resampled, y_resampled = smote.fit_resample(
                            df_processed.drop(columns=[target_column]), target
                        )
                        df_processed = pd.DataFrame(X_resampled, columns=df_processed.drop(columns=[target_column]).columns)
                        target = y_resampled
                        print(f"    Resampled to: {len(df_processed)} samples")
            except Exception as e:
                print(f"    ⚠️ Imbalance handling failed: {e}")
        
        return df_processed, target
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'rfe', 
                       n_features: Optional[int] = None) -> List[str]:
        """Advanced feature selection"""
        print(f"🎯 Feature selection using {method}...")
        
        if n_features is None:
            n_features = min(10, len(X.columns))
        
        try:
            if method == 'rfe':
                # Recursive Feature Elimination
                if self.problem_type == 'classification':
                    estimator = RandomForestClassifier(n_estimators=50, random_state=42)
                else:
                    estimator = RandomForestRegressor(n_estimators=50, random_state=42)
                
                rfe = RFE(estimator=estimator, n_features_to_select=n_features)
                rfe.fit(X, y)
                selected_features = X.columns[rfe.support_].tolist()
                
            elif method == 'selectkbest':
                # SelectKBest
                if self.problem_type == 'classification':
                    selector = SelectKBest(score_func=f_classif, k=n_features)
                else:
                    selector = SelectKBest(score_func=f_classif, k=n_features)
                
                selector.fit(X, y)
                selected_features = X.columns[selector.get_support()].tolist()
                
            elif method == 'correlation':
                # Correlation-based selection
                if self.problem_type == 'classification':
                    # For classification, use correlation with target
                    correlations = X.corrwith(y).abs().sort_values(ascending=False)
                else:
                    correlations = X.corrwith(y).abs().sort_values(ascending=False)
                
                selected_features = correlations.head(n_features).index.tolist()
                
            else:
                # Default: use all features
                selected_features = X.columns.tolist()
            
            print(f"  ✅ Selected {len(selected_features)} features: {selected_features}")
            return selected_features
            
        except Exception as e:
            print(f"  ⚠️ Feature selection failed: {e}")
            return X.columns.tolist()
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, 
                    models_to_train: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """Train multiple models and compare performance"""
        print("🤖 Training models...")
        
        if models_to_train is None:
            models_to_train = ['Random Forest', 'Logistic Regression', 'Decision Tree', 'SVM', 'KNN']
        
        results = {}
        
        # Define model configurations
        model_configs = {
            'Random Forest': {
                'classification': RandomForestClassifier(n_estimators=100, random_state=42),
                'regression': RandomForestRegressor(n_estimators=100, random_state=42)
            },
            'Logistic Regression': {
                'classification': LogisticRegression(random_state=42, max_iter=1000),
                'regression': LinearRegression()
            },
            'Decision Tree': {
                'classification': DecisionTreeClassifier(random_state=42),
                'regression': DecisionTreeRegressor(random_state=42)
            },
            'SVM': {
                'classification': SVC(random_state=42, probability=True),
                'regression': SVC(kernel='rbf', random_state=42)
            },
            'KNN': {
                'classification': KNeighborsClassifier(n_neighbors=5),
                'regression': KNeighborsRegressor(n_neighbors=5)
            },
            'Naive Bayes': {
                'classification': GaussianNB(),
                'regression': None
            }
        }
        
        # Add advanced models if available
        if XGBOOST_AVAILABLE:
            model_configs['XGBoost'] = {
                'classification': xgb.XGBClassifier(random_state=42),
                'regression': xgb.XGBRegressor(random_state=42)
            }
        
        if LIGHTGBM_AVAILABLE:
            model_configs['LightGBM'] = {
                'classification': lgb.LGBMClassifier(random_state=42, verbose=-1),
                'regression': lgb.LGBMRegressor(random_state=42, verbose=-1)
            }
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if self.problem_type == 'classification' else None
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        for model_name in models_to_train:
            if model_name not in model_configs:
                print(f"  ⚠️ Unknown model: {model_name}")
                continue
            
            model_config = model_configs[model_name]
            if self.problem_type not in model_config or model_config[self.problem_type] is None:
                print(f"  ⚠️ {model_name} not available for {self.problem_type}")
                continue
            
            try:
                print(f"  🔄 Training {model_name}...")
                start_time = time.time()
                
                model = model_config[self.problem_type]
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                metrics = self.calculate_metrics(y_test, y_pred, model, X_test_scaled)
                metrics['training_time'] = time.time() - start_time
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy' if self.problem_type == 'classification' else 'r2')
                metrics['cv_mean'] = float(cv_scores.mean())
                metrics['cv_std'] = float(cv_scores.std())
                
                results[model_name] = metrics
                print(f"    ✅ {model_name}: {metrics.get('accuracy', metrics.get('r2_score', 0)):.4f}")
                
            except Exception as e:
                print(f"    ❌ {model_name} failed: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         model: Any, X_test: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive metrics"""
        metrics = {}
        
        try:
            if self.problem_type == 'classification':
                metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
                metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
                metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
                metrics['f1_score'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
                
                # ROC AUC for binary classification
                if len(np.unique(y_true)) == 2 and hasattr(model, 'predict_proba'):
                    try:
                        y_proba = model.predict_proba(X_test)[:, 1]
                        metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
                    except:
                        metrics['roc_auc'] = 0.0
                else:
                    metrics['roc_auc'] = 0.0
                
                # Set regression metrics to 0
                metrics['r2_score'] = 0.0
                metrics['mse'] = 0.0
                metrics['rmse'] = 0.0
                
            else:  # regression
                metrics['r2_score'] = float(r2_score(y_true, y_pred))
                metrics['mse'] = float(mean_squared_error(y_true, y_pred))
                metrics['rmse'] = float(np.sqrt(metrics['mse']))
                
                # Set classification metrics to 0
                metrics['accuracy'] = 0.0
                metrics['precision'] = 0.0
                metrics['recall'] = 0.0
                metrics['f1_score'] = 0.0
                metrics['roc_auc'] = 0.0
                
        except Exception as e:
            print(f"    ⚠️ Metrics calculation failed: {e}")
            # Set default values
            for key in ['accuracy', 'precision', 'recall', 'f1_score', 'r2_score', 'mse', 'rmse', 'roc_auc']:
                metrics[key] = 0.0
        
        return metrics
    
    def generate_recommendations(self, data_quality: Dict[str, Any], 
                               results: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Data quality recommendations
        missing_pct = data_quality.get('missing_percentage', {})
        high_missing = [col for col, pct in missing_pct.items() if pct > 20]
        if high_missing:
            recommendations.append(f"Consider removing columns with high missing values: {', '.join(high_missing)}")
        
        outliers = data_quality.get('outliers', {})
        shape = data_quality.get('shape', (0, 0))
        if isinstance(shape, tuple):
            total_rows = shape[0]
        else:
            total_rows = 0
        high_outliers = [col for col, count in outliers.items() if count > total_rows * 0.1]
        if high_outliers:
            recommendations.append(f"High outlier count detected in: {', '.join(high_outliers)}. Consider outlier treatment.")
        
        # Model performance recommendations
        if results:
            best_model = max(results.items(), key=lambda x: x[1].get('accuracy', x[1].get('r2_score', 0)))
            recommendations.append(f"Best performing model: {best_model[0]} with score {best_model[1].get('accuracy', best_model[1].get('r2_score', 0)):.4f}")
            
            # Check for overfitting
            for model_name, metrics in results.items():
                if 'cv_mean' in metrics and 'accuracy' in metrics:
                    if metrics['accuracy'] - metrics['cv_mean'] > 0.1:
                        recommendations.append(f"{model_name} shows signs of overfitting. Consider regularization.")
        
        # Feature recommendations
        if len(data_quality.get('numeric_columns', [])) > 20:
            recommendations.append("High number of features detected. Consider feature selection or dimensionality reduction.")
        
        return recommendations
    
    def run_complete_analysis(self, df: pd.DataFrame, target_column: str,
                            models_to_train: List[str] = None,
                            handle_imbalance: bool = True,
                            handle_outliers: bool = True,
                            feature_selection_method: str = 'rfe') -> Dict[str, Any]:
        """Run complete enhanced analysis pipeline"""
        print("🚀 Starting Enhanced ML Analysis...")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"🎯 Target column: {target_column}")
        
        try:
            # Step 1: Data Quality Analysis
            data_quality = self.analyze_data_quality(df)
            
            # Step 2: Detect Problem Type
            self.problem_type = self.detect_problem_type(df[target_column])
            print(f"🎯 Problem type: {self.problem_type}")
            
            # Step 3: Preprocess Data
            df_processed, target = self.preprocess_data(
                df, target_column, 
                handle_missing=True, 
                handle_outliers=handle_outliers,
                handle_imbalance=handle_imbalance
            )
            
            # Step 4: Feature Selection
            X = df_processed.drop(columns=[target_column])
            selected_features = self.select_features(X, target, method=feature_selection_method)
            X_selected = X[selected_features]
            
            # Step 5: Train Models
            results = self.train_models(X_selected, target, models_to_train)
            
            # Step 6: Find Best Model
            if results:
                best_model_name = max(results.items(), 
                                    key=lambda x: x[1].get('accuracy', x[1].get('r2_score', 0)))[0]
                best_score = results[best_model_name].get('accuracy', results[best_model_name].get('r2_score', 0))
            else:
                best_model_name = "No models trained successfully"
                best_score = 0
            
            # Step 7: Generate Recommendations
            recommendations = self.generate_recommendations(data_quality, results)
            
            # Step 8: Prepare Results
            analysis_results = {
                'success': True,
                'problem_type': self.problem_type,
                'results': results,
                'best_model': best_model_name,
                'best_score': float(best_score),
                'selected_features': selected_features,
                'data_quality': data_quality,
                'recommendations': recommendations,
                'dataset_info': {
                    'original_shape': df.shape,
                    'processed_shape': df_processed.shape,
                    'target_column': target_column,
                    'feature_count': len(selected_features)
                }
            }
            
            print("✅ Enhanced analysis completed successfully!")
            print(f"🏆 Best model: {best_model_name} (Score: {best_score:.4f})")
            print(f"📈 Features selected: {len(selected_features)}")
            
            return analysis_results
            
        except Exception as e:
            print(f"❌ Enhanced analysis failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'problem_type': 'unknown',
                'results': {},
                'best_model': 'None',
                'best_score': 0,
                'selected_features': [],
                'data_quality': {},
                'recommendations': [f"Analysis failed: {str(e)}"],
                'dataset_info': {
                    'original_shape': df.shape if df is not None else (0, 0),
                    'processed_shape': (0, 0),
                    'target_column': target_column,
                    'feature_count': 0
                }
            }


# Create a global instance for easy access
enhanced_analyzer = EnhancedMLAnalyzer()
