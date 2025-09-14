"""
Enhanced ML Analysis Module - Based on Comprehensive Diabetes Prediction Analysis
Integrates advanced preprocessing, feature selection, and model comparison techniques
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.utils.multiclass import type_of_target
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
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.over_sampling import RandomOverSampler, SMOTE
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class EnhancedMLAnalyzer:
    """
    Enhanced ML Analysis class incorporating advanced techniques from diabetes prediction analysis
    """
    
    def __init__(self):
        self.results = {}
        self.best_model = None
        self.best_score = 0
        self.problem_type = None
        self.feature_importance = None
        self.correlation_matrix = None
        
    def analyze_data_quality(self, df):
        """Comprehensive data quality analysis"""
        analysis = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
            'categorical_summary': {}
        }
        
        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns
        for col in categorical_cols:
            analysis['categorical_summary'][col] = {
                'unique_values': df[col].nunique(),
                'value_counts': df[col].value_counts().head(10).to_dict(),
                'missing_percentage': (df[col].isnull().sum() / len(df)) * 100
            }
        
        return analysis
    
    def detect_outliers(self, df, method='iqr', threshold=3):
        """Advanced outlier detection using multiple methods"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outliers = df[z_scores > threshold]
                
            outlier_info[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(df)) * 100,
                'method': method
            }
        
        return outlier_info
    
    def handle_outliers(self, df, method='iqr', action='cap'):
        """Handle outliers using various methods"""
        df_clean = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                if action == 'cap':
                    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                elif action == 'remove':
                    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
                    
            elif method == 'zscore':
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                
                if action == 'cap':
                    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                elif action == 'remove':
                    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        return df_clean
    
    def handle_class_imbalance(self, X, y, method='undersample', sampling_strategy='auto'):
        """Handle class imbalance using various techniques"""
        if method == 'undersample':
            if sampling_strategy == 'auto':
                sampler = RandomUnderSampler(random_state=42)
            else:
                sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
                
        elif method == 'oversample':
            if sampling_strategy == 'auto':
                sampler = RandomOverSampler(random_state=42)
            else:
                sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
                
        elif method == 'smote':
            sampler = SMOTE(random_state=42)
            
        elif method == 'nearmiss':
            sampler = NearMiss(version=1)
            
        else:
            return X, y
            
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def feature_selection(self, X, y, method='rfe', n_features=None, problem_type='classification'):
        """Advanced feature selection using multiple methods"""
        if n_features is None:
            n_features = min(10, X.shape[1])
        
        if method == 'rfe':
            if problem_type == 'classification':
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=100, random_state=42)
                
            selector = RFE(estimator, n_features_to_select=n_features)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.support_].tolist()
            
        elif method == 'univariate':
            if problem_type == 'classification':
                selector = SelectKBest(score_func=f_classif, k=n_features)
            else:
                selector = SelectKBest(score_func=f_classif, k=n_features)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        else:
            return X, X.columns.tolist()
        
        return pd.DataFrame(X_selected, columns=selected_features), selected_features
    
    def get_available_models(self, problem_type='classification'):
        """Get available models based on problem type"""
        models = {}
        
        if problem_type == 'classification':
            models = {
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'SVM': SVC(random_state=42, probability=True),
                'KNN': KNeighborsClassifier(),
                'Naive Bayes': GaussianNB()
            }
            
            if XGBOOST_AVAILABLE:
                models['XGBoost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
                
        else:  # regression
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'KNN': KNeighborsRegressor()
            }
            
            if XGBOOST_AVAILABLE:
                models['XGBoost'] = xgb.XGBRegressor(random_state=42)
        
        return models
    
    def train_and_evaluate_models(self, X, y, models, problem_type='classification', cv_folds=5):
        """Train and evaluate multiple models with cross-validation"""
        results = {}
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features for models that need it
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        for name, model in models.items():
            try:
                # Use scaled data for models that benefit from it
                if name in ['SVM', 'KNN', 'Logistic Regression']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if problem_type == 'classification' else None
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1] if problem_type == 'classification' else None
                
                # Calculate metrics
                if problem_type == 'classification':
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
                    
                    results[name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'predictions': y_pred,
                        'probabilities': y_pred_proba,
                        'model': model
                    }
                    
                    # Calculate ROC-AUC if binary classification
                    if len(np.unique(y)) == 2 and y_pred_proba is not None:
                        roc_auc = roc_auc_score(y_test, y_pred_proba)
                        results[name]['roc_auc'] = roc_auc
                        
                else:  # regression
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
                    
                    results[name] = {
                        'mse': mse,
                        'rmse': rmse,
                        'r2_score': r2,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'predictions': y_pred,
                        'model': model
                    }
                
                print(f"✅ {name} - Score: {results[name].get('accuracy', results[name].get('r2_score', 0)):.4f}")
                
            except Exception as e:
                print(f"❌ Error training {name}: {str(e)}")
                continue
        
        return results, X_test, y_test
    
    def find_best_model(self, results, problem_type='classification'):
        """Find the best model based on primary metric"""
        if not results:
            return None, 0
        
        if problem_type == 'classification':
            # Use F1-score as primary metric for balanced evaluation
            best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
            best_score = results[best_model_name]['f1_score']
        else:
            # Use R² score for regression
            best_model_name = max(results.keys(), key=lambda x: results[x]['r2_score'])
            best_score = results[best_model_name]['r2_score']
        
        return best_model_name, best_score
    
    def generate_comprehensive_report(self, df, target_column, results, best_model_name):
        """Generate comprehensive analysis report"""
        report = {
            'data_quality': self.analyze_data_quality(df),
            'outlier_analysis': self.detect_outliers(df),
            'model_comparison': {},
            'best_model': {
                'name': best_model_name,
                'metrics': results[best_model_name] if best_model_name in results else {}
            },
            'recommendations': []
        }
        
        # Model comparison summary
        for name, metrics in results.items():
            report['model_comparison'][name] = {
                'primary_metric': metrics.get('f1_score', metrics.get('r2_score', 0)),
                'cross_validation': {
                    'mean': metrics.get('cv_mean', 0),
                    'std': metrics.get('cv_std', 0)
                }
            }
        
        # Generate recommendations
        if report['data_quality']['duplicates'] > 0:
            report['recommendations'].append("Consider removing duplicate records")
        
        if any(analysis['percentage'] > 5 for analysis in report['outlier_analysis'].values()):
            report['recommendations'].append("Consider outlier treatment for high-outlier columns")
        
        if len(df[target_column].value_counts()) == 2:
            class_balance = df[target_column].value_counts()
            imbalance_ratio = max(class_balance) / min(class_balance)
            if imbalance_ratio > 3:
                report['recommendations'].append("Consider class imbalance handling techniques")
        
        return report
    
    def run_complete_analysis(self, df, target_column, models_to_train=None, 
                            handle_imbalance=True, handle_outliers=True, 
                            feature_selection_method='rfe'):
        """Run complete ML analysis pipeline"""
        print("🚀 Starting Enhanced ML Analysis Pipeline")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"🎯 Target column: {target_column}")
        
        # Step 1: Data Quality Analysis
        print("\n📋 Step 1: Data Quality Analysis")
        data_quality = self.analyze_data_quality(df)
        print(f"✅ Missing values: {sum(data_quality['missing_values'].values())}")
        print(f"✅ Duplicates: {data_quality['duplicates']}")
        
        # Step 2: Determine problem type
        target_unique = df[target_column].nunique()
        target_dtype = df[target_column].dtype
        
        # Check if target is categorical (integer with few unique values)
        if target_dtype in ['int64', 'int32', 'object'] and target_unique <= 10:
            # Check if values are discrete (integers)
            if df[target_column].dtype in ['int64', 'int32'] or df[target_column].apply(lambda x: str(x).replace('.', '').isdigit()).all():
                problem_type = 'classification'
                print("🎯 Problem type: Classification")
                print(f"📊 Target unique values: {target_unique}")
            else:
                problem_type = 'regression'
                print("🎯 Problem type: Regression")
        else:
            problem_type = 'regression'
            print("🎯 Problem type: Regression")
        
        # Step 3: Handle outliers
        if handle_outliers:
            print("\n🔧 Step 2: Outlier Treatment")
            outlier_info = self.detect_outliers(df)
            high_outlier_cols = [col for col, info in outlier_info.items() if info['percentage'] > 5]
            if high_outlier_cols:
                print(f"⚠️ High outlier columns: {high_outlier_cols}")
                df = self.handle_outliers(df, method='iqr', action='cap')
                print("✅ Outliers capped using IQR method")
        
        # Step 4: Prepare features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Preprocess target variable for classification
        if problem_type == 'classification':
            # Convert target to integer if it's not already
            if y.dtype == 'object':
                y = y.astype('category').cat.codes
            elif y.dtype in ['float64', 'float32']:
                # Check if float values are actually integers
                if y.apply(lambda x: x.is_integer()).all():
                    y = y.astype(int)
                else:
                    # Convert continuous values to discrete classes
                    y = pd.cut(y, bins=5, labels=False)
                    print("🔄 Converted continuous target to discrete classes")
            
            print(f"📊 Target unique values: {sorted(y.unique())}")
            print(f"📊 Problem type: Classification")
        
        # Handle categorical features
        categorical_columns = X.select_dtypes(include=['object', 'string']).columns
        if len(categorical_columns) > 0:
            X_encoded = X.copy()
            for col in categorical_columns:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            X = X_encoded
        
        # Step 5: Handle class imbalance
        if handle_imbalance and problem_type == 'classification':
            print("\n⚖️ Step 3: Class Imbalance Handling")
            class_counts = y.value_counts()
            imbalance_ratio = max(class_counts) / min(class_counts)
            print(f"📊 Class imbalance ratio: {imbalance_ratio:.2f}")
            
            if imbalance_ratio > 3:
                X, y = self.handle_class_imbalance(X, y, method='undersample')
                print(f"✅ Applied undersampling - New shape: {X.shape}")
        
        # Step 6: Feature Selection
        print("\n🎯 Step 4: Feature Selection")
        X_selected, selected_features = self.feature_selection(
            X, y, method=feature_selection_method, 
            problem_type=problem_type
        )
        print(f"✅ Selected {len(selected_features)} features: {selected_features}")
        
        # Step 7: Get models to train
        available_models = self.get_available_models(problem_type)
        if models_to_train:
            models = {name: model for name, model in available_models.items() 
                     if name in models_to_train}
        else:
            models = available_models
        
        print(f"\n🤖 Step 5: Training {len(models)} Models")
        
        # Step 8: Train and evaluate models
        results, X_test, y_test = self.train_and_evaluate_models(
            X_selected, y, models, problem_type
        )
        
        # Step 9: Find best model
        best_model_name, best_score = self.find_best_model(results, problem_type)
        print(f"\n🏆 Best Model: {best_model_name} (Score: {best_score:.4f})")
        
        # Step 10: Generate comprehensive report
        report = self.generate_comprehensive_report(df, target_column, results, best_model_name)
        
        self.results = results
        self.best_model = best_model_name
        self.best_score = best_score
        self.problem_type = problem_type
        
        return {
            'results': results,
            'best_model': best_model_name,
            'best_score': best_score,
            'problem_type': problem_type,
            'selected_features': selected_features,
            'report': report,
            'test_data': {'X_test': X_test, 'y_test': y_test}
        }

# Example usage function
def run_enhanced_analysis(df, target_column, models_to_train=None):
    """Run enhanced analysis on diabetes prediction dataset"""
    analyzer = EnhancedMLAnalyzer()
    
    # Run complete analysis
    analysis_results = analyzer.run_complete_analysis(
        df=df,
        target_column=target_column,
        models_to_train=models_to_train,
        handle_imbalance=True,
        handle_outliers=True,
        feature_selection_method='rfe'
    )
    
    return analysis_results

if __name__ == "__main__":
    # Example with diabetes dataset
    print("🧪 Testing Enhanced ML Analysis with Sample Data")
    
    # Create sample diabetes-like data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.normal(45, 15, n_samples),
        'bmi': np.random.normal(27, 5, n_samples),
        'glucose': np.random.normal(140, 40, n_samples),
        'blood_pressure': np.random.normal(80, 15, n_samples),
        'diabetes': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    }
    
    df = pd.DataFrame(data)
    
    # Run analysis
    results = run_enhanced_analysis(df, 'diabetes', ['Logistic Regression', 'Random Forest', 'XGBoost'])
    
    print("\n📊 Analysis Complete!")
    print(f"Best Model: {results['best_model']}")
    print(f"Best Score: {results['best_score']:.4f}")
