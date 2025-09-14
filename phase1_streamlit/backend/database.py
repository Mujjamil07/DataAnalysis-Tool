"""
Real-time Database System for AutoML
"""

import sqlite3
import json
import pandas as pd
from datetime import datetime
import os
from typing import Dict, List, Optional, Any

class AutoMLDatabase:
    def __init__(self, db_path: str = "automl_database.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Datasets table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    session_id TEXT UNIQUE NOT NULL,
                    filename TEXT NOT NULL,
                    data_hash TEXT UNIQUE,
                    rows INTEGER,
                    columns INTEGER,
                    column_names TEXT,  -- JSON array
                    data_types TEXT,    -- JSON object
                    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'uploaded',
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Training sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id INTEGER,
                    session_id TEXT UNIQUE NOT NULL,
                    target_column TEXT NOT NULL,
                    problem_type TEXT,  -- 'classification' or 'regression'
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    status TEXT DEFAULT 'running',  -- 'running', 'completed', 'failed'
                    total_models INTEGER DEFAULT 0,
                    successful_models INTEGER DEFAULT 0,
                    best_model TEXT,
                    best_score REAL,
                    FOREIGN KEY (dataset_id) REFERENCES datasets (id)
                )
            ''')
            
            # Model results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    training_session_id INTEGER,
                    model_name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    r2_score REAL,
                    rmse REAL,
                    mae REAL,
                    mse REAL,
                    training_time REAL,
                    feature_importance TEXT,  -- JSON object
                    predictions TEXT,        -- JSON array
                    confusion_matrix TEXT,   -- JSON array
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (training_session_id) REFERENCES training_sessions (id)
                )
            ''')
            
            # Reports table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    training_session_id INTEGER,
                    report_type TEXT NOT NULL,  -- 'summary', 'detailed', 'comparison'
                    report_data TEXT,           -- JSON object
                    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (training_session_id) REFERENCES training_sessions (id)
                )
            ''')
            
            # Data processing logs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processing_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id INTEGER,
                    step_name TEXT NOT NULL,
                    step_status TEXT DEFAULT 'running',  -- 'running', 'completed', 'failed'
                    details TEXT,  -- JSON object
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    FOREIGN KEY (dataset_id) REFERENCES datasets (id)
                )
            ''')
            
            conn.commit()
    
    def save_dataset(self, session_id: str, filename: str, data: pd.DataFrame, user_id: int = 1) -> int:
        """Save uploaded dataset to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create data hash for uniqueness
            data_hash = str(hash(data.to_string()))
            
            # Prepare data
            rows = len(data)
            columns = len(data.columns)
            column_names = json.dumps(data.columns.tolist())
            data_types = json.dumps(data.dtypes.astype(str).to_dict())
            
            cursor.execute('''
                INSERT INTO datasets (user_id, session_id, filename, data_hash, rows, columns, column_names, data_types, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, session_id, filename, data_hash, rows, columns, column_names, data_types, 'uploaded'))
            
            dataset_id = cursor.lastrowid
            conn.commit()
            
            print(f"✅ Dataset saved to database: ID {dataset_id}, Session {session_id}")
            return dataset_id
    
    def create_training_session(self, dataset_id: int, session_id: str, target_column: str, problem_type: str = None) -> int:
        """Create a new training session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO training_sessions (dataset_id, session_id, target_column, problem_type, status)
                VALUES (?, ?, ?, ?, ?)
            ''', (dataset_id, session_id, target_column, problem_type, 'running'))
            
            training_session_id = cursor.lastrowid
            conn.commit()
            
            print(f"✅ Training session created: ID {training_session_id}, Session {session_id}")
            return training_session_id
    
    def save_model_result(self, training_session_id: int, model_name: str, model_type: str, 
                         metrics: Dict[str, Any], feature_importance: Dict = None, 
                         predictions: List = None, confusion_matrix: List = None) -> int:
        """Save model training results"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO model_results (
                    training_session_id, model_name, model_type, accuracy, precision, recall, f1_score,
                    r2_score, rmse, mae, mse, training_time, feature_importance, predictions, confusion_matrix
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                training_session_id, model_name, model_type,
                metrics.get('accuracy'), metrics.get('precision'), metrics.get('recall'), metrics.get('f1_score'),
                metrics.get('r2_score'), metrics.get('rmse'), metrics.get('mae'), metrics.get('mse'),
                metrics.get('training_time'),
                json.dumps(feature_importance) if feature_importance else None,
                json.dumps(predictions) if predictions else None,
                json.dumps(confusion_matrix) if confusion_matrix else None
            ))
            
            model_result_id = cursor.lastrowid
            conn.commit()
            
            print(f"✅ Model result saved: {model_name} - ID {model_result_id}")
            return model_result_id
    
    def update_training_session(self, training_session_id: int, status: str = 'completed', 
                               best_model: str = None, best_score: float = None):
        """Update training session status"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE training_sessions 
                SET status = ?, end_time = CURRENT_TIMESTAMP, best_model = ?, best_score = ?
                WHERE id = ?
            ''', (status, best_model, best_score, training_session_id))
            
            conn.commit()
            print(f"✅ Training session updated: ID {training_session_id}, Status: {status}")
    
    def get_dataset_by_session(self, session_id: str) -> Optional[Dict]:
        """Get dataset information by session ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, filename, rows, columns, column_names, data_types, upload_time, status
                FROM datasets WHERE session_id = ?
            ''', (session_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'filename': row[1],
                    'rows': row[2],
                    'columns': row[3],
                    'column_names': json.loads(row[4]),
                    'data_types': json.loads(row[5]),
                    'upload_time': row[6],
                    'status': row[7]
                }
            return None
    
    def get_training_results(self, training_session_id: int) -> List[Dict]:
        """Get all model results for a training session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT model_name, model_type, accuracy, precision, recall, f1_score,
                       r2_score, rmse, mae, mse, training_time, feature_importance
                FROM model_results 
                WHERE training_session_id = ?
                ORDER BY created_at DESC
            ''', (training_session_id,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'model_name': row[0],
                    'model_type': row[1],
                    'accuracy': row[2],
                    'precision': row[3],
                    'recall': row[4],
                    'f1_score': row[5],
                    'r2_score': row[6],
                    'rmse': row[7],
                    'mae': row[8],
                    'mse': row[9],
                    'training_time': row[10],
                    'feature_importance': json.loads(row[11]) if row[11] else None
                })
            
            return results
    
    def generate_report(self, training_session_id: int, report_type: str = 'summary') -> Dict:
        """Generate a comprehensive report"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get training session info
            cursor.execute('''
                SELECT ts.*, d.filename, d.rows, d.columns
                FROM training_sessions ts
                JOIN datasets d ON ts.dataset_id = d.id
                WHERE ts.id = ?
            ''', (training_session_id,))
            
            session_row = cursor.fetchone()
            if not session_row:
                return None
            
            # Get model results
            model_results = self.get_training_results(training_session_id)
            
            # Generate report
            report = {
                'training_session_id': training_session_id,
                'dataset_info': {
                    'filename': session_row[8],
                    'rows': session_row[9],
                    'columns': session_row[10]
                },
                'training_info': {
                    'target_column': session_row[3],
                    'problem_type': session_row[4],
                    'start_time': session_row[5],
                    'end_time': session_row[6],
                    'status': session_row[7],
                    'best_model': session_row[11],
                    'best_score': session_row[12]
                },
                'model_results': model_results,
                'summary': {
                    'total_models': len(model_results),
                    'best_model': session_row[11],
                    'best_score': session_row[12],
                    'average_training_time': sum(r['training_time'] for r in model_results if r['training_time']) / len(model_results) if model_results else 0
                }
            }
            
            # Save report to database
            cursor.execute('''
                INSERT INTO reports (training_session_id, report_type, report_data)
                VALUES (?, ?, ?)
            ''', (training_session_id, report_type, json.dumps(report)))
            
            conn.commit()
            return report
    
    def get_user_datasets(self, user_id: int = 1) -> List[Dict]:
        """Get all datasets for a user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, session_id, filename, rows, columns, upload_time, status
                FROM datasets 
                WHERE user_id = ?
                ORDER BY upload_time DESC
            ''', (user_id,))
            
            datasets = []
            for row in cursor.fetchall():
                datasets.append({
                    'id': row[0],
                    'session_id': row[1],
                    'filename': row[2],
                    'rows': row[3],
                    'columns': row[4],
                    'upload_time': row[5],
                    'status': row[6]
                })
            
            return datasets
    
    def get_user_training_sessions(self, user_id: int = 1) -> List[Dict]:
        """Get all training sessions for a user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT ts.id, ts.session_id, ts.target_column, ts.problem_type, 
                       ts.start_time, ts.end_time, ts.status, ts.best_model, ts.best_score,
                       d.filename
                FROM training_sessions ts
                JOIN datasets d ON ts.dataset_id = d.id
                JOIN users u ON d.user_id = u.id
                WHERE u.id = ?
                ORDER BY ts.start_time DESC
            ''', (user_id,))
            
            sessions = []
            for row in cursor.fetchall():
                sessions.append({
                    'id': row[0],
                    'session_id': row[1],
                    'target_column': row[2],
                    'problem_type': row[3],
                    'start_time': row[4],
                    'end_time': row[5],
                    'status': row[6],
                    'best_model': row[7],
                    'best_score': row[8],
                    'dataset_filename': row[9]
                })
            
            return sessions

# Global database instance
db = AutoMLDatabase()

    def save_session(self, session_id: str, session_data: Dict[str, Any]):
        """Save session data to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Save session info
                cursor.execute('''
                    INSERT OR REPLACE INTO sessions 
                    (session_id, filename, uploaded_at, data_info, target_column, feature_names, training_info)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session_id,
                    session_data.get('filename'),
                    session_data.get('uploaded_at'),
                    json.dumps(session_data.get('data_info', {})),
                    session_data.get('target_column'),
                    json.dumps(session_data.get('feature_names', [])),
                    json.dumps(session_data.get('training_info', {}))
                ))
                
                # Save FULL dataset if it exists - NO TRUNCATION
                if 'data' in session_data and session_data['data'] is not None:
                    df = session_data['data']
                    print(f"💾 Saving FULL dataset to database: {df.shape[0]:,} rows, {df.shape[1]} columns")
                    csv_data = df.to_csv(index=False)
                    data_size = len(csv_data)
                    print(f"   Database storage size: {data_size / 1024 / 1024:.2f} MB")
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO dataset_storage 
                        (session_id, data_csv, data_size)
                        VALUES (?, ?, ?)
                    ''', (session_id, csv_data, data_size))
                
                conn.commit()
                return True
        except Exception as e:
            print(f"Error saving session: {e}")
            return False
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get session info
                cursor.execute('''
                    SELECT filename, uploaded_at, data_info, target_column, feature_names, training_info
                    FROM sessions WHERE session_id = ?
                ''', (session_id,))
                
                session_row = cursor.fetchone()
                if not session_row:
                    return None
                
                # Get dataset
                cursor.execute('''
                    SELECT data_csv FROM dataset_storage WHERE session_id = ?
                ''', (session_id,))
                
                dataset_row = cursor.fetchone()
                df = None
                if dataset_row:
                    from io import StringIO
                    print(f"📂 Loading FULL dataset from database...")
                    df = pd.read_csv(StringIO(dataset_row[0]))
                    print(f"   ✅ Loaded FULL dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
                
                return {
                    'filename': session_row[0],
                    'uploaded_at': session_row[1],
                    'data_info': json.loads(session_row[2]),
                    'target_column': session_row[3],
                    'feature_names': json.loads(session_row[4]),
                    'training_info': json.loads(session_row[5]),
                    'data': df
                }
        except Exception as e:
            print(f"Error getting session: {e}")
            return None
    
    def save_training_results(self, session_id: str, results: Dict[str, Any]):
        """Save training results to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for model_name, result in results.items():
                    cursor.execute('''
                        INSERT INTO training_results 
                        (session_id, model_name, accuracy, precision, recall, f1_score, training_time, confusion_matrix, feature_importance)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        session_id,
                        model_name,
                        result.get('accuracy', 0),
                        result.get('precision', 0),
                        result.get('recall', 0),
                        result.get('f1_score', 0),
                        result.get('training_time', 0),
                        json.dumps(result.get('confusion_matrix', [])),
                        json.dumps(result.get('feature_importance', {}))
                    ))
                
                conn.commit()
                return True
        except Exception as e:
            print(f"Error saving training results: {e}")
            return False
    
    def get_training_results(self, session_id: str) -> Dict[str, Any]:
        """Get training results from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT model_name, accuracy, precision, recall, f1_score, training_time, confusion_matrix, feature_importance
                    FROM training_results WHERE session_id = ?
                    ORDER BY accuracy DESC
                ''', (session_id,))
                
                results = {}
                for row in cursor.fetchall():
                    results[row[0]] = {
                        'accuracy': row[1],
                        'precision': row[2],
                        'recall': row[3],
                        'f1_score': row[4],
                        'training_time': row[5],
                        'confusion_matrix': json.loads(row[6]),
                        'feature_importance': json.loads(row[7])
                    }
                
                return results
        except Exception as e:
            print(f"Error getting training results: {e}")
            return {}
    
    def list_sessions(self) -> list:
        """List all sessions"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT session_id, filename, uploaded_at, data_info, target_column, training_info
                    FROM sessions ORDER BY created_at DESC
                ''')
                
                sessions = []
                for row in cursor.fetchall():
                    sessions.append({
                        'session_id': row[0],
                        'filename': row[1],
                        'uploaded_at': row[2],
                        'data_info': json.loads(row[3]),
                        'target_column': row[4],
                        'training_info': json.loads(row[5]) if row[5] else {}
                    })
                
                return sessions
        except Exception as e:
            print(f"Error listing sessions: {e}")
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session and all related data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete training results
                cursor.execute('DELETE FROM training_results WHERE session_id = ?', (session_id,))
                
                # Delete dataset storage
                cursor.execute('DELETE FROM dataset_storage WHERE session_id = ?', (session_id,))
                
                # Delete session
                cursor.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
                
                conn.commit()
                return True
        except Exception as e:
            print(f"Error deleting session: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count sessions
                cursor.execute('SELECT COUNT(*) FROM sessions')
                total_sessions = cursor.fetchone()[0]
                
                # Count training results
                cursor.execute('SELECT COUNT(*) FROM training_results')
                total_results = cursor.fetchone()[0]
                
                # Database size
                db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                
                return {
                    'total_sessions': total_sessions,
                    'total_training_results': total_results,
                    'database_size_mb': round(db_size / (1024 * 1024), 2),
                    'database_path': self.db_path
                }
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {}

# Global database instance
db_manager = DatabaseManager()
