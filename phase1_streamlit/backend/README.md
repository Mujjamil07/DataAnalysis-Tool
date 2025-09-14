# AutoML Backend API

A FastAPI-based backend for real machine learning workflows with data processing, model training, and visualization generation.

## 🚀 Features

- **Real ML Training**: scikit-learn, XGBoost, LightGBM
- **Data Processing**: Automatic cleaning, encoding, scaling
- **Multiple Models**: Logistic Regression, Random Forest, SVM, XGBoost, LightGBM
- **Real Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualizations**: Data distributions, correlation matrices, performance plots
- **Session Management**: Track multiple training sessions
- **REST API**: Full RESTful API with automatic documentation

## 📋 Requirements

- Python 3.8+
- All dependencies listed in `requirements.txt`

## 🛠️ Installation

1. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 🚀 Running the Backend

### Option 1: Using the startup script
```bash
python start_backend.py
```

### Option 2: Direct uvicorn
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 3: Using Python
```bash
python main.py
```

## 📚 API Endpoints

### Core Endpoints
- `GET /` - Health check
- `POST /upload-data` - Upload CSV/Excel file
- `POST /train-models` - Train ML models
- `GET /get-results/{session_id}` - Get training results
- `POST /generate-visualizations/{session_id}` - Generate plots
- `GET /sessions` - List all sessions
- `DELETE /sessions/{session_id}` - Delete session

### API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔧 Usage Example

### 1. Upload Data
```bash
curl -X POST "http://localhost:8000/upload-data" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_dataset.csv"
```

### 2. Train Models
```bash
curl -X POST "http://localhost:8000/train-models" \
     -H "accept: application/json" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "session_id=your_session_id&target_column=target"
```

### 3. Get Results
```bash
curl -X GET "http://localhost:8000/get-results/your_session_id"
```

## 🎯 Supported Models

1. **Logistic Regression** - Linear classification
2. **Random Forest** - Ensemble method
3. **Support Vector Machine** - Kernel-based classification
4. **XGBoost** - Gradient boosting
5. **LightGBM** - Light gradient boosting

## 📊 Metrics Calculated

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: Detailed classification results

## 🔍 Data Processing

- **Missing Values**: Filled with median
- **Categorical Encoding**: Label encoding
- **Feature Scaling**: StandardScaler
- **Train-Test Split**: 80-20 with stratification

## 📈 Visualizations

- **Data Distributions**: Histograms of numerical features
- **Correlation Matrix**: Feature relationships
- **Model Performance**: Accuracy comparison bar chart

## 🛡️ Error Handling

- Comprehensive error messages
- Input validation
- Graceful failure handling
- Session management

## 🔄 Session Management

- Unique session IDs for each upload
- Persistent data storage during session
- Automatic cleanup capabilities
- Session listing and deletion

## 🚀 Production Considerations

- Use Redis for session storage
- Add authentication/authorization
- Implement rate limiting
- Add logging and monitoring
- Use PostgreSQL for persistent storage
- Add Docker containerization
