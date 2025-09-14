# 🚀 AutoML Full Stack Application

A complete machine learning platform with **React Frontend** and **Python FastAPI Backend** for real ML workflows.

## 🎯 Features

### Frontend (React)
- 📊 Modern UI with real-time updates
- 📁 Drag & drop file upload
- 🎯 Target column selection
- 🤖 Model training interface
- 📈 Live training progress
- 📊 Results visualization
- 📋 Model comparison tables

### Backend (FastAPI)
- 🔧 Real ML algorithms (scikit-learn, XGBoost, LightGBM)
- 📊 Data preprocessing & cleaning
- 🎯 Multiple model training
- 📈 Real metrics calculation
- 🖼️ Visualization generation
- 🔄 Session management
- 📚 Auto-generated API docs

## 🏗️ Architecture

```
┌─────────────────┐    HTTP/JSON    ┌─────────────────┐
│   React App     │ ◄─────────────► │  FastAPI Backend│
│  (Frontend)     │                 │   (Python)      │
│                 │                 │                 │
│ • File Upload   │                 │ • Data Processing│
│ • UI Components │                 │ • ML Training   │
│ • State Mgmt    │                 │ • Model Storage │
│ • Visualizations│                 │ • API Endpoints │
└─────────────────┘                 └─────────────────┘
        │                                    │
        │                                    │
        ▼                                    ▼
┌─────────────────┐                 ┌─────────────────┐
│   Local Storage │                 │  Session Storage│
│  (Zustand)      │                 │  (Memory/Redis) │
└─────────────────┘                 └─────────────────┘
```

## 🛠️ Installation

### Prerequisites
- **Node.js** (v16+)
- **Python** (3.8+)
- **npm** or **yarn**

### 1. Clone Repository
```bash
git clone <repository-url>
cd automl-saas
```

### 2. Frontend Setup
```bash
# Install dependencies
npm install

# Start frontend only
npm start
```

### 3. Backend Setup
```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start backend only
python start_backend.py
```

### 4. Full Stack Setup (Recommended)
```bash
# From root directory
python start_full_stack.py
```

## 🚀 Quick Start

### Option 1: Full Stack (Easiest)
```bash
python start_full_stack.py
```
This starts both frontend and backend automatically.

### Option 2: Manual Start
```bash
# Terminal 1 - Backend
cd backend
python start_backend.py

# Terminal 2 - Frontend
npm start
```

## 📱 Usage

### 1. Access Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### 2. Upload Data
- Drag & drop CSV/Excel files
- Select target column
- View data preview

### 3. Train Models
- Choose models to train
- Monitor real-time progress
- View live results

### 4. Analyze Results
- Compare model performance
- View confusion matrices
- Generate visualizations

## 🔧 API Endpoints

### Core Endpoints
```
POST /upload-data          # Upload dataset
POST /train-models         # Train ML models
GET  /get-results/{id}     # Get training results
POST /generate-visualizations/{id}  # Create plots
GET  /sessions             # List sessions
DELETE /sessions/{id}      # Delete session
```

### Example Usage
```bash
# Upload data
curl -X POST "http://localhost:8000/upload-data" \
     -F "file=@dataset.csv"

# Train models
curl -X POST "http://localhost:8000/train-models" \
     -F "session_id=abc123" \
     -F "target_column=target"

# Get results
curl "http://localhost:8000/get-results/abc123"
```

## 🎯 Supported Models

| Model | Type | Description |
|-------|------|-------------|
| **Logistic Regression** | Linear | Binary classification |
| **Random Forest** | Ensemble | Multiple decision trees |
| **SVM** | Kernel | Support Vector Machine |
| **XGBoost** | Boosting | Gradient boosting |
| **LightGBM** | Boosting | Light gradient boosting |

## 📊 Metrics Calculated

- **Accuracy**: Overall correctness
- **Precision**: True positives / (TP + FP)
- **Recall**: True positives / (TP + FN)
- **F1-Score**: Harmonic mean of precision & recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: Detailed classification results

## 🔍 Data Processing

### Automatic Processing
- **Missing Values**: Filled with median
- **Categorical Encoding**: Label encoding
- **Feature Scaling**: StandardScaler
- **Train-Test Split**: 80-20 with stratification

### Supported Formats
- **CSV** (.csv)
- **Excel** (.xlsx, .xls)

## 📈 Visualizations

### Generated Plots
- **Data Distributions**: Histograms of features
- **Correlation Matrix**: Feature relationships
- **Model Performance**: Accuracy comparison
- **Confusion Matrices**: Classification results

## 🛡️ Error Handling

### Frontend
- Network error handling
- File validation
- Progress tracking
- User feedback

### Backend
- Input validation
- Exception handling
- Graceful failures
- Detailed error messages

## 🔄 Session Management

### Features
- **Unique Session IDs**: UUID-based
- **Data Persistence**: During session lifetime
- **Session Listing**: View all active sessions
- **Cleanup**: Automatic session deletion

## 🚀 Production Deployment

### Backend Considerations
```python
# Use Redis for sessions
REDIS_URL = "redis://localhost:6379"

# Add authentication
JWT_SECRET = "your-secret-key"

# Database for persistence
DATABASE_URL = "postgresql://user:pass@localhost/db"

# Rate limiting
RATE_LIMIT = "100/minute"
```

### Frontend Considerations
```javascript
// Environment variables
REACT_APP_API_URL = "https://api.yourdomain.com"

// Error boundaries
// Service workers
// Progressive web app
```

## 🐳 Docker Deployment

### Backend Dockerfile
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend Dockerfile
```dockerfile
FROM node:16-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
CMD ["npm", "start"]
```

## 🔧 Development

### Backend Development
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development
```bash
npm install
npm start
```

### Testing
```bash
# Backend tests
cd backend
pytest

# Frontend tests
npm test
```

## 📚 Documentation

- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Backend README**: `backend/README.md`

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

- **Issues**: GitHub Issues
- **Documentation**: README files
- **API Docs**: Swagger UI

---

**🎉 Ready to build amazing ML applications!**
