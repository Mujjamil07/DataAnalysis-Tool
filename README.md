# 🚀 AutoML SaaS Platform

A comprehensive Machine Learning platform with automated model training, enhanced analysis, and real-time results visualization.

## ✨ Features

### 🎯 **Core ML Capabilities**
- **Automated Model Training**: Linear Regression, Logistic Regression, Random Forest, SVM, Decision Tree, KNN, Naive Bayes
- **Enhanced Analysis**: Advanced preprocessing, feature selection, and model comparison
- **Real-time Results**: Live training progress and instant model performance metrics
- **Data Quality Analysis**: Comprehensive data profiling and quality assessment

### 🔧 **Technical Stack**
- **Backend**: FastAPI, Python, Scikit-learn, XGBoost, LightGBM
- **Frontend**: React, Zustand, Framer Motion, Tailwind CSS
- **ML Libraries**: Pandas, NumPy, Imbalanced-learn, Matplotlib, Seaborn

### 📊 **Advanced Features**
- **Enhanced ML Analysis**: Based on comprehensive diabetes prediction analysis
- **Feature Selection**: RFE, SelectKBest, correlation analysis
- **Data Preprocessing**: Missing value handling, outlier detection, standardization
- **Model Comparison**: Cross-validation, performance metrics, best model selection
- **Data Visualization**: Interactive charts and graphs

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/automl-saas.git
   cd automl-saas
   ```

2. **Backend Setup**
   ```bash
   cd phase1_streamlit
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   pip install -r requirements.txt
   ```

3. **Frontend Setup**
   ```bash
   cd phase1_streamlit
   npm install
   ```

4. **Start the Application**
   ```bash
   # Start both servers
   .\start_both.bat  # Windows
   ```

## 📁 Project Structure

```
automl-saas/
├── main_fresh.py                 # FastAPI backend server
├── enhanced_ml_analysis.py       # Enhanced ML analysis module
├── phase1_streamlit/             # Frontend React application
│   ├── src/
│   │   ├── pages/               # React components
│   │   ├── services/            # API services
│   │   ├── store/               # Zustand state management
│   │   └── components/          # Reusable components
│   ├── public/
│   └── package.json
├── start_both.bat               # Start script for both servers
└── README.md
```

## 🎯 Usage

### 1. **Data Upload**
- Upload CSV files through the web interface
- Automatic data type detection and validation
- Real-time data preview and statistics

### 2. **Model Training**
- Select target column and models to train
- Real-time training progress monitoring
- Automatic problem type detection (classification/regression)

### 3. **Enhanced Analysis**
- Advanced preprocessing and feature selection
- Comprehensive model comparison
- Data quality assessment and recommendations

### 4. **Results Visualization**
- Interactive model performance charts
- Detailed metrics and statistics
- Export capabilities for results

## 🔧 API Endpoints

### Core Endpoints
- `GET /health` - Health check
- `POST /upload` - Upload dataset
- `POST /train` - Train models
- `POST /enhanced-analysis` - Run enhanced analysis
- `GET /sessions` - List active sessions

### Data Processing
- `POST /clean-data` - Clean and preprocess data
- `GET /analyze-data/{session_id}` - Analyze dataset quality

## 🛠️ Development

### Backend Development
```bash
cd automl-saas
python main_fresh.py
```

### Frontend Development
```bash
cd phase1_streamlit
npm start
```

### Testing
```bash
# Backend health check
curl http://localhost:8005/health

# Frontend
http://localhost:3005
```

## 📊 Enhanced Analysis Features

The enhanced analysis module provides:

- **Automatic Problem Detection**: Binary, multiclass, or regression
- **Feature Selection**: Multiple algorithms (RFE, SelectKBest, correlation)
- **Data Quality Assessment**: Missing values, outliers, distributions
- **Model Comparison**: Cross-validation, performance metrics
- **Recommendations**: Data preprocessing suggestions

## 🎨 UI/UX Features

- **Modern Design**: Clean, responsive interface
- **Real-time Updates**: Live progress indicators
- **Interactive Charts**: Dynamic data visualization
- **Smooth Animations**: Framer Motion transitions
- **Dark/Light Theme**: User preference support

## 🔒 Security & Performance

- **CORS Configuration**: Proper cross-origin resource sharing
- **Data Validation**: Input sanitization and validation
- **Error Handling**: Comprehensive error management
- **Performance Optimization**: Efficient data processing

## 📈 Future Enhancements

- [ ] Model deployment capabilities
- [ ] Advanced visualization options
- [ ] User authentication and authorization
- [ ] Cloud deployment support
- [ ] API rate limiting
- [ ] Model versioning

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Scikit-learn for ML algorithms
- FastAPI for backend framework
- React for frontend framework
- Framer Motion for animations

---

**Built with ❤️ for the ML community**
