# 🚀 Quick Start Guide

## One-Click Startup Options

### Option 1: PowerShell Script (Recommended)
```powershell
.\start_app.ps1
```

### Option 2: Batch File
```cmd
start_app.bat
```

### Option 3: Manual Startup
1. **Start Backend:**
   ```cmd
   cd backend
   py main.py
   ```

2. **Start Frontend (in new terminal):**
   ```cmd
   npm start
   ```

## ✅ What's Fixed

1. **Models.map Error**: Fixed the `models.map is not a function` error by adding proper array checks
2. **Backend Connection**: Backend is running on port 8002 and responding correctly
3. **Frontend**: React app is running on port 3000
4. **API Service**: Configured to connect to backend on port 8002

## 🔧 Current Status

- ✅ Backend: Running on http://localhost:8002
- ✅ Frontend: Running on http://localhost:3000
- ✅ API Connection: Configured and working
- ✅ Error Handling: Fixed models.map error

## 🎯 How to Use

1. **Upload Data**: Go to Data Upload page and upload your CSV/Excel file
2. **Select Target**: Choose your target column for prediction
3. **Train Models**: Click "Train All Models" to train multiple ML models
4. **View Results**: Check Model Results page for performance metrics
5. **Visualizations**: Explore data visualizations and model comparisons

## 🐛 Troubleshooting

If you see "Backend server is not available":
1. Make sure both servers are running
2. Check that backend is on port 8002
3. Try refreshing the page
4. Use the startup scripts above

## 📊 Features

- **Real ML Models**: Logistic Regression, Random Forest, SVM
- **Real Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Data Processing**: Missing values, categorical encoding, scaling
- **Visualizations**: Distribution plots, correlation matrix, model comparison
- **Session Management**: Multiple datasets and model sessions

## 🎉 Ready to Use!

Both servers are now running and the application should work perfectly!
