# 🚀 AutoML SaaS - Complete Setup Instructions

## ⚡ Quick Start (Recommended)

1. **Install Dependencies**
   ```bash
   install_dependencies.bat
   ```

2. **Start Both Servers**
   ```bash
   start_both.bat
   ```

3. **Open Browser**
   - Frontend: http://localhost:3005
   - Backend: http://localhost:8005

## 🔧 Manual Setup (If Quick Start Fails)

### Backend Setup
```bash
cd phase1_streamlit
venv\Scripts\activate
cd backend
pip install -r requirements.txt
python main_fresh.py
```

### Frontend Setup (New Terminal)
```bash
cd phase1_streamlit
npm install
npm start
```

## 📁 Test Files

Use `test_data.csv` for testing the upload functionality.

## 🔍 Troubleshooting

### Backend Issues
- Check: http://localhost:8005/health
- Debug: http://localhost:8005/debug
- Logs: Check terminal running backend

### Frontend Issues
- Check browser console (F12)
- Verify backend is running first
- Clear browser cache if needed

### Common Errors

1. **"Cannot connect to server"**
   - Start backend first: `cd backend && python main_fresh.py`

2. **"File parsing error"**
   - Ensure file is valid CSV/Excel
   - Check file encoding (use UTF-8)

3. **"FormData cloning error"**
   - This should be fixed in the latest version
   - Try refreshing the page

## 📋 Features Working

✅ File Upload (CSV, XLS, XLSX)
✅ Data Preview
✅ Session Management
✅ Error Handling
✅ File Validation
✅ Backend Health Check

## 🎯 Next Steps

After successful upload:
1. Go to "Model Training" page
2. Select target column
3. Train models
4. View results








