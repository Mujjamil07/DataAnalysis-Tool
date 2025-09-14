# Watson AI Studio - AutoML SaaS Platform

A professional data analysis and machine learning platform with Watson-style interface built using **React** (Frontend) and **Streamlit** (Backend).

## 🎨 UI Versions

This project now includes **two UI versions**:

1. **React Frontend** (Modern, Interactive) - `start_react.bat`
2. **Streamlit Backend** (Original) - `run_project.bat`

## 🚀 Quick Start

### Option 1: React Frontend (Recommended)
1. Navigate to the project directory in Command Prompt or PowerShell
2. Run the React startup script:
   ```cmd
   start_react.bat
   ```
3. The React app will open at: http://localhost:3000

### Option 2: Streamlit Backend
1. Navigate to the project directory in Command Prompt or PowerShell
2. Run the Streamlit startup script:
   ```cmd
   run_project.bat
   ```
3. The Streamlit app will open at: http://localhost:8501

## 🚀 Quick Start

### Windows Users
1. Navigate to the project directory in Command Prompt or PowerShell
2. Double-click `run_project.bat` or run:
   ```cmd
   run_project.bat
   ```

### Mac/Linux Users
1. Navigate to the project directory in Terminal
2. Make the script executable and run:
   ```bash
   chmod +x run_project.sh
   ./run_project.sh
   ```

### Manual Setup (Alternative)
If you prefer to run manually:

1. **Navigate to the project directory:**
   ```bash
   cd /path/to/your/project
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 📋 Prerequisites

### For React Frontend:
- Node.js 16 or higher
- npm (Node package manager)
- Web browser

### For Streamlit Backend:
- Python 3.8 or higher
- pip (Python package installer)
- Web browser

## 🛠️ Features

- **Professional UI**: Watson AI Studio-style interface
- **Data Upload**: CSV file upload and analysis
- **AutoML**: Automatic machine learning model training
- **Data Visualization**: Interactive charts and graphs
- **Model Comparison**: Performance metrics and model comparison
- **Export Results**: Download analysis reports

## 📁 Project Structure

```
phase1_streamlit/
├── app.py                 # Main Streamlit application (Backend)
├── requirements.txt       # Python dependencies
├── run_project.bat       # Streamlit startup script
├── run_project.sh        # Unix/Mac startup script
├── start_react.bat       # React startup script
├── package.json          # React dependencies
├── tailwind.config.js    # Tailwind CSS configuration
├── postcss.config.js     # PostCSS configuration
├── public/               # React public assets
│   └── index.html        # Main HTML file
├── src/                  # React source code
│   ├── components/       # React components
│   ├── pages/           # React pages
│   ├── store/           # State management
│   ├── App.js           # Main React app
│   ├── index.js         # React entry point
│   └── index.css        # Main CSS file
└── README.md            # This file
```

## 🔧 Configuration

### React Frontend:
- **URL**: http://localhost:3000
- **Port**: 3000
- **Host**: localhost

### Streamlit Backend:
- **URL**: http://localhost:8501
- **Port**: 8501
- **Host**: localhost

## 📦 Dependencies

### React Frontend:
- react>=18.2.0
- react-dom>=18.2.0
- react-router-dom>=6.3.0
- framer-motion>=10.12.0
- lucide-react>=0.263.1
- react-dropzone>=14.2.3
- react-hot-toast>=2.4.1
- zustand>=4.3.8
- tailwindcss>=3.3.0
- axios>=1.4.0

### Streamlit Backend:
- streamlit>=1.28.0
- pandas>=2.0.0
- numpy>=1.24.0
- scikit-learn>=1.3.0
- xgboost>=1.7.0
- lightgbm>=4.0.0
- plotly>=5.15.0
- seaborn>=0.12.0
- matplotlib>=3.7.0
- shap>=0.42.0
- reportlab>=4.0.0
- openpyxl>=3.1.0

## 🚨 Troubleshooting

### Common Issues

1. **Python not found**
   - Ensure Python is installed and added to PATH
   - Try running `python --version` to verify

2. **Port already in use**
   - The application uses port 8501 by default
   - If busy, modify the port in the startup scripts

3. **Dependencies installation fails**
   - Try upgrading pip: `python -m pip install --upgrade pip`
   - Check your internet connection
   - Some packages may require additional system dependencies

4. **Virtual environment issues**
   - Delete the `venv` folder and let the script recreate it
   - Ensure you have sufficient disk space

### Getting Help

If you encounter issues:
1. Check the console output for error messages
2. Ensure you're in the correct project directory
3. Verify Python version compatibility
4. Check that all dependencies are properly installed

## 🎯 Usage

1. **Upload Data**: Use the "Data Upload" section to upload your CSV file
2. **Select Target**: Choose the column you want to predict
3. **Train Models**: Use the "Model Training" section to train AI models
4. **View Results**: Analyze model performance in the "Model Results" section

## 📄 License

This project is for educational and demonstration purposes.

---

**Note**: The startup scripts will automatically handle virtual environment creation, dependency installation, and application startup. Just make sure you're in the correct project directory when running them.
