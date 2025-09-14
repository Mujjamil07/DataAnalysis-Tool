@echo off
echo 🔧 Installing Backend Dependencies...
cd /d "C:\Users\Asus\automl-saas\phase1_streamlit"

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing Python packages...
cd backend
pip install -r requirements.txt

echo.
echo 🔧 Installing Frontend Dependencies...
cd ..
npm install

echo.
echo ✅ All dependencies installed!
echo You can now run: start_both.bat
pause








