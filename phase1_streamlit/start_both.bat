@echo off
echo 🚀 Starting AutoML SaaS - Frontend & Backend...

:: Start Backend in new window
echo Starting Backend...
start "Backend Server" cmd /k "cd /d C:\Users\Asus\automl-saas\phase1_streamlit && call venv\Scripts\activate.bat && cd /d C:\Users\Asus\automl-saas && python main_fresh.py"

:: Wait a moment for backend to start
timeout /t 3 /nobreak > nul

:: Start Frontend in new window  
echo Starting Frontend...
start "Frontend Server" cmd /k "cd /d C:\Users\Asus\automl-saas\phase1_streamlit && set PORT=3005 && npm start"

echo.
echo ✅ Both servers are starting...
echo 🌐 Frontend: http://localhost:3005
echo 🔧 Backend: http://localhost:8005
echo.
echo Press any key to close this window...
pause > nul
