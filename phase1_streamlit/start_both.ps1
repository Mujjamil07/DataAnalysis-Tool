Write-Host "🚀 Starting AutoML SaaS - Frontend & Backend..." -ForegroundColor Green

# Start Backend in new window
Write-Host "Starting Backend..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\Users\Asus\automl-saas\phase1_streamlit'; .\venv\Scripts\Activate.ps1; cd backend; python main_fresh.py" -WindowStyle Normal

# Wait a moment for backend to start
Start-Sleep -Seconds 3

# Start Frontend in new window
Write-Host "Starting Frontend..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\Users\Asus\automl-saas\phase1_streamlit'; npm start" -WindowStyle Normal

Write-Host ""
Write-Host "✅ Both servers are starting..." -ForegroundColor Green
Write-Host "🌐 Frontend: http://localhost:3005" -ForegroundColor Cyan
Write-Host "🔧 Backend: http://localhost:8005" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to close this window..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")








