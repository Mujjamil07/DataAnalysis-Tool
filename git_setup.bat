@echo off
echo ========================================
echo AutoML SaaS - Git Setup and Push
echo ========================================

echo.
echo Step 1: Initialize Git Repository
git init

echo.
echo Step 2: Add all files
git add .

echo.
echo Step 3: Create initial commit
git commit -m "Initial commit: AutoML SaaS with Enhanced Analysis

- Complete FastAPI backend with ML training
- React frontend with modern UI
- Enhanced ML analysis module
- CORS and serialization fixes
- Diabetes prediction analysis integration
- Real-time model training and results"

echo.
echo Step 4: Add remote repository (replace YOUR_USERNAME with your GitHub username)
echo git remote add origin https://github.com/YOUR_USERNAME/automl-saas.git

echo.
echo Step 5: Push to GitHub (uncomment after adding remote)
echo git branch -M main
echo git push -u origin main

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Create a new repository on GitHub
echo 2. Replace YOUR_USERNAME in the remote URL above
echo 3. Uncomment the push commands
echo 4. Run this script again
echo.
pause
