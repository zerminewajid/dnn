why@echo off
echo Starting Weathering App...

:: Start backend
start "Backend" cmd /k "cd /d %~dp0backend && .venv\Scripts\activate && python main.py"

:: Wait a moment then start frontend
timeout /t 2 /nobreak >nul
start "Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:5173
echo.
timeout /t 3 /nobreak >nul
start http://localhost:5173
