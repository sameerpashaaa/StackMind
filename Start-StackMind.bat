@echo off
setlocal

set "ROOT=%~dp0"
set "FRONTEND_PORT=3001"

echo Starting StackMind backend and frontend...

start "StackMind API" cmd /k "cd /d ""%ROOT%"" && if exist ""%ROOT%venv\Scripts\python.exe"" (""%ROOT%venv\Scripts\python.exe"" ""%ROOT%main.py"" --api) else (python ""%ROOT%main.py"" --api)"
start "StackMind UI" cmd /k "cd /d ""%ROOT%frontend"" && npm run dev -- --port %FRONTEND_PORT%"
start "" "http://localhost:%FRONTEND_PORT%"

endlocal