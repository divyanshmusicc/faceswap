@echo off
rem One-click push of the Kaggle notebook, then opens it in your browser.
rem One-time setup: pip install kaggle  +  place kaggle.json from
rem https://www.kaggle.com/settings (API section) at %USERPROFILE%\.kaggle\kaggle.json
setlocal
cd /d "%~dp0"
where kaggle >nul 2>nul
if errorlevel 1 (
  echo [faceswap] 'kaggle' CLI not found. Run:  pip install kaggle
  echo [faceswap] Then download kaggle.json from https://www.kaggle.com/settings
  echo [faceswap] and save it to %%USERPROFILE%%\.kaggle\kaggle.json
  pause
  exit /b 1
)
if not exist "%USERPROFILE%\.kaggle\kaggle.json" (
  echo [faceswap] Missing %%USERPROFILE%%\.kaggle\kaggle.json
  echo [faceswap] Get it from https://www.kaggle.com/settings ^(API section^)
  pause
  exit /b 1
)
kaggle kernels push -p kaggle
if errorlevel 1 (
  echo [faceswap] Push failed - see error above.
  pause
  exit /b 1
)
start "" "https://www.kaggle.com/code/divyanshmusicc/faceswap-kaggle"
