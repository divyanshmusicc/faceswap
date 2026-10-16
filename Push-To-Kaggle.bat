@echo off
rem One-click push of the Kaggle notebook, then opens it in your browser.
rem One-time setup: pip install -U kaggle  +  save your API token from
rem https://www.kaggle.com/settings (API section) to %USERPROFILE%\.kaggle\access_token
rem (Kaggle no longer issues kaggle.json files - the token file replaces it.)
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
rem New-style token auth (kaggle.json is legacy): env var or access_token file.
if defined KAGGLE_API_TOKEN goto have_auth
if exist "%USERPROFILE%\.kaggle\access_token" goto have_auth
if exist "%USERPROFILE%\.kaggle\kaggle.json" goto have_auth
  echo [faceswap] No Kaggle credentials found.
  echo [faceswap] Save your API token from https://www.kaggle.com/settings ^(API section^)
  echo [faceswap] to %%USERPROFILE%%\.kaggle\access_token  -or-  set KAGGLE_API_TOKEN
  pause
  exit /b 1
:have_auth
kaggle kernels push -p kaggle
if errorlevel 1 (
  echo [faceswap] Push failed - see error above.
  pause
  exit /b 1
)
start "" "https://www.kaggle.com/code/divyanshmusicc/faceswap-kaggle"
