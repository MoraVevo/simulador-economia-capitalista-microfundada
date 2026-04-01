@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_EXE=py -3.13"
set "VENV_DIR=%~dp0.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"

if not exist "%VENV_PYTHON%" (
    echo Creando entorno virtual...
    %PYTHON_EXE% -m venv "%VENV_DIR%"
    if errorlevel 1 goto :error
)

echo Verificando dependencias...
"%VENV_PYTHON%" -m pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo Instalando dependencias del simulador...
    "%VENV_PYTHON%" -m pip install --upgrade pip
    if errorlevel 1 goto :error
    "%VENV_PYTHON%" -m pip install -e .
    if errorlevel 1 goto :error
)

echo Abriendo simulador...
"%VENV_PYTHON%" -m streamlit run app.py --server.headless true --server.showEmailPrompt false --browser.gatherUsageStats false
goto :eof

:error
echo No se pudo iniciar el simulador.
pause
exit /b 1
