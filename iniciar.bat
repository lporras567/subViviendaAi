@echo off
setlocal enabledelayedexpansion

pushd "%~dp0" >nul
set "PROJECT_DIR=%CD%"
popd >nul

title SubVivienda IA

echo.
echo  ============================================
echo   SubVivienda IA
echo  ============================================
echo.
echo  Directorio: %PROJECT_DIR%
echo.

REM ============================================
REM 1. VERIFICAR PYTHON
REM ============================================

echo [PASO 1] Verificando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no encontrado.
    pause
    exit /b 1
)
echo [OK] Python encontrado.

REM ============================================
REM 2. SETUP (solo si es necesario)
REM ============================================

echo [PASO 2] Verificando entorno virtual...
if exist "%PROJECT_DIR%\venv\Scripts\activate.bat" (
    echo [OK] Entorno virtual encontrado. Saltando instalacion.
    goto INICIAR_BACKEND
)

echo [SETUP] Creando entorno virtual por primera vez...
python -m venv "%PROJECT_DIR%\venv"
if errorlevel 1 (
    echo [ERROR] No se pudo crear el entorno virtual.
    pause
    exit /b 1
)

echo [SETUP] Instalando dependencias...
"%PROJECT_DIR%\venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
"%PROJECT_DIR%\venv\Scripts\pip.exe" install -r "%PROJECT_DIR%\requirements.txt"
if errorlevel 1 (
    echo [ERROR] Fallo la instalacion.
    pause
    exit /b 1
)
echo [OK] Entorno listo.

REM ============================================
REM 3. ENTRENAR (solo 1ra vez)
REM ============================================

if not exist "%PROJECT_DIR%\data\Subsidios_De_Vivienda.xlsx" (
    echo [ERROR] No se encuentra data\Subsidios_De_Vivienda.xlsx
    pause
    exit /b 1
)

echo [ENTRENAMIENTO] Entrenando modelos...
"%PROJECT_DIR%\venv\Scripts\python.exe" "%PROJECT_DIR%\train_and_save.py"
if errorlevel 1 (
    echo [ERROR] Entrenamiento fallo.
    pause
    exit /b 1
)
echo [OK] Entrenamiento completado.

REM ============================================
REM 4. INICIAR BACKEND
REM ============================================

:INICIAR_BACKEND
echo.
echo [PASO 3] Iniciando backend...
echo          Ruta: %PROJECT_DIR%\backend
start "SubVivienda IA - Backend" cmd /k "cd /d "%PROJECT_DIR%\backend" && call "%PROJECT_DIR%\venv\Scripts\activate.bat" && set PYTHONPATH=%PROJECT_DIR%\backend && python -m uvicorn main:app --reload --host 127.0.0.1 --port 8001"

REM ============================================
REM 5. ESPERAR BACKEND
REM ============================================

echo.
echo [PASO 4] Esperando 25 segundos para que el backend cargue...
timeout /t 25 /nobreak

REM ============================================
REM 6. INICIAR FRONTEND
REM ============================================

echo.
echo [PASO 5] Iniciando frontend...
echo          Ruta: %PROJECT_DIR%\frontend
start "SubVivienda IA - Frontend" cmd /k "cd /d "%PROJECT_DIR%\frontend" && call "%PROJECT_DIR%\venv\Scripts\activate.bat" && streamlit run app.py --server.port 8501"

echo.
echo  ============================================
echo   App lista!
echo  ============================================
echo.
echo   Backend:   http://localhost:8001
echo   Frontend:  http://localhost:8501
echo.
echo   NO cierres las terminales del backend y frontend.
echo.
pause
exit /b 0
