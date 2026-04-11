@echo off
setlocal enabledelayedexpansion

pushd "%~dp0" >nul
set "PROJECT_DIR=%CD%"
popd >nul

title SubVivienda IA - Setup y lanzamiento automatico

echo.
echo  ==========================================
echo   SubVivienda IA - Instalacion y Lanzamiento
echo  ==========================================
echo.

REM ============================================
REM 1. VERIFICACIONES INICIALES
REM ============================================

echo [1/6] Verificando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no instalado o no en PATH
    echo Descargalo desde python.org
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VER=%%i
echo [OK] Python %PYTHON_VER% encontrado

echo [2/6] Verificando requirements.txt...
if not exist requirements.txt (
    echo ERROR: No se encuentra requirements.txt
    pause
    exit /b 1
)
echo [OK] requirements.txt encontrado

REM ============================================
REM 2. CONFIRMACION DEL USUARIO
REM ============================================

echo.
echo  Este script:
echo  1. Crea el entorno virtual
echo  2. Instala todas las dependencias
echo  3. Abre 3 terminales automaticamente:
echo     - Terminal 1: Entrenamiento (train_and_save.py)
echo     - Terminal 2: Backend API (puerto 8001)
echo     - Terminal 3: Frontend Streamlit (puerto 8501)
echo.
echo  IMPORTANTE: Pon Subsidios_De_Vivienda.xlsx
echo             en la carpeta data\ ANTES de entrenar
echo.

set /p CONFIRM=" Continuar? (S/N): "
if /i not "%CONFIRM%"=="S" (
    echo Cancelado.
    pause
    exit /b 0
)

REM ============================================
REM 3. CREAR ENTORNO VIRTUAL
REM ============================================

echo.
echo [3/6] Creando entorno virtual...
if exist venv (
    echo El entorno virtual ya existe. Eliminando antiguo...
    rmdir /s /q venv
)
python -m venv venv
if errorlevel 1 (
    echo ERROR: No se pudo crear el entorno virtual
    pause
    exit /b 1
)
echo [OK] Entorno virtual creado

REM ============================================
REM 4. ACTIVAR E INSTALAR DEPENDENCIAS
REM ============================================

echo [4/6] Instalando dependencias en el entorno virtual...
echo Esto puede tomar varios minutos...

REM Usar la ruta directa al pip del entorno virtual
venv\Scripts\python -m pip install --upgrade pip >nul
venv\Scripts\pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Fallo la instalacion de dependencias
    pause
    exit /b 1
)
echo [OK] Dependencias instaladas

REM ============================================
REM 5. VERIFICAR ARCHIVO DE DATOS
REM ============================================

echo [5/6] Verificando archivo de datos...
if not exist "data\Subsidios_De_Vivienda.xlsx" (
    echo.
    echo ADVERTENCIA: No se encuentra data\Subsidios_De_Vivienda.xlsx
    echo.
    echo El entrenamiento fallara sin este archivo.
    echo.
    set /p CONTINUAR=" Crear carpeta data y continuar de todas formas? (S/N): "
    if /i "!CONTINUAR!"=="S" (
        if not exist data mkdir data
        echo Carpeta data creada. Por favor copia el archivo manualmente.
        timeout /t 3 >nul
    ) else (
        echo Cancelado por falta de archivo de datos
        pause
        exit /b 1
    )
) else (
    echo [OK] Archivo de datos encontrado
)

REM ============================================
REM 6. ABRIR TERMINALES AUTOMATICAMENTE
REM ============================================

echo [6/6] Abriendo terminales...
echo.

REM La variable PROJECT_DIR ya se estableció al inicio del script.

REM ===== TERMINAL 1: Entrenamiento =====
echo Abriendo Terminal 1: Entrenamiento de modelos...
start "SubVivienda IA - Entrenamiento" cmd /k "title SubVivienda IA - Entrenamiento && echo ========================================== && echo  Entrenamiento de modelos SubVivienda IA && echo ========================================== && echo. && cd /d ^"%PROJECT_DIR%^" && echo [1] Activando entorno virtual... && call ^"%PROJECT_DIR%\venv\Scripts\activate^" && echo [2] Ejecutando entrenamiento... && echo. && python train_and_save.py && echo. && echo ========================================== && echo  Entrenamiento completado && echo ========================================== && echo. && echo Puedes cerrar esta ventana o seguir usandola && echo Para ejecutar comandos adicionales: && echo   conda activate && echo   python script.py && echo. && echo Presiona Ctrl+C para salir..."

timeout /t 2 >nul

REM ===== TERMINAL 2: Backend =====
echo Abriendo Terminal 2: Backend API (puerto 8001)...
start "SubVivienda IA - Backend" cmd /k "title SubVivienda IA - Backend && echo ========================================== && echo  Backend API - SubVivienda IA && echo ========================================== && echo. && cd /d ^"%PROJECT_DIR%\backend^" 2>nul || (echo ERROR: Carpeta backend no encontrada && echo Creando carpeta backend... && mkdir ^"%PROJECT_DIR%\backend^" && cd /d ^"%PROJECT_DIR%\backend^") && echo [1] Activando entorno virtual... && call ^"%PROJECT_DIR%\venv\Scripts\activate^" && set PYTHONPATH=%PROJECT_DIR%\backend && echo [2] Iniciando servidor FastAPI en puerto 8001... && echo. && echo IMPORTANTE: Manten esta terminal abierta && echo Presiona Ctrl+C para detener el servidor && echo. && python -m uvicorn main:app --reload --host 127.0.0.1 --port 8001"

timeout /t 2 >nul

REM ===== TERMINAL 3: Frontend =====
echo Abriendo Terminal 3: Frontend Streamlit (puerto 8501)...
start "SubVivienda IA - Frontend" cmd /k "title SubVivienda IA - Frontend && echo ========================================== && echo  Frontend Streamlit - SubVivienda IA && echo ========================================== && echo. && cd /d ^"%PROJECT_DIR%\frontend^" 2>nul || (echo ERROR: Carpeta frontend no encontrada && echo Creando carpeta frontend... && mkdir ^"%PROJECT_DIR%\frontend^" && cd /d ^"%PROJECT_DIR%\frontend^") && echo [1] Activando entorno virtual... && call ^"%PROJECT_DIR%\venv\Scripts\activate^" && echo [2] Iniciando Streamlit en puerto 8501... && echo. && echo IMPORTANTE: Manten esta terminal abierta && echo La app se abrira automaticamente en tu navegador && echo Presiona Ctrl+C para detener Streamlit && echo. && streamlit run app.py --server.port 8501"

echo.
echo ==========================================
echo  INSTALACION Y LANZAMIENTO COMPLETADO
echo ==========================================
echo.
echo Las 3 terminales se han abierto:
echo.
echo   [1] Terminal de ENTRENAMIENTO
echo       - Ejecuta train_and_save.py
echo       - Espera a que termine (puede tomar minutos)
echo.
echo   [2] Terminal de BACKEND (puerto 8001)
echo       - Debe mostrar: "Application startup complete"
echo       - Si falla, espera a que termine el entrenamiento
echo.
echo   [3] Terminal de FRONTEND (puerto 8501)
echo       - Se abrira automaticamente en tu navegador
echo       - Si no se abre, ve a: http://localhost:8501
echo.
echo ==========================================
echo  IMPORTANTE:
echo ==========================================
echo.
echo  - NO cierres las terminales mientras usas la app
echo  - Presiona Ctrl+C en cada terminal para detener
echo  - El orden recomendado es:
echo    1. Esperar a que termine el entrenamiento
echo    2. Verificar que el backend inicio correctamente
echo    3. Usar el frontend
echo.
echo  Si el backend falla con "no module named 'backend'":
echo    cd "%PROJECT_DIR%"
echo    set PYTHONPATH=%PROJECT_DIR%
echo    uvicorn backend.main:app --reload --port 8001
echo.
echo Presiona cualquier tecla para salir...
pause >nul
exit /b 0