@echo off
setlocal enabledelayedexpansion

pushd "%~dp0" >nul
set "PROJECT_DIR=%CD%"
popd >nul

title SubVivienda IA - Lanzamiento Rapido

echo.
echo ==========================================
echo  SubVivienda IA - Lanzamiento Rapido
echo ==========================================
echo.

REM ============================================
REM 1. VERIFICAR ENTORNO LISTO
REM ============================================

if not exist "%PROJECT_DIR%\venv\Scripts\activate" (
    echo ERROR: Entorno virtual no encontrado.
    echo Ejecuta primero setup_auto.bat para instalar el entorno.
    pause
    exit /b 1
)
echo [OK] Entorno virtual encontrado.

REM ============================================
REM 2. INICIAR BACKEND
REM ============================================

echo.
echo [1/2] Abriendo Backend (puerto 8001)...
start "SubVivienda IA - Backend" cmd /k "title Backend SubVivienda IA && cd /d ^"%PROJECT_DIR%\backend^" && call ^"%PROJECT_DIR%\venv\Scripts\activate^" && set PYTHONPATH=%PROJECT_DIR%\backend && echo Iniciando servidor FastAPI... && python -m uvicorn main:app --reload --host 127.0.0.1 --port 8001"

REM ============================================
REM 3. ESPERAR A QUE EL BACKEND RESPONDA
REM ============================================

echo.
echo Esperando que el backend este listo en http://localhost:8001 ...

set BACKEND_READY=0
set INTENTOS=0

:ESPERAR_BACKEND
set /a INTENTOS+=1
if %INTENTOS% GTR 90 (
    echo.
    echo ADVERTENCIA: El backend no respondio en 3 minutos.
    echo Verifica la terminal del backend y luego abre manualmente:
    echo   http://localhost:8501
    pause
    exit /b 1
)

timeout /t 2 >nul

REM Verificar si el backend responde usando PowerShell
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:8001' -UseBasicParsing -TimeoutSec 3; exit 0 } catch { exit 1 }" >nul 2>&1
if errorlevel 1 (
    set /a MOD=!INTENTOS! %% 5
    if !MOD!==0 echo   ... esperando backend (!INTENTOS!/90)
    goto ESPERAR_BACKEND
)

echo [OK] Backend respondiendo!

REM ============================================
REM 4. INICIAR FRONTEND
REM ============================================

echo.
echo [2/2] Abriendo Frontend Streamlit (puerto 8501)...
start "SubVivienda IA - Frontend" cmd /k "title Frontend SubVivienda IA && cd /d ^"%PROJECT_DIR%\frontend^" && call ^"%PROJECT_DIR%\venv\Scripts\activate^" && echo Iniciando Streamlit... && streamlit run app.py --server.port 8501"

REM ============================================
REM 5. ABRIR NAVEGADOR AUTOMATICAMENTE
REM ============================================

echo.
echo Esperando que Streamlit arranque (5 segundos)...
timeout /t 5 >nul

echo Abriendo la app en el navegador...
start http://localhost:8501

echo.
echo ==========================================
echo  APP LISTA
echo ==========================================
echo.
echo  Backend:  http://localhost:8001
echo  Frontend: http://localhost:8501
echo.
echo  NO cierres las terminales mientras usas la app.
echo  Para detener: Ctrl+C en cada terminal.
echo.
pause
exit /b 0
