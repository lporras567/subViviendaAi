@echo off
setlocal

echo.
echo  ==========================================
echo   SubVivienda IA - Setup del proyecto
echo  ==========================================
echo.
echo  Este script:
echo  1. Crea el entorno virtual
echo  2. Instala todas las dependencias
echo.
echo  REQUISITO: Pon Subsidios_De_Vivienda.xlsx
echo             en la carpeta data\ antes de
echo             ejecutar train_and_save.py
echo.

set /p CONFIRM=" Continuar? (S/N): "
if /i not "%CONFIRM%"=="S" (
    echo Cancelado.
    pause
    exit /b 0
)

echo.
echo [1/3] Creando entorno virtual...
python -m venv venv
if errorlevel 1 (
    echo ERROR: No se pudo crear el entorno virtual.
    echo Verifica que Python esta instalado correctamente.
    pause
    exit /b 1
)
echo [OK] Entorno virtual creado

echo.
echo [2/3] Activando entorno virtual...
call venv\Scripts\activate
echo [OK] Entorno virtual activo

echo.
echo [3/3] Instalando dependencias...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Fallo la instalacion de dependencias.
    pause
    exit /b 1
)
echo [OK] Dependencias instaladas

echo.
echo  ==========================================
echo   Instalacion completada exitosamente
echo  ==========================================
echo.
echo  Proximos pasos:
echo.
echo  1. Copia Subsidios_De_Vivienda.xlsx en: data\
echo.
echo  2. Entrena los modelos (una sola vez):
echo     venv\Scripts\activate
echo     python train_and_save.py
echo.
echo  3. Terminal 1 - Backend:
echo     venv\Scripts\activate
echo     cd backend
echo     uvicorn main:app --reload --port 8000
echo.
echo  4. Terminal 2 - Frontend:
echo     venv\Scripts\activate
echo     cd frontend
echo     streamlit run app.py
echo.
pause
