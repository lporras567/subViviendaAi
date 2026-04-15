#!/bin/bash

# Directorio donde vive este script
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo " ============================================"
echo "  SubVivienda IA"
echo " ============================================"
echo ""
echo " Directorio: $PROJECT_DIR"
echo ""

# ============================================
# Funcion para abrir una terminal nueva
# ============================================

abrir_terminal() {
    local TITULO="$1"
    local COMANDO="$2"

    if [[ "$OSTYPE" == "darwin"* ]]; then
        # Mac
        osascript -e "tell application \"Terminal\" to do script \"$COMANDO\""
    elif command -v gnome-terminal &>/dev/null; then
        gnome-terminal --title="$TITULO" -- bash -c "$COMANDO; exec bash"
    elif command -v xterm &>/dev/null; then
        xterm -title "$TITULO" -e bash -c "$COMANDO; exec bash" &
    elif command -v konsole &>/dev/null; then
        konsole --title "$TITULO" -e bash -c "$COMANDO; exec bash" &
    else
        echo "[ERROR] No se encontro un emulador de terminal compatible."
        echo "        Instala gnome-terminal o xterm e intenta de nuevo."
        exit 1
    fi
}

# ============================================
# 1. VERIFICAR PYTHON
# ============================================

echo "[PASO 1] Verificando Python..."
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python3 no encontrado. Instalalo desde python.org"
    exit 1
fi
PYTHON_VER=$(python3 --version 2>&1 | awk '{print $2}')
echo "[OK] Python $PYTHON_VER"

# ============================================
# 2. SETUP (solo si es necesario)
# ============================================

echo "[PASO 2] Verificando entorno virtual..."

# Verificar si el venv existe Y funciona en este sistema operativo
VENV_OK=false
if [ -f "$PROJECT_DIR/venv/bin/activate" ]; then
    if "$PROJECT_DIR/venv/bin/python3" --version &>/dev/null; then
        VENV_OK=true
        echo "[OK] Entorno virtual encontrado. Saltando instalacion."
    else
        echo "[AVISO] Entorno virtual incompatible (probablemente copiado de otro SO)."
        echo "        Eliminando y recreando para este sistema..."
        rm -rf "$PROJECT_DIR/venv"
    fi
fi

if [ "$VENV_OK" = false ]; then
    echo "[SETUP] Creando entorno virtual por primera vez..."
    python3 -m venv "$PROJECT_DIR/venv"
    if [ $? -ne 0 ]; then
        echo "[ERROR] No se pudo crear el entorno virtual."
        exit 1
    fi

    echo "[SETUP] Instalando dependencias..."
    "$PROJECT_DIR/venv/bin/pip" install --upgrade pip --quiet
    "$PROJECT_DIR/venv/bin/pip" install -r "$PROJECT_DIR/requirements.txt"
    if [ $? -ne 0 ]; then
        echo "[ERROR] Fallo la instalacion de dependencias."
        exit 1
    fi
    echo "[OK] Entorno listo."

    # ============================================
    # 3. ENTRENAR (solo 1ra vez)
    # ============================================

    if [ ! -f "$PROJECT_DIR/data/Subsidios_De_Vivienda.xlsx" ]; then
        echo "[ERROR] No se encuentra data/Subsidios_De_Vivienda.xlsx"
        exit 1
    fi

    echo "[ENTRENAMIENTO] Entrenando modelos..."
    "$PROJECT_DIR/venv/bin/python3" "$PROJECT_DIR/train_and_save.py"
    if [ $? -ne 0 ]; then
        echo "[ERROR] Entrenamiento fallo."
        exit 1
    fi
    echo "[OK] Entrenamiento completado."
fi

# ============================================
# 4. INICIAR BACKEND
# ============================================

echo ""
echo "[PASO 3] Iniciando backend..."
abrir_terminal "Backend SubVivienda IA" \
    "cd '$PROJECT_DIR/backend' && source '$PROJECT_DIR/venv/bin/activate' && PYTHONPATH='$PROJECT_DIR/backend' python3 -m uvicorn main:app --reload --host 127.0.0.1 --port 8001"

# ============================================
# 5. ESPERAR BACKEND
# ============================================

echo ""
echo "[PASO 4] Esperando 25 segundos para que el backend cargue..."
sleep 25

# ============================================
# 6. INICIAR FRONTEND
# ============================================

echo ""
echo "[PASO 5] Iniciando frontend..."
abrir_terminal "Frontend SubVivienda IA" \
    "cd '$PROJECT_DIR/frontend' && source '$PROJECT_DIR/venv/bin/activate' && streamlit run app.py --server.port 8501"

echo ""
echo " ============================================"
echo "  App lista!"
echo " ============================================"
echo ""
echo "  Backend:   http://localhost:8001"
echo "  Frontend:  http://localhost:8501"
echo ""
echo "  NO cierres las terminales del backend y frontend."
echo ""
