# SubVivienda IA 🏠
**Sistema Inteligente de Prediccion de Subsidios de Vivienda**
MVCT Colombia · Dataset v2.0 · 84,680 registros · 2003-2025

---

## Estructura del proyecto

```
subvivienda_ia/
├── backend/
│   ├── main.py              # FastAPI — endpoint POST /predict
│   ├── model_loader.py      # Carga modelos al iniciar
│   ├── schemas.py           # Estructuras request/response
│   └── requirements.txt     # Dependencias del backend
├── frontend/
│   ├── app.py               # Streamlit completo
│   └── requirements.txt     # Dependencias del frontend
├── models/                  # Generados por train_and_save.py
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl    # 20 arboles (deploy) / 100 arboles (local)
│   ├── xgboost.pkl          # Modelo de produccion
│   ├── gradient_boosting.pkl
│   ├── mlp_model.h5
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── meta.json
├── data/
│   └── Subsidios_De_Vivienda.xlsx  (NO incluido en repo)
├── train_and_save.py
├── render.yaml              # Configuracion para Render
├── .gitignore
└── README.md
```

---

## Deploy en la nube (gratuito)

### Backend → Render.com

1. Crear cuenta en https://render.com
2. New → Web Service → conectar repositorio GitHub
3. Configurar:
   - **Root Directory:** `backend`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Runtime:** Python 3.10
4. Deploy → copiar la URL asignada (ej: `https://subvivienda-ia-api.onrender.com`)

### Frontend → Streamlit Cloud

1. Crear cuenta en https://share.streamlit.io
2. New app → conectar repositorio GitHub
3. Configurar:
   - **Branch:** main
   - **Main file path:** `frontend/app.py`
4. Antes del deploy: actualizar `API_URL` en `frontend/app.py`:
   ```python
   API_URL = "https://subvivienda-ia-api.onrender.com"
   ```

---

## Ejecucion local

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_and_save.py

:: Terminal 1
cd backend && uvicorn main:app --reload --port 8001

:: Terminal 2
cd frontend && streamlit run app.py
```

---

## Modelos

| Modelo | F1 | AUC | Rol |
|--------|-----|-----|-----|
| Logistic Regression | 0.8637 | 0.8623 | Comparacion |
| Random Forest | ~0.91 | ~0.875 | Comparacion (20 arboles en deploy) |
| **XGBoost** | **0.9274** | **0.9110** | **Produccion / Referencia fija** |
| Gradient Boosting | 0.9266 | 0.9072 | Comparacion |
| Red MLP | 0.9215 | 0.8973 | Comparacion |
