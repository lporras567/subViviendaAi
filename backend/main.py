"""
SubVivienda IA — Backend FastAPI
Endpoint: POST /predict
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from schemas import PredictRequest, PredictResponse, ModelResult
from model_loader import (load_all, build_features, predict_one,
                           CLUSTER_INFO, METRICAS_HISTORICAS,
                           clasificar, interpretar)

# ── Startup: cargar modelos una sola vez ──────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all()
    yield

app = FastAPI(
    title="SubVivienda IA API",
    description="Sistema Inteligente de Prediccion de Subsidios de Vivienda — MVCT Colombia",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "sistema": "SubVivienda IA",
        "version": "1.0.0",
        "estado": "activo",
        "modelos_disponibles": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
            "Red MLP"
        ],
        "referencia_fija": "XGBoost"
    }


@app.get("/health")
def health():
    from model_loader import _cache
    return {
        "status": "ok",
        "modelos_cargados": [k for k in _cache if k not in ("scaler","led","lem","lep","meta")]
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    cache = load_all()

    # Validar modelo
    modelos_validos = ["Logistic Regression", "Random Forest", "XGBoost", "Red MLP"]
    if req.modelo not in modelos_validos:
        raise HTTPException(
            status_code=400,
            detail=f"Modelo '{req.modelo}' no valido. Opciones: {modelos_validos}"
        )

    # Validar entradas
    meta = cache["meta"]
    if req.departamento not in meta["departamentos"]:
        raise HTTPException(status_code=400, detail=f"Departamento '{req.departamento}' no encontrado")
    if req.municipio not in meta["municipios"]:
        raise HTTPException(status_code=400, detail=f"Municipio '{req.municipio}' no encontrado")
    if req.programa not in meta["programas"]:
        raise HTTPException(status_code=400, detail=f"Programa '{req.programa}' no encontrado")

    # Construir features
    X = build_features(req, cache)

    # Prediccion modelo seleccionado
    prob_sel, _ = predict_one(req.modelo, X, cache)
    clas_sel, riesgo_sel = clasificar(prob_sel)

    # Prediccion XGBoost (referencia fija — modelo de produccion)
    prob_gb, _ = predict_one("XGBoost", X, cache)
    clas_gb, riesgo_gb = clasificar(prob_gb)

    # Cluster del municipio
    cl = meta.get("mpio_cluster", {}).get(req.municipio, None)
    cl_int = int(cl) if cl is not None else None
    cl_info = CLUSTER_INFO.get(cl_int, {})

    # Metricas historicas
    met_sel = METRICAS_HISTORICAS.get(req.modelo, {"F1-Score":0,"AUC-ROC":0})
    met_gb  = METRICAS_HISTORICAS["XGBoost"]

    return PredictResponse(
        modelo_seleccionado=ModelResult(
            modelo        = req.modelo,
            probabilidad  = round(prob_sel, 4),
            clasificacion = clas_sel,
            nivel_riesgo  = riesgo_sel,
            f1_historico  = met_sel["F1-Score"],
            auc_historico = met_sel["AUC-ROC"],
            cluster       = cl_int,
            cluster_perfil= cl_info.get("perfil"),
            cluster_tasa  = cl_info.get("tasa"),
        ),
        gradient_boosting=ModelResult(
            modelo        = "XGBoost",
            probabilidad  = round(prob_gb, 4),
            clasificacion = clas_gb,
            nivel_riesgo  = riesgo_gb,
            f1_historico  = met_gb["F1-Score"],
            auc_historico = met_gb["AUC-ROC"],
            cluster       = cl_int,
            cluster_perfil= cl_info.get("perfil"),
            cluster_tasa  = cl_info.get("tasa"),
        ),
        diferencia_prob = round(abs(prob_sel - prob_gb), 4),
        interpretacion  = interpretar(prob_sel, prob_gb, req.modelo),
    )


@app.get("/metadata")
def metadata():
    cache = load_all()
    meta  = cache["meta"]
    return {
        "departamentos": meta["departamentos"],
        "programas":     meta["programas"],
        "anio_min":      meta["anio_min"],
        "anio_max":      meta["anio_max"],
        "kpis":          meta["kpis"],
        "metricas":      METRICAS_HISTORICAS,
    }


@app.get("/municipios/{departamento}")
def municipios_por_departamento(departamento: str):
    cache = load_all()
    meta  = cache["meta"]
    mpios = meta.get("dept_mpio", {}).get(departamento, [])
    if not mpios:
        raise HTTPException(status_code=404, detail=f"Departamento '{departamento}' no encontrado")
    return {"departamento": departamento, "municipios": mpios}
