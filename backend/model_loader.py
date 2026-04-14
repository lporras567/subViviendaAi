import pickle, json
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent / "models"

_cache = {}

def load_all():
    if _cache:
        return _cache

    with open(BASE/"logistic_regression.pkl","rb") as f: _cache["Logistic Regression"] = pickle.load(f)
    with open(BASE/"random_forest.pkl","rb") as f:       _cache["Random Forest"]       = pickle.load(f)
    with open(BASE/"xgboost.pkl","rb") as f:             _cache["XGBoost"]             = pickle.load(f)
    with open(BASE/"gradient_boosting.pkl","rb") as f:   _cache["Gradient Boosting"]   = pickle.load(f)
    with open(BASE/"scaler.pkl","rb") as f:              _cache["scaler"]              = pickle.load(f)
    with open(BASE/"label_encoders.pkl","rb") as f:
        enc = pickle.load(f)
        _cache["led"] = enc["led"]
        _cache["lem"] = enc["lem"]
        _cache["lep"] = enc["lep"]

    # TF 2.15 en Windows — cargamos en formato .h5 (maxima compatibilidad)
    try:
        from tensorflow.keras.models import load_model
        _cache["Red MLP"] = load_model(str(BASE / "mlp_model.h5"))
    except Exception as e:
        print(f"[WARN] No se pudo cargar MLP: {e}")
        _cache["Red MLP"] = None

    with open(BASE/"meta.json") as f:
        _cache["meta"] = json.load(f)

    print("[INFO] Todos los modelos cargados correctamente.")
    return _cache


def build_features(req, cache):
    """
    Construye el vector de 12 features a partir del request.
    Replica exactamente el pipeline del Colab.
    """
    meta      = cache["meta"]
    led       = cache["led"]
    lem       = cache["lem"]
    lep       = cache["lep"]
    mpio_stats= meta["mpio_stats"]
    prog_stats= meta["prog_stats"]

    # Encoding categórico
    d_enc = int(led.transform([req.departamento])[0]) if req.departamento in led.classes_ else 0
    m_enc = int(lem.transform([req.municipio])[0])    if req.municipio    in lem.classes_ else 0
    p_enc = int(lep.transform([req.programa])[0])     if req.programa     in lep.classes_ else 0

    # Variables históricas del municipio
    ms    = mpio_stats.get(req.municipio, {})
    ha    = float(ms.get("hist_ap", meta["kpis"]["tasa_aprobacion"]))
    hh    = float(ms.get("hist_hh", 2.0))
    nr    = float(ms.get("n_reg",   1.0))

    # Variables históricas del programa
    ps    = prog_stats.get(req.programa, {})
    pa    = float(ps.get("prog_ap", meta["kpis"]["tasa_aprobacion"]))
    ph    = float(ps.get("prog_hh", 2.0))

    # Variables numéricas
    log_h = np.log1p(req.hogares)
    log_v = np.log1p(req.valor_cop)
    vxh   = req.valor_cop / max(req.hogares, 1)

    X = np.array([[d_enc, m_enc, p_enc,
                   req.anio,
                   log_h, log_v, vxh,
                   ha, hh, nr, pa, ph]])
    return X


def predict_one(nombre_modelo, X_raw, cache):
    """
    Ejecuta la predicción para un modelo dado.
    Retorna (probabilidad, prediccion_binaria).
    """
    sc    = cache["scaler"]
    model = cache[nombre_modelo]

    if model is None:
        return 0.5, 0

    # LR y MLP necesitan datos escalados
    if nombre_modelo in ("Logistic Regression", "Red MLP"):
        X = sc.transform(X_raw)
    else:
        X = X_raw

    if nombre_modelo == "Red MLP":
        # TF 2.13: predict() devuelve shape (n,1) — aplanar con flatten()
        prob = float(model.predict(X, verbose=0).flatten()[0])
    else:
        prob = float(model.predict_proba(X)[0][1])

    pred = 1 if prob >= 0.5 else 0
    return prob, pred


CLUSTER_INFO = {
    0: {"perfil": "Alta eficiencia rural",   "tasa": 0.930, "accion": "Priorizar para escalar"},
    1: {"perfil": "Rendimiento medio-alto",  "tasa": 0.812, "accion": "Candidato a mayor volumen"},
    2: {"perfil": "Alto volumen urbano",      "tasa": 0.781, "accion": "Monitoreo continuo"},
    3: {"perfil": "Bajo rendimiento",         "tasa": 0.584, "accion": "Intervencion previa obligatoria"},
    4: {"perfil": "Rendimiento alto",         "tasa": 0.837, "accion": "Asignaciones directas confiables"},
}

METRICAS_HISTORICAS = {
    "Logistic Regression": {"F1-Score": 0.8637, "AUC-ROC": 0.8623},
    "Random Forest":       {"F1-Score": 0.9100, "AUC-ROC": 0.8750},  # 20 arboles para deploy
    "XGBoost":             {"F1-Score": 0.9274, "AUC-ROC": 0.9110},
    "Gradient Boosting":   {"F1-Score": 0.9266, "AUC-ROC": 0.9072},
    "Red MLP":             {"F1-Score": 0.9214, "AUC-ROC": 0.8953},
}

def clasificar(prob):
    if prob >= 0.70: return "APROBADO",       "bajo"
    if prob >= 0.45: return "EN RIESGO",      "medio"
    return              "NO RECOMENDADO", "alto"

def interpretar(prob_sel, prob_xgb, nombre_sel):
    diff = prob_sel - prob_xgb
    if abs(diff) < 0.03:
        return f"{nombre_sel} y XGBoost coinciden en la evaluacion — señal robusta."
    elif diff > 0:
        return f"{nombre_sel} es mas optimista que XGBoost en {diff*100:.1f} pp."
    else:
        return f"XGBoost es mas optimista que {nombre_sel} en {abs(diff)*100:.1f} pp — confie en el modelo de produccion."
