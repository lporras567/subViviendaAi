from pydantic import BaseModel
from typing import Optional

class PredictRequest(BaseModel):
    departamento: str
    municipio: str
    programa: str
    anio: int
    hogares: int
    valor_cop: float
    modelo: str  # "Logistic Regression" | "Random Forest" | "XGBoost" | "Red MLP"

class ModelResult(BaseModel):
    modelo: str
    probabilidad: float
    clasificacion: str   # "APROBADO" | "EN RIESGO" | "NO RECOMENDADO"
    nivel_riesgo: str    # "bajo" | "medio" | "alto"
    f1_historico: float
    auc_historico: float
    cluster: Optional[int] = None
    cluster_perfil: Optional[str] = None
    cluster_tasa: Optional[float] = None

class PredictResponse(BaseModel):
    modelo_seleccionado: ModelResult
    gradient_boosting:   ModelResult
    diferencia_prob:     float
    interpretacion:      str
