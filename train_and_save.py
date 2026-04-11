"""
SubVivienda IA — Script de entrenamiento y serialización de modelos
Ejecutar UNA sola vez para generar los archivos en models/
"""
import os, pickle, json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)

print("=" * 60)
print("SubVivienda IA — Entrenamiento de modelos")
print("=" * 60)

# ── 1. CARGA Y LIMPIEZA ────────────────────────────────────────────
print("\n[1/6] Cargando y limpiando dataset...")
df = pd.read_excel('data/Subsidios_De_Vivienda.xlsx')
df.columns = ['Departamento','Cod_Dpto','Municipio','Cod_Mpio','Programa',
              'Anio_raw','Anio_real','Estado','Hogares','Valor_Asignado','Extra']
df = df.drop(columns=['Extra'])

df['Anio_int'] = df['Anio_real'].round().astype(int)
df['Hogares']  = pd.to_numeric(df['Hogares'], errors='coerce').fillna(0).astype(int)
df['Valor_Num']= pd.to_numeric(
    df['Valor_Asignado'].astype(str).str.replace('[$,]','',regex=True),
    errors='coerce').fillna(0)
df['Aprobado'] = df['Estado'].str.upper().apply(lambda x: 1 if 'ASIGNAD' in x else 0)
df['Log_H']    = np.log1p(df['Hogares'])
df['Log_V']    = np.log1p(df['Valor_Num'])
df['VxH']      = np.where(df['Hogares']>0, df['Valor_Num']/df['Hogares'], 0)
print(f"   Registros: {len(df):,} | Municipios: {df['Municipio'].nunique()} | Tasa aprobacion: {df['Aprobado'].mean()*100:.1f}%")

# ── 2. FEATURE ENGINEERING + ENCODING ─────────────────────────────
print("\n[2/6] Feature engineering y encoding...")
ms = df.groupby('Municipio').agg(
    hist_ap=('Aprobado','mean'), hist_hh=('Hogares','mean'),
    n_reg=('Aprobado','count')).reset_index()
ps = df.groupby('Programa').agg(
    prog_ap=('Aprobado','mean'), prog_hh=('Hogares','mean')).reset_index()
df2 = df.merge(ms, on='Municipio', how='left').merge(ps, on='Programa', how='left')

led=LabelEncoder(); lem=LabelEncoder(); lep=LabelEncoder()
df2['d_enc'] = led.fit_transform(df2['Departamento'])
df2['m_enc'] = lem.fit_transform(df2['Municipio'])
df2['p_enc'] = lep.fit_transform(df2['Programa'])

FEAT = ['d_enc','m_enc','p_enc','Anio_int','Log_H','Log_V','VxH',
        'hist_ap','hist_hh','n_reg','prog_ap','prog_hh']
dm = df2[FEAT+['Aprobado']].dropna()
X  = dm[FEAT]; y = dm['Aprobado']
X_tr,X_te,y_tr,y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = MinMaxScaler(); sc.fit(X_tr)
X_tr_sc = sc.transform(X_tr)
X_te_sc = sc.transform(X_te)

# Guardar listas de categorias para el predictor
meta = {
    'departamentos': sorted(df['Departamento'].unique().tolist()),
    'municipios':    sorted(df['Municipio'].unique().tolist()),
    'programas':     sorted(df['Programa'].unique().tolist()),
    'anio_min':      int(df['Anio_int'].min()),
    'anio_max':      int(df['Anio_int'].max()),
    'hogares_max':   int(df['Hogares'].max()),
    'valor_max':     float(df['Valor_Num'].max()),
    'FEAT':          FEAT,
    # Stats para lookup en predictor
    'mpio_stats':    ms.set_index('Municipio').to_dict(orient='index'),
    'prog_stats':    ps.set_index('Programa').to_dict(orient='index'),
    'dept_mpio':     df.groupby('Departamento')['Municipio'].apply(lambda x: sorted(x.unique().tolist())).to_dict(),
    # Metricas reales del colab ejecutado
    'metricas': {
        'Logistic Regression': {'Accuracy':0.7947,'Precision':0.9354,'Recall':0.8022,'F1-Score':0.8637,'AUC-ROC':0.8623},
        'Random Forest':       {'Accuracy':0.8725,'Precision':0.9057,'Recall':0.9406,'F1-Score':0.9228,'AUC-ROC':0.8897},
        'XGBoost':             {'Accuracy':0.8791,'Precision':0.9034,'Recall':0.9528,'F1-Score':0.9274,'AUC-ROC':0.9110},
        'Gradient Boosting':   {'Accuracy':0.8772,'Precision':0.8988,'Recall':0.9563,'F1-Score':0.9266,'AUC-ROC':0.9072},
        'Red MLP':             {'Accuracy':0.8686,'Precision':None,  'Recall':None,  'F1-Score':0.9214,'AUC-ROC':0.8953},
    },
    'kpis': {
        'registros': 84680, 'municipios': 984, 'departamentos': 33,
        'tasa_aprobacion': 0.811, 'tasa_noejec': 0.189,
        'hogares_totales': 927512,
        'valor_total_b': 19.92, 'valor_perdido_mm': 890,
        'programas': 25, 'estados': 32,
    }
}

with open('models/meta.json','w') as f:
    json.dump(meta, f, ensure_ascii=False, default=str)
print("   Meta guardado")

# ── 3. MODELOS ML ──────────────────────────────────────────────────
print("\n[3/6] Entrenando modelos ML...")
resultados = {}

# Logistic Regression — ESCALADO
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_tr_sc, y_tr)
yp = lr.predict(X_te_sc); ypr = lr.predict_proba(X_te_sc)[:,1]
resultados['Logistic Regression'] = {'Accuracy':accuracy_score(y_te,yp),'F1-Score':f1_score(y_te,yp),'AUC-ROC':roc_auc_score(y_te,ypr)}
print(f"   LR          F1={resultados['Logistic Regression']['F1-Score']:.4f} | AUC={resultados['Logistic Regression']['AUC-ROC']:.4f}")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
yp = rf.predict(X_te); ypr = rf.predict_proba(X_te)[:,1]
resultados['Random Forest'] = {'Accuracy':accuracy_score(y_te,yp),'F1-Score':f1_score(y_te,yp),'AUC-ROC':roc_auc_score(y_te,ypr)}
print(f"   RF          F1={resultados['Random Forest']['F1-Score']:.4f} | AUC={resultados['Random Forest']['AUC-ROC']:.4f}")

# XGBoost
xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0)
xgb.fit(X_tr, y_tr)
yp = xgb.predict(X_te); ypr = xgb.predict_proba(X_te)[:,1]
resultados['XGBoost'] = {'Accuracy':accuracy_score(y_te,yp),'F1-Score':f1_score(y_te,yp),'AUC-ROC':roc_auc_score(y_te,ypr)}
print(f"   XGBoost     F1={resultados['XGBoost']['F1-Score']:.4f} | AUC={resultados['XGBoost']['AUC-ROC']:.4f}")

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
gb.fit(X_tr, y_tr)
yp = gb.predict(X_te); ypr = gb.predict_proba(X_te)[:,1]
resultados['Gradient Boosting'] = {'Accuracy':accuracy_score(y_te,yp),'F1-Score':f1_score(y_te,yp),'AUC-ROC':roc_auc_score(y_te,ypr)}
print(f"   GB          F1={resultados['Gradient Boosting']['F1-Score']:.4f} | AUC={resultados['Gradient Boosting']['AUC-ROC']:.4f}")

# ── 4. RED MLP ──────────────────────────────────────────────────────
print("\n[4/6] Entrenando Red MLP...")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

mlp = Sequential([
    Dense(128, activation='relu', input_shape=(len(FEAT),)),
    BatchNormalization(), Dropout(0.3),
    Dense(64,  activation='relu'), Dropout(0.2),
    Dense(32,  activation='relu'), Dropout(0.2),
    Dense(1,   activation='sigmoid')
])
mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
mlp.fit(X_tr_sc, y_tr, validation_split=0.15,
        epochs=50, batch_size=512, callbacks=[es], verbose=0)
# TF 2.13 en Windows: predict() devuelve shape (n,1) — aplanar con flatten()
pred_raw = mlp.predict(X_te_sc, verbose=0).flatten()
yp_m  = (pred_raw > 0.5).astype(int)
ypr_m = pred_raw
resultados['Red MLP'] = {'Accuracy':accuracy_score(y_te,yp_m),'F1-Score':f1_score(y_te,yp_m),'AUC-ROC':roc_auc_score(y_te,ypr_m)}
print(f"   MLP         F1={resultados['Red MLP']['F1-Score']:.4f} | AUC={resultados['Red MLP']['AUC-ROC']:.4f}")

# ── 5. CLUSTERING ──────────────────────────────────────────────────
print("\n[5/6] Clustering K-Means...")
mp = df2.groupby('Municipio').agg(
    tasa=('Aprobado','mean'), hh_mean=('Hogares','mean'),
    val_mean=('Valor_Num','mean'), n_reg=('Aprobado','count'),
    n_prog=('Programa','nunique')).reset_index()
for c in ['hh_mean','val_mean','n_reg']:
    mp[c+'_n'] = mp[c]/(mp[c].max()+1e-8)
Xcl = mp[['tasa','hh_mean_n','val_mean_n','n_reg_n','n_prog']].fillna(0)
Xcl_s = StandardScaler().fit_transform(Xcl)
km5 = KMeans(n_clusters=5, random_state=42, n_init=10)
mp['cluster'] = km5.fit_predict(Xcl_s)
mpio_cluster = mp.set_index('Municipio')['cluster'].to_dict()
meta['mpio_cluster'] = {k: int(v) for k,v in mpio_cluster.items()}
with open('models/meta.json','w') as f:
    json.dump(meta, f, ensure_ascii=False, default=str)
print(f"   K-Means k=5 completado")

# ── 6. SERIALIZAR TODO ─────────────────────────────────────────────
print("\n[6/6] Guardando modelos...")

with open('models/logistic_regression.pkl','wb') as f: pickle.dump(lr, f)
with open('models/random_forest.pkl','wb') as f:       pickle.dump(rf, f)
with open('models/xgboost.pkl','wb') as f:             pickle.dump(xgb, f)
with open('models/gradient_boosting.pkl','wb') as f:   pickle.dump(gb, f)
with open('models/scaler.pkl','wb') as f:              pickle.dump(sc, f)
with open('models/label_encoders.pkl','wb') as f:
    pickle.dump({'led':led,'lem':lem,'lep':lep}, f)
# TF 2.15 soporta tanto .keras como .h5
# Usamos .h5 para maxima compatibilidad con Windows
mlp.save('models/mlp_model.h5')

print("\n" + "=" * 60)
print("Modelos guardados exitosamente en models/")
print("=" * 60)
for nombre, res in resultados.items():
    print(f"  {nombre:<22} F1={res['F1-Score']:.4f} | AUC={res['AUC-ROC']:.4f}")
print("\nListo para ejecutar la aplicacion.")
