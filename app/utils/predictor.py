"""
predictor.py
Carga el modelo ganador y expone la función de predicción.
Usado tanto por app.py como por shap_explainer.py
"""

import joblib
import json
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / 'models'
METADATA_PATH = MODELS_DIR / 'model_metadata.json'

def cargar_metadata() -> dict:
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def cargar_modelo():
    metadata = cargar_metadata()
    ruta_modelo = MODELS_DIR / metadata['archivo']
    return joblib.load(ruta_modelo)

METADATA = cargar_metadata()
MODELO = cargar_modelo()
FEATURES = METADATA['features']
UMBRAL = METADATA['umbral_optimo']

def predecir(datos_cliente: dict) -> dict:
    """
    Recibe un diccionario con los datos de un cliente,
    retorna probabilidad de default y clasificación
    
    Parámetros
    ----------
    datos_cliente : dict
        Claves = nombres de features, valores = datos del cliente

    Retorna
    -------
    dict con:
        - probabilidad : float (0-1)
        - clasificación : 'MOROSO' | 'NO MOROSO'
        - umbral : float
        - riesgo : 'Alto' | 'Medio' | 'Bajo'
    """
    x = pd.DataFrame([datos_cliente])[FEATURES]

    prob = float(MODELO.predict_proba(x)[0, 1])

    if prob >= UMBRAL:
        clasificacion = 'MOROSO'
    else:
        clasificacion = 'NO MOROSO'

    if prob >= 0.60:
        riesgo = 'Alto'
    elif prob >= 0.30:
        riesgo = 'Medio'
    else:
        riesgo = 'Bajo'

    return {
        'probabilidad'  : round(prob, 4),
        'clasificacion' : clasificacion,
        'umbral'        : UMBRAL,
        'riesgo'        : riesgo,
    }


