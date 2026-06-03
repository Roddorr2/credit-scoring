"""
shap_explainer.py
Genera los valores SHAP y el gráfico waterfall para un cliente individual.
Usado por app.py para mostrar la explicación de cada predicción.
"""

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from utils.predictor import MODELO, FEATURES

EXPLAINER = shap.TreeExplainer(MODELO)

def _extraer_sv(shap_raw) -> np.ndarray:
    """Normaliza la salida de shap_values a array 1D para un solo cliente."""
    if isinstance(shap_raw, list):
        return np.array(shap_raw[1][0], dtype=float)
    elif np.array(shap_raw).ndim == 3:
        return np.array(shap_raw)[0, :, 1].astype(float)
    else:
        return np.array(shap_raw)[0].astype(float)


def _extraer_base_value() -> float:
    """Extrae el base_value como escalar float."""
    ev = EXPLAINER.expected_value
    if isinstance(ev, (list, np.ndarray)):
        return float(np.array(ev).flat[1])
    return float(ev)

def calcular_shap(datos_cliente: dict) -> dict:
    """
    Calcula los valores SHAP para un cliente individual.

    Retorna
    -------
    dict con:
        - shap_values   : dict {feature: valor_shap}
        - base_value    : float (predicción base del modelo)
        - top_positivos : list de (feature, shap) que más empujan hacia moroso
        - top_negativos : list de (feature, shap) que más empujan hacia no moroso
    """
    X = pd.DataFrame([datos_cliente])[FEATURES]

    shap_raw   = EXPLAINER.shap_values(X)
    sv         = _extraer_sv(shap_raw)
    base_value = _extraer_base_value()

    shap_dict = dict(zip(FEATURES, sv.tolist()))

    top_pos = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)[:3]
    top_neg = sorted(shap_dict.items(), key=lambda x: x[1])[:3]

    return {
        'shap_values'   : shap_dict,
        'base_value'    : base_value,
        'top_positivos' : top_pos,
        'top_negativos' : top_neg,
    }

def generar_waterfall(datos_cliente: dict, prob: float) -> plt.Figure:
    """
    Genera el gráfico waterfall SHAP para un cliente.
    Retorna un objeto Figure de matplotlib listo para st.pyplot().
    """
    X = pd.DataFrame([datos_cliente])[FEATURES]

    shap_raw   = EXPLAINER.shap_values(X)
    sv         = _extraer_sv(shap_raw)
    base_value = _extraer_base_value()

    explanation = shap.Explanation(
        values        = sv,
        base_values   = base_value,
        data          = X.iloc[0].values.astype(float),
        feature_names = FEATURES
    )

    fig, ax = plt.subplots(figsize=(11, 7))
    shap.plots.waterfall(explanation, show=False)
    plt.title(
        f'Explicación SHAP — P(default) = {prob:.2%}',
        fontsize=11, fontweight='bold', pad=12
    )
    plt.tight_layout()
    return fig