"""
predict.py
Carga el modelo ganador, ejecuta predicciones sobre un dataset
y exporta los resultados a data/exports/predictions.csv para Power BI

Uso:
    python src/predict.py
    python src/predict.py --input data/processed/data_test.py
    python src/predict.py --input  data/raw/CreditScoring.csv --output data/exports/predictions_full.csv
"""

import argparse
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'
EXPORTS_DIR = DATA_DIR / 'exports'
METADATA_PATH = MODELS_DIR / 'model_metadata.json'

DEFAULT_INPUT = DATA_DIR / 'processed' / 'data_test.csv'
DEFAULT_OUTPUT = EXPORTS_DIR / 'predictions.csv'

def cargar_modelo():
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    modelo = joblib.load(MODELS_DIR / metadata['archivo'])
    features = metadata['features']
    umbral = metadata['umbral_optimo']
    nombre = metadata['modelo']

    return modelo, features, umbral

def segmentar_riesgo(prob: float) -> str:
    if prob >= 0.60:
        return 'Alto'
    elif prob >= 0.30:
        return 'Medio'
    else:
        return 'Bajo'

def generar_predicciones(ruta_input: Path, ruta_output: Path):
    modelo, features, umbral = cargar_modelo()

    df = pd.read_csv(ruta_input)

    if 'ID' in df.columns:
        ids = df['ID'].values
        df = df.drop(columns=['ID'])
    else:
        ids = np.arange(1, len(df) + 1)
    
    TARGET = 'SeriousDlqin2yrs'
    if TARGET in df.columns:
        y_real = df[TARGET].values
        X = df[features]
    else:
        y_real = None
        X = df[features]
    
    probs = modelo.predict_proba(X)[:, 1]
    clasificacion = (probs >= umbral).astype(int)
    riesgo = [segmentar_riesgo(p) for p in probs]

    df_resultado = X.copy()
    df_resultado.insert(0, 'cliente_id', ids)
    df_resultado['prob_default'] = probs.round(4)
    df_resultado['clasificacion'] = ['MOROSO' if c == 1 else 'NO MOROSO' for c in clasificacion]
    df_resultado['riesgo'] = riesgo

    if y_real is not None:
        df_resultado['default_real'] = y_real
        df_resultado['acierto'] = (clasificacion == y_real).astype(int)

    df_resultado.rename(columns={
        'cliente_id'                           : 'ID Cliente',
        'RevolvingUtilizationOfUnsecuredLines' : 'Utilización Crédito',
        'age'                                  : 'Edad',
        'NumberOfTime30-59DaysPastDueNotWorse' : 'Atrasos 30-59 días',
        'DebtRatio'                            : 'Ratio de Deuda',
        'MonthlyIncome'                        : 'Ingreso Mensual',
        'NumberOfOpenCreditLinesAndLoans'      : 'Líneas de Crédito',
        'NumberOfTimes90DaysLate'              : 'Atrasos +90 días',
        'NumberRealEstateLoansOrLines'         : 'Préstamos Inmobiliarios',
        'NumberOfDependents'                   : 'Dependientes',
        'prob_default'                         : 'Probabilidad Default',
        'clasificacion'                        : 'Clasificación',
        'riesgo'                               : 'Nivel de Riesgo',
        'default_real'                         : 'Default Real',
        'acierto'                              : 'Acierto del Modelo',
    }, inplace=True)
    
    
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df_resultado.to_csv(ruta_output, index=False)

    return df_resultado

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Genera predicciones de riesgo crediticio')
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT, help='Ruta al CSV de entrada')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='Ruta al CSV de salida')
    args = parser.parse_args()

    generar_predicciones(args.input, args.output)