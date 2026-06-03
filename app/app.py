"""
app.py
Simulador de riesgo crediticio — Credit Scoring
Entrada: datos del cliente → Salida: probabilidad de default → explicación SHAP
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from utils.predictor import predecir, METADATA, FEATURES, UMBRAL
from utils.shap_explainer import calcular_shap, generar_waterfall


st.set_page_config(
    page_title='Credit Scoring',
    page_icon='💳',
    layout='wide'
)

COLORES_RIESGO = {
    'Alto' : '#E55C5C',
    'Medio' : '#F0A500',
    'Bajo' : '#27ae60'
}

st.title('💳 Simulador de Riesgo Crediticio')
st.markdown(
    f'**Modelo:** {METADATA["modelo"]} · '
    f'**AUC-ROC:** {METADATA["auc_roc"]} · '
    f'**Recall:** {METADATA["recall"]} · '
    f'**Umbral:** {UMBRAL}'
)
st.divider()

st.sidebar.header('Datos del cliente')
st.sidebar.markdown('Ingrese los datos para evaluar el riesgo crediticio.')

with st.sidebar:
    utilizacion = st.slider(
        'Utilización de crédito revolving (0-1)',
        min_value=0.0, max_value=1.0, value=0.3, step=0.01,
        help='Porcentaje del crédito disponible que está siendo usado'
    )
    edad = st.number_input(
        'Edad', min_value=18, max_value=100, value=40,
        help='Edad del cliente en años'
    )
    atrasos_30_59 = st.number_input(
        'Atrasos 30-59 días (últimos 2 años)',
        min_value=0, max_value=20, value=0
    )
    debt_ratio = st.slider(
        'Ratio de deuda (DebtRatio)',
        min_value=0.0, max_value=5.0, value=0.3, step=0.01,
        help='Pagos mensuales de deuda / ingreso mensual'
    )
    ingreso = st.number_input(
        'Ingreso mensual (USD)',
        min_value=0, max_value=100000, value=5400, step=100
    )
    lineas_credito = st.number_input(
        'Líneas de crédito abiertas',
        min_value=0, max_value=50, value=5
    )
    atrasos_90 = st.number_input(
        'Veces con 90+ días de atraso',
        min_value=0, max_value=20, value=0
    )
    prestamos_inmuebles = st.number_input(
        'Préstamos inmobiliarios',
        min_value=0, max_value=20, value=1
    )
    dependientes = st.number_input(
        'Número de dependientes',
        min_value=0, max_value=20, value=0
    )

    evaluar = st.button('Evaluar cliente', use_container_width=True, type='primary')

datos_cliente = {
    'RevolvingUtilizationOfUnsecuredLines'  : utilizacion,
    'age'                                   : edad,
    'NumberOfTime30-59DaysPastDueNotWorse'  : atrasos_30_59,
    'DebtRatio'                             : debt_ratio,
    'MonthlyIncome'                         : ingreso,
    'NumberOfOpenCreditLinesAndLoans'       : lineas_credito,
    'NumberOfTimes90DaysLate'               : atrasos_90,
    'NumberRealEstateLoansOrLines'          : prestamos_inmuebles,
    'NumberOfDependents'                    : dependientes,
}

if evaluar:
    with st.spinner('Calculando riesgo...'):
        resultado = predecir(datos_cliente)
        shap_info = calcular_shap(datos_cliente)
        fig_wf = generar_waterfall(datos_cliente, resultado['probabilidad'])
    
    prob = resultado['probabilidad']
    clasif = resultado['clasificacion']
    nivel_riesgo = resultado['riesgo']
    color_riesgo = COLORES_RIESGO[nivel_riesgo]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric('Probabilidad de default', f'{prob:.2%}')
    with col2:
        st.markdown(
            f'<div style="text-align:center">'
            f'<p style="font-size:14px;color:gray;margin-bottom:4px">Clasificación</p>'
            f'<p style="font-size:28px;font-weight:bold;color:{color_riesgo}">{clasif}</p>'
            f'</div>',
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f'<div style="text-align:center">'
            f'<p style="font-size:14px;color:gray;margin-bottom:4px">Nivel de riesgo</p>'
            f'<p style="font-size:28px;font-weight:bold;color:{color_riesgo}">{nivel_riesgo}</p>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    st.progress(min(prob, 1.0))
    st.caption(f'Umbral de clasificación: {UMBRAL} — por encima se clasifica como moroso')

    col_izq, col_der = st.columns([1.2, 1])

    with col_izq:
        st.subheader('Explicación SHAP')
        st.caption('Cada barra muestra cuánto empuja cada variable la predicción hacia moroso (rojo) o no moroso (azul)')
        st.pyplot(fig_wf, width='stretch')

    with col_der:
        st.subheader('Factores que aumentan el riesgo')
        for feature, val in shap_info['top_positivos']:
            valor_real = datos_cliente[feature]
            st.markdown(
                f'**{feature}**'
                f'Valor: `{valor_real}` · Impacto SHAP: `+{val:.4f}`'
            )

        st.divider()

        st.subheader('Factores que reducen el riesgo')
        for feature, val in shap_info['top_negativos']:
            valor_real = datos_cliente[feature]
            st.markdown(
                f'**{feature}** \n'
                f'Valor: `{valor_real}` · Impacto SHAP: `{val:.4f}`'
            )

    st.divider()

    with st.expander('Ver tabla completa de contribuciones SHAP'):
        df_shap =pd.DataFrame(
            shap_info['shap_values'].items(),
            columns=['Variable', 'Contribución SHAP']
        ).sort_values('Contribución SHAP', key=abs, ascending=False)
        df_shap['Valor del cliente'] = df_shap['Variable'].map(datos_cliente)
        df_shap['Dirección'] = df_shap['Contribución SHAP'].apply(
            lambda x: '↑ Aumenta riesgo' if x > 0 else '↓ Reduce riesgo'
        )
        st.dataframe(df_shap, width='stretch', hide_index=True)
else:
    st.info('Ingresa los datos del cliente en el panel izquierdo y presiona **Evaluar cliente**')

    st.markdown("""
    ### ¿Cómo funciona esta herramienta?
    
    1. **Ingresa los datos** del cliente en el panel lateral
    2. El modelo calula la **probabilidad de que el cliente incurra en 90+ días de morosidad** en los próximos 2 años
    3. Se muestra la **clasificación** (moroso / no moroso) según el umbral óptimo
    4. Los gráficos **SHAP** explican qué variables llevaron al modelo a esa decisión

    ### Sobre el modelo
    | Parámetro | Valor |
    |---|---|
    | Algoritmo | {modelo} |
    | AUC-ROC | {auc} |
    | Recall (morosos detectados) | {recall} |
    | Umbral de clasificación | {umbral} |
    | Dataset de entrenamiento | 150,000 clientes |
    """.format(
        modelo = METADATA['modelo'],
        auc    = METADATA['auc_roc'],
        recall = METADATA['recall'],
        umbral = UMBRAL                
    ))
