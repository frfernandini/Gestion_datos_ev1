import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from urllib.parse import quote
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging
import joblib
import requests
import time
import random

# Importar pipeline local
from validacion import validar_estructura_y_semantica
from limpieza import limpiar_datos
from transformacion import transformar_datos

# [WARNING] LIMITAR CPU
# Establecer threads ANTES de importar numpy/pandas
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Configuración de página
st.set_page_config(
    page_title="Dashboard - Detección de Fraude",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .alert-error {
        background-color: #ffebee;
        padding: 15px;
        border-radius: 5px;
        color: #c62828;
    }
    .alert-success {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 5px;
        color: #2e7d32;
    }
    </style>
""", unsafe_allow_html=True)

# Configuración de la API para el simulador
API_KEY = os.getenv("API_KEY", "")
API_URL = "http://localhost:8001/predecir"

def obtener_headers():
    """Retorna headers con autenticación si está configurada"""
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers

COLUMNAS_SIMULADOR = [
    'amt', 'gender', 'city_pop', 'unix_time', 'flag_invalid_amt', 'flag_fake_location', 
    'trans_hour', 'trans_day_of_week', 'trans_month', 'age', 'distance_km',
    'category_food_dining', 'category_gas_transport', 'category_grocery_net', 
    'category_grocery_pos', 'category_health_fitness', 'category_home', 
    'category_kids_pets', 'category_misc_net', 'category_misc_pos', 
    'category_personal_care', 'category_shopping_net', 'category_shopping_pos', 
    'category_travel'
]

def alinear_columnas_simulador(df_transformado: pd.DataFrame) -> pd.DataFrame:
    for col in COLUMNAS_SIMULADOR:
        if col not in df_transformado.columns:
            df_transformado[col] = 0
    return df_transformado[COLUMNAS_SIMULADOR]

def convertir_a_dict_valido(fila_series: pd.Series) -> dict:
    int_fields = ['gender', 'city_pop', 'unix_time', 'flag_invalid_amt', 'flag_fake_location', 
                  'trans_hour', 'trans_day_of_week', 'trans_month', 'age',
                  'category_food_dining', 'category_gas_transport', 'category_grocery_net',
                  'category_grocery_pos', 'category_health_fitness', 'category_home',
                  'category_kids_pets', 'category_misc_net', 'category_misc_pos',
                  'category_personal_care', 'category_shopping_net', 'category_shopping_pos',
                  'category_travel']
    float_fields = ['amt', 'distance_km']
    transaccion = {}
    for key in fila_series.index:
        value = fila_series[key]
        if pd.isna(value): value = 0
        if key in int_fields: transaccion[key] = int(value)
        elif key in float_fields: transaccion[key] = float(value)
        else: transaccion[key] = value
    return transaccion

# ==========================================
# CONEXIÓN A SUPABASE
# ==========================================


@st.cache_resource
def get_db_connection():
    """Conecta a Supabase PostgreSQL"""
    try:
        DB_USER = os.getenv("DB_USER")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_HOST = os.getenv("DB_HOST")
        DB_PORT = os.getenv("DB_PORT", "5432")
        DB_NAME = os.getenv("DB_NAME", "postgres")
        
        URL_SUPABASE = (
            f"postgresql://"
            f"{quote(DB_USER, safe='')}:"
            f"{quote(DB_PASSWORD, safe='')}@"
            f"{DB_HOST}:{DB_PORT}/{DB_NAME}"
            f"?sslmode=require"
        )
        
        engine = create_engine(URL_SUPABASE)
        logger.info("[OK] Conexión a Supabase exitosa")
        return engine
    except Exception as e:
        logger.error(f"[ERROR] No se pudo conectar a Supabase: {e}")
        return None

# ==========================================
# FUNCIONES DE CARGA DE DATOS
# ==========================================

@st.cache_data(ttl=300)
def cargar_datos_predicciones():
    """Carga las predicciones desde Supabase"""
    engine = get_db_connection()
    if engine is None:
        return pd.DataFrame()
    
    try:
        query = """
            SELECT * FROM transacciones_ml_procesadas 
            ORDER BY unix_time DESC 
            LIMIT 10000
        """
        df = pd.read_sql(query, engine)
        logger.info(f"[OK] Cargadas {len(df)} predicciones desde BD")
        return df
    except Exception as e:
        logger.error(f"[ERROR] Error cargando predicciones: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def cargar_histórico_diario():
    """Carga el histórico diario de predicciones"""
    engine = get_db_connection()
    if engine is None:
        return pd.DataFrame()
    
    try:
        query = """
            SELECT 
                DATE(to_timestamp(unix_time)) as fecha,
                COUNT(*) as total_predicciones,
                SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) as fraudes_detectados
            FROM transacciones_ml_procesadas
            WHERE unix_time IS NOT NULL
            GROUP BY DATE(to_timestamp(unix_time))
            ORDER BY fecha DESC
            LIMIT 30
        """
        df = pd.read_sql(query, engine)
        logger.info(f"[OK] Cargado histórico de {len(df)} días")
        return df
    except Exception as e:
        logger.error(f"[ERROR] Error cargando histórico: {e}")
        return pd.DataFrame()

# ==========================================
# MODELO DE ML
# ==========================================

MODELO_PATH = 'modelo_fraude_base_500k_datos.pkl'

COLUMNAS_MODELO = [
    'amt', 'gender', 'city_pop', 'unix_time', 'flag_invalid_amt', 'flag_fake_location',
    'trans_hour', 'trans_day_of_week', 'trans_month', 'age', 'distance_km',
    'category_food_dining', 'category_gas_transport', 'category_grocery_net',
    'category_grocery_pos', 'category_health_fitness', 'category_home',
    'category_kids_pets', 'category_misc_net', 'category_misc_pos',
    'category_personal_care', 'category_shopping_net', 'category_shopping_pos',
    'category_travel',
]

@st.cache_resource
def cargar_modelo():
    """Carga el modelo entrenado desde disco"""
    if not os.path.exists(MODELO_PATH):
        logger.error(f"[ERROR] Modelo no encontrado en {MODELO_PATH}")
        return None
    try:
        modelo = joblib.load(MODELO_PATH)
        if not hasattr(modelo, 'predict'):
            logger.error("[ERROR] El modelo cargado no tiene método predict")
            return None
        logger.info(f"[OK] Modelo cargado: {type(modelo).__name__}")
        return modelo
    except Exception as e:
        logger.error(f"[ERROR] Error cargando modelo: {e}")
        return None

def preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Alinea columnas del DataFrame con las que espera el modelo"""
    df_features = df.copy()
    for col in COLUMNAS_MODELO:
        if col not in df_features.columns:
            df_features[col] = 0
    return df_features[COLUMNAS_MODELO]

@st.cache_data(ttl=300)
def generar_predicciones(df: pd.DataFrame):
    """Genera predicciones del modelo. Retorna (y_true, y_pred) o (None, None)."""
    if df.empty or 'is_fraud' not in df.columns:
        return None, None

    modelo = cargar_modelo()
    if modelo is None:
        return None, None

    try:
        y_true = df['is_fraud'].astype(int).values
        y_pred = modelo.predict(preparar_features(df)).astype(int)
        return y_true, y_pred
    except Exception as e:
        logger.warning(f"[WARNING] Error generando predicciones: {e}")
        return None, None

# ==========================================
# FUNCIONES DE CÁLCULO DE MÉTRICAS
# ==========================================

def calcular_métricas(df):
    """Calcula accuracy, precision, recall y F1 usando el modelo entrenado"""
    y_true, y_pred = generar_predicciones(df)
    if y_true is None:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "modelo_disponible": False}

    try:
        return {
            "accuracy": accuracy_score(y_true, y_pred) * 100,
            "precision": precision_score(y_true, y_pred, zero_division=0) * 100,
            "recall": recall_score(y_true, y_pred, zero_division=0) * 100,
            "f1": f1_score(y_true, y_pred, zero_division=0) * 100,
            "modelo_disponible": True,
        }
    except Exception as e:
        logger.warning(f"[WARNING] Error calculando métricas: {e}")
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "modelo_disponible": False}

def crear_matriz_confusión(df):
    """Crea matriz de confusión con predicciones del modelo"""
    y_true, y_pred = generar_predicciones(df)
    if y_true is None:
        return None

    try:
        return confusion_matrix(y_true, y_pred, labels=[0, 1])
    except Exception as e:
        logger.warning(f"[WARNING] Error creando matriz: {e}")
        return None

def crear_tabla_errores(df):
    """Filtra transacciones donde el modelo se equivocó"""
    y_true, y_pred = generar_predicciones(df)
    if y_true is None:
        return pd.DataFrame()

    try:
        df_errores = df.copy()
        df_errores['predicción'] = y_pred
        df_errores['es_error'] = df_errores['is_fraud'].astype(int) != df_errores['predicción']

        errores = df_errores[df_errores['es_error']][
            ['amt', 'gender', 'age', 'distance_km', 'is_fraud', 'predicción', 'trans_hour']
        ].head(100)

        return errores
    except Exception as e:
        logger.warning(f"[WARNING] Error creando tabla de errores: {e}")
        return pd.DataFrame()

# ==========================================
# PÁGINA: OVERVIEW
# ==========================================

def page_overview():
    st.title("[DASHBOARD] Detección de Fraude - Overview")
    st.markdown("---")
    
    # Cargar datos
    df = cargar_datos_predicciones()
    
    if df.empty:
        st.error("[ERROR] No hay datos disponibles. Verifica la conexión a Supabase.")
        return
    
    # Calcular métricas
    métricas = calcular_métricas(df)

    if not métricas.get('modelo_disponible'):
        st.warning(
            f"[WARNING] No se pudo cargar el modelo (`{MODELO_PATH}`). "
            "Las métricas no están disponibles."
        )
    
    # KPIs principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("[METRIC] Accuracy", f"{métricas['accuracy']:.2f}%")
    
    with col2:
        st.metric("[METRIC] Precision", f"{métricas['precision']:.2f}%")
    
    with col3:
        st.metric("[METRIC] Recall", f"{métricas['recall']:.2f}%")
    
    with col4:
        st.metric("[METRIC] F1-Score", f"{métricas['f1']:.2f}%")
    
    st.markdown("---")
    
    # Estadísticas generales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("[STAT] Total de Transacciones", f"{len(df):,}")
    
    with col2:
        fraudes_reales = df['is_fraud'].sum() if 'is_fraud' in df.columns else 0
        porcentaje = (fraudes_reales / len(df) * 100) if len(df) > 0 else 0
        st.metric(
            "[STAT] Fraudes Reales (etiqueta)",
            f"{int(fraudes_reales):,}",
            delta=f"{porcentaje:.2f}% del total"
        )
    
    with col3:
        _, y_pred = generar_predicciones(df)
        fraudes_predichos = int(y_pred.sum()) if y_pred is not None else 0
        pct_pred = (fraudes_predichos / len(df) * 100) if len(df) > 0 else 0
        st.metric(
            "[STAT] Fraudes Predichos (modelo)",
            f"{fraudes_predichos:,}",
            delta=f"{pct_pred:.2f}% del total"
        )
    
    st.markdown("---")
    
    # Gráficos en dos columnas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("[CHART] Distribución de Fraude")
        etiquetas_fraude = {0: 'Legítima', 1: 'Fraude'}
        distribución = df['is_fraud'].value_counts().sort_index()
        fig_pie = px.pie(
            values=distribución.values,
            names=[etiquetas_fraude.get(k, str(k)) for k in distribución.index],
            color_discrete_sequence=['#2ecc71', '#e74c3c'],
            hole=0.3
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("[CHART] Distribución por Género")
        if 'gender' in df.columns:
            género = df.groupby('gender').size()
            fig_bar = px.bar(
                x=['Femenino', 'Masculino'],
                y=género.values,
                color_discrete_sequence=['#3498db']
            )
            fig_bar.update_layout(showlegend=False, xaxis_title="Género", yaxis_title="Cantidad")
            st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# PÁGINA: MATRIZ DE CONFUSIÓN
# ==========================================

def page_matriz_confusion():
    st.title("[HEATMAP] Matriz de Confusión")
    st.markdown("Muestra aciertos (diagonal) y errores (fuera de diagonal)")
    st.markdown("---")
    
    df = cargar_datos_predicciones()
    
    if df.empty:
        st.error("[ERROR] No hay datos disponibles.")
        return

    if cargar_modelo() is None:
        st.warning(
            f"[WARNING] No se pudo cargar el modelo (`{MODELO_PATH}`). "
            "No se puede generar la matriz de confusión."
        )
        return
    
    cm = crear_matriz_confusión(df)
    
    if cm is None:
        st.error("[ERROR] No se pudo crear la matriz de confusión.")
        return
    
    # Crear heatmap con Plotly
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=['Legítima (Pred)', 'Fraude (Pred)'],
        y=['Legítima (Real)', 'Fraude (Real)'],
        colorscale='RdYlGn_r',
        showscale=True,
        text=cm,
        texttemplate='%{text}'
    )
    
    fig.update_layout(
        title="Matriz de Confusión - Modelo de Fraude",
        xaxis_title="Predicción",
        yaxis_title="Real",
        height=500,
        width=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Análisis detallado
    st.subheader("[ANALYSIS] Análisis Detallado")
    
    col1, col2 = st.columns(2)
    
    with col1:
        tn, fp, fn, tp = cm.ravel()
        st.write(f"[OK] Verdaderos Negativos (TN): {tn}")
        st.write(f"[WARNING] Falsos Positivos (FP): {fp}")
    
    with col2:
        st.write(f"[WARNING] Falsos Negativos (FN): {fn}")
        st.write(f"[OK] Verdaderos Positivos (TP): {tp}")

# ==========================================
# PÁGINA: TABLA DE ERRORES
# ==========================================

def page_errores():
    st.title("[TABLE] Tabla de Errores")
    st.markdown("Casos donde el modelo se equivocó")
    st.markdown("---")
    
    df = cargar_datos_predicciones()
    
    if df.empty:
        st.error("[ERROR] No hay datos disponibles.")
        return

    if cargar_modelo() is None:
        st.warning(
            f"[WARNING] No se pudo cargar el modelo (`{MODELO_PATH}`). "
            "No se puede generar la tabla de errores."
        )
        return
    
    errores = crear_tabla_errores(df)
    
    if errores.empty:
        st.info("[INFO] No hay errores registrados o datos insuficientes.")
        return
    
    # Filtros
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filtro_monto = st.slider(
            "[FILTER] Monto mínimo",
            float(errores['amt'].min()),
            float(errores['amt'].max()),
            float(errores['amt'].min())
        )
    
    with col2:
        filtro_edad = st.slider(
            "[FILTER] Edad mínima",
            int(errores['age'].min()) if 'age' in errores.columns else 18,
            int(errores['age'].max()) if 'age' in errores.columns else 100,
            int(errores['age'].min()) if 'age' in errores.columns else 18
        )
    
    with col3:
        filtro_distancia = st.slider(
            "[FILTER] Distancia (km)",
            0.0,
            float(errores['distance_km'].max()) if 'distance_km' in errores.columns else 100.0,
            0.0
        )
    
    # Aplicar filtros
    errores_filtrados = errores[
        (errores['amt'] >= filtro_monto) &
        (errores['age'] >= filtro_edad) &
        (errores['distance_km'] >= filtro_distancia)
    ]
    
    st.subheader(f"[RESULTS] {len(errores_filtrados)} Errores encontrados")
    
    # Mostrar tabla
    st.dataframe(
        errores_filtrados,
        use_container_width=True,
        height=400
    )
    
    # Descargar CSV
    csv = errores_filtrados.to_csv(index=False)
    st.download_button(
        label="[DOWNLOAD] Descargar como CSV",
        data=csv,
        file_name="errores_fraude.csv",
        mime="text/csv"
    )

# ==========================================
# PÁGINA: HISTÓRICO
# ==========================================

def page_historico():
    st.title("[CHART] Histórico de Predicciones")
    st.markdown("Evolución del modelo en los últimos 30 días")
    st.markdown("---")
    
    histórico = cargar_histórico_diario()
    
    if histórico.empty:
        st.info("[INFO] No hay datos históricos disponibles.")
        return
    
    histórico = histórico.sort_values('fecha')
    
    # Gráfico de línea: Predicciones por día
    fig_línea = go.Figure()
    
    fig_línea.add_trace(go.Scatter(
        x=histórico['fecha'],
        y=histórico['total_predicciones'],
        mode='lines+markers',
        name='Total Predicciones',
        line=dict(color='#3498db', width=3),
        marker=dict(size=8)
    ))
    
    fig_línea.add_trace(go.Scatter(
        x=histórico['fecha'],
        y=histórico['fraudes_detectados'],
        mode='lines+markers',
        name='Fraudes Detectados',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=8)
    ))
    
    fig_línea.update_layout(
        title="[TREND] Evolución de Predicciones",
        xaxis_title="Fecha",
        yaxis_title="Cantidad",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig_línea, use_container_width=True)
    
    # Estadísticas
    st.subheader("[STATS] Resumen del Período")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "[STAT] Total de Predicciones",
            f"{histórico['total_predicciones'].sum():,}"
        )
    
    with col2:
        st.metric(
            "[STAT] Fraudes Totales",
            f"{histórico['fraudes_detectados'].sum():,}"
        )
    
    with col3:
        promedio_fraudes = histórico['fraudes_detectados'].mean()
        st.metric(
            "[STAT] Promedio Fraudes/Día",
            f"{promedio_fraudes:.0f}"
        )
    
    with col4:
        porcentaje_fraude = (histórico['fraudes_detectados'].sum() / histórico['total_predicciones'].sum() * 100)
        st.metric(
            "[STAT] % Fraude",
            f"{porcentaje_fraude:.2f}%"
        )

# ==========================================
# PÁGINA: VOLUMEN PIPELINE
# ==========================================

def page_volumen():
    st.title("[STATS] Volumen del Pipeline")
    st.markdown("Monitoreo de datos procesados en cada etapa")
    st.markdown("---")
    
    df = cargar_datos_predicciones()
    
    if df.empty:
        st.error("[ERROR] No hay datos disponibles.")
        return
    
    # Simular etapas del pipeline
    etapas = {
        "Ingesta": len(df),
        "Limpieza": int(len(df) * 0.98),  # 2% descartados
        "Transformación": int(len(df) * 0.98),
        "Validación": int(len(df) * 0.97),  # 1% descartado más
        "Carga BD": int(len(df) * 0.97)
    }
    
    # Gráfico de barras
    fig_barras = px.bar(
        x=list(etapas.keys()),
        y=list(etapas.values()),
        color_discrete_sequence=['#2ecc71'],
        text=list(etapas.values())
    )
    
    fig_barras.update_traces(textposition='auto')
    fig_barras.update_layout(
        title="[FLOW] Registros por Etapa del Pipeline",
        xaxis_title="Etapa",
        yaxis_title="Cantidad de Registros",
        showlegend=False,
        height=500
    )
    
    st.plotly_chart(fig_barras, use_container_width=True)
    
    # Tabla detallada
    st.subheader("[TABLE] Detalle de Pérdida de Datos")
    
    pérdidas = []
    registros_anteriores = len(df)
    
    for etapa, registros in etapas.items():
        pérdida = registros_anteriores - registros
        porcentaje_pérdida = (pérdida / registros_anteriores * 100) if registros_anteriores > 0 else 0
        
        pérdidas.append({
            "Etapa": etapa,
            "Registros": registros,
            "Perdidos": pérdida,
            "% Pérdida": f"{porcentaje_pérdida:.2f}%"
        })
        
        registros_anteriores = registros
    
        df_pérdidas = pd.DataFrame(pérdidas)
    st.dataframe(df_pérdidas, use_container_width=True)

# ==========================================
# PÁGINA: SIMULADOR (DEMO)
# ==========================================

def page_simulador():
    st.title("[DEMO] Simulador de Fraude en Tiempo Real")
    st.markdown("Carga un CSV crudo para procesarlo a través del pipeline completo y predecir con la API.")
    st.markdown("---")

    archivo_subido = st.file_uploader("[FILE] Elige un archivo CSV", type="csv")

    if archivo_subido is not None:
        try:
            df_datos = pd.read_csv(archivo_subido)
            st.success(f"[OK] Archivo cargado: {len(df_datos)} transacciones.")
        except Exception as e:
            st.error(f"[ERROR] Al leer el archivo: {e}")
            return

        col1, col2 = st.columns(2)
        with col1:
            iniciar = st.button("▶ Iniciar Streaming", type="primary", use_container_width=True)
        with col2:
            detener = st.button("⏹ Detener", use_container_width=True)

        metricas_placeholder = st.empty()
        tabla_placeholder = st.empty()

        if 'historial' not in st.session_state:
            st.session_state.historial = []
            st.session_state.total_transacciones = 0
            st.session_state.total_fraudes = 0
            st.session_state.total_latencia = 0.0

        if iniciar:
            for index, fila in df_datos.iterrows():
                if detener:
                    st.warning("[INFO] Simulación detenida por el usuario.")
                    break
                
                time.sleep(0.5)
                inicio_latencia = time.time()
                
                # Pipeline local
                df_evento = pd.DataFrame([fila])
                df_valido = validar_estructura_y_semantica(df_evento)
                if df_valido.empty: continue
                    
                df_limpio = limpiar_datos(df_valido)
                if df_limpio.empty: continue
                    
                df_trans = transformar_datos(df_limpio)
                df_final = alinear_columnas_simulador(df_trans)
                transaccion = convertir_a_dict_valido(df_final.iloc[0])
                
                # Enviar a la API local (FastAPI)
                try:
                    res = requests.post(API_URL, json=transaccion, timeout=30, headers=obtener_headers())
                    if res.status_code == 200:
                        data_api = res.json()
                        estado = "🚨 FRAUDE" if data_api["es_fraude"] else "✅ LEGÍTIMA"
                        
                        st.session_state.total_transacciones += 1
                        if data_api["es_fraude"]:
                            st.session_state.total_fraudes += 1
                        
                        fin_latencia = time.time()
                        latencia = round(fin_latencia - inicio_latencia, 3)
                        st.session_state.total_latencia += latencia
                        
                        st.session_state.historial.insert(0, {
                            "ID": index,
                            "Monto ($)": round(fila.get('amt', 0), 2),
                            "Categoría": fila.get('category', 'N/A'),
                            "Estado": estado,
                            "Latencia (s)": latencia
                        })
                        
                        if len(st.session_state.historial) > 10:
                            st.session_state.historial.pop()
                    else:
                        st.error(f"[API ERROR] Status: {res.status_code}")
                        break
                        
                except Exception as e:
                    st.error(f"[CONN ERROR] No se pudo conectar con la API: {e}")
                    break
                    
                # Actualizar UI
                latencia_prom = round(st.session_state.total_latencia / st.session_state.total_transacciones, 3)
                with metricas_placeholder.container():
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Evaluadas", st.session_state.total_transacciones)
                    k2.metric("Fraudes", st.session_state.total_fraudes)
                    k3.metric("Latencia Prom.", f"{latencia_prom}s")
                    
                with tabla_placeholder.container():
                    st.dataframe(pd.DataFrame(st.session_state.historial), use_container_width=True)
                
                time.sleep(random.uniform(0.1, 0.4))

    else:
        st.info("Sube un archivo CSV para comenzar la simulación.")

# ==========================================
# MENU PRINCIPAL
# ==========================================


def main():
    # Sidebar
    st.sidebar.title("[DASHBOARD] Menú Principal")
    st.sidebar.markdown("---")
    
    página = st.sidebar.radio(
    "[SELECT] Selecciona una página:",
    [
    "[HOME] Overview",
        "[HEATMAP] Matriz de Confusión",
        "[TABLE] Tabla de Errores",
        "[TREND] Histórico",
        "[FLOW] Volumen Pipeline",
        "[DEMO] Simulador Streaming"
    ],
    index=0
    )

    
    st.sidebar.markdown("---")
    
    # Información
    st.sidebar.subheader("[INFO] Información")
    st.sidebar.write("**Modelo**: modelo_fraude_base_500k_datos.pkl")
    st.sidebar.write("**Datos**: Supabase PostgreSQL")
    st.sidebar.write("**Actualizado**: Cada 5 minutos")
    
    st.sidebar.markdown("---")
    
    # Botón de refresh
    if st.sidebar.button("[REFRESH] Actualizar Datos"):
        st.cache_data.clear()
        st.rerun()
    
    # Renderizar página
    if página == "[HOME] Overview":
        page_overview()
    elif página == "[HEATMAP] Matriz de Confusión":
        page_matriz_confusion()
    elif página == "[TABLE] Tabla de Errores":
        page_errores()
    elif página == "[TREND] Histórico":
        page_historico()
    elif página == "[FLOW] Volumen Pipeline":
        page_volumen()
    elif página == "[DEMO] Simulador Streaming":
        page_simulador()


if __name__ == "__main__":
    main()
