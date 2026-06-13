import streamlit as st
import pandas as pd
import requests
import time
import random
import os
from dotenv import load_dotenv

from validacion import validar_estructura_y_semantica
from limpieza import limpiar_datos
from transformacion import transformar_datos

# Cargar variables de entorno
load_dotenv()

# Configurar URL según el ambiente
API_KEY = os.getenv("API_KEY", "")
API_URL_DEV = "http://localhost:8001/predecir"
API_URL_PROD = "http://localhost:8001/predecir"  # En Render, ambos están en el mismo contenedor

# Usar la misma URL en ambos ambientes (mismo contenedor)
API_URL = API_URL_PROD

# Headers para autenticación
def obtener_headers():
    """Retorna headers con autenticación si está configurada"""
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers


COLUMNAS_ESPERADAS = [
    'amt', 'gender', 'city_pop', 'unix_time', 'flag_invalid_amt', 'flag_fake_location', 
    'trans_hour', 'trans_day_of_week', 'trans_month', 'age', 'distance_km',
    'category_food_dining', 'category_gas_transport', 'category_grocery_net', 
    'category_grocery_pos', 'category_health_fitness', 'category_home', 
    'category_kids_pets', 'category_misc_net', 'category_misc_pos', 
    'category_personal_care', 'category_shopping_net', 'category_shopping_pos', 
    'category_travel'
]

def alinear_columnas(df_transformado: pd.DataFrame) -> pd.DataFrame:
    """Asegura que el DataFrame tenga exactamente las columnas que el modelo espera."""
    for col in COLUMNAS_ESPERADAS:
        if col not in df_transformado.columns:
            df_transformado[col] = 0
    return df_transformado[COLUMNAS_ESPERADAS]

def convertir_a_dict_valido(fila_series: pd.Series) -> dict:
    """Convierte una fila pandas a dict con tipos correctos para la API."""
    transaccion = fila_series.to_dict()
    
    # Validaciones y conversiones de tipos
    for key, value in transaccion.items():
        # Convertir NaN a 0
        if pd.isna(value):
            transaccion[key] = 0
        # Campos que deben ser INT
        elif key in ['gender', 'city_pop', 'unix_time', 'flag_invalid_amt', 'flag_fake_location', 
                     'trans_hour', 'trans_day_of_week', 'trans_month', 'age',
                     'category_food_dining', 'category_gas_transport', 'category_grocery_net',
                     'category_grocery_pos', 'category_health_fitness', 'category_home',
                     'category_kids_pets', 'category_misc_net', 'category_misc_pos',
                     'category_personal_care', 'category_shopping_net', 'category_shopping_pos',
                     'category_travel']:
            transaccion[key] = int(value)
        # Campos que deben ser FLOAT
        elif key in ['amt', 'distance_km']:
            transaccion[key] = float(value)
    
    st.write(f"DEBUG - Datos enviados: {transaccion}")  # Temporalmente para debug
    return transaccion

st.set_page_config(page_title="Simulador de Fraude", layout="wide")
st.title("Simulador de Detección de Fraude en Tiempo Real")

st.markdown("Sube tu archivo CSV **crudo** (ej. `02_fraudTest.csv`):")
archivo_subido = st.file_uploader("Elige un archivo CSV", type="csv")

if archivo_subido is not None:
    try:
        df_datos = pd.read_csv(archivo_subido)
        st.success(f"Archivo cargado correctamente. {len(df_datos)} transacciones listas para procesar.")
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        iniciar = st.button("▶ Iniciar Streaming y Pipeline", type="primary", use_container_width=True)
    with col2:
        detener = st.button("⏹ Detener", use_container_width=True)

    metricas_placeholder = st.empty()
    tabla_placeholder = st.empty()

    if 'historial' not in st.session_state:
        st.session_state.historial = []
        st.session_state.total_transacciones = 0
        st.session_state.total_fraudes = 0
        st.session_state.total_latencia = 0.0

    if iniciar and not detener:
        for index, fila in df_datos.iterrows():
            if detener:
                break
            
            inicio_latencia = time.time()
            
            # --- 1. PASAMOS LA FILA ÚNICA POR EL PIPELINE ---
            df_evento = pd.DataFrame([fila])
            
            # Paso A: Validación
            df_valido = validar_estructura_y_semantica(df_evento)
            if df_valido.empty:
                st.warning(f"Fila {index} ignorada por validación.")
                continue
                
            # Paso B: Limpieza
            df_limpio = limpiar_datos(df_valido)
            if df_limpio.empty:
                continue
                
            # Paso C: Transformación
            df_trans = transformar_datos(df_limpio)
            
            # Paso D: Alineación
            df_final = alinear_columnas(df_trans)
            
            # Convertir a dict con validación de tipos
            transaccion = convertir_a_dict_valido(df_final.iloc[0])
            
            # --- 2. ENVIAMOS A RENDER ---
            try:
                res = requests.post(
                    API_URL, 
                    json=transaccion, 
                    timeout=30,
                    headers=obtener_headers()
                )
                if res.status_code == 200:
                    data_api = res.json()
                    estado = "🚨 FRAUDE" if data_api["es_fraude"] else "✅ OK"
                    
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
                        "Distancia (km)": round(df_final.iloc[0]['distance_km'], 1),
                        "Latencia (s)": latencia,
                        "Predicción": estado
                    })
                    
                    if len(st.session_state.historial) > 20: # Mostramos solo las últimas 20 en la UI
                        st.session_state.historial.pop()
                else:
                    st.error(f"Error {res.status_code} desde la API.")
                    
            except Exception as e:
                st.error(f"Error conectando a la API: {e}")
                break
                
            # --- 3. ACTUALIZAR DASHBOARD ---
            latencia_promedio = round(st.session_state.total_latencia / st.session_state.total_transacciones, 3) if st.session_state.total_transacciones > 0 else 0
            with metricas_placeholder.container():
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("Transacciones Evaluadas", st.session_state.total_transacciones)
                kpi2.metric("Fraudes Detectados", st.session_state.total_fraudes)
                kpi3.metric("Latencia Promedio (s)", f"{latencia_promedio}s")
                
            with tabla_placeholder.container():
                st.dataframe(pd.DataFrame(st.session_state.historial), use_container_width=True, hide_index=True)
                
            time.sleep(random.uniform(0.1, 0.8)) # Pausa para simular fluidez
else:
    st.info("Esperando archivo CSV crudo...")
