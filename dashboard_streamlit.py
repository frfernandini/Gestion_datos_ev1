import streamlit as st
import pandas as pd
import requests
import time

# URL de tu API original que procesalos datos
API_URL = "https://gestion-datos-ev1.onrender.com/predecir"

st.set_page_config(page_title="Simulador de Fraude Streaming", layout="wide")

st.title("🛡️ Simulador de Detección de Fraude en Tiempo Real")
st.markdown("Este panel lee eventos transaccionales, aplica el pipeline de ingeniería de datos y consulta el modelo alojado en Render.")

# Botones de control
col1, col2 = st.columns(2)
with col1:
    iniciar = st.button("▶ Iniciar Streaming", type="primary", use_container_width=True)
with col2:
    detener = st.button("⏹ Detener Streaming", use_container_width=True)

# Contenedores para actualizar datos dinámicamente
metricas_placeholder = st.empty()
tabla_placeholder = st.empty()

# Lista para guardar los resultados recientes
if 'historial' not in st.session_state:
    st.session_state.historial = []

if iniciar and not detener:
    try:
        # Usamos tu archivo preprocesado (o puedes usar el pipeline igual que antes)
        df_datos = pd.read_csv("datos_preprocesados.csv") 
    except:
        st.error("No se encontró el archivo csv para la simulación.")
        st.stop()
        
    for index, fila in df_datos.iterrows():
        if detener:
            break
            
        transaccion = fila.drop(labels=['is_fraud'], errors='ignore').to_dict()
        
        try:
            # Enviamos datos a la API original
            res = requests.post(API_URL, json=transaccion, timeout=3)
            if res.status_code == 200:
                data_api = res.json()
                
                estado = "🚨 FRAUDE" if data_api["es_fraude"] else "✅ OK"
                # Guardamos en historial
                st.session_state.historial.insert(0, {
                    "ID Evento": index,
                    "Monto ($)": transaccion.get('amt', 0),
                    "Predicción": estado
                })
                # Mantenemos solo los últimos 20
                if len(st.session_state.historial) > 20:
                    st.session_state.historial.pop()
                    
                # Acutal                
        except Exception as e:
            st.error(f"Error conectando a la API: {e}")
            break
            
        # 1. Actualizar Métricas en vivo
        fraudes = sum(1 for item in st.session_state.historial if "FRAUDE" in item["Predicción"])
        with metricas_placeholder.container():
            kpi1, kpi2 = st.columns(2)
            kpi1.metric("Transacciones Evaluadas", len(st.session_state.historial))
            kpi2.metric("Fraudes Detectados", fraudes)
            
        # 2. Actualizar Tabla en vivo
        df_mostrar = pd.DataFrame(st.session_state.historial)
        with tabla_placeholder.container():
            st.dataframe(df_mostrar, use_container_width=True, hide_index=True)
            
        # Pausa para el siguiente stream
        time.sleep(0.5)