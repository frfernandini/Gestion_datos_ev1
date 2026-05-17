from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import logging

# Configurar logging básico
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Inicializar la aplicación FastAPI
app = FastAPI(
    title="API de Detección de Fraude",
    description="API para evaluar transacciones en tiempo real",
    version="1.0.0"
)

# 2. Cargar el modelo en memoria al iniciar la API
try:
    # Reemplaza con el nombre exacto de tu archivo
    modelo = joblib.load('modelo_fraude_base_500k_datos.pkl') 
    logger.info("Modelo cargado exitosamente.")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {e}")
    modelo = None

# 3. Crear el endpoint (URL) para hacer predicciones
@app.post("/predecir")
def predecir_fraude(datos_transaccion: dict):
    if modelo is None:
        raise HTTPException(status_code=500, detail="El modelo no está disponible.")
    
    try:
        # Convertir el JSON (diccionario) que recibimos a un DataFrame de 1 fila
        df_nueva_transaccion = pd.DataFrame([datos_transaccion])
        
        # Realizar la predicción
        prediccion = modelo.predict(df_nueva_transaccion)
        
        # Extraer el resultado (0 o 1)
        resultado = int(prediccion[0])
        es_fraude = bool(resultado == 1)
        
        return {
            "estado": "éxito",
            "es_fraude": es_fraude,
            "codigo_prediccion": resultado
        }
        
    except Exception as e:
        logger.error(f"Error durante la predicción: {e}")
        raise HTTPException(status_code=400, detail=f"Error al procesar los datos: {str(e)}")