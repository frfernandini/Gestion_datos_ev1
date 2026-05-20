from fastapi import FastAPI, HTTPException, Body
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
    modelo = joblib.load('modelo_fraude_base_500k_datos.pkl') 
    logger.info("Modelo cargado exitosamente.")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {e}")
    modelo = None

# 3. Crear el endpoint (URL) para hacer predicciones
@app.post("/predecir")
def predecir_fraude(datos_transaccion: dict = Body(
    ...,
        example={
            "amt": 45.50,
            "gender": 1,
            "city_pop": 120500,
            "unix_time": 1371816893,
            "flag_invalid_amt": 0,
            "flag_fake_location": 0,
            "trans_hour": 14,
            "trans_day_of_week": 3,
            "trans_month": 10,
            "age": 34,
            "distance_km": 12.4,
            "category_food_dining": 1,
            "category_gas_transport": 0,
            "category_grocery_net": 0,
            "category_grocery_pos": 0,
            "category_health_fitness": 0,
            "category_home": 0,
            "category_kids_pets": 0,
            "category_misc_net": 0,
            "category_misc_pos": 0,
            "category_personal_care": 0,
            "category_shopping_net": 0,
            "category_shopping_pos": 0,
            "category_travel": 0
        }
    )
    ):
    
    if modelo is None:
        raise HTTPException(status_code=500, detail="El modelo no está disponible.")
    
    try:
        # Convertir el JSON (diccionario) 
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