import pandas as pd
import requests
import time
import random

# Importamos las piezas de tu pipeline existente
from validacion import validar_estructura_y_semantica
from limpieza import limpiar_datos
from transformacion import transformar_datos

API_URL = "https://gestion-datos-ev1.onrender.com/predecir" 

# 🔴 ESTO ES VITAL: Lista de TODAS las columnas que tu modelo entrenado espera
# (Asegúrate de que estas sean exactamente las columnas con las que entrenaste, en el mismo orden)
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
    """
    Como en streaming llega de a 1 fila, get_dummies no creará todas las columnas categóricas.
    Esta función rellena con 0 las columnas faltantes para que el modelo no colapse.
    """
    for col in COLUMNAS_ESPERADAS:
        if col not in df_transformado.columns:
            df_transformado[col] = 0
            
    # Retornar exactamente en el orden que espera el modelo
    return df_transformado[COLUMNAS_ESPERADAS]

def simular_streaming_crudo():
    print("Iniciando simulador de streaming utilizando el pipeline completo...")
    
    # 1. Leemos el archivo en CRUDO (simulando eventos históricos)
    df_raw = pd.read_csv('datos/02_fraudTest.csv')
    
    # 2. Iteramos fila por fila para simular que llegan eventos secuenciales
    for index, fila in df_raw.iterrows():
        # Convertimos la fila en un DataFrame de 1 solo registro
        df_evento = pd.DataFrame([fila])
        
        try:
            # ========================================================
            #  AQUÍ APLICAMOS TU PIPELINE EN MICRO-BATCH (1 FILA)
            # ========================================================
            
            # Paso A: Validación
            df_valido = validar_estructura_y_semantica(df_evento)
            if df_valido.empty:
                print(f"[Transacción {index}] Ignorada: Falló validación")
                continue
                
            # Paso B: Limpieza
            df_limpio = limpiar_datos(df_valido)
            if df_limpio.empty:
                continue
                
            # Paso C: Transformación (feature engineering: edad, distancias, etc.)
            df_trans = transformar_datos(df_limpio)
            
            # Paso D: Alineación de Columnas (Arregla el problema de get_dummies)
            df_final = alinear_columnas(df_trans)
            
            # Convertimos la fila única ya procesada a diccionario
            datos_prediccion = df_final.iloc[0].to_dict()
            
            # ========================================================
            # 🌐 ENVIAMOS A LA API EN RENDER
            # ========================================================
            respuesta = requests.post(API_URL, json=datos_prediccion, timeout=5)
            
            if respuesta.status_code == 200:
                resultado = respuesta.json()
                estado = "🚨 FRAUDE 🚨" if resultado['es_fraude'] else "✅ OK"
                print(f"[Evento {index:05d}] | Monto original: ${fila['amt']:.2f} | Predicción: {estado}")
            else:
                print(f"[Evento {index:05d}] | ERROR API: {respuesta.status_code}")
                
        except Exception as e:
            print(f"[Evento {index:05d}] | Error en procesamiento del pipeline: {e}")
            
        # Pausa para simular el tráfico en tiempo real humano
        time.sleep(random.uniform(0.1, 1.2))

if __name__ == "__main__":
    simular_streaming_crudo()