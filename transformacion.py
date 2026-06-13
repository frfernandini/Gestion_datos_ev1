import os

# ⚠️ LIMITAR CPU - Establecer threads ANTES de importar numpy/pandas
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'

import pandas as pd
import numpy as np
import logging
import re
logger = logging.getLogger(__name__)

def calcular_distancia_haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def transformar_datos(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Iniciando proceso de transformación de datos (Feature Engineering)...")
    df_trans = df.copy()

    try:
        # 1. Variables Temporales
        logger.info("Extrayendo características temporales...")
        df_trans['trans_hour'] = df_trans['trans_date_trans_time'].dt.hour
        df_trans['trans_day_of_week'] = df_trans['trans_date_trans_time'].dt.dayofweek
        df_trans['trans_month'] = df_trans['trans_date_trans_time'].dt.month
        
        # 2. Edad
        logger.info("Calculando la edad del titular...")
        df_trans['age'] = (df_trans['trans_date_trans_time'] - df_trans['dob']).dt.days // 365

        # 3. Distancia
        logger.info("Calculando distancia entre residencia y comercio (Haversine)...")
        df_trans['distance_km'] = calcular_distancia_haversine(
            df_trans['lat'], df_trans['long'], 
            df_trans['merch_lat'], df_trans['merch_long']
        )

        # 4. Encoding
        logger.info("Aplicando encoding a variables categóricas...")
        if 'gender' in df_trans.columns:
            df_trans['gender'] = df_trans['gender'].map({'M': 1, 'F': 0})
            
        if 'category' in df_trans.columns:
            #Agregamos dtype=int para forzar 1 y 0 en lugar de True/False
            df_trans = pd.get_dummies(df_trans, columns=['category'], drop_first=True, dtype=int)

        # 5. Eliminación de Columnas
        columnas_a_eliminar = [
            'trans_date_trans_time', 'dob', 'first', 'last', 
            'street', 'city', 'zip', 'trans_num', 'cc_num', 
            'lat', 'long', 'merch_lat', 'merch_long',
            'merchant', 'state', 'job', 'Unnamed: 0'  # <-- Agregamos 'Unnamed: 0' aquí para borrarla si existe
        ]
        columnas_existentes = [col for col in columnas_a_eliminar if col in df_trans.columns]
        df_trans = df_trans.drop(columns=columnas_existentes)
        
        # 6. Estandarización de nombres de columnas
        logger.info("Estandarizando nombres de columnas para compatibilidad con ML...")
        df_trans = df_trans.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))

        logger.info(f"Transformación finalizada. Dimensiones finales: {df_trans.shape[0]} filas, {df_trans.shape[1]} columnas.")
        
    except Exception as e:
        logger.error(f"Error inesperado durante la transformación: {e}", exc_info=True)
        
    return df_trans