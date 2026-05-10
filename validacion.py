import pandas as pd
import logging

# Instanciamos el logger para este módulo
logger = logging.getLogger(__name__)

def validar_estructura_y_semantica(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza pruebas de calidad de datos (Data Quality).
    Falla y detiene el proceso si la estructura es incorrecta.
    Crea 'flags' (banderas) cuando detecta reglas de negocio violadas, 
    para que el modelo aprenda de estos posibles intentos de fraude.
    """
    logger.info("Iniciando validación estructural y semántica de los datos...")
    df_valido = df.copy()
    
    # ==========================================
    # 1. VALIDACIÓN ESTRUCTURAL
    # ==========================================
    logger.info("Comprobando estructura (esquema de columnas)...")
    
    columnas_esperadas = [
        'trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt', 
        'first', 'last', 'gender', 'street', 'city', 'state', 'zip', 
        'lat', 'long', 'city_pop', 'job', 'dob', 'trans_num', 'unix_time', 
        'merch_lat', 'merch_long', 'is_fraud'
    ]
    
    columnas_faltantes = [col for col in columnas_esperadas if col not in df_valido.columns]
    
    if columnas_faltantes:
        logger.error(f"Fallo Estructural Crítico: Faltan las siguientes columnas: {columnas_faltantes}")
        raise ValueError(f"El dataset no tiene la estructura esperada. Faltan: {columnas_faltantes}")
    else:
        logger.info("[+] Validación estructural aprobada: Todas las columnas requeridas están presentes.")

    # ==========================================
    # 2. VALIDACIÓN SEMÁNTICA (Reglas de Negocio con Flags)
    # ==========================================
    logger.info("Comprobando semántica y marcando anomalías (creación de flags)...")

    # Regla A: El monto (amt) debe ser >= 0. Si es negativo, marcamos.
    df_valido['flag_invalid_amt'] = 0
    idx_monto_invalido = df_valido[df_valido['amt'] < 0].index
    if len(idx_monto_invalido) > 0:
        logger.warning(f"¡Alerta! {len(idx_monto_invalido)} transacciones con monto negativo. Se marcarán como sospechosas.")
        df_valido.loc[idx_monto_invalido, 'flag_invalid_amt'] = 1
        # Neutralizamos el valor pasándolo a nulo para que limpieza lo trate
        df_valido.loc[idx_monto_invalido, 'amt'] = pd.NA

    # Regla B: La variable objetivo (is_fraud) debe ser 0 o 1.
    # Esta variable SÍ debemos filtrarla si está mala, porque no podemos entrenar a un modelo si no sabemos si fue fraude o no.
    filas_antes = df_valido.shape[0]
    df_valido = df_valido[df_valido['is_fraud'].isin([0, 1])]
    target_eliminados = filas_antes - df_valido.shape[0]
    if target_eliminados > 0:
        logger.warning(f"Se eliminaron {target_eliminados} registros porque la etiqueta 'is_fraud' no era 0 ni 1 (No sirven para entrenar).")

    # Regla C: Coordenadas planetarias (Lat: -90 a 90 | Long: -180 a 180)
    df_valido['flag_fake_location'] = 0
    
    idx_fake_titular = df_valido[
        (df_valido['lat'] < -90) | (df_valido['lat'] > 90) |
        (df_valido['long'] < -180) | (df_valido['long'] > 180)
    ].index
    
    idx_fake_merch = df_valido[
        (df_valido['merch_lat'] < -90) | (df_valido['merch_lat'] > 90) |
        (df_valido['merch_long'] < -180) | (df_valido['merch_long'] > 180)
    ].index
    
    indices_sospechosos = idx_fake_titular.union(idx_fake_merch)
    if len(indices_sospechosos) > 0:
        logger.warning(f"¡Alerta! {len(indices_sospechosos)} transacciones con coordenadas falsas. Se marcarán como sospechosas.")
        df_valido.loc[indices_sospechosos, 'flag_fake_location'] = 1
        # Neutralizamos para que no rompa la fórmula de Haversine después
        df_valido.loc[idx_fake_titular, ['lat', 'long']] = pd.NA
        df_valido.loc[idx_fake_merch, ['merch_lat', 'merch_long']] = pd.NA

    # Regla D: Género M o F
    genero_invalido = df_valido[~df_valido['gender'].isin(['M', 'F'])].shape[0]
    if genero_invalido > 0:
         logger.warning(f"Detectados {genero_invalido} registros con género distinto a M/F. Se pasarán a nulo.")
         df_valido.loc[~df_valido['gender'].isin(['M', 'F']), 'gender'] = pd.NA

    logger.info(f"Validación finalizada. Dataset listo con {df_valido.shape[0]} filas y {df_valido.shape[1]} columnas (incluyendo flags).")
    
    return df_valido