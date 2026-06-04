import pandas as pd
import logging
import hashlib

logger = logging.getLogger(__name__)

def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Iniciando proceso de limpieza de datos...")
    df_clean = df.copy()
    
    # 1. Eliminación de registros duplicados por error de sistema
    filas_iniciales = df_clean.shape[0]
    if 'trans_num' in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=['trans_num'])
        filas_duplicadas = filas_iniciales - df_clean.shape[0]
        if filas_duplicadas > 0:
            logger.warning(f"Se eliminaron {filas_duplicadas} registros duplicados de sistema (mismo ID de transacción).")
        else:
            logger.info("No se encontraron registros duplicados de sistema.")
    else:
        logger.warning("No se encontró la columna 'trans_num' para validar duplicados.")

    # 2. Corrección de Tipos de Datos Base
    if 'cc_num' in df_clean.columns:
        df_clean['cc_num'] = df_clean['cc_num'].astype(str)
        logger.info("Columna de tarjetas 'cc_num' convertida a formato texto.")
    if 'zip' in df_clean.columns:
        df_clean['zip'] = df_clean['zip'].astype(str)
        logger.info("Columna 'zip' convertida a formato texto.")
        
    if 'trans_date_trans_time' in df_clean.columns:
        df_clean['trans_date_trans_time'] = pd.to_datetime(df_clean['trans_date_trans_time'], errors='coerce')
    if 'dob' in df_clean.columns:
        df_clean['dob'] = pd.to_datetime(df_clean['dob'], errors='coerce')
    logger.info("Variables temporales convertidas a datetime.")

    # 3. Tratamiento de Valores Nulos
    nulos_por_columna = df_clean.isnull().sum()
    columnas_con_nulos = nulos_por_columna[nulos_por_columna > 0]
    
    if not columnas_con_nulos.empty:
        logger.warning(f"Se encontraron valores nulos en: {list(columnas_con_nulos.index)}")
        for col in columnas_con_nulos.index:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna('Desconocido')
            elif pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            elif pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                df_clean = df_clean.dropna(subset=[col])
        logger.info("Valores nulos tratados (mediana para numéricos, 'Desconocido' para texto).")
    else:
        logger.info("No se detectaron valores nulos en el dataset.")

    # 4. Anonimización de Datos Sensibles (PII)
    logger.info("Iniciando enmascaramiento de datos sensibles...")
    
    # 4.1 Enmascarar la tarjeta de crédito (solo conservar últimos 4 dígitos)
    if 'cc_num' in df_clean.columns:
        df_clean['cc_num'] = df_clean['cc_num'].astype(str).apply(
            lambda x: '*' * (len(x) - 4) + x[-4:] if len(x) > 4 else x
        )
        logger.info("Tarjetas de crédito enmascaradas con éxito.")
        
    # 4.2 Hashing de nombres y apellidos (irreversible)
    def aplicar_hash(texto):
        if pd.isna(texto): return texto
        return hashlib.sha256(str(texto).encode('utf-8')).hexdigest()

    if 'first' in df_clean.columns:
        df_clean['first'] = df_clean['first'].apply(aplicar_hash)
    if 'last' in df_clean.columns:
        df_clean['last'] = df_clean['last'].apply(aplicar_hash)
        
    logger.info("Nombres de titulares anonimizados vía hashing (SHA-256).")
        
    logger.info(f"Limpieza finalizada. Dimensiones actuales: {df_clean.shape[0]} filas, {df_clean.shape[1]} columnas.")
    return df_clean