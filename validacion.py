import os

# ⚠️ LIMITAR CPU - Establecer threads ANTES de importar numpy/pandas
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'

import pandas as pd
import logging


logger = logging.getLogger(__name__)

def validar_estructura_y_semantica(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Iniciando validación de datos transformados (pre-carga)...")
    df_valido = df.copy()
    
    # Validar que tenga las columnas de ML features (después de transformación)
    # COLUMNAS NÚCLEO (siempre deben estar)
    columnas_nucleos = [
        'amt', 'gender', 'city_pop', 'unix_time',
        'trans_hour', 'trans_day_of_week', 'trans_month', 'age', 'distance_km'
    ]
    
    # CATEGORÍAS (dinámicas, pueden no todas en cada batch)
    columnas_categorias_posibles = [
        'category_food_dining', 'category_gas_transport', 'category_grocery_net',
        'category_grocery_pos', 'category_health_fitness', 'category_home',
        'category_kids_pets', 'category_misc_net', 'category_misc_pos',
        'category_personal_care', 'category_shopping_net', 'category_shopping_pos',
        'category_travel'
    ]
    
    # Verificar estructura - COLUMNAS NÚCLEO
    logger.info("Verificando estructura de datos transformados...")
    columnas_faltantes_nucleos = [col for col in columnas_nucleos if col not in df_valido.columns]
    
    if columnas_faltantes_nucleos:
        logger.error(f"❌ Faltan columnas NÚCLEO: {columnas_faltantes_nucleos}")
        raise ValueError(f"Dataset incompleto. Faltan columnas críticas: {columnas_faltantes_nucleos}")
    else:
        logger.info(f"✅ Columnas NÚCLEO validadas ({len(columnas_nucleos)} presentes)")
    
    # Verificar categorías (solo las que existan en este batch)
    categorias_presentes = [col for col in columnas_categorias_posibles if col in df_valido.columns]
    if categorias_presentes:
        logger.info(f"✅ Categorías encontradas: {len(categorias_presentes)} de {len(columnas_categorias_posibles)}")
    else:
        logger.warning(f"⚠️  No se encontraron columnas de categorías en este batch")
    
    # Validar que no haya valores nulos en columnas críticas
    logger.info("Verificando valores nulos en columnas críticas...")
    columnas_criticas = ['amt', 'gender', 'age', 'distance_km', 'trans_hour']
    
    for col in columnas_criticas:
        if col in df_valido.columns:
            nulos = df_valido[col].isna().sum()
            if nulos > 0:
                logger.warning(f"⚠️  {nulos} valores nulos en columna '{col}'. Se eliminarán esas filas.")
                df_valido = df_valido[df_valido[col].notna()]
    
    # Validar rangos de valores
    logger.info("Verificando rangos de valores...")
    validaciones = [
        ('amt', lambda x: x > 0, "Monto debe ser > 0"),
        ('gender', lambda x: x.isin([0, 1]), "Género debe ser 0 o 1"),
        ('age', lambda x: (x > 0) & (x <= 150), "Edad debe estar entre 1 y 150"),
        ('trans_hour', lambda x: (x >= 0) & (x <= 23), "Hora debe estar entre 0 y 23"),
        ('distance_km', lambda x: x >= 0, "Distancia no puede ser negativa"),
    ]
    
    filas_descartadas = 0
    for col, validacion, msg in validaciones:
        if col in df_valido.columns:
            mask_invalido = ~validacion(df_valido[col])
            filas_invalidas = mask_invalido.sum()
            if filas_invalidas > 0:
                logger.warning(f"⚠️  {filas_invalidas} filas inválidas en '{col}': {msg}")
                df_valido = df_valido[~mask_invalido]
                filas_descartadas += filas_invalidas
    
    if filas_descartadas > 0:
        logger.info(f"Se descartaron {filas_descartadas} filas por validación de rangos.")
    
    # Validar que no esté vacío
    if df_valido.empty:
        logger.error("❌ CRÍTICO: Dataset vacío después de validación")
        raise ValueError("Dataset está vacío después de validación")
    
    logger.info(f"✅ Validación completada. {len(df_valido)} filas listas para carga a BD.")
    return df_valido