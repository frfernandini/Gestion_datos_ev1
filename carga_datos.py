import pandas as pd
import logging

logger = logging.getLogger(__name__)

def cargar_datos_supabase(df_limpio: pd.DataFrame, nombre_tabla: str, url_conexion: str):
    logger.info("Iniciando fase de Carga de Datos a Supabase...")
    
    # 1. Aseguramos el formato correcto de la URL
    url_corregida = url_conexion.replace("postgres://", "postgresql://")
    
    try:
        logger.info(f"Intentando cargar {len(df_limpio)} registros a la tabla '{nombre_tabla}'...")
        
        # 2. EL TRUCO: Le pasamos la URL directamente a Pandas en "con="
        df_limpio.to_sql(
            name=nombre_tabla, 
            con=url_corregida, 
            if_exists='append', 
            index=False
        )
        
        logger.info("✅ Carga exitosa. Transacción confirmada en Supabase.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error crítico durante la inserción en la BD: {e}")
        # Guardamos el CSV como respaldo en caso de fallo, tal como pide tu evaluación
        df_limpio.to_csv("datos_rechazados.csv", index=False)
        logger.warning("Los registros que fallaron fueron guardados en 'datos_rechazados.csv'")
        return False