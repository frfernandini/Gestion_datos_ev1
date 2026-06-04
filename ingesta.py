import pandas as pd
import logging

# Instancia logger
logger = logging.getLogger(__name__)

def ingestar_datos_csv(ruta_archivo: str) -> pd.DataFrame:
    logger.info(f"Iniciando ingesta de datos desde: '{ruta_archivo}'")
    try:
        df_raw = pd.read_csv(ruta_archivo)
        logger.info(f"Ingesta exitosa: Se cargaron {df_raw.shape[0]} registros y {df_raw.shape[1]} variables.")
        return df_raw
        
    except FileNotFoundError:
        logger.error(f"Error de ingesta: No se encontró el archivo '{ruta_archivo}'.")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"Error de ingesta: El archivo '{ruta_archivo}' está vacío.")
        return None
    except Exception as e:
        logger.error(f"Error inesperado durante la ingesta: {e}", exc_info=True)
        return None

# --- Ejecución del módulo ---
ruta_fuente = 'datos/02_fraudTest_10k.csv'
df_fraude_raw = ingestar_datos_csv(ruta_fuente)