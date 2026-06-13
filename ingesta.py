import os

# ⚠️ LIMITAR CPU - Establecer threads ANTES de importar numpy/pandas
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'

import pandas as pd
import logging

# Instancia logger
logger = logging.getLogger(__name__)

def ingestar_datos_csv(ruta_archivo: str) -> pd.DataFrame:
    """
    Ingesta datos desde CSV.
    Nota: Los datos se optimizan en exportación con Parquet (si disponible).
    """
    logger.info(f"📥 Cargando desde CSV: '{ruta_archivo}'")
    try:
        df_raw = pd.read_csv(ruta_archivo)
        logger.info(f"✅ Ingesta exitosa: {df_raw.shape[0]} registros, {df_raw.shape[1]} variables.")
        return df_raw
        
    except FileNotFoundError:
        logger.error(f"❌ Error: No se encontró '{ruta_archivo}'")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"❌ Error: El archivo '{ruta_archivo}' está vacío.")
        return None
    except Exception as e:
        logger.error(f"❌ Error inesperado en ingesta: {e}", exc_info=True)
        return None

# --- Ejecución del módulo (solo si se ejecuta directamente) ---
if __name__ == "__main__":
    ruta_fuente = 'datos/02_fraudTest_10k.csv'
    df_fraude_raw = ingestar_datos_csv(ruta_fuente)