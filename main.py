import logging
import pandas as pd

# Importamos nuestros 4 módulos
from ingesta import ingestar_datos_csv
from validacion import validar_estructura_y_semantica
from limpieza import limpiar_datos
from transformacion import transformar_datos
from carga_datos import cargar_datos_supabase
# ==========================================
# CONFIGURACIÓN CENTRALIZADA DE LOGS
# ==========================================
# 1. Guardar en un archivo de texto

URL_SUPABASE = "postgresql://postgres.bcdfoultzlgkmucgvdct:Fernandini810.@aws-1-us-east-2.pooler.supabase.com:5432/postgres"
logging.basicConfig(
    filename='ejecucion_pipeline.log', 
    level=logging.INFO,
    format='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 2. Imprimir simultáneamente en la consola/terminal
consola = logging.StreamHandler()
consola.setLevel(logging.INFO)
formato_consola = logging.Formatter('%(name)-20s: %(levelname)-8s %(message)s')
consola.setFormatter(formato_consola)
logging.getLogger('').addHandler(consola)

# ==========================================
# ORQUESTADOR DEL PIPELINE
# ==========================================
def ejecutar_pipeline():
    logging.info("="*60)
    logging.info("   INICIANDO PIPELINE DE DETECCIÓN DE FRAUDE (V2)   ")
    logging.info("="*60)
    
    ruta_fuente = 'datos/02_fraudTest.csv'
    
    try:
        # Paso 1: Ingesta
        df_raw = ingestar_datos_csv(ruta_fuente)
        
        if df_raw is not None:
            # Paso 2: Validación (Data Quality y Flags)
            df_valido = validar_estructura_y_semantica(df_raw)
            
            # Paso 3: Limpieza (Trata nulos y duplicados de sistema)
            df_limpio = limpiar_datos(df_valido)
            
            # Paso 4: Transformación (Feature Engineering)
            df_transformado = transformar_datos(df_limpio)
            
            # (Opcional) Paso 5: Exportar el resultado final para dárselo al modelo
            df_transformado.to_csv('datos_preprocesados.csv', index=False)
            
            exito = cargar_datos_supabase(df_transformado, "transacciones_ml_procesadas", URL_SUPABASE)
            logging.info("Dataset final exportado a 'datos_preprocesados.csv'")
            
            logging.info("="*60)
            logging.info("   PIPELINE FINALIZADO EXITOSAMENTE                 ")
            logging.info("="*60)
            
        else:
            logging.error("El pipeline se detuvo en la capa de ingesta. Verifica el archivo.")
            
    except ValueError as ve:
        # Atrapa errores críticos de estructura (ej. archivo incorrecto)
        logging.critical(f"FALLO CRÍTICO DE CALIDAD DE DATOS: {ve}")
    except Exception as e:
        # Atrapa cualquier otro error imprevisto para que no colapse silenciosamente
        logging.critical(f"Fallo inesperado en el orquestador principal: {e}", exc_info=True)

if __name__ == "__main__":
    # Ejecutamos el pipeline
    ejecutar_pipeline()
