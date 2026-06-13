import logging
import pandas as pd
import time
import psutil
from urllib.parse import quote


from ingesta import ingestar_datos_csv
from validacion import validar_estructura_y_semantica
from limpieza import limpiar_datos
from transformacion import transformar_datos
from carga_datos import cargar_datos_supabase
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# ==========================================
# CONFIGURACIÓN CENTRALIZADA DE LOGS
# ==========================================
# 1. Guardar en un archivo de texto

# Construir URL de conexión de forma segura
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")

URL_SUPABASE = (
    f"postgresql://"
    f"{quote(DB_USER, safe='')}:"
    f"{quote(DB_PASSWORD, safe='')}@"
    f"{DB_HOST}:{DB_PORT}/{DB_NAME}"
    f"?sslmode=require"
)
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


def medir_etapa(nombre_etapa, func, *args, **kwargs):
    proceso = psutil.Process()
    # Llamamos a cpu_percent una vez para inicializar el contador
    proceso.cpu_percent(interval=None)
    ram_inicio = proceso.memory_info().rss / (1024 * 1024)
    tiempo_inicio = time.time()
    
    resultado = func(*args, **kwargs)
    
    tiempo_fin = time.time()
    ram_fin = proceso.memory_info().rss / (1024 * 1024)
    cpu_promedio = proceso.cpu_percent(interval=None)
    
    latencia = tiempo_fin - tiempo_inicio
    
    logging.info(f"--- MÉTRICAS [{nombre_etapa}] ---")
    logging.info(f"Latencia / Tiempo de ejecución: {latencia:.4f} seg")
    logging.info(f"Consumo CPU: {cpu_promedio:.2f}%")
    logging.info(f"Consumo RAM: {ram_fin:.2f} MB (Variación: {ram_fin - ram_inicio:+.2f} MB)")
    logging.info("-" * 40)
    
    return resultado

# ==========================================
# ORQUESTADOR DEL PIPELINE
# ==========================================
def ejecutar_pipeline():
    logging.info("="*60)
    logging.info("   INICIANDO PIPELINE DE DETECCIÓN DE FRAUDE (V2)   ")
    logging.info("="*60)
    
    tiempo_inicio_total = time.time()
    proceso = psutil.Process()
    ram_inicio_total = proceso.memory_info().rss / (1024 * 1024)
    proceso.cpu_percent(interval=None)
    
    ruta_fuente = 'datos/02_fraudTest.csv'
    
    try:
        # Paso 1: Ingesta
        df_raw = medir_etapa("1. Ingesta", ingestar_datos_csv, ruta_fuente)
        
        if df_raw is not None:
            # Paso 2: Validación (Data Quality y Flags)
            df_valido = medir_etapa("2. Validación", validar_estructura_y_semantica, df_raw)
            
            # Paso 3: Limpieza (Trata nulos y duplicados de sistema)
            df_limpio = medir_etapa("3. Limpieza y Anonimización", limpiar_datos, df_valido)
            
            # Paso 4: Transformación (Feature Engineering)
            df_transformado = medir_etapa("4. Transformación", transformar_datos, df_limpio)
            
            # Paso 5: Exportar el resultado final
            def exportar_local():
                df_transformado.to_csv('datos_preprocesados.csv', index=False)
            medir_etapa("5.1 Exportación Local CSV", exportar_local)
            
            exito = medir_etapa("5.2 Carga a Supabase", cargar_datos_supabase, df_transformado, "transacciones_ml_procesadas", URL_SUPABASE)
            logging.info("Dataset final exportado a 'datos_preprocesados.csv'")
            
            tiempo_fin_total = time.time()
            ram_fin_total = proceso.memory_info().rss / (1024 * 1024)
            cpu_promedio_total = proceso.cpu_percent(interval=None)
            latencia_total = tiempo_fin_total - tiempo_inicio_total

            logging.info("="*60)
            logging.info("   RESUMEN GENERAL DE RENDIMIENTO DEL PIPELINE   ")
            logging.info("="*60)
            logging.info(f"Tiempo Total (Latencia): {latencia_total:.4f} seg")
            logging.info(f"Consumo CPU Promedio:    {cpu_promedio_total:.2f}%")
            logging.info(f"Consumo RAM Final:       {ram_fin_total:.2f} MB (Inicio: {ram_inicio_total:.2f} MB | Diff: {ram_fin_total - ram_inicio_total:+.2f} MB)")
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
