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

def ejecutar_pipeline_con_batches(tamaño_batch=1000):
    """
    Ejecuta el pipeline procesando datos en batches para mantener CPU/RAM estables.
    Orden CORRECTO: Ingesta → Limpieza → Transformación → Validación (pre-carga) → Carga
    """
    logging.info("="*60)
    logging.info("   INICIANDO PIPELINE DE DETECCIÓN DE FRAUDE (V2 - BATCHES)   ")
    logging.info("="*60)
    
    # ⚠️ LIMITAR CPU A 2 CORES FÍSICOS (throttle a nivel de proceso)
    proceso = psutil.Process()
    try:
        # Obtener cores disponibles y limitar a 2
        num_cores = psutil.cpu_count(logical=False) or 2
        cores_a_usar = min(2, num_cores)
        cores_asignados = list(range(cores_a_usar))
        proceso.cpu_affinity(cores_asignados)
        logging.info(f"🔧 Limitando proceso a {cores_a_usar} cores físicos: {cores_asignados}")
    except Exception as e:
        logging.warning(f"⚠️  No se pudo aplicar cpu_affinity: {e}")
    
    tiempo_inicio_total = time.time()
    ram_inicio_total = proceso.memory_info().rss / (1024 * 1024)
    proceso.cpu_percent(interval=None)
    
    ruta_fuente = 'datos/02_fraudTest_10k.csv'
    
    try:
        # Paso 1: Ingesta completa (una sola vez)
        logging.info(f"📥 Ingesta con procesamiento en batches de {tamaño_batch} filas...")
        df_raw = medir_etapa("1. Ingesta", ingestar_datos_csv, ruta_fuente)
        
        if df_raw is not None:
            total_filas = len(df_raw)
            num_batches = (total_filas + tamaño_batch - 1) // tamaño_batch
            
            logging.info(f"📊 Total de filas: {total_filas} | Batches: {num_batches} x {tamaño_batch}")
            
            # Procesar en batches
            dfs_procesados = []
            
            for i in range(num_batches):
                inicio = i * tamaño_batch
                fin = min((i + 1) * tamaño_batch, total_filas)
                batch = df_raw.iloc[inicio:fin].copy()
                
                logging.info(f"⚙️  Procesando batch {i+1}/{num_batches} (filas {inicio}-{fin})...")
                
                # ORDEN CORRECTO: Limpieza → Transformación → Validación
                
                # Paso 2: Limpieza por batch
                batch_limpio = limpiar_datos(batch)
                
                if batch_limpio.empty:
                    logging.warning(f"  ⚠️  Batch {i+1} quedó vacío después de limpieza")
                    continue
                
                # Paso 3: Transformación por batch
                batch_transformado = transformar_datos(batch_limpio)
                
                if batch_transformado.empty:
                    logging.warning(f"  ⚠️  Batch {i+1} quedó vacío después de transformación")
                    continue
                
                # Paso 4: Validación por batch (DESPUÉS de transformación, antes de cargar)
                batch_validado = validar_estructura_y_semantica(batch_transformado)
                
                if batch_validado.empty:
                    logging.warning(f"  ⚠️  Batch {i+1} quedó vacío después de validación")
                    continue
                
                dfs_procesados.append(batch_validado)
                
                # Log de progreso
                ram_actual = proceso.memory_info().rss / (1024 * 1024)
                cpu_actual = proceso.cpu_percent(interval=None)
                logging.info(f"  ✅ Batch {i+1} procesado | CPU: {cpu_actual:.1f}% | RAM: {ram_actual:.1f} MB")
                
                # ⏸️ THROTTLE: Pausar 0.3s entre batches para evitar picos de CPU
                if i < num_batches - 1:  # No pausar después del último batch
                    time.sleep(0.3)
            
            # Combinar todos los batches procesados
            logging.info("🔀 Combinando batches procesados...")
            df_transformado = pd.concat(dfs_procesados, ignore_index=True)
            logging.info(f"📦 Dataset final: {len(df_transformado)} filas")
            
            # ⏸️ THROTTLE: Pausar antes de exportación para dejar que CPU descanse
            logging.info("⏸️  Preparando exportación (throttle CPU)...")
            time.sleep(0.5)
            
            # Paso 5: Exportar el resultado final
            logging.info("💾 Exportando datos procesados...")
            def exportar_local():
                try:
                    # Intentar Parquet primero (columnar, 3-5x más eficiente)
                    logging.info("📦 Intentando formato Parquet...")
                    df_transformado.to_parquet('datos_preprocesados.parquet', compression='snappy', index=False)
                    logging.info("✅ Exportado a Parquet exitosamente (columnar comprimido)")
                except Exception as e:
                    # Fallback a CSV si Parquet falla
                    logging.warning(f"⚠️  Parquet falló ({type(e).__name__}), usando CSV...")
                    try:
                        df_transformado.to_csv('datos_preprocesados.csv', index=False, encoding='utf-8')
                        logging.info("✅ Exportado a CSV exitosamente")
                    except Exception as e2:
                        logging.error(f"❌ Error en exportación CSV: {e2}", exc_info=True)
            medir_etapa("5.1 Exportación Local", exportar_local)
            
            # ⏸️ THROTTLE: Pausar después de exportación
            logging.info("⏸️  Descansando después de exportación...")
            time.sleep(1.0)
            
            # Paso 6: Carga a Supabase
            logging.info("📤 Cargando a Supabase...")
            exito = medir_etapa("5.2 Carga a Supabase", cargar_datos_supabase, df_transformado, "transacciones_ml_procesadas", URL_SUPABASE)
            logging.info("✅ Dataset final exportado (formato optimizado)")
            
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
            logging.error("❌ El pipeline se detuvo en la capa de ingesta. Verifica el archivo.")
            
    except ValueError as ve:
        logging.critical(f"❌ FALLO CRÍTICO DE CALIDAD DE DATOS: {ve}")
    except Exception as e:
        logging.critical(f"❌ Fallo inesperado en el orquestador principal: {e}", exc_info=True)


# Mantener función antigua por compatibilidad
def ejecutar_pipeline():
    logging.info("="*60)
    logging.info("   INICIANDO PIPELINE DE DETECCIÓN DE FRAUDE (V2)   ")
    logging.info("="*60)
    
    tiempo_inicio_total = time.time()
    proceso = psutil.Process()
    ram_inicio_total = proceso.memory_info().rss / (1024 * 1024)
    proceso.cpu_percent(interval=None)
    
    ruta_fuente = 'datos/02_fraudTest_10k.csv'
    
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
                try:
                    logging.info("📦 Probando formato Parquet...")
                    df_transformado.to_parquet('datos_preprocesados.parquet', engine='pyarrow', compression='snappy', index=False)
                    logging.info("✅ Exportación a Parquet exitosa (columnar comprimido)")
                except Exception as e:
                    logging.warning(f"⚠️  Parquet no disponible ({e}), usando CSV...")
                    df_transformado.to_csv('datos_preprocesados.csv', index=False)
                    logging.info("✅ Exportación a CSV exitosa")
            medir_etapa("5.1 Exportación Local", exportar_local)
            
            exito = medir_etapa("5.2 Carga a Supabase", cargar_datos_supabase, df_transformado, "transacciones_ml_procesadas", URL_SUPABASE)
            logging.info("Dataset final exportado (formato optimizado)")
            
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
    # Ejecutamos el pipeline CON BATCHES para optimizar CPU/RAM
    # Tamaño 3000 = 3-4 batches (optimización máxima: RAM↑↑ overhead↓↓)
    ejecutar_pipeline_con_batches(tamaño_batch=3000)
