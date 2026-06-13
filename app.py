from fastapi import FastAPI, HTTPException, Body, Depends, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os

# [WARNING] LIMITAR CPU - Establecer threads antes de importar numpy/pandas
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'

import pandas as pd
import joblib
import logging
import hashlib
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Cargar variables de entorno
load_dotenv()

# Configurar logging básico
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ FUNCIONES DE SEGURIDAD ============

def mask_credential(value: str, show_chars: int = 5) -> str:
    """Oculta credenciales en logs mostrando solo primeros N caracteres"""
    if not value or len(value) <= show_chars:
        return "***"
    return f"{value[:show_chars]}...{value[-3:]}"

def validar_integridad_modelo(ruta: str) -> bool:
    """Verifica que el archivo del modelo sea válido y no corrupto"""
    if not os.path.exists(ruta):
        logger.error(f"[ERROR] CRÍTICO: Modelo no encontrado en {ruta}")
        return False
    
    try:
        # Intentar cargar el modelo
        modelo_test = joblib.load(ruta)
        
        # Verificar que tenga el método predict
        if not hasattr(modelo_test, 'predict'):
            logger.error(f"[ERROR] CRÍTICO: Modelo no tiene método 'predict'")
            return False
        
        logger.info(f"[OK] Modelo validado correctamente: {type(modelo_test).__name__}")
        return True
    except Exception as e:
        logger.error(f"[ERROR] CRÍTICO: Error al validar modelo: {e}")
        return False

# Cargar clave API desde variables de entorno
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    logger.warning("[WARNING] API_KEY no configurada en variables de entorno - autenticación deshabilitada")
else:
    logger.info(f"[OK] API_KEY cargada: {mask_credential(API_KEY)}")  # Masked para seguridad

# Configurar rate limiting (5 requests por minuto por IP)
limiter = Limiter(key_func=get_remote_address)

# Modelo de validación para las transacciones
class TransaccionPredecir(BaseModel):
    amt: float = Field(..., gt=0, description="Monto de la transacción (debe ser > 0)")
    gender: int = Field(..., ge=0, le=1, description="Género: 0 (Femenino) o 1 (Masculino)")
    city_pop: int = Field(..., ge=0, description="Población de la ciudad")
    unix_time: int = Field(..., ge=0, description="Timestamp Unix")
    flag_invalid_amt: int = Field(..., ge=0, le=1, description="Bandera de monto inválido (0 o 1)")
    flag_fake_location: int = Field(..., ge=0, le=1, description="Bandera de ubicación falsa (0 o 1)")
    trans_hour: int = Field(..., ge=0, le=23, description="Hora de la transacción (0-23)")
    trans_day_of_week: int = Field(..., ge=0, le=6, description="Día de la semana (0-6)")
    trans_month: int = Field(..., ge=1, le=12, description="Mes (1-12)")
    age: int = Field(..., gt=0, le=150, description="Edad del usuario")
    distance_km: float = Field(..., ge=0, description="Distancia en km (>= 0)")
    category_food_dining: int = Field(..., ge=0, le=1)
    category_gas_transport: int = Field(..., ge=0, le=1)
    category_grocery_net: int = Field(..., ge=0, le=1)
    category_grocery_pos: int = Field(..., ge=0, le=1)
    category_health_fitness: int = Field(..., ge=0, le=1)
    category_home: int = Field(..., ge=0, le=1)
    category_kids_pets: int = Field(..., ge=0, le=1)
    category_misc_net: int = Field(..., ge=0, le=1)
    category_misc_pos: int = Field(..., ge=0, le=1)
    category_personal_care: int = Field(..., ge=0, le=1)
    category_shopping_net: int = Field(..., ge=0, le=1)
    category_shopping_pos: int = Field(..., ge=0, le=1)
    category_travel: int = Field(..., ge=0, le=1)

# Función para verificar autenticación
def verificar_api_key(authorization: str = Header(None)):
    """Verifica que el header Authorization contenga una clave API válida"""
    if not API_KEY:
        # Si no hay clave configurada, permitir acceso (para desarrollo)
        return True
    
    if not authorization:
        logger.warning("[WARNING] SEGURIDAD: Sin Authorization header")
        raise HTTPException(status_code=401, detail="Authorization header faltante")
    
    # Esperar formato: "Bearer tu_clave_api"
    partes = authorization.split(" ")
    if len(partes) != 2 or partes[0] != "Bearer":
        logger.warning(f"[WARNING] SEGURIDAD: Formato de Authorization inválido")
        raise HTTPException(status_code=401, detail="Formato de Authorization inválido. Usa: Bearer <API_KEY>")
    
    token = partes[1]
    if token != API_KEY:
        logger.error(f"[ERROR] SEGURIDAD: Token incorrecto. Recibido: {mask_credential(token)} | Esperado: {mask_credential(API_KEY)}")
        raise HTTPException(status_code=403, detail="Clave API inválida")
    
    logger.info("[OK] Autenticación exitosa")
    return True

# 1. Inicializar la aplicación FastAPI
app = FastAPI(
    title="API de Detección de Fraude",
    description="API para evaluar transacciones en tiempo real",
    version="1.0.0"
)

# ============ CONFIGURAR CORS ============
# Permitir solo orígenes específicos (máximo 1 en producción)
ALLOWED_ORIGINS = [
    "http://localhost:8000",       # Desarrollo local
    "http://localhost:3000",       # Desarrollo alternativo
    "https://gestion-datos-ev1.onrender.com",  # Render producción
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST"],  # Solo POST para /predecir
    allow_headers=["Content-Type", "Authorization"],
)

logger.info(f"[OK] CORS configurado para: {ALLOWED_ORIGINS}")

# ============ HEADERS DE SEGURIDAD ============
# Middleware que agrega headers de seguridad a todas las responses
from fastapi import Request
from fastapi.responses import Response

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Agrega headers de seguridad HTTP a todas las respuestas"""
    response = await call_next(request)
    
    # Headers de seguridad
    response.headers["X-Content-Type-Options"] = "nosniff"  # Prevenir MIME type sniffing
    response.headers["X-Frame-Options"] = "DENY"  # Prevenir clickjacking
    response.headers["X-XSS-Protection"] = "1; mode=block"  # XSS protection
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"  # HSTS
    response.headers["Content-Security-Policy"] = "default-src 'none'"  # CSP restrictivo
    
    return response

# ============ MANEJADORES DE EXCEPCIONES ============

# Manejador personalizado para errores de validación (422)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Retorna error genérico de validación sin exponer detalles internos"""
    logger.error(f"[ERROR] Error de validación: {len(exc.errors())} campo(s) inválido(s)")
    # No mostrar detalles específicos en producción
    return JSONResponse(
        status_code=422,
        content={"detail": "Los datos proporcionados no son válidos. Verifica el formato de tu solicitud."}
    )

# Manejador personalizado para rate limit exceeded (429)
@app.exception_handler(RateLimitExceeded)
async def rate_limit_exception_handler(request, exc):
    """Retorna error 429 cuando se supera el rate limit"""
    logger.warning(f"[WARNING] Rate limit excedido para IP: {request.client.host}")
    return JSONResponse(
        status_code=429,
        content={"detail": "Demasiadas solicitudes. Límite: 20 por minuto por IP"}
    )

# Agregar rate limiting a la app
app.state.limiter = limiter

# 2. Cargar y validar el modelo en memoria al iniciar la API
MODELO_PATH = 'modelo_fraude_base_500k_datos.pkl'
modelo = None

if validar_integridad_modelo(MODELO_PATH):
    try:
        modelo = joblib.load(MODELO_PATH)
        logger.info(f"[OK] Modelo cargado exitosamente: {type(modelo).__name__}")
    except Exception as e:
        logger.error(f"[ERROR] CRÍTICO: Error al cargar modelo validado: {e}")
        modelo = None
else:
    logger.error(f"[ERROR] CRÍTICO: Validación de integridad del modelo FALLÓ")
    modelo = None

# 3. Crear el endpoint (URL) para hacer predicciones
@app.post("/predecir")
@limiter.limit("500/minute")
def predecir_fraude(
    request: Request,
    datos_transaccion: TransaccionPredecir,
    _auth: bool = Depends(verificar_api_key)
):
    """
    Realiza predicción de fraude.
    
    Requiere:
    - Header: Authorization: Bearer <API_KEY>
    - Body: JSON con datos de la transacción validados
    
    Límite: 20 solicitudes por minuto por IP
    """
    
    if modelo is None:
        logger.error(f"[ERROR] CRÍTICO: Modelo no disponible para request desde {request.client.host}")
        raise HTTPException(status_code=503, detail="Servicio temporalmente no disponible. Intenta más tarde.")
    
    try:
        logger.info(f"[OK] Predicción solicitada desde IP: {request.client.host}")
        
        # Convertir el modelo Pydantic a diccionario y luego a DataFrame
        datos_dict = datos_transaccion.dict()
        logger.debug(f"Datos validados por Pydantic: {list(datos_dict.keys())}")
        
        df_nueva_transaccion = pd.DataFrame([datos_dict])
        
        # Realizar la predicción
        prediccion = modelo.predict(df_nueva_transaccion)
        
        # Extraer el resultado (0 o 1)
        resultado = int(prediccion[0])
        es_fraude = bool(resultado == 1)
        
        logger.info(f"[OK] Predicción completada: Fraude={es_fraude}")
        
        return {
            "estado": "éxito",
            "es_fraude": es_fraude,
            "codigo_prediccion": resultado
        }
        
    except ValueError as e:
        # Error en conversión de tipos
        logger.error(f"[ERROR] Error de tipo en predicción: {str(e)}")
        raise HTTPException(status_code=400, detail="Formato de datos inválido.")
    except Exception as e:
        # Error interno - NO exponer detalles
        import traceback
        logger.error(f"[ERROR] ERROR INTERNO en predicción")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Devolver error genérico sin detalles
        raise HTTPException(status_code=500, detail="Error al procesar la solicitud. Contacta al administrador.")