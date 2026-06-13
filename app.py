from fastapi import FastAPI, HTTPException, Body, Depends, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import os

# ⚠️ LIMITAR CPU - Establecer threads antes de importar numpy/pandas
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'

import pandas as pd
import joblib
import logging
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

# Cargar clave API desde variables de entorno
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    logger.warning("API_KEY no configurada en .env - autenticación deshabilitada")
else:
    logger.info(f"API_KEY cargada: {API_KEY[:10]}...{API_KEY[-5:]}")  # Log parcial por seguridad

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
        logger.warning("Sin Authorization header")
        raise HTTPException(status_code=401, detail="Authorization header faltante")
    
    # Esperar formato: "Bearer tu_clave_api"
    partes = authorization.split(" ")
    if len(partes) != 2 or partes[0] != "Bearer":
        logger.warning(f"Formato de Authorization inválido: {authorization[:20]}...")
        raise HTTPException(status_code=401, detail="Formato de Authorization inválido. Usa: Bearer <API_KEY>")
    
    token = partes[1]
    if token != API_KEY:
        logger.error(f"Token incorrecto. Recibido: {token[:10]}...{token[-5:]} | Esperado: {API_KEY[:10]}...{API_KEY[-5:]}")
        raise HTTPException(status_code=403, detail="Clave API inválida")
    
    logger.info("Autenticación exitosa")
    return True

# 1. Inicializar la aplicación FastAPI
app = FastAPI(
    title="API de Detección de Fraude",
    description="API para evaluar transacciones en tiempo real",
    version="1.0.0"
)

# Manejador personalizado para errores de validación (422)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Retorna detalles completos de errores de validación"""
    logger.error(f"Error de validación: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body if hasattr(exc, 'body') else None}
    )

# Manejador personalizado para rate limit exceeded (429)
@app.exception_handler(RateLimitExceeded)
async def rate_limit_exception_handler(request, exc):
    """Retorna error 429 cuando se supera el rate limit"""
    logger.warning(f"⚠️ Rate limit excedido para IP: {request.client.host}")
    return JSONResponse(
        status_code=429,
        content={"detail": "Demasiadas solicitudes. Límite: 5 por minuto por IP"}
    )

# Agregar rate limiting a la app
app.state.limiter = limiter

# 2. Cargar el modelo en memoria al iniciar la API
try:
    modelo = joblib.load('modelo_fraude_base_500k_datos.pkl') 
    logger.info("Modelo cargado exitosamente.")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {e}")
    modelo = None

# 3. Crear el endpoint (URL) para hacer predicciones
@app.post("/predecir")
@limiter.limit("5/minute")
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
    
    Límite: 5 solicitudes por minuto por IP
    """
    
    if modelo is None:
        raise HTTPException(status_code=500, detail="El modelo no está disponible.")
    
    try:
        logger.info(f"Predicción solicitada desde IP: {request.client.host}")
        
        # Convertir el modelo Pydantic a diccionario y luego a DataFrame
        datos_dict = datos_transaccion.dict()
        logger.debug(f"Datos recibidos: {datos_dict}")
        
        df_nueva_transaccion = pd.DataFrame([datos_dict])
        logger.debug(f"DataFrame creado con shape: {df_nueva_transaccion.shape}")
        
        # Realizar la predicción
        prediccion = modelo.predict(df_nueva_transaccion)
        logger.debug(f"Predicción realizada: {prediccion}")
        
        # Extraer el resultado (0 o 1)
        resultado = int(prediccion[0])
        es_fraude = bool(resultado == 1)
        
        logger.info(f"Predicción exitosa: Fraude={es_fraude}")
        
        return {
            "estado": "éxito",
            "es_fraude": es_fraude,
            "codigo_prediccion": resultado
        }
        
    except Exception as e:
        import traceback
        logger.error(f"❌ ERROR en predicción: {str(e)}")
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")