#!/bin/bash

echo "=== Iniciando Fraud Detection Pipeline ==="

# ⚠️ LIMITAR CPU - Restringir threads de numpy/pandas
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

echo "CPU Limiting:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
echo "  MKL_NUM_THREADS=$MKL_NUM_THREADS"

echo "Variables de entorno:"
echo "API_KEY: ${API_KEY:0:10}...${API_KEY: -5}"
echo "DB_USER: $DB_USER"
echo "DB_HOST: $DB_HOST"

# Verificar que el modelo existe
if [ ! -f "modelo_fraude_base_500k_datos.pkl" ]; then
    echo "❌ ERROR: Modelo no encontrado!"
    exit 1
fi

echo "✅ Modelo encontrado"

# Iniciar la API (FastAPI) en el puerto 8001 - logs van a stdout/stderr
echo "📡 Iniciando FastAPI en puerto 8001..."
uvicorn app:app --host 0.0.0.0 --port 8001 --log-level info &
FASTAPI_PID=$!
echo "FastAPI PID: $FASTAPI_PID"

# Darle 5 segundos a la API para que inicie
sleep 5

# Verificar que FastAPI está corriendo
if ! kill -0 $FASTAPI_PID 2>/dev/null; then
    echo "❌ ERROR: FastAPI no se inició correctamente (PID $FASTAPI_PID no existe)"
    exit 1
fi

echo "✅ FastAPI está corriendo (PID: $FASTAPI_PID)"
ps aux | grep uvicorn

# Iniciar el Dashboard (Streamlit)
echo "📊 Iniciando Dashboard en puerto ${PORT:-8000}..."
exec streamlit run dashboard.py --server.port ${PORT:-8000} --server.address 0.0.0.0
