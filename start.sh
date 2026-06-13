#!/bin/bash

echo "=== Iniciando Fraud Detection Pipeline ==="
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

# Iniciar la API (FastAPI) en el puerto 8001 con logging visible
echo "📡 Iniciando FastAPI en puerto 8001..."
uvicorn app:app --host 0.0.0.0 --port 8001 > /tmp/fastapi.log 2>&1 &
FASTAPI_PID=$!
echo "FastAPI PID: $FASTAPI_PID"

# Darle 5 segundos a la API para que inicie
sleep 5

# Verificar que FastAPI está corriendo
if ! kill -0 $FASTAPI_PID 2>/dev/null; then
    echo "❌ ERROR: FastAPI no se inició correctamente"
    echo "=== FastAPI Logs ==="
    cat /tmp/fastapi.log
    exit 1
fi

echo "✅ FastAPI iniciado correctamente"

# Iniciar el Dashboard (Streamlit) en el puerto 8000
echo "🎨 Iniciando Streamlit en puerto 8000..."
streamlit run simulador_streaming_pipeline.py --server.port 8000 --server.address 0.0.0.0
