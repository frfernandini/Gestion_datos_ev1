#!/bin/bash

# Iniciar la API (FastAPI) en el puerto 8000 en segundo plano (&)
uvicorn app:app --host 0.0.0.0 --port 8000 &

# Darle 3 segundos a la API para que cargue el modelo .pkl
sleep 3

# Iniciar el Dashboard (Streamlit) en el puerto que Render asigna ($PORT)
streamlit run simulador_streaming_pipeline.py --server.port $PORT --server.address 0.0.0.0
