#!/bin/bash

# Iniciar la API (FastAPI) en el puerto 8001 en segundo plano (&) (Solo interno)
uvicorn app:app --host 0.0.0.0 --port 8001 &

# Darle 3 segundos a la API para que cargue el modelo
sleep 3

# Iniciar el Dashboard (Streamlit) en el puerto 8000 (Cara externa a internet declarada en Dockerfile)
streamlit run simulador_streaming_pipeline.py --server.port 8000 --server.address 0.0.0.0
