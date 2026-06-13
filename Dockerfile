
FROM python:3.10-slim

#  SEGURIDAD: Instalar dependencias necesarias
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

#  SEGURIDAD: Crear usuario no-root
RUN groupadd -r fraud_user && useradd -r -g fraud_user fraud_user

WORKDIR /app

#  SEGURIDAD: Copiar con permisos restrictivos
COPY --chown=fraud_user:fraud_user requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

#  SEGURIDAD: Copiar código con permisos restrictivos
COPY --chown=fraud_user:fraud_user . .

#  SEGURIDAD: Crear directorio temporal con permisos adecuados
RUN mkdir -p /tmp && chmod 1777 /tmp && \
    chown -R fraud_user:fraud_user /app

EXPOSE 8000

#  SEGURIDAD: Ejecutar como usuario no-root
USER fraud_user

CMD ["bash", "start.sh"]