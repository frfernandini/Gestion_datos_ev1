# Gestion_datos_ev1

Este proyecto implementa un pipeline completo de procesamiento de datos en Python. El sistema realiza la ingesta, limpieza, validacion, transformacion y carga de datos, y ademas cuenta con capacidades para simular un procesamiento en streaming.

## Estructura del Proyecto

Los modulos principales del pipeline son los siguientes:

* **ingesta.py**: Modulo encargado de la lectura inicial de los datos (por ejemplo, desde el archivo `datos/02_fraudTest.csv`).
* **limpieza.py**: Modulo dedicado al tratamiento de valores nulos o atipicos y correccion de formatos.
* **validacion.py**: Encargado de verificar si los datos cumplen ciertas reglas. Los datos anomalos pueden ser enviados a archivos como `datos_rechazados.csv`.
* **transformacion.py**: Encargado de aplicar la logica de negocio y generar la informacion final, por ejemplo `datos_preprocesados.csv`.
* **carga_datos.py**: Envia o almacena los datos una vez han superado todas las etapas anteriores.
* **simulador_streaming_pipeline.py**: Un script disenado para probar o emular escenarios de entrada continua de datos.
* **main.py** / **app.py**: Puntos de entrada principales para iniciar el proceso o lanzar la aplicacion.

## Despliegue y Ejecucion

El proyecto tambien esta preparado para ser contenerizado. Para ello se cuenta con:

* **requirements.txt**: Listado con las dependencias necesarias de Python.
* **Dockerfile** y **start.sh**: Archivos y scripts para construir la imagen e iniciar la ejecucion del contenedor Docker.

### Uso en Entorno Local

1. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Iniciar el pipeline de datos principal:
   ```bash
   python main.py
   ```

### Uso con Docker

Construir la imagen y ejecutar el contenedor:
```bash
docker build -t gestion_datos_ev1 .
docker run gestion_datos_ev1
```