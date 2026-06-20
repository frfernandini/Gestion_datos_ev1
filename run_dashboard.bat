@echo off
REM Script para ejecutar el Dashboard de Detección de Fraude

echo [INFO] ===================================================
echo [INFO]  DASHBOARD - DETECCIÓN DE FRAUDE
echo [INFO] ===================================================
echo.

REM Verificar si Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no está instalado o no está en PATH
    pause
    exit /b 1
)

echo [OK] Python encontrado: 
python --version
echo.

REM Verificar si pip está instalado
pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip no está instalado
    pause
    exit /b 1
)

echo [OK] pip encontrado
echo.

REM Instalar/actualizar dependencias
echo [INSTALL] Actualizando dependencias...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Fallo al instalar dependencias
    pause
    exit /b 1
)

echo [OK] Dependencias instaladas
echo.

REM Ejecutar dashboard
echo [START] Iniciando Dashboard Streamlit...
echo [INFO] Dashboard disponible en: http://localhost:8501
echo [INFO] Presiona Ctrl+C para detener
echo.

streamlit run dashboard.py --logger.level=warning

pause
