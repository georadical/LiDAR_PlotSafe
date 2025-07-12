# LiDAR PlotSafe

© 2025 LiDAR PlotSafe Project. All rights reserved.

## Descripción

LiDAR PlotSafe es una herramienta de procesamiento para nubes de puntos LiDAR de parcelas forestales. Automatiza la extracción de métricas como diámetro a la altura del pecho (DBH), altura total, sweep, ramas y características de tronco siguiendo los estándares PlotSafe.

## Requisitos

- Python 3.10 o superior
- Windows 11 (64 bit) o Linux (Ubuntu 20.04+)
- Dependencias listadas en `requirements.txt`

## Instalación

1. Clone el repositorio
2. Cree un entorno virtual:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
3. Instale las dependencias:
```powershell
pip install -r requirements.txt
```

## Uso

### Interfaz Gráfica

Para iniciar la interfaz gráfica:

```powershell
python -m src.gui.launcher
```

La interfaz permite:
- Seleccionar archivos de nubes de puntos (LAS/LAZ/PLY)
- Configurar parámetros básicos
- Ejecutar el procesamiento
- Visualizar el progreso

### Configuración

Los parámetros del pipeline pueden configurarse en `config.yaml` o mediante la interfaz gráfica.

## Licencia

Este software es propietario. Todos los derechos reservados.
