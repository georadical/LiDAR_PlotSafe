#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
© 2025 LiDAR PlotSafe Project. All rights reserved.

Pytest configuration for LiDAR PlotSafe tests.

Configuración de pytest para pruebas de LiDAR PlotSafe.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path for imports
# Agregar directorio src al path de Python para importaciones
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
