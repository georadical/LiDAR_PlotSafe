#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
© 2025 LiDAR PlotSafe Project. All rights reserved.

Simple test execution for compute_eps function.
Ejecución simple de prueba para la función compute_eps.
"""

import sys
import os
import numpy as np

# Add src directory to path
# Agregar directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from pipeline.clustering import compute_eps
    
    print("Testing compute_eps with grid pattern...")
    print("Probando compute_eps con patrón de cuadrícula...")
    print("-" * 50)
    
    # Create grid with 0.04 m spacing → expect eps ≈ 0.08
    # Crear cuadrícula con espaciado 0.04 m → esperar eps ≈ 0.08
    xy = np.mgrid[0:1:0.04, 0:1:0.04].reshape(2, -1).T
    print(f"Grid created with {len(xy)} points")
    print(f"Cuadrícula creada con {len(xy)} puntos")
    
    # Test the function
    # Probar la función
    eps = compute_eps(xy, k=8, factor=2.0)
    print(f"Computed epsilon: {eps:.4f}")
    print(f"Epsilon calculado: {eps:.4f}")
    
    # Check the assertion
    # Verificar la aserción
    if 0.07 < eps < 0.10:
        print("✅ TEST PASSED: Epsilon is in expected range [0.07, 0.10]")
        print("✅ PRUEBA APROBADA: Epsilon está en el rango esperado [0.07, 0.10]")
        result = "PASS"
    else:
        print(f"❌ TEST FAILED: Expected 0.07 < eps < 0.10, got {eps}")
        print(f"❌ PRUEBA FALLIDA: Se esperaba 0.07 < eps < 0.10, se obtuvo {eps}")
        result = "FAIL"
    
    print("-" * 50)
    print(f"Final result: {result}")
    print(f"Resultado final: {result}")
    
except Exception as e:
    print(f"Error running test: {e}")
    print(f"Error ejecutando prueba: {e}")
    import traceback
    traceback.print_exc()
