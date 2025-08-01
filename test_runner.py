#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
© 2025 LiDAR PlotSafe Project. All rights reserved.

Test runner for compute_eps function validation.

Ejecutor de pruebas para validación de la función compute_eps.
"""

import sys
import os

# Add src directory to Python path for imports
# Agregar directorio src al path de Python para importaciones
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from pipeline.clustering import compute_eps

def test_compute_eps_reasonable():
    """
    Test epsilon calculation with regular grid to validate expected values.
    
    Prueba el cálculo de epsilon con cuadrícula regular para validar valores esperados.
    """
    print("Running test_compute_eps_reasonable...")
    print("Ejecutando test_compute_eps_reasonable...")
    
    # Create a grid with 0.04 m spacing → expect eps ≈ 0.08
    # Crear una cuadrícula con espaciado de 0.04 m → esperar eps ≈ 0.08
    xy = np.mgrid[0:1:0.04, 0:1:0.04].reshape(2, -1).T
    print(f"Created grid with {len(xy)} points")
    print(f"Cuadrícula creada con {len(xy)} puntos")
    
    # Calculate epsilon with k=8 neighbors and factor=2.0
    # Calcular epsilon con k=8 vecinos y factor=2.0
    eps = compute_eps(xy, k=8, factor=2.0)
    print(f"Computed epsilon: {eps:.4f}")
    print(f"Epsilon calculado: {eps:.4f}")
    
    # Validate the result is within expected range
    # Validar que el resultado esté dentro del rango esperado
    if 0.07 < eps < 0.10:
        print("✅ TEST PASSED: Epsilon is within expected range [0.07, 0.10]")
        print("✅ PRUEBA APROBADA: Epsilon está dentro del rango esperado [0.07, 0.10]")
        return True
    else:
        print(f"❌ TEST FAILED: Expected epsilon between 0.07 and 0.10, got {eps}")
        print(f"❌ PRUEBA FALLIDA: Se esperaba epsilon entre 0.07 y 0.10, se obtuvo {eps}")
        return False

def run_all_tests():
    """
    Run all test functions and report results.
    
    Ejecutar todas las funciones de prueba y reportar resultados.
    """
    print("=" * 60)
    print("LiDAR PlotSafe - compute_eps Unit Test Runner")
    print("Ejecutor de Pruebas Unitarias - compute_eps")
    print("=" * 60)
    
    tests = [
        test_compute_eps_reasonable
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ TEST ERROR: {test_func.__name__} - {str(e)}")
            print(f"❌ ERROR EN PRUEBA: {test_func.__name__} - {str(e)}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results / Resultados de Pruebas:")
    print(f"Passed / Aprobadas: {passed}")
    print(f"Failed / Fallidas: {failed}")
    print(f"Total: {passed + failed}")
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
