# 2025 LiDAR PlotSafe Project. All rights reserved.

"""
Pruebas unitarias para extracción de rebanadas adaptativas.

Unit tests for adaptive slice extraction.
"""

import sys
import os
import numpy as np
import pytest

# Add src directory to Python path for imports
# Agregar directorio src al path de Python para imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline.slice import extract_adaptive_slice


def test_adaptive_slice_density():
    """
    Prueba la extracción de rebanada adaptativa con control de densidad.
    
    Test adaptive slice extraction with density control.
    """
    # Generate point cloud with dense region at bottom and sparse at top
    # Generar nube de puntos con región densa abajo y escasa arriba
    np.random.seed(42)
    z_dense = np.random.uniform(0, 3, 800)  # Dense region at bottom
    z_sparse = np.random.uniform(20, 25, 50)  # Sparse region at top
    z = np.concatenate([z_dense, z_sparse])
    pts = np.column_stack([np.random.rand(z.size), np.random.rand(z.size), z])
    
    # Test adaptive slice extraction at sparse region (should fallback)
    # Probar extracción de rebanada adaptativa en región escasa (debería usar fallback)
    sl, h, w = extract_adaptive_slice(pts, preferred_height=22, thickness=0.2, min_points=200)
    
    # Should find some points (best available slice)
    # Debería encontrar algunos puntos (mejor rebanada disponible)
    assert len(sl) > 0  # Should find some points
    # Should have warning since it couldn't meet min_points requirement
    # Debería tener advertencia ya que no pudo cumplir requisito de min_points
    assert w is not None
    assert "Low density" in w


def test_adaptive_slice_preferred_height():
    """
    Prueba que la función use la altura preferida cuando hay suficientes puntos.
    
    Test that the function uses preferred height when enough points are available.
    """
    # Generate point cloud with dense region at preferred height
    # Generar nube de puntos con región densa a altura preferida
    np.random.seed(42)  # For reproducible test
    z = np.random.uniform(1.2, 1.4, 1000)  # Dense region around 1.3m
    pts = np.column_stack([np.random.rand(z.size), np.random.rand(z.size), z])
    
    # Test with preferred height that should work
    # Probar con altura preferida que debería funcionar
    sl, h, w = extract_adaptive_slice(pts, preferred_height=1.3, thickness=0.2, min_points=200)
    
    # Should use preferred height
    # Debería usar altura preferida
    assert h == 1.3
    assert w is None  # No warning expected
    assert len(sl) >= 200


def test_adaptive_slice_fallback_percentiles():
    """
    Prueba el mecanismo de respaldo usando percentiles.
    
    Test the fallback mechanism using percentiles.
    """
    # Generate skewed point cloud (most points at bottom)
    # Generar nube de puntos sesgada (mayoría de puntos abajo)
    np.random.seed(123)
    z_dense = np.random.uniform(0, 2, 1500)  # Dense at bottom (more points)
    z_sparse = np.random.uniform(8, 10, 100)  # Sparse at top
    z = np.concatenate([z_dense, z_sparse])
    pts = np.column_stack([np.random.rand(z.size), np.random.rand(z.size), z])
    
    # Request slice at sparse region
    # Solicitar rebanada en región escasa
    sl, h, w = extract_adaptive_slice(pts, preferred_height=9.0, thickness=0.2, min_points=200)
    
    # Should find some points (best available slice)
    # Debería encontrar algunos puntos (mejor rebanada disponible) 
    assert len(sl) > 0  # Should find some points
    # Should have warning since it likely couldn't meet min_points at preferred height
    # Debería tener advertencia ya que probablemente no pudo cumplir min_points en altura preferida
    if len(sl) < 200:
        assert w is not None
        assert "Low density" in w


def test_adaptive_slice_small_dataset():
    """
    Prueba el manejo de conjuntos de datos pequeños.
    
    Test handling of small datasets.
    """
    # Small point cloud (less than min_points)
    # Nube de puntos pequeña (menos que min_points)
    np.random.seed(456)
    z = np.random.uniform(1, 2, 50)  # Only 50 points
    pts = np.column_stack([np.random.rand(z.size), np.random.rand(z.size), z])
    
    # Should handle gracefully with small dataset
    # Debería manejar con elegancia conjunto de datos pequeño
    sl, h, w = extract_adaptive_slice(pts, preferred_height=1.5, thickness=0.2, min_points=200)
    
    # Should return what it can find
    # Debería devolver lo que puede encontrar
    assert len(sl) <= 50  # Can't find more than available
    assert w is not None  # Should have warning about low density
    assert "Low density" in w
