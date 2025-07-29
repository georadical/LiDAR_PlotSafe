# 2025 LiDAR PlotSafe Project. All rights reserved.

"""
Pruebas unitarias para utilidades de procesamiento de nubes de puntos.

Unit tests for point cloud processing utilities.
"""

import sys
import os
import numpy as np
import pytest

# Add src directory to Python path for imports
# Agregar directorio src al path de Python para imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline.utils import downsample_open3d


def test_open3d_downsample():
    """
    Prueba la función de submuestreo Open3D con atributos.
    
    Test the Open3D downsampling function with attributes.
    """
    # Generate test data with random points and intensity values
    # Generar datos de prueba con puntos aleatorios y valores de intensidad
    xyz = np.random.rand(1_000_000, 3).astype(np.float32)
    intensity = np.random.randint(0, 255, size=(1_000_000,), dtype=np.uint8)
    
    # Test downsampling with attributes
    # Probar submuestreo con atributos
    xyz_ds, attrs_ds = downsample_open3d(xyz, 0.05, {"intensity": intensity})
    
    # Verify that downsampling reduces the number of points
    # Verificar que el submuestreo reduce el número de puntos
    assert xyz_ds.shape[0] < xyz.shape[0]
    
    # Verify that attributes are preserved and have correct shape
    # Verificar que los atributos se preservan y tienen la forma correcta
    assert attrs_ds and attrs_ds["intensity"].shape[0] == xyz_ds.shape[0]
    
    # Verify output data types
    # Verificar tipos de datos de salida
    assert xyz_ds.dtype == np.float64  # Open3D returns float64
    assert attrs_ds["intensity"].dtype == np.uint8


def test_open3d_downsample_without_attributes():
    """
    Prueba la función de submuestreo Open3D sin atributos.
    
    Test the Open3D downsampling function without attributes.
    """
    # Generate test data with random points only
    # Generar datos de prueba solo con puntos aleatorios
    xyz = np.random.rand(10_000, 3).astype(np.float32)
    
    # Test downsampling without attributes
    # Probar submuestreo sin atributos
    xyz_ds, attrs_ds = downsample_open3d(xyz, 0.1)
    
    # Verify that downsampling reduces the number of points
    # Verificar que el submuestreo reduce el número de puntos
    assert xyz_ds.shape[0] < xyz.shape[0]
    
    # Verify that no attributes are returned
    # Verificar que no se devuelven atributos
    assert attrs_ds is None
    
    # Verify output shape and data type
    # Verificar forma y tipo de datos de salida
    assert xyz_ds.shape[1] == 3
    assert xyz_ds.dtype == np.float64


def test_open3d_downsample_small_voxel_size():
    """
    Prueba la función de submuestreo con tamaño de voxel muy pequeño.
    
    Test the downsampling function with very small voxel size.
    """
    # Generate small test dataset
    # Generar conjunto de datos de prueba pequeño
    xyz = np.random.rand(1_000, 3).astype(np.float32)
    
    # Test with very small voxel size (should retain most points)
    # Probar con tamaño de voxel muy pequeño (debería retener la mayoría de puntos)
    xyz_ds, attrs_ds = downsample_open3d(xyz, 0.001)
    
    # With very small voxel size, most points should be retained
    # Con tamaño de voxel muy pequeño, la mayoría de puntos deberían retenerse
    assert xyz_ds.shape[0] >= xyz.shape[0] * 0.8  # At least 80% retained
    assert attrs_ds is None
