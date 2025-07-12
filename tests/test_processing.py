#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
© 2025 LiDAR PlotSafe Project. All rights reserved.

Tests for the processing module.
"""

import os
import pytest
import numpy as np
import tempfile
import shutil
import laspy
from unittest import mock

from src.processing import (
    crop_circular_plot, 
    calculate_point_density, 
    calculate_circle_area
)


class TestProcessingModule:
    """Test cases for the processing module."""

    @pytest.fixture
    def test_las_file(self):
        """Create a temporary LAS file for testing."""
        # Crear un archivo LAS temporal para pruebas
        
        # Create a temporary directory
        # Crear un directorio temporal
        temp_dir = tempfile.mkdtemp()
        
        # Create a simple test LAS file
        # Crear un archivo LAS de prueba simple
        test_file = os.path.join(temp_dir, "test_points.las")
        
        # Create points in a 20x20 grid
        # Crear puntos en una cuadrícula de 20x20
        x_coords = []
        y_coords = []
        z_coords = []
        
        for x in range(20):
            for y in range(20):
                x_coords.append(float(x))
                y_coords.append(float(y))
                z_coords.append(float(x + y) / 20.0)
        
        # Convert to numpy arrays
        # Convertir a arrays de numpy
        x = np.array(x_coords)
        y = np.array(y_coords)
        z = np.array(z_coords)
        
        # Create LAS file with these points
        # Crear archivo LAS con estos puntos
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.offsets = [0, 0, 0]
        header.scales = [0.001, 0.001, 0.001]
        
        las = laspy.LasData(header)
        las.x = x
        las.y = y
        las.z = z
        
        # Add some dummy scalar fields for testing
        # Añadir algunos campos escalares ficticios para pruebas
        intensity = np.random.randint(0, 100, size=len(x), dtype=np.uint16)
        las.intensity = intensity
        
        # Write the test file
        # Escribir el archivo de prueba
        las.write(test_file)
        
        yield test_file
        
        # Cleanup after test
        # Limpiar después de la prueba
        shutil.rmtree(temp_dir)
    
    def test_crop_circular_plot_normal_case(self, test_las_file):
        """Test cropping with normal parameters."""
        # Probar recorte con parámetros normales
        
        # Create output file in same directory as input
        # Crear archivo de salida en el mismo directorio que la entrada
        output_file = os.path.join(os.path.dirname(test_las_file), "cropped.las")
        
        # Crop circle at center of grid with radius 5
        # Recortar círculo en el centro de la cuadrícula con radio 5
        result = crop_circular_plot(
            test_las_file,
            output_file,
            center_x=10.0,
            center_y=10.0,
            radius=5.0
        )
        
        # Verify output file exists
        # Verificar que existe el archivo de salida
        assert os.path.exists(output_file)
        
        # Read the cropped file and verify point count
        # Leer el archivo recortado y verificar el conteo de puntos
        cropped_las = laspy.read(output_file)
        
        # All points should be within radius
        # Todos los puntos deben estar dentro del radio
        distances = np.sqrt((cropped_las.x - 10.0)**2 + (cropped_las.y - 10.0)**2)
        assert np.all(distances <= 5.0)
        
        # Verify stats are returned correctly
        # Verificar que las estadísticas se devuelven correctamente
        assert 'total_points' in result
        assert result['total_points'] == len(cropped_las.x)
        assert 'radius' in result
        assert result['radius'] == 5.0
        assert 'center' in result
        assert result['center'] == (10.0, 10.0)
        
        # Verify all scalar fields were preserved
        # Verificar que todos los campos escalares se han preservado
        assert hasattr(cropped_las, 'intensity')

    def test_crop_circular_plot_edge_case(self, test_las_file):
        """Test cropping with edge case parameters (small radius)."""
        # Probar recorte con parámetros de caso límite (radio pequeño)
        
        output_file = os.path.join(os.path.dirname(test_las_file), "edge_case.las")
        
        # Small radius should return few points
        # Un radio pequeño debería devolver pocos puntos
        result = crop_circular_plot(
            test_las_file,
            output_file,
            center_x=10.0,
            center_y=10.0,
            radius=0.5
        )
        
        assert os.path.exists(output_file)
        assert result['total_points'] < 10  # Should have very few points
    
    def test_crop_circular_plot_no_points(self, test_las_file):
        """Test cropping with parameters that result in no points."""
        # Probar recorte con parámetros que no resultan en puntos
        
        output_file = os.path.join(os.path.dirname(test_las_file), "no_points.las")
        
        # Center outside the grid should yield no points
        # Centro fuera de la cuadrícula no debería devolver puntos
        with pytest.raises(ValueError, match="No points found"):
            crop_circular_plot(
                test_las_file,
                output_file,
                center_x=100.0,
                center_y=100.0,
                radius=1.0
            )
    
    def test_crop_circular_plot_invalid_input(self):
        """Test cropping with invalid inputs."""
        # Probar recorte con entradas inválidas
        
        with pytest.raises(FileNotFoundError):
            crop_circular_plot(
                "non_existent_file.las",
                "output.las",
                center_x=0.0,
                center_y=0.0,
                radius=1.0
            )
        
        with pytest.raises(ValueError, match="Radius must be positive"):
            crop_circular_plot(
                "any_file.las",  # Won't be accessed due to validation
                "output.las",
                center_x=0.0,
                center_y=0.0,
                radius=-1.0
            )
    
    def test_calculate_point_density(self):
        """Test point density calculation."""
        # Probar cálculo de densidad de puntos
        
        # Normal case
        # Caso normal
        assert calculate_point_density(100, 10.0) == 10.0
        
        # Edge case - zero area
        # Caso límite - área cero
        assert calculate_point_density(100, 0.0) == 0.0
        
        # Edge case - negative area (should handle gracefully)
        # Caso límite - área negativa (debe manejar correctamente)
        assert calculate_point_density(100, -1.0) == 0.0
    
    def test_calculate_circle_area(self):
        """Test circle area calculation."""
        # Probar cálculo de área de círculo
        
        # Test with radius of 1
        # Probar con radio de 1
        assert calculate_circle_area(1.0) == pytest.approx(np.pi)
        
        # Test with radius of 0
        # Probar con radio de 0
        assert calculate_circle_area(0.0) == 0.0
        
        # Test with large radius
        # Probar con radio grande
        assert calculate_circle_area(1000.0) == pytest.approx(np.pi * 1000.0 * 1000.0)
