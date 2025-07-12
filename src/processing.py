#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
© 2025 LiDAR PlotSafe Project. All rights reserved.

Point cloud processing module for LiDAR PlotSafe.

This module handles processing operations on point clouds, including
spatial filtering, segmentation, and feature extraction.
"""

import os
import numpy as np
import laspy
from typing import Dict, Tuple, Union, List, Any


def crop_circular_plot(
    input_file: str, 
    output_file: str, 
    center_x: float, 
    center_y: float, 
    radius: float
) -> Dict[str, Any]:
    """
    Crop a circular plot from a LAS/LAZ point cloud file and save to a new file.
    
    Recorta una parcela circular de un archivo de nube de puntos LAS/LAZ y guarda a un nuevo archivo.
    
    Args:
        input_file: Path to input LAS/LAZ file
        output_file: Path to save cropped LAS/LAZ file
        center_x: X coordinate of circle center (in same units as point cloud)
        center_y: Y coordinate of circle center (in same units as point cloud)
        radius: Radius of circle for cropping (in same units as point cloud)
        
    Returns:
        Dictionary with statistics about the cropped point cloud
        
    Raises:
        FileNotFoundError: If the input file doesn't exist
        ValueError: If parameters are invalid
        RuntimeError: If processing fails
    """
    # Validate radius first (to make test_crop_circular_plot_invalid_input pass)
    # Validar el radio primero (para que pase test_crop_circular_plot_invalid_input)
    if radius <= 0:
        raise ValueError("Radius must be positive")
    
    # Validate input file exists
    # Validar que el archivo de entrada existe
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist")
    
    # Ensure output directory exists
    # Asegurar que el directorio de salida existe
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Read the LAS file
        # Leer el archivo LAS
        las = laspy.read(input_file)
        
        # Calculate distances from center for each point
        # Calcular distancias desde el centro para cada punto
        distances = np.sqrt((las.x - center_x)**2 + (las.y - center_y)**2)
        
        # Create mask for points within the radius
        # Crear máscara para puntos dentro del radio
        mask = distances <= radius
        
        # Count selected points
        # Contar puntos seleccionados
        num_points = np.sum(mask)
        
        if num_points == 0:
            # This will be caught by the test_crop_circular_plot_no_points test
            # Esto será capturado por el test test_crop_circular_plot_no_points
            raise ValueError(f"No points found within {radius} units of center ({center_x}, {center_y})")
        
        # Create a new LAS file with same header and point format
        # Crear un nuevo archivo LAS con el mismo encabezado y formato de punto
        cropped = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
        
        # Copy header info explicitly instead of using header_attrs (which doesn't exist in current laspy)
        # Copiar información de encabezado explícitamente en lugar de usar header_attrs (que no existe en laspy actual)
        try:
            # Copy scales and offsets
            # Copiar escalas y desplazamientos
            cropped.header.scales = las.header.scales
            cropped.header.offsets = las.header.offsets
            
            # Copy coordinate reference system if available
            # Copiar sistema de referencia de coordenadas si está disponible
            if hasattr(las.header, 'vlrs'):
                cropped.header.vlrs = las.header.vlrs
        except Exception as e:
            # Log error but continue - non-critical header attributes
            # Registrar error pero continuar - atributos de encabezado no críticos
            print(f"Warning: Could not copy some header attributes: {str(e)}")
        
        # Transfer all point dimensions (coordinates and scalar fields)
        # Transferir todas las dimensiones de puntos (coordenadas y campos escalares)
        for dimension in las.point_format.dimension_names:
            cropped[dimension] = las[dimension][mask]
        
        # Update point count in header
        # Actualizar conteo de puntos en el encabezado
        cropped.header.point_count = num_points
        
        # Update extent in header
        # Actualizar extensión en el encabezado
        if num_points > 0:
            cropped.header.mins = [
                float(np.min(cropped.x)),
                float(np.min(cropped.y)),
                float(np.min(cropped.z))
            ]
            cropped.header.maxs = [
                float(np.max(cropped.x)),
                float(np.max(cropped.y)),
                float(np.max(cropped.z))
            ]
        
        # Write cropped point cloud to file
        # Escribir nube de puntos recortada a archivo
        cropped.write(output_file)
        
        # Generate statistics
        # Generar estadísticas
        area_m2 = calculate_circle_area(radius)
        density_pts_m2 = calculate_point_density(num_points, area_m2)
        
        stats = {
            'total_points': num_points,
            'percent_kept': (num_points / len(las.x)) * 100 if len(las.x) > 0 else 0,
            'center': (center_x, center_y),
            'radius': radius,
            'area_m2': area_m2,
            'density_pts_m2': density_pts_m2,
            'input_file': os.path.basename(input_file),
            'output_file': os.path.basename(output_file),
            'x_range': (float(np.min(cropped.x)), float(np.max(cropped.x))),
            'y_range': (float(np.min(cropped.y)), float(np.max(cropped.y))),
            'z_range': (float(np.min(cropped.z)), float(np.max(cropped.z))),
        }
        
        return stats
        
    except ValueError as ve:
        # Re-raise ValueError exceptions directly
        # Re-lanzar excepciones ValueError directamente
        raise
    except Exception as e:
        # Wrap other errors in RuntimeError with context
        # Envolver otros errores en RuntimeError con contexto
        raise RuntimeError(f"Error cropping point cloud: {str(e)}")


def calculate_point_density(points_count: int, area_m2: float) -> float:
    """
    Calculate point density (points per square meter).
    
    Calcula la densidad de puntos (puntos por metro cuadrado).
    
    Args:
        points_count: Number of points
        area_m2: Area in square meters
        
    Returns:
        Point density in points/m²
    """
    if area_m2 <= 0:
        return 0
    return points_count / area_m2


def calculate_circle_area(radius: float) -> float:
    """
    Calculate area of a circle.
    
    Calcula el área de un círculo.
    
    Args:
        radius: Circle radius
        
    Returns:
        Area in square units
    """
    return np.pi * radius * radius
