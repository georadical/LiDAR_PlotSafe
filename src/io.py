#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
© 2025 LiDAR PlotSafe Project. All rights reserved.

Point cloud input/output module for LiDAR PlotSafe.

This module handles loading, reading and basic information extraction from point cloud files.
"""

import os
import laspy
import numpy as np
from typing import Dict, Tuple, Union, List


def load_point_cloud(filepath: str) -> Tuple[np.ndarray, Dict]:
    """
    Load and summarize LiDAR point cloud data.
    
    Carga y resume los datos de nube de puntos LiDAR.
    
    Args:
        filepath: Path to .las/.laz file.
        
    Returns:
        Tuple containing:
            - points: Nx3 array of point coordinates
            - summary: Dictionary with point cloud statistics
            
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is not supported
    """
    # Validate file exists
    # Validar que el archivo existe
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    
    # Validate file extension
    # Validar extensión del archivo
    _, ext = os.path.splitext(filepath)
    if ext.lower() not in ['.las', '.laz']:
        raise ValueError(f"Unsupported file format: {ext}. Only .las and .laz files are supported.")
    
    try:
        # Load file
        # Cargar archivo
        las = laspy.read(filepath)
        
        # Extract points
        # Extraer puntos
        points = np.vstack((las.x, las.y, las.z)).transpose()
        
        # Calculate statistics
        # Calcular estadísticas
        summary = {
            'total_points': len(points),
            'x_range': (float(np.min(las.x)), float(np.max(las.x))),
            'y_range': (float(np.min(las.y)), float(np.max(las.y))),
            'z_range': (float(np.min(las.z)), float(np.max(las.z))),
            'point_density': float(len(points)/((np.max(las.x)-np.min(las.x))*(np.max(las.y)-np.min(las.y)))),
            'file_size_mb': os.path.getsize(filepath) / (1024 * 1024),
        }
        
        return points, summary
        
    except Exception as e:
        raise RuntimeError(f"Error loading point cloud: {str(e)}")


def get_file_info(filepath: str) -> Dict:
    """
    Get basic information about a point cloud file without loading all points.
    
    Obtiene información básica sobre un archivo de nube de puntos sin cargar todos los puntos.
    
    Args:
        filepath: Path to .las/.laz file.
        
    Returns:
        Dictionary with file information
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is not supported
    """
    # Validate file exists
    # Validar que el archivo existe
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    
    # Validate file extension
    # Validar extensión del archivo
    _, ext = os.path.splitext(filepath)
    if ext.lower() not in ['.las', '.laz']:
        raise ValueError(f"Unsupported file format: {ext}. Only .las and .laz files are supported.")
    
    # Get file info
    # Obtener información del archivo
    file_size = os.path.getsize(filepath)
    file_info = {
        'filepath': filepath,
        'filename': os.path.basename(filepath),
        'file_size_bytes': file_size,
        'file_size_mb': file_size / (1024 * 1024),
        'file_type': ext.lower(),
    }
    
    try:
        # Read header info only (faster than loading all points)
        # Leer solo información del encabezado (más rápido que cargar todos los puntos)
        with laspy.open(filepath) as f:
            header = f.header
            file_info.update({
                'point_count': header.point_count,
                'version': f"{header.major_version}.{header.minor_version}",
                'point_format': header.point_format.id,
            })
    except Exception as e:
        file_info['header_error'] = str(e)
        
    return file_info


def is_valid_point_cloud_file(filepath: str) -> bool:
    """
    Check if a file is a valid point cloud file that can be loaded.
    
    Verifica si un archivo es un archivo de nube de puntos válido que puede ser cargado.
    
    Args:
        filepath: Path to the file.
        
    Returns:
        True if the file is valid and can be loaded, False otherwise.
    """
    try:
        # Check if file exists
        # Verificar si el archivo existe
        if not os.path.exists(filepath):
            return False
        
        # Check extension
        # Verificar extensión
        _, ext = os.path.splitext(filepath)
        if ext.lower() not in ['.las', '.laz']:
            return False
        
        # Try to open and read header
        # Intentar abrir y leer el encabezado
        with laspy.open(filepath) as f:
            # Just access header to see if it works
            # Solo acceder al encabezado para ver si funciona
            _ = f.header.point_count
        
        return True
    except:
        return False


def get_supported_extensions() -> List[str]:
    """
    Get list of supported point cloud file extensions.
    
    Obtiene la lista de extensiones de archivo de nube de puntos soportadas.
    
    Returns:
        List of supported file extensions (lowercase without dot)
    """
    return ['las', 'laz']
