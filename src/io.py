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

def save_segmented_point_cloud(trees: List[np.ndarray], output_file: str) -> None:
    """
    Save segmented trees to a point cloud file.
    
    Guarda árboles segmentados en un archivo de nube de puntos.
    
    Args:
        trees: List of arrays, each representing a tree as a set of points
        output_file: Path where to save the segmented point cloud
        
    Raises:
        ValueError: If the output format is not supported
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Validate file extension
    # Validar extensión del archivo
    _, ext = os.path.splitext(output_file)
    if ext.lower() not in ['.las', '.laz']:
        raise ValueError(f"Unsupported output format: {ext}. Only .las and .laz files are supported.")
    
    try:
        # Create a new LAS file with tree ID classification
        # Crear un nuevo archivo LAS con clasificación de ID de árbol
        
        # First, combine all points and create a tree_id array
        # Primero, combinar todos los puntos y crear un array de tree_id
        all_points = []
        tree_ids = []
        
        # IDs start from 1 as requested
        # Los IDs comienzan desde 1 como se solicitó
        for tree_id, tree_points in enumerate(trees, start=1):
            all_points.append(tree_points)
            tree_ids.extend([tree_id] * len(tree_points))
        
        if not all_points:
            raise ValueError("No tree points to save")
            
        # Combine all points into a single array
        # Combinar todos los puntos en un único array
        points = np.vstack(all_points)
        tree_ids = np.array(tree_ids)
        
        # Create a new LAS file with custom dimension for tree IDs
        # Crear un nuevo archivo LAS con dimensión personalizada para IDs de árboles
        header = laspy.LasHeader(point_format=3)
        header.scales = [0.001, 0.001, 0.001]
        
        # Add custom dimension for tree_id with no limit (32-bit integer)
        # Añadir dimensión personalizada para tree_id sin límite (entero de 32 bits)
        extra_dims = [
            laspy.ExtraBytesParams(name="tree_id", type="int32")
        ]
        header.add_extra_dims(extra_dims)
        
        # Create the point data with the custom dimension
        # Crear los datos de puntos con la dimensión personalizada
        las = laspy.LasData(header)
        las.x = points[:, 0]
        las.y = points[:, 1]
        las.z = points[:, 2]
        
        # Assign tree_ids to the custom dimension
        # Asignar tree_ids a la dimensión personalizada
        las.tree_id = tree_ids
        
        # Write to file
        # Escribir en archivo
        las.write(output_file)
        
        logger.info(f"Saved {len(trees)} trees to {output_file} with IDs 1-{len(trees)}")
        logger.info("Tree IDs are stored in the custom 'tree_id' field")
        
        return True
        
    except Exception as e:
        raise RuntimeError(f"Error saving segmented point cloud: {str(e)}")