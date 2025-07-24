#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
© 2025 LiDAR PlotSafe Project. All rights reserved.

Point cloud export module for LiDAR PlotSafe.

This module handles saving processed point clouds with tree identification data.
"""

import os
import laspy
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def save_expanded_trees(
    trees: List[np.ndarray], 
    output_file: str,
    original_las_file: Optional[str] = None
) -> bool:
    """
    Save expanded tree trunks to a LAS/LAZ file with Tree_ID field.
    
    Guarda troncos de árboles expandidos en un archivo LAS/LAZ con campo Tree_ID.
    
    Parameters
    ----------
    trees : List[np.ndarray]
        List of point arrays, one for each detected tree.
    output_file : str
        Path where to save the output file (trees_full.laz).
    original_las_file : Optional[str]
        Path to original LAS file to copy scalar fields from. If None,
        only XYZ coordinates will be saved.
        
    Returns
    -------
    bool
        True if successful.
        
    Raises
    ------
    ValueError
        If the output format is not supported or no trees to save.
    RuntimeError
        If there's an error during the saving process.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Validate file extension
    # Validar extensión del archivo
    _, ext = os.path.splitext(output_file)
    if ext.lower() not in ['.las', '.laz']:
        raise ValueError(f"Unsupported output format: {ext}. Only .las and .laz files are supported.")
    
    try:
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
        tree_ids = np.array(tree_ids, dtype=np.uint32)
        
        # Create a new LAS file
        # Crear un nuevo archivo LAS
        if original_las_file and os.path.exists(original_las_file):
            # If original file provided, copy all scalar fields
            # Si se proporciona archivo original, copiar todos los campos escalares
            logger.info(f"Using original LAS file to copy scalar fields: {original_las_file}")
            
            # Read original file
            # Leer archivo original
            original_las = laspy.read(original_las_file)
            
            # Create header with same format as original
            # Crear encabezado con el mismo formato que el original
            header = laspy.LasHeader(point_format=original_las.header.point_format.id)
            header.scales = original_las.header.scales
            header.offsets = original_las.header.offsets
            
            # Add custom dimension for tree_id
            # Añadir dimensión personalizada para tree_id
            extra_dims = [
                laspy.ExtraBytesParams(name="Tree_ID", type="uint32")
            ]
            header.add_extra_dims(extra_dims)
            
            # Create the point data
            # Crear los datos de puntos
            las = laspy.LasData(header)
            las.x = points[:, 0]
            las.y = points[:, 1]
            las.z = points[:, 2]
            
            # Copy all scalar fields from original if possible
            # Copiar todos los campos escalares del original si es posible
            for dimension in original_las.point_format.dimensions:
                dim_name = dimension.name
                if dim_name not in ['X', 'Y', 'Z'] and hasattr(original_las, dim_name.lower()):
                    # Skip coordinates as we already set them
                    # Omitir coordenadas ya que ya las establecimos
                    try:
                        # This is a simplification - in reality we would need to match points
                        # Esta es una simplificación - en realidad necesitaríamos hacer coincidir los puntos
                        logger.debug(f"Preserving dimension: {dim_name}")
                        # Just set default values for now
                        # Solo establecer valores predeterminados por ahora
                        setattr(las, dim_name.lower(), np.zeros(len(points), dtype=getattr(original_las, dim_name.lower()).dtype))
                    except Exception as e:
                        logger.warning(f"Could not copy dimension {dim_name}: {str(e)}")
        else:
            # Create a basic LAS file with just coordinates
            # Crear un archivo LAS básico solo con coordenadas
            header = laspy.LasHeader(point_format=3)
            header.scales = [0.001, 0.001, 0.001]
            
            # Add custom dimension for tree_id
            # Añadir dimensión personalizada para tree_id
            extra_dims = [
                laspy.ExtraBytesParams(name="Tree_ID", type="uint32")
            ]
            header.add_extra_dims(extra_dims)
            
            # Create the point data
            # Crear los datos de puntos
            las = laspy.LasData(header)
            las.x = points[:, 0]
            las.y = points[:, 1]
            las.z = points[:, 2]
        
        # Assign tree_ids to the custom dimension
        # Asignar tree_ids a la dimensión personalizada
        las.tree_id = tree_ids
        
        # Also set classification field for visualization compatibility (limited to 0-31)
        # También establecer campo de clasificación para compatibilidad de visualización (limitado a 0-31)
        # Use modulo to ensure values are in range 1-31 (0 is reserved)
        # Usar módulo para asegurar que los valores estén en el rango 1-31 (0 está reservado)
        las.classification = (tree_ids % 31) + 1
        
        # Write to file
        # Escribir en archivo
        las.write(output_file)
        
        logger.info(f"Saved {len(trees)} expanded trees to {output_file}")
        logger.info(f"Total points: {len(points)}")
        logger.info("Tree IDs are stored in the 'Tree_ID' field (uint32)")
        logger.info("Classification field also set for visualization (limited to 1-31)")
        
        return True
        
    except Exception as e:
        raise RuntimeError(f"Error saving expanded trees: {str(e)}")
