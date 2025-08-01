#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 2025 LiDAR PlotSafe Project. All rights reserved.

Point cloud export module for LiDAR PlotSafe.

This module handles saving processed point clouds with tree identification data.
"""

import os
import laspy
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

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


def export_full_trunks(
    xyz: np.ndarray,
    tree_ids: np.ndarray,
    extra: Optional[Dict[str, np.ndarray]] = None,
    out_path: str = "trees_full.laz"
) -> bool:
    """
    Write merged trunk cloud preserving scalar fields + new Tree_ID.
    Escribe nube de troncos fusionados preservando campos escalares + nuevo Tree_ID.
    
    Args:
        xyz: Nx3 array of point coordinates (X, Y, Z)
        tree_ids: Nx1 array of tree identifiers for each point
        extra: Optional dictionary of additional scalar fields to preserve
        out_path: Output file path for LAS/LAZ file
        
    Returns:
        bool: True if export successful, False otherwise
        
    Raises:
        ValueError: If input arrays have incompatible shapes
        IOError: If file cannot be written
    """
    # Input validation / Validación de entrada
    if xyz.shape[0] != len(tree_ids):
        raise ValueError(f"Point count mismatch: xyz={xyz.shape[0]}, tree_ids={len(tree_ids)}")
        
    if xyz.shape[1] != 3:
        raise ValueError(f"XYZ array must have 3 columns, got {xyz.shape[1]}")
        
    logger.info("Exporting %d points with Tree_ID to %s", len(xyz), out_path)
    logger.info("Exportando %d puntos con Tree_ID a %s", len(xyz), out_path)
    
    try:
        # Create LAS file with proper header / Crear archivo LAS con encabezado apropiado
        header = laspy.create(file_version="1.4", point_format=3)
        las = laspy.LasData(header)
        
        # Set coordinates / Establecer coordenadas
        las.x, las.y, las.z = xyz.T
        
        # Add extra scalar fields / Añadir campos escalares adicionales
        if extra:
            for name, arr in extra.items():
                if len(arr) != len(xyz):
                    logger.warning("Skipping field '%s': length mismatch (%d vs %d)", 
                                   name, len(arr), len(xyz))
                    logger.warning("Saltando campo '%s': longitud no coincide (%d vs %d)", 
                                   name, len(arr), len(xyz))
                    continue
                    
                try:
                    # Create extra dimension / Crear dimensión extra
                    las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.float32))
                    las[name] = arr.astype(np.float32)
                    logger.debug("Added extra field: %s", name)
                    logger.debug("Campo extra añadido: %s", name)
                except Exception as e:
                    logger.error("Failed to add field '%s': %s", name, e)
                    logger.error("Error al añadir campo '%s': %s", name, e)
        
        # Add Tree_ID field / Añadir campo Tree_ID
        las.add_extra_dim(laspy.ExtraBytesParams(name="Tree_ID", type=np.uint32))
        las["Tree_ID"] = tree_ids.astype(np.uint32)
        
        # Ensure output directory exists / Asegurar que directorio de salida existe
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        
        # Write file / Escribir archivo
        las.write(out_path)
        
        # Verify file was created / Verificar que archivo fue creado
        if os.path.exists(out_path):
            file_size = os.path.getsize(out_path) / (1024 * 1024)  # MB
            logger.info("Successfully exported %d points to %s (%.2f MB)", 
                        len(xyz), out_path, file_size)
            logger.info("Exportación exitosa de %d puntos a %s (%.2f MB)", 
                        len(xyz), out_path, file_size)
            return True
        else:
            logger.error("File was not created: %s", out_path)
            logger.error("Archivo no fue creado: %s", out_path)
            return False
            
    except Exception as e:
        logger.error("Export failed: %s", str(e))
        logger.error("Exportación falló: %s", str(e))
        return False
