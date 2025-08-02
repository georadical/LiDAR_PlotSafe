# 2025 LiDAR PlotSafe Project. All rights reserved.

"""
Verticality Filter Module for LiDAR PlotSafe Pipeline.

This module provides functionality to filter point clouds based on the verticality
of local surface normals computed using Principal Component Analysis (PCA).
Trees and vertical structures typically have surfaces with normals close to vertical.

Este módulo proporciona funcionalidad para filtrar nubes de puntos basándose en la
verticalidad de las normales de superficie locales calculadas usando Análisis de
Componentes Principales (PCA).
"""

import numpy as np
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def compute_point_normals(
    xyz: np.ndarray,
    radius: float = 0.10,
    min_neighbors: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute surface normals for each point using PCA on local neighborhoods.
    
    EN: Uses KDTree to find neighbors within radius, then applies PCA to estimate
        the local surface normal (3rd principal component with smallest variance).
    ES: Usa KDTree para encontrar vecinos dentro del radio, luego aplica PCA para
        estimar la normal de superficie local (3er componente principal con menor varianza).
    
    Args:
        xyz: Point cloud coordinates (N, 3)
        radius: Search radius for neighborhood detection
        min_neighbors: Minimum neighbors required for PCA computation
        
    Returns:
        normals: Computed normal vectors (N, 3), zero vectors for invalid points
        valid_mask: Boolean mask indicating which points have valid normals
    """
    if len(xyz) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=bool)
    
    n_points = len(xyz)
    normals = np.zeros((n_points, 3), dtype=np.float32)
    valid_mask = np.zeros(n_points, dtype=bool)
    
    # Build KDTree for efficient neighbor search
    # Construir KDTree para búsqueda eficiente de vecinos
    logger.debug("Building KDTree for %d points with radius %.3f", n_points, radius)
    kdt = KDTree(xyz)
    neighbor_indices = kdt.query_radius(xyz, r=radius)
    
    valid_count = 0
    for i, neighbors in enumerate(neighbor_indices):
        if len(neighbors) < min_neighbors:
            continue
            
        # Extract neighborhood points for PCA
        # Extraer puntos del vecindario para PCA
        neighborhood = xyz[neighbors]
        
        try:
            # Center the neighborhood and compute PCA
            # Centrar el vecindario y calcular PCA
            centered = neighborhood - neighborhood.mean(axis=0)
            
            # Check if all points are identical (degenerate case)
            # Verificar si todos los puntos son idénticos (caso degenerado)
            if np.allclose(centered, 0, atol=1e-10):
                continue
            
            # Compute covariance matrix and eigendecomposition
            # Calcular matriz de covarianza y descomposición eigen
            cov_matrix = np.cov(centered, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # The normal is the eigenvector with smallest eigenvalue
            # La normal es el vector propio con menor valor propio
            normal_idx = np.argmin(eigenvalues)
            normal = eigenvectors[:, normal_idx]
            
            # Ensure consistent orientation (point upward if possible)
            # Asegurar orientación consistente (apuntar hacia arriba si es posible)
            if normal[2] < 0:
                normal = -normal
                
            normals[i] = normal
            valid_mask[i] = True
            valid_count += 1
            
        except Exception as e:
            logger.warning("PCA failed for point %d: %s", i, str(e))
            continue
    
    logger.info(
        "Computed normals for %d/%d points (%.1f%%) with radius %.3f",
        valid_count, n_points, 100.0 * valid_count / n_points, radius
    )
    
    return normals, valid_mask


def vertical_mask(
    xyz: np.ndarray,
    radius: float = 0.10,
    cos_thresh: float = 0.85,
    min_neighbors: int = 10,
    precomputed_normals: Optional[np.ndarray] = None,
    precomputed_valid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Return boolean mask of points whose local normal is close to vertical.

    EN: Computes PCA on neighbors within radius; keeps points with
        normals PERPENDICULAR to Z-axis (horizontal normals = vertical surfaces).
        For tree trunks, surface normals point horizontally outward.
    ES: Calcula PCA en vecinos dentro del radio; conserva puntos con
        normales PERPENDICULARES al eje Z (normales horizontales = superficies verticales).
        Para troncos de árboles, las normales apuntan horizontalmente hacia afuera.
    
    Args:
        xyz: Point cloud coordinates (N, 3)
        radius: Search radius for neighborhood detection (meters)
        cos_thresh: Maximum cosine similarity with Z-axis for vertical surfaces (lower = more vertical)
        min_neighbors: Minimum neighbors required for PCA computation
        precomputed_normals: Optional pre-computed normal vectors (N, 3)
        precomputed_valid: Optional pre-computed validity mask for normals
        
    Returns:
        Boolean mask indicating which points belong to vertical surfaces
    """
    if len(xyz) == 0:
        return np.empty(0, dtype=bool)
    
    if precomputed_normals is not None and precomputed_valid is not None:
        # Use precomputed normals to avoid redundant computation
        # Usar normales precomputadas para evitar cálculo redundante
        normals = precomputed_normals
        valid_mask = precomputed_valid
        logger.debug("Using precomputed normals for verticality filtering")
    else:
        # Compute normals on-demand
        # Calcular normales bajo demanda
        normals, valid_mask = compute_point_normals(xyz, radius, min_neighbors)
    
    # Initialize verticality mask
    # Inicializar máscara de verticalidad
    vertical_points = np.zeros(len(xyz), dtype=bool)
    
    if not valid_mask.any():
        logger.warning("No valid normals computed, returning empty mask")
        return vertical_points
    
    # Z-axis unit vector for vertical reference
    # Vector unitario del eje Z para referencia vertical
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    
    # Compute cosine similarity with Z-axis for valid normals
    # Calcular similitud coseno con el eje Z para normales válidas
    valid_normals = normals[valid_mask]
    cos_angles = np.abs(np.dot(valid_normals, z_axis))
    
    # Apply threshold to determine vertical surfaces
    # For vertical surfaces, normals should be perpendicular to Z (small cos_angle)
    # Para superficies verticales, las normales deben ser perpendiculares a Z (cos_angle pequeño)
    vertical_valid = cos_angles <= cos_thresh
    
    # Map back to original point indices
    # Mapear de vuelta a índices de puntos originales
    valid_indices = np.where(valid_mask)[0]
    vertical_indices = valid_indices[vertical_valid]
    vertical_points[vertical_indices] = True
    
    kept_count = vertical_points.sum()
    total_count = len(xyz)
    percentage = 100.0 * kept_count / total_count if total_count > 0 else 0.0
    
    logger.info(
        "Verticality filter: kept %d/%d points (%.1f%%) with cos_thresh=%.2f",
        kept_count, total_count, percentage, cos_thresh
    )
    
    return vertical_points


def apply_verticality_filter(
    xyz: np.ndarray,
    radius: float = 0.10,
    cos_thresh: float = 0.85,
    min_neighbors: int = 10,
    return_normals: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Apply verticality filter and return filtered points with indices.
    
    EN: Convenience function that combines normal computation and filtering
        in a single call. Returns filtered point cloud and mapping indices.
    ES: Función de conveniencia que combina cálculo de normales y filtrado
        en una sola llamada. Devuelve nube filtrada e índices de mapeo.
    
    Args:
        xyz: Input point cloud coordinates (N, 3)
        radius: Search radius for neighborhood detection
        cos_thresh: Minimum cosine similarity with Z-axis
        min_neighbors: Minimum neighbors for PCA computation
        return_normals: Whether to return computed normals for filtered points
        
    Returns:
        filtered_xyz: Filtered point cloud coordinates (M, 3)
        kept_indices: Original indices of kept points (M,)
        normals: Normal vectors for kept points (M, 3) if return_normals=True
    """
    logger.debug(
        "Applying verticality filter: radius=%.3f, cos_thresh=%.2f, min_neighbors=%d",
        radius, cos_thresh, min_neighbors
    )
    
    # Compute verticality mask
    # Calcular máscara de verticalidad
    mask = vertical_mask(xyz, radius, cos_thresh, min_neighbors)
    
    if not mask.any():
        logger.warning("Verticality filter removed all points!")
        return np.empty((0, 3)), np.array([]), None
    
    # Apply filter to get subset
    # Aplicar filtro para obtener subconjunto
    filtered_xyz = xyz[mask]
    kept_indices = np.where(mask)[0]
    
    result_normals = None
    if return_normals:
        # Compute normals for filtered points only
        # Calcular normales solo para puntos filtrados
        normals, _ = compute_point_normals(filtered_xyz, radius, min_neighbors)
        result_normals = normals
    
    logger.info(
        "Applied verticality filter: %d points kept from %d original (%.1f%%)",
        len(filtered_xyz), len(xyz), 100.0 * len(filtered_xyz) / len(xyz)
    )
    
    return filtered_xyz, kept_indices, result_normals
