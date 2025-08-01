# -*- coding: utf-8 -*-
"""
Vertical expansion and duplicate merging for tree trunk clusters.
Expansión vertical y fusión de duplicados para clusters de troncos de árboles.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExpansionConfig:
    """
    Configuration parameters for trunk expansion and merging.
    Parámetros de configuración para expansión y fusión de troncos.
    """
    expansion_radius: float = 0.5      # Radius for vertical expansion / Radio para expansión vertical
    merge_tolerance: float = 0.25      # Distance threshold for merging / Umbral de distancia para fusión
    min_points_per_trunk: int = 10     # Minimum points required per trunk / Puntos mínimos requeridos por tronco
    max_expansion_ratio: float = 5.0   # Max ratio of expanded to original points / Ratio máximo de puntos expandidos a originales


def expand_cluster_to_trunk(
    cluster_xy: np.ndarray,
    full_pc: np.ndarray,
    radius: float = 0.5,
    min_points: int = 10,
    max_ratio: float = 5.0
) -> Optional[np.ndarray]:
    """
    Expand slice cluster to full trunk using XY radius buffer.
    Expande cluster de corte a tronco completo usando buffer de radio XY.
    
    Args:
        cluster_xy: 2D/3D points from slice cluster (N, 2/3)
        full_pc: Complete point cloud (M, 3+)  
        radius: Expansion radius in meters
        min_points: Minimum points required for valid trunk
        max_ratio: Maximum expansion ratio to prevent over-expansion
        
    Returns:
        Expanded trunk points or None if invalid
        
    Raises:
        ValueError: If input arrays are invalid
    """
    # Input validation / Validación de entrada
    if cluster_xy.shape[0] == 0:
        logger.warning("Empty cluster provided for expansion")
        return None
        
    if cluster_xy.shape[1] < 2:
        raise ValueError("Cluster must have at least XY coordinates")
        
    if full_pc.shape[0] == 0 or full_pc.shape[1] < 3:
        raise ValueError("Full point cloud must have XYZ coordinates")
        
    try:
        # Calculate cluster centroid / Calcula centroide del cluster
        centroid = cluster_xy[:, :2].mean(axis=0)
        logger.debug(f"Cluster centroid at ({centroid[0]:.3f}, {centroid[1]:.3f})")
        
        # Find points within radius / Encuentra puntos dentro del radio
        distances_sq = np.sum((full_pc[:, :2] - centroid) ** 2, axis=1)
        mask = distances_sq <= (radius ** 2)
        expanded_points = full_pc[mask]
        
        # Validate expansion / Valida expansión
        if expanded_points.shape[0] < min_points:
            logger.warning(f"Expanded trunk has only {expanded_points.shape[0]} points, minimum {min_points}")
            return None
            
        expansion_ratio = expanded_points.shape[0] / cluster_xy.shape[0]
        if expansion_ratio > max_ratio:
            logger.warning(f"Expansion ratio {expansion_ratio:.1f} exceeds maximum {max_ratio}")
            return None
            
        logger.debug(f"Expanded {cluster_xy.shape[0]} → {expanded_points.shape[0]} points (ratio: {expansion_ratio:.1f})")
        return expanded_points
        
    except Exception as e:
        logger.error(f"Error expanding cluster: {e}")
        return None


def merge_overlapping_trunks(
    trunks: List[np.ndarray],
    merge_tolerance: float = 0.25
) -> List[np.ndarray]:
    """
    Merge trunk clusters with nearby centroids using Union-Find algorithm.
    Fusiona clusters de troncos con centroides cercanos usando algoritmo Union-Find.
    
    Args:
        trunks: List of trunk point arrays
        merge_tolerance: Distance threshold for merging centroids
        
    Returns:
        List of merged trunk clusters
    """
    if not trunks or len(trunks) <= 1:
        return trunks
        
    try:
        # Calculate centroids / Calcula centroides  
        centroids = []
        valid_indices = []
        
        for i, trunk in enumerate(trunks):
            if trunk.shape[0] > 0 and trunk.shape[1] >= 2:
                centroids.append(trunk[:, :2].mean(axis=0))
                valid_indices.append(i)
            else:
                logger.warning(f"Skipping invalid trunk {i} with shape {trunk.shape}")
                
        if len(centroids) <= 1:
            return [trunks[i] for i in valid_indices]
            
        centroids = np.array(centroids)
        
        # Union-Find data structure / Estructura de datos Union-Find
        parent = list(range(len(centroids)))
        
        def find_root(i: int) -> int:
            """Find root with path compression / Encuentra raíz con compresión de ruta"""
            if parent[i] != i:
                parent[i] = find_root(parent[i])
            return parent[i]
            
        def union(i: int, j: int) -> None:
            """Union two components / Une dos componentes"""
            root_i, root_j = find_root(i), find_root(j)
            if root_i != root_j:
                parent[root_j] = root_i
                
        # Merge nearby centroids / Fusiona centroides cercanos
        merge_count = 0
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                distance = np.linalg.norm(centroids[i] - centroids[j])
                if distance < merge_tolerance:
                    union(i, j)
                    merge_count += 1
                    logger.debug(f"Merging trunks {i}-{j}, distance: {distance:.3f}m")
        
        # Group by root component / Agrupa por componente raíz
        groups = {}
        for idx in range(len(centroids)):
            root = find_root(idx)
            if root not in groups:
                groups[root] = []
            groups[root].append(valid_indices[idx])
            
        # Merge trunk points / Fusiona puntos de troncos
        merged_trunks = []
        for group_indices in groups.values():
            if len(group_indices) == 1:
                merged_trunks.append(trunks[group_indices[0]])
            else:
                # Stack points from all trunks in group / Apila puntos de todos los troncos en el grupo
                group_points = [trunks[i] for i in group_indices]
                merged_trunk = np.vstack(group_points)
                merged_trunks.append(merged_trunk)
                
        logger.info(f"Merged {len(trunks)} trunks → {len(merged_trunks)} final trunks ({merge_count} merges)")
        return merged_trunks
        
    except Exception as e:
        logger.error(f"Error merging trunks: {e}")
        return trunks


def expand_and_merge_clusters(
    slice_clusters: List[np.ndarray],
    full_pc: np.ndarray,
    config: ExpansionConfig
) -> Tuple[List[np.ndarray], dict]:
    """
    Complete pipeline: expand slice clusters to trunks and merge duplicates.
    Pipeline completo: expande clusters de corte a troncos y fusiona duplicados.
    
    Args:
        slice_clusters: List of 2D slice cluster point arrays
        full_pc: Complete 3D point cloud
        config: Expansion configuration parameters
        
    Returns:
        Tuple of (merged_trunks, metadata_dict)
    """
    if not slice_clusters:
        logger.warning("No slice clusters provided for expansion")
        return [], {}
        
    logger.info(f"Starting expansion of {len(slice_clusters)} slice clusters")
    
    # Step 1: Expand each slice cluster / Paso 1: Expande cada cluster de corte
    expanded_trunks = []
    expansion_stats = {
        'successful_expansions': 0,
        'failed_expansions': 0,
        'total_points_before': 0,
        'total_points_after': 0
    }
    
    for i, cluster in enumerate(slice_clusters):
        expansion_stats['total_points_before'] += cluster.shape[0]
        
        expanded = expand_cluster_to_trunk(
            cluster,
            full_pc,
            radius=config.expansion_radius,
            min_points=config.min_points_per_trunk,
            max_ratio=config.max_expansion_ratio
        )
        
        if expanded is not None:
            expanded_trunks.append(expanded)
            expansion_stats['successful_expansions'] += 1
            expansion_stats['total_points_after'] += expanded.shape[0]
        else:
            expansion_stats['failed_expansions'] += 1
            logger.debug(f"Failed to expand cluster {i}")
    
    # Step 2: Merge overlapping trunks / Paso 2: Fusiona troncos superpuestos  
    if expanded_trunks:
        merged_trunks = merge_overlapping_trunks(
            expanded_trunks,
            merge_tolerance=config.merge_tolerance
        )
    else:
        merged_trunks = []
        
    # Compile metadata / Compila metadatos
    metadata = {
        'input_clusters': len(slice_clusters),
        'expanded_trunks': len(expanded_trunks),
        'final_trunks': len(merged_trunks),
        'expansion_stats': expansion_stats,
        'config': config
    }
    
    logger.info(f"Expansion complete: {len(slice_clusters)} clusters → {len(merged_trunks)} final trunks")
    return merged_trunks, metadata


def expand_and_merge(
    slice_clusters: List[np.ndarray],
    full_pc: np.ndarray,
    radius: float = 0.5,
    merge_tol: float = 0.25
) -> List[np.ndarray]:
    """
    Simplified interface: expand slice clusters to trunks and merge duplicates.
    Interfaz simplificada: expande clusters de corte a troncos y fusiona duplicados.
    
    Args:
        slice_clusters: List of 2D slice cluster point arrays
        full_pc: Complete 3D point cloud
        radius: Expansion radius in meters
        merge_tol: Distance threshold for merging centroids
        
    Returns:
        List of merged trunk clusters
    """
    # Create default configuration / Crear configuración por defecto
    config = ExpansionConfig(
        expansion_radius=radius,
        merge_tolerance=merge_tol,
        min_points_per_trunk=10,
        max_expansion_ratio=5.0
    )
    
    # Use the complete pipeline / Usar el pipeline completo
    trunks, metadata = expand_and_merge_clusters(slice_clusters, full_pc, config)
    
    logger.info("Simplified expand_and_merge: %d clusters → %d trunks", 
                len(slice_clusters), len(trunks))
    logger.info("expand_and_merge simplificado: %d clusters → %d troncos", 
                len(slice_clusters), len(trunks))
    
    return trunks
