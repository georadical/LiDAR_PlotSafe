# © 2025 LiDAR PlotSafe Project. All rights reserved.

"""
Módulo de podado iterativo para mejorar la segmentación de árboles.

Iterative pruning module for improving tree segmentation.
"""

import numpy as np
import logging
from typing import List, Optional
from pipeline.verticality import vertical_mask
from pipeline.clustering import dbscan_trunks

logger = logging.getLogger(__name__)


def iterative_prune(
    slice_pts: np.ndarray,
    eps: float,
    min_samples: int,
    iterations: int = 2,
    vert_radius: float = 0.10,
    cos_thresh: float = 0.85
) -> List[np.ndarray]:
    """
    Aplica repetidamente vertical-mask + DBSCAN para refinar clusters.
    
    Repeatedly apply vertical-mask + DBSCAN to refine clusters.
    
    This iterative approach helps to separate tree stems that are initially
    connected by branches or other bridge-like structures. Each iteration
    applies vertical filtering followed by DBSCAN clustering, using the
    results as input for the next iteration.
    
    Este enfoque iterativo ayuda a separar troncos de árboles que inicialmente
    están conectados por ramas u otras estructuras tipo puente. Cada iteración
    aplica filtrado vertical seguido de clustering DBSCAN, usando los
    resultados como entrada para la siguiente iteración.

    Args:
        slice_pts: Nx3 array of slice points to process
                   Array Nx3 de puntos de rebanada para procesar
        eps: Maximum distance between points in DBSCAN clustering
             Distancia máxima entre puntos en clustering DBSCAN
        min_samples: Minimum number of points to form a cluster
                     Número mínimo de puntos para formar un cluster
        iterations: Number of pruning iterations to perform (default: 2)
                   Número de iteraciones de podado a realizar (predeterminado: 2)
        vert_radius: Search radius for verticality filter in meters
                    Radio de búsqueda para filtro de verticalidad en metros
        cos_thresh: Cosine threshold for vertical surface detection
                   Umbral de coseno para detección de superficies verticales

    Returns:
        List of final slice clusters (arrays of points)
        Lista de clusters finales de rebanada (arrays de puntos)
    """
    # Validar entrada de datos
    # Validate input data
    if len(slice_pts) == 0:
        logger.warning("Empty input points for iterative pruning")
        logger.warning("Puntos de entrada vacíos para podado iterativo")
        return []
    
    if iterations < 1:
        logger.warning("Invalid iterations count: %d, using 1", iterations)
        logger.warning("Número de iteraciones inválido: %d, usando 1", iterations)
        iterations = 1
    
    logger.info("Starting iterative pruning with %d iterations on %d points", 
                iterations, len(slice_pts))
    logger.info("Iniciando podado iterativo con %d iteraciones en %d puntos", 
                iterations, len(slice_pts))
    
    current = slice_pts.copy()  # Work with a copy to avoid modifying original
    clusters = []
    
    for iteration in range(iterations):
        logger.debug("Pruning iteration %d/%d: processing %d points", 
                     iteration + 1, iterations, len(current))
        logger.debug("Iteración de podado %d/%d: procesando %d puntos", 
                     iteration + 1, iterations, len(current))
        
        # Aplicar filtro de verticalidad
        # Apply verticality filter
        mask = vertical_mask(
            current, 
            radius=vert_radius, 
            cos_thresh=cos_thresh
        )
        vert_pts = current[mask]
        
        # Verificar si quedan puntos después del filtrado
        # Check if points remain after filtering
        if len(vert_pts) == 0:
            logger.info("No vertical points remain after iteration %d, stopping", 
                       iteration + 1)
            logger.info("No quedan puntos verticales después de iteración %d, deteniendo", 
                       iteration + 1)
            break
        
        logger.debug("Verticality filter: %d/%d points retained", 
                     len(vert_pts), len(current))
        logger.debug("Filtro de verticalidad: %d/%d puntos retenidos", 
                     len(vert_pts), len(current))
        
        # Aplicar clustering DBSCAN
        # Apply DBSCAN clustering
        _, iter_clusters = dbscan_trunks(
            vert_pts, 
            eps=eps, 
            min_samples=min_samples
        )
        
        # Verificar si se encontraron clusters
        # Check if clusters were found
        if not iter_clusters:
            logger.info("No clusters found in iteration %d, stopping", iteration + 1)
            logger.info("No se encontraron clusters en iteración %d, deteniendo", iteration + 1)
            break
        
        clusters = iter_clusters
        logger.debug("Iteration %d produced %d clusters", 
                     iteration + 1, len(clusters))
        logger.debug("Iteración %d produjo %d clusters", 
                     iteration + 1, len(clusters))
        
        # Si es la última iteración, no necesitamos preparar para la siguiente
        # If it's the last iteration, we don't need to prepare for the next
        if iteration < iterations - 1:
            # Aplanar clusters de vuelta en array para la siguiente ronda
            # Flatten clusters back into array for next round
            if clusters:
                try:
                    current = np.vstack(clusters)
                    logger.debug("Flattened %d clusters into %d points for next iteration", 
                                len(clusters), len(current))
                    logger.debug("Aplanados %d clusters en %d puntos para siguiente iteración", 
                                len(clusters), len(current))
                except ValueError as e:
                    logger.error("Error flattening clusters: %s", str(e))
                    logger.error("Error aplanando clusters: %s", str(e))
                    break
            else:
                logger.info("No clusters to flatten, stopping iterations")
                logger.info("No hay clusters para aplanar, deteniendo iteraciones")
                break
    
    # Registrar resultados finales
    # Log final results
    final_cluster_count = len(clusters) if clusters else 0
    if clusters:
        total_points = sum(len(cluster) for cluster in clusters)
        avg_points = total_points / final_cluster_count if final_cluster_count > 0 else 0
        logger.info("Iterative pruning completed: %d final clusters, %d total points, %.1f avg points per cluster", 
                   final_cluster_count, total_points, avg_points)
        logger.info("Podado iterativo completado: %d clusters finales, %d puntos totales, %.1f puntos promedio por cluster", 
                   final_cluster_count, total_points, avg_points)
    else:
        logger.info("Iterative pruning completed: no clusters found")
        logger.info("Podado iterativo completado: no se encontraron clusters")
    
    return clusters if clusters else []


def validate_pruning_parameters(
    iterations: int,
    vert_radius: float,
    cos_thresh: float
) -> bool:
    """
    Valida los parámetros de podado iterativo.
    
    Validates iterative pruning parameters.
    
    Args:
        iterations: Number of pruning iterations
                   Número de iteraciones de podado
        vert_radius: Verticality filter search radius
                    Radio de búsqueda del filtro de verticalidad
        cos_thresh: Cosine threshold for verticality
                   Umbral de coseno para verticalidad
    
    Returns:
        True if parameters are valid, False otherwise
        True si los parámetros son válidos, False en caso contrario
    """
    # Validar número de iteraciones
    # Validate iteration count
    if iterations < 1 or iterations > 10:
        logger.error("Invalid iterations count: %d (must be 1-10)", iterations)
        logger.error("Número de iteraciones inválido: %d (debe ser 1-10)", iterations)
        return False
    
    # Validar radio de verticalidad
    # Validate verticality radius
    if vert_radius <= 0 or vert_radius > 1.0:
        logger.error("Invalid verticality radius: %.3f (must be 0 < radius <= 1.0)", vert_radius)
        logger.error("Radio de verticalidad inválido: %.3f (debe ser 0 < radio <= 1.0)", vert_radius)
        return False
    
    # Validar umbral de coseno
    # Validate cosine threshold
    if cos_thresh < 0.0 or cos_thresh > 1.0:
        logger.error("Invalid cosine threshold: %.3f (must be 0.0 <= threshold <= 1.0)", cos_thresh)
        logger.error("Umbral de coseno inválido: %.3f (debe ser 0.0 <= umbral <= 1.0)", cos_thresh)
        return False
    
    return True
