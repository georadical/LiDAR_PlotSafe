# 2025 LiDAR PlotSafe Project. All rights reserved.

"""
Módulo de clustering para LiDAR PlotSafe.

Clustering module for LiDAR PlotSafe.
"""

import numpy as np
import logging
from typing import Tuple, List, Optional, Dict, Any
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


def compute_eps(
    slice_xy: np.ndarray, 
    k: int = 8, 
    factor: float = 2.0,
    min_eps: float = 0.02,
    max_eps: float = 1.0
) -> float:
    """
    Calcula un radio adaptativo de DBSCAN basado en la densidad local XY.
    
    Calculates an adaptive DBSCAN radius based on local XY density.

    EN: Uses mean k-NN distance * factor (default 2×).
    ES: Calcula ε = factor × distancia media a los k vecinos más cercanos.

    Parameters
    ----------
    slice_xy : np.ndarray
        (N,2) array of XY coords of points in the horizontal slice.
    k : int, optional
        Number of neighbours to consider (≥ 4), by default 8
    factor : float, optional
        Multiplicative factor applied to mean distance, by default 2.0
    min_eps : float, optional
        Minimum allowed epsilon value, by default 0.02
    max_eps : float, optional
        Maximum allowed epsilon value, by default 1.0

    Returns
    -------
    float
        Recommended ε for DBSCAN.
        
    Raises
    ------
    ValueError
        If k < 1 or factor <= 0
    """
    # Parameter validation
    # Validación de parámetros
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if factor <= 0:
        raise ValueError(f"factor must be > 0, got {factor}")
    if len(slice_xy) == 0:
        logger.warning("Empty point set, returning min_eps=%.3f", min_eps)
        logger.warning("Conjunto de puntos vacío, devolviendo min_eps=%.3f", min_eps)
        return min_eps
    
    # Handle case where we have fewer points than k neighbors
    # Manejar caso donde tenemos menos puntos que k vecinos
    if len(slice_xy) <= k:
        logger.warning("Too few points (%d) for k=%d neighbors, using min_eps=%.3f",
                      len(slice_xy), k, min_eps)
        logger.warning("Muy pocos puntos (%d) para k=%d vecinos, usando min_eps=%.3f",
                      len(slice_xy), k, min_eps)
        return min_eps
    
    try:
        # Build k-NN tree for efficient neighbor search
        # Construir árbol k-NN para búsqueda eficiente de vecinos
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(slice_xy)
        distances, _ = nbrs.kneighbors(slice_xy)
        
        # Calculate mean distance to k-th neighbor
        # Calcular distancia media al k-ésimo vecino
        mean_d = distances[:, -1].mean()
        
        # Apply factor and clamp to allowed range
        # Aplicar factor y limitar al rango permitido
        eps = factor * mean_d
        eps = np.clip(eps, min_eps, max_eps)
        
        logger.info("Computed adaptive eps: %.3f (mean k-dist=%.3f, factor=%.1f)",
                   eps, mean_d, factor)
        logger.info("Eps adaptativo calculado: %.3f (k-dist media=%.3f, factor=%.1f)",
                   eps, mean_d, factor)
        
        return float(eps)
        
    except Exception as e:
        logger.error("Error computing adaptive eps: %s. Using min_eps=%.3f", 
                    str(e), min_eps)
        logger.error("Error calculando eps adaptativo: %s. Usando min_eps=%.3f", 
                    str(e), min_eps)
        return min_eps


def dbscan_trunks(
    slice_xy: np.ndarray, 
    eps: float, 
    min_samples: int = 5
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Wrapper que realiza DBSCAN 2-D en coordenadas de rebanada.
    
    Wrapper that performs 2-D DBSCAN on slice coordinates.

    Returns labels (-1 = noise) and unique clusters.
    Retorna etiquetas (-1 = ruido) y clusters únicos.
    
    Parameters
    ----------
    slice_xy : np.ndarray
        (N,2) array of XY coordinates
    eps : float
        Maximum distance between points in a cluster
    min_samples : int, optional
        Minimum number of points to form a cluster, by default 5
        
    Returns
    -------
    labels : np.ndarray
        Cluster labels for each point (-1 indicates noise)
    clusters : List[np.ndarray]
        List of point arrays, one for each cluster (excluding noise)
        
    Raises
    ------
    ValueError
        If eps <= 0 or min_samples < 1
    """
    # Parameter validation
    # Validación de parámetros
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")
    if min_samples < 1:
        raise ValueError(f"min_samples must be >= 1, got {min_samples}")
    
    if len(slice_xy) == 0:
        logger.warning("Empty point set for DBSCAN clustering")
        logger.warning("Conjunto de puntos vacío para clustering DBSCAN")
        return np.array([]), []
    
    try:
        # Perform DBSCAN clustering
        # Realizar clustering DBSCAN
        dbs = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbs.fit_predict(slice_xy)
        
        # Extract unique clusters (excluding noise label -1)
        # Extraer clusters únicos (excluyendo etiqueta de ruido -1)
        unique_labels = np.unique(labels)
        clusters = []
        
        for lab in unique_labels:
            if lab != -1:  # Skip noise points
                cluster_points = slice_xy[labels == lab]
                clusters.append(cluster_points)
        
        # Log clustering results
        # Registrar resultados de clustering
        n_noise = np.sum(labels == -1)
        logger.info("DBSCAN clustering: %d clusters, %d noise points (%.1f%% noise)",
                   len(clusters), n_noise, 100 * n_noise / len(slice_xy) if len(slice_xy) > 0 else 0)
        logger.info("Clustering DBSCAN: %d clusters, %d puntos ruido (%.1f%% ruido)",
                   len(clusters), n_noise, 100 * n_noise / len(slice_xy) if len(slice_xy) > 0 else 0)
        
        if len(clusters) > 0:
            cluster_sizes = [len(cluster) for cluster in clusters]
            logger.debug("Cluster sizes: min=%d, max=%d, avg=%.1f",
                        min(cluster_sizes), max(cluster_sizes), np.mean(cluster_sizes))
        
        return labels, clusters
        
    except Exception as e:
        logger.error("Error in DBSCAN clustering: %s", str(e))
        logger.error("Error en clustering DBSCAN: %s", str(e))
        return np.array([]), []


def adaptive_dbscan_trunks(
    slice_xy: np.ndarray,
    k: int = 8,
    factor: float = 2.0,
    min_samples: int = 5,
    eps_config: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, List[np.ndarray], float]:
    """
    Realiza clustering DBSCAN con cálculo automático de epsilon.
    
    Performs DBSCAN clustering with automatic epsilon calculation.
    
    Parameters
    ----------
    slice_xy : np.ndarray
        (N,2) array of XY coordinates
    k : int, optional
        Number of neighbors for epsilon calculation, by default 8
    factor : float, optional
        Factor to multiply mean k-distance, by default 2.0
    min_samples : int, optional
        Minimum samples for DBSCAN, by default 5
    eps_config : Dict[str, Any], optional
        Configuration for epsilon calculation (min_eps, max_eps)
        
    Returns
    -------
    labels : np.ndarray
        Cluster labels for each point (-1 indicates noise)
    clusters : List[np.ndarray]
        List of point arrays, one for each cluster
    eps : float
        The epsilon value that was computed and used
    """
    # Set up epsilon configuration
    # Configurar configuración de epsilon
    eps_cfg = {'min_eps': 0.02, 'max_eps': 1.0}
    if eps_config:
        eps_cfg.update(eps_config)
    
    # Compute adaptive epsilon
    # Calcular epsilon adaptativo
    eps = compute_eps(
        slice_xy, 
        k=k, 
        factor=factor, 
        min_eps=eps_cfg['min_eps'], 
        max_eps=eps_cfg['max_eps']
    )
    
    # Perform clustering with computed epsilon
    # Realizar clustering con el epsilon calculado
    labels, clusters = dbscan_trunks(slice_xy, eps, min_samples)
    
    return labels, clusters, eps
