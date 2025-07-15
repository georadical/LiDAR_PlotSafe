# 2025 LiDAR PlotSafe Project. All rights reserved.

"""
Módulo de segmentación de árboles para LiDAR PlotSafe.

Tree segmentation module for LiDAR PlotSafe.
"""

import numpy as np
import logging
import time
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


def downsample_point_cloud(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Submuestrea la nube de puntos usando un tamaño de voxel fijo.
    
    Downsamples the point cloud using a fixed voxel size.
    
    Args:
        points: Nx3 array of point coordinates
        voxel_size: Size of voxels for downsampling
        
    Returns:
        Downsampled point cloud as Nx3 array
    """
    # Esta es una implementación simplificada; en producción usaríamos Open3D
    # This is a simplified implementation; in production we would use Open3D
    
    # Calcular límites de la nube de puntos
    # Calculate point cloud bounds
    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)
    
    # Discretizar puntos en voxels
    # Discretize points into voxels
    voxel_indices = np.floor((points - min_bounds) / voxel_size).astype(int)
    
    # Identificar voxels únicos y tomar un representante de cada uno
    # Identify unique voxels and take one representative from each
    voxel_dict = {}
    for i, idx in enumerate(voxel_indices):
        key = tuple(idx)
        if key not in voxel_dict:
            voxel_dict[key] = i
    
    # Construir nube submuestreada
    # Build downsampled cloud
    indices = list(voxel_dict.values())
    downsampled = points[indices]
    
    logger.info("Downsampled from %d to %d points", len(points), len(downsampled))
    logger.info("Submuestreado de %d a %d puntos", len(points), len(downsampled))
    
    return downsampled


def extract_horizontal_slice(
    points: np.ndarray, 
    slice_height: float, 
    slice_thickness: float
) -> np.ndarray:
    """
    Extrae una rebanada horizontal de la nube de puntos a una altura específica.
    
    Extracts a horizontal slice from the point cloud at a specific height.
    
    Args:
        points: Nx3 array of point coordinates
        slice_height: Height at which to extract the slice
        slice_thickness: Thickness of the slice
        
    Returns:
        Points within the horizontal slice as Mx3 array
    """
    # Calcular límites superior e inferior de la rebanada
    # Calculate upper and lower bounds of the slice
    lower_bound = slice_height - slice_thickness / 2
    upper_bound = slice_height + slice_thickness / 2
    
    # Extraer puntos dentro del rango de altura
    # Extract points within the height range
    mask = (points[:, 2] >= lower_bound) & (points[:, 2] <= upper_bound)
    slice_points = points[mask]
    
    logger.info("Extracted %d points in horizontal slice at height %.2f m (±%.2f m)",
                len(slice_points), slice_height, slice_thickness/2)
    logger.info("Extraídos %d puntos en rebanada horizontal a altura %.2f m (±%.2f m)",
                len(slice_points), slice_height, slice_thickness/2)
    
    return slice_points


def cluster_trunks(
    points: np.ndarray, 
    eps: float, 
    min_samples: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Agrupa puntos para identificar troncos individuales usando DBSCAN.
    
    Clusters points to identify individual trunks using DBSCAN.
    
    Args:
        points: Nx3 array of point coordinates
        eps: Maximum distance between points in DBSCAN clustering
        min_samples: Minimum number of points to form a cluster
        
    Returns:
        Tuple containing:
        - Nx1 array of cluster labels (-1 for noise)
        - List of point arrays, one for each cluster
    """
    if len(points) == 0:
        return np.array([]), []
    
    # Aplicar DBSCAN solo a coordenadas X e Y (ignorando altura)
    # Apply DBSCAN only to X and Y coordinates (ignoring height)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points[:, :2])
    
    # Extraer clusters individuales
    # Extract individual clusters
    unique_labels = np.unique(labels)
    clusters = []
    
    for label in unique_labels:
        if label == -1:  # Ignorar ruido / Ignore noise
            continue
        
        # Extraer puntos del cluster
        # Extract cluster points
        cluster_points = points[labels == label]
        clusters.append(cluster_points)
    
    logger.info("Found %d clusters with DBSCAN", len(clusters))
    logger.info("Encontrados %d clusters con DBSCAN", len(clusters))
    
    return labels, clusters


def filter_tree_clusters(
    clusters: List[np.ndarray], 
    min_tree_height: float = 1.5,
    min_points: int = 50,
    full_point_cloud: Optional[np.ndarray] = None
) -> List[np.ndarray]:
    """
    Filtra clusters para eliminar falsos positivos basándose en altura y número de puntos.
    
    Filters clusters to remove false positives based on height and number of points.
    
    Args:
        clusters: List of point clusters (each is a numpy array of shape (n, 3))
                Lista de clusters de puntos (cada uno es un array numpy de forma (n, 3))
        min_tree_height: Minimum height to consider a cluster as a tree
                        Altura mínima para considerar un cluster como árbol
        min_points: Minimum number of points for a valid tree cluster
                   Número mínimo de puntos para un cluster de árbol válido
        full_point_cloud: Optional full point cloud for height calculation
                         Nube de puntos completa opcional para cálculo de altura
    
    Returns:
        List of clusters that passed the filter
        Lista de clusters que pasaron el filtro
    """
    logger.info("Filtering %d clusters (min_height=%.2f, min_points=%d, full_point_cloud=%s)",
                len(clusters), min_tree_height, min_points, "provided" if full_point_cloud is not None else "not provided")
    logger.info("Filtrando %d clusters (altura_min=%.2f, puntos_min=%d, nube_completa=%s)",
                len(clusters), min_tree_height, min_points, "proporcionada" if full_point_cloud is not None else "no proporcionada")
    
    valid_clusters = []
    
    for i, cluster in enumerate(clusters):
        # Verificar número mínimo de puntos
        # Check minimum number of points
        if len(cluster) < min_points:
            logger.debug("Cluster %d rejected: too few points (%d < %d)", 
                        i, len(cluster), min_points)
            continue
        
        # Calcular la altura utilizando la nube de puntos completa si está disponible
        # Calculate height using full point cloud if available
        if full_point_cloud is not None:
            # Calcular centro del cluster (solo coordenadas xy)
            # Calculate cluster center (xy coordinates only)
            center_xy = np.mean(cluster[:, :2], axis=0)
            
            # Radio de búsqueda para puntos cercanos (un poco más grande que el radio típico del tronco)
            # Search radius for nearby points (slightly larger than typical trunk radius)
            search_radius = 0.5  # metros / meters
            
            # Encontrar todos los puntos de la nube completa que están cerca del centro del cluster
            # Find all points from the full cloud that are near the cluster center
            distances_xy = np.sqrt(np.sum((full_point_cloud[:, :2] - center_xy) ** 2, axis=1))
            nearby_points = full_point_cloud[distances_xy < search_radius]
            
            if len(nearby_points) > 0:
                # Calcular altura como la diferencia entre el punto más alto y el más bajo
                # Calculate height as the difference between highest and lowest points
                min_z = np.min(nearby_points[:, 2])
                max_z = np.max(nearby_points[:, 2])
                height = max_z - min_z
                
                logger.debug("Cluster %d: calculated height from full cloud = %.2f m (%d nearby points)", 
                            i, height, len(nearby_points))
            else:
                # Si no hay puntos cercanos en la nube completa, usar solo el cluster actual
                # If no nearby points in full cloud, use only the current cluster
                min_z = np.min(cluster[:, 2])
                max_z = np.max(cluster[:, 2])
                height = max_z - min_z
                
                logger.debug("Cluster %d: no nearby points in full cloud, using cluster height = %.2f m", 
                            i, height)
        else:
            # Si no hay nube de puntos completa, calcular altura solo con el cluster
            # If no full point cloud provided, calculate height using only the cluster
            min_z = np.min(cluster[:, 2])
            max_z = np.max(cluster[:, 2])
            height = max_z - min_z
            
            logger.debug("Cluster %d: using cluster height = %.2f m (no full cloud provided)", 
                        i, height)
        
        # Verificar altura mínima
        # Check minimum height
        if height < min_tree_height:
            logger.debug("Cluster %d rejected: too short (%.2f < %.2f m)", 
                        i, height, min_tree_height)
            continue
        
        # El cluster pasa todos los filtros
        # Cluster passes all filters
        logger.debug("Cluster %d accepted: height = %.2f m, points = %d", 
                    i, height, len(cluster))
        valid_clusters.append(cluster)
    
    logger.info("Filtering complete: %d/%d clusters passed", 
                len(valid_clusters), len(clusters))
    logger.info("Filtrado completo: %d/%d clusters pasaron", 
                len(valid_clusters), len(clusters))
    
    return valid_clusters


def segment_trees(
    points: np.ndarray,
    voxel_size: float = 0.02,
    eps: float = 0.1,
    min_samples: int = 10,
    slice_height: float = 1.3,
    slice_thickness: float = 0.1,
    min_tree_height: float = 1.5,
    min_points: int = 50
) -> Tuple[List[np.ndarray], Dict]:
    """
    Segmenta árboles individuales de una nube de puntos LiDAR.
    
    Segments individual trees from a LiDAR point cloud.
    
    Args:
        points: Nx3 array of point coordinates
                Array Nx3 de coordenadas de puntos
        voxel_size: Size of voxels for downsampling
                    Tamaño de voxeles para submuestreo
        eps: Maximum distance between points in DBSCAN clustering
             Distancia máxima entre puntos en clustering DBSCAN
        min_samples: Minimum number of points to form a cluster
                     Número mínimo de puntos para formar un cluster
        slice_height: Height at which to extract the horizontal slice
                     Altura a la que extraer la rebanada horizontal
        slice_thickness: Thickness of the horizontal slice
                        Grosor de la rebanada horizontal
        min_tree_height: Minimum height to consider a cluster as a tree
                        Altura mínima para considerar un cluster como árbol
        min_points: Minimum points for a valid tree cluster
                   Puntos mínimos para un cluster de árbol válido
    
    Returns:
        Tuple containing:
        - List of point arrays, one for each detected tree
        - Dictionary with metadata about the segmentation process
        
        Tupla que contiene:
        - Lista de arrays de puntos, uno por cada árbol detectado
        - Diccionario con metadatos sobre el proceso de segmentación
    """
    # Tiempo de inicio para medir rendimiento
    # Start time to measure performance
    start_time = time.time()
    
    # 1. Submuestreo opcional para acelerar procesamiento
    # Optional downsampling to speed up processing
    if voxel_size > 0:
        logger.info("Downsampling point cloud with voxel size %.3f m", voxel_size)
        logger.info("Submuestreando nube de puntos con tamaño de voxel %.3f m", voxel_size)
        points = downsample_point_cloud(points, voxel_size)
    
    # 2. Extraer rebanada horizontal para identificación de troncos
    # Extract horizontal slice for trunk identification
    logger.info("Extracting horizontal slice at height %.2f m (±%.2f m)", 
                slice_height, slice_thickness/2)
    logger.info("Extrayendo rebanada horizontal a altura %.2f m (±%.2f m)", 
                slice_height, slice_thickness/2)
    slice_points = extract_horizontal_slice(points, slice_height, slice_thickness)
    
    if len(slice_points) == 0:
        logger.warning("No points found in the horizontal slice")
        logger.warning("No se encontraron puntos en la rebanada horizontal")
        return [], {"error": "No points in slice", "trees_found": 0}
    
    # 3. Aplicar clustering para identificar troncos individuales
    # Apply clustering to identify individual trunks
    logger.info("Clustering points with DBSCAN (eps=%.3f, min_samples=%d)", 
                eps, min_samples)
    logger.info("Agrupando puntos con DBSCAN (eps=%.3f, min_samples=%d)", 
                eps, min_samples)
    _, clusters = cluster_trunks(slice_points, eps, min_samples)
    
    # 4. Filtrar clusters para eliminar falsos positivos
    # Filter clusters to remove false positives
    logger.info("Filtering clusters (min_height=%.2f m, min_points=%d)", 
                min_tree_height, min_points)
    logger.info("Filtrando clusters (altura_min=%.2f m, puntos_min=%d)", 
                min_tree_height, min_points)
    tree_clusters = filter_tree_clusters(clusters, min_tree_height, min_points, full_point_cloud=points)

    # 5. Empaquetar resultados y metadatos
    # Package results and metadata
    elapsed_time = time.time() - start_time
    metadata = {
        "elapsed_time": elapsed_time,
        "n_trees": len(tree_clusters),
        "trees_found": len(tree_clusters),  # For backwards compatibility
        "parameters": {
            "voxel_size": voxel_size,
            "eps": eps,
            "min_samples": min_samples,
            "slice_height": slice_height,
            "slice_thickness": slice_thickness
        }
    }
    
    logger.info("Tree segmentation complete: found %d trees in %.2f seconds", 
                len(tree_clusters), elapsed_time)
    logger.info("Segmentación de árboles completa: %d árboles encontrados en %.2f segundos", 
                len(tree_clusters), elapsed_time)
    
    return tree_clusters, metadata


def assign_tree_ids(clusters: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Asigna identificadores únicos a cada árbol detectado.
    
    Assigns unique identifiers to each detected tree.
    
    Args:
        clusters: List of point arrays, one for each detected tree
        
    Returns:
        Tuple containing:
        - Nx3 array with all tree points
        - Nx1 array with tree IDs for each point
    """
    if not clusters:
        return np.empty((0, 3)), np.empty((0,), dtype=int)
    
    # Contar puntos totales
    # Count total points
    total_points = sum(len(cluster) for cluster in clusters)
    
    # Preparar arrays para puntos e IDs
    # Prepare arrays for points and IDs
    all_points = np.zeros((total_points, 3))
    tree_ids = np.zeros(total_points, dtype=int)
    
    # Llenar arrays
    # Fill arrays
    point_idx = 0
    for tree_id, cluster in enumerate(clusters):
        n_points = len(cluster)
        all_points[point_idx:point_idx + n_points] = cluster
        tree_ids[point_idx:point_idx + n_points] = tree_id  # IDs empiezan en 0 / IDs start at 0
        point_idx += n_points
    
    logger.info("Assigned IDs to %d trees with %d total points", len(clusters), total_points)
    logger.info("Asignados IDs a %d árboles con %d puntos totales", len(clusters), total_points)
    
    return all_points, tree_ids
