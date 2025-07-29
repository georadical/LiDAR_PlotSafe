# 2025 LiDAR PlotSafe Project. All rights reserved.

"""
Utilidades para el procesamiento de nubes de puntos LiDAR.

Utilities for LiDAR point cloud processing.
"""

import numpy as np
import open3d as o3d
import logging

logger = logging.getLogger(__name__)

def downsample_open3d(
    xyz: np.ndarray,
    voxel_size: float = 0.05,
    keep_attributes: dict[str, np.ndarray] | None = None,
):
    """
    Fast voxel down-sample with Open3D while keeping extra attributes.
    EN: Downsamples xyz with voxel_down_sample_and_trace.  
    ES: Submuestrea xyz usando voxel_down_sample_and_trace.

    Parameters
    ----------
    xyz : np.ndarray
        (N,3) array of point coordinates
    voxel_size : float, optional
        Size of voxels for downsampling, by default 0.05
    keep_attributes : dict[str, np.ndarray] | None, optional
        Dictionary of attributes to keep, by default None

    Returns
    -------
    xyz_ds : np.ndarray
        (M,3) array of downsampled point coordinates
    attrs_ds : dict[str, np.ndarray] | None
        Dictionary of downsampled attributes with same keys as input
    """
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.astype(np.float32)))
    pcd_ds, _, idx_map = pcd.voxel_down_sample_and_trace(
        voxel_size=voxel_size,
        min_bound=xyz.min(0) - 1e-3,
        max_bound=xyz.max(0) + 1e-3,
    )
    
    # Extract first index from each voxel's index list
    # Extraer primer índice de la lista de índices de cada voxel
    idx_keep = np.array([indices[0] if len(indices) > 0 else 0 for indices in idx_map])
    xyz_ds = np.asarray(pcd_ds.points)
    
    logger.info("Downsampled from %d to %d points", len(xyz), len(xyz_ds))
    logger.info("Submuestreado de %d a %d puntos", len(xyz), len(xyz_ds))
    
    if keep_attributes is None:
        return xyz_ds, None
    attrs_ds = {k: v[idx_keep] for k, v in keep_attributes.items()}
    return xyz_ds, attrs_ds
