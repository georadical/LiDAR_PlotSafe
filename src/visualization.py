"""
Visualization module for LiDAR PlotSafe.

Módulo de visualización para LiDAR PlotSafe.

© 2025 LiDAR PlotSafe Project. All rights reserved.
"""
import numpy as np
import open3d as o3d
import logging

logger = logging.getLogger(__name__)

def downsample_point_cloud(points, target_points=500000):
    """
    Downsample a point cloud to a target number of points.
    
    Submuestrea una nube de puntos a un número objetivo de puntos.
    
    Args:
        points (numpy.ndarray): Points array with shape (n, 3) for XYZ coordinates
        target_points (int): Target number of points after downsampling
        
    Returns:
        numpy.ndarray: Downsampled point cloud
    """
    # Create Open3D point cloud object
    # Crear objeto de nube de puntos Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # If point cloud is already small enough, return it as is
    # Si la nube de puntos ya es suficientemente pequeña, devolverla tal cual
    if len(points) <= target_points:
        logger.info(f"Point cloud already has {len(points)} points, no downsampling needed")
        return points
    
    # Calculate voxel size based on point density
    # Calcular tamaño de voxel basado en la densidad de puntos
    x_range = np.max(points[:, 0]) - np.min(points[:, 0])
    y_range = np.max(points[:, 1]) - np.min(points[:, 1])
    z_range = np.max(points[:, 2]) - np.min(points[:, 2])
    volume = x_range * y_range * z_range
    
    # Heuristic for voxel size: larger voxels for more aggressive downsampling
    # Heurística para tamaño de voxel: voxels más grandes para submuestreo más agresivo
    point_density = len(points) / volume
    voxel_size = (len(points) / target_points) ** (1/3) / point_density ** (1/3)
    
    # Ensure voxel_size is reasonable
    # Asegurar que el tamaño de voxel sea razonable
    min_voxel_size = min(x_range, y_range, z_range) / 100
    voxel_size = max(voxel_size, min_voxel_size)
    
    # Downsample using voxel grid
    # Submuestreo usando rejilla de voxel
    logger.info(f"Downsampling from {len(points)} to ~{target_points} points with voxel size {voxel_size:.4f}")
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    
    # Convert back to numpy array
    # Convertir de vuelta a array numpy
    downsampled_points = np.asarray(downsampled_pcd.points)
    logger.info(f"Downsampled to {len(downsampled_points)} points")
    
    return downsampled_points

def visualize_point_cloud(points, title="LiDAR PlotSafe - Point Cloud Preview", point_size=1):
    """
    Visualize a point cloud in an Open3D window.
    
    Visualiza una nube de puntos en una ventana Open3D.
    
    Args:
        points (numpy.ndarray): Points array with shape (n, 3) for XYZ coordinates
        title (str): Window title
        point_size (float): Size of points to display
    """
    # Create Open3D point cloud object
    # Crear objeto de nube de puntos Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Create visualization window
    # Crear ventana de visualización
    vis = o3d.visualization.Visualizer()
    vis.create_window(title, width=1024, height=768)
    
    # Add geometry to the visualizer
    # Añadir geometría al visualizador
    vis.add_geometry(pcd)
    
    # Set rendering options
    # Establecer opciones de renderizado
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark gray background
    
    # Set initial camera position
    # Establecer posición inicial de la cámara
    view_control = vis.get_view_control()
    view_control.set_zoom(0.8)
    
    # Display the visualization
    # Mostrar la visualización
    vis.run()
    vis.destroy_window()
