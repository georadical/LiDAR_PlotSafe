"""
Visualization module for LiDAR PlotSafe.

Módulo de visualización para LiDAR PlotSafe.

© 2025 LiDAR PlotSafe Project. All rights reserved.
"""
import numpy as np
import open3d as o3d
import logging
import threading
import traceback
import multiprocessing
import os

# Set multiprocessing start method to 'spawn' for Windows compatibility
# Configurar método de inicio de multiprocessing a 'spawn' para compatibilidad con Windows
if __name__ == "__main__" and multiprocessing.get_start_method() != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)

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

# Define visualization function at module level for multiprocessing compatibility
# Definir función de visualización a nivel de módulo para compatibilidad con multiprocessing
def _run_visualization_process(points_data, window_title, point_size):
    """
    Run visualization in a separate process.
    
    Ejecutar visualización en un proceso separado.
    
    Args:
        points_data (numpy.ndarray): Points array
        window_title (str): Window title
        point_size (float): Size of points to display
    """
    try:
        # Create point cloud in separate process
        # Crear nube de puntos en proceso separado
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_data)
        
        # Apply colorful visualization using height (Z coordinate) for color mapping
        # Aplicar visualización colorida usando la altura (coordenada Z) para mapeo de colores
        if len(points_data) > 0:
            # Get Z coordinates for color mapping
            # Obtener coordenadas Z para mapeo de colores
            z = points_data[:, 2]
            
            # Normalize Z values between 0 and 1
            # Normalizar valores Z entre 0 y 1
            min_z = np.min(z)
            max_z = np.max(z)
            if max_z > min_z:
                normalized_z = (z - min_z) / (max_z - min_z)
            else:
                normalized_z = np.zeros_like(z)
            
            # Create colormap: blue for low values, red for high values
            # Crear mapa de colores: azul para valores bajos, rojo para valores altos
            colors = np.zeros((len(normalized_z), 3))
            
            # Create a colorful rainbow effect
            # Crear un efecto arcoíris colorido
            colors[:, 0] = np.abs(np.sin(normalized_z * np.pi))  # Red
            colors[:, 1] = np.abs(np.sin(normalized_z * np.pi + 2*np.pi/3))  # Green
            colors[:, 2] = np.abs(np.sin(normalized_z * np.pi + 4*np.pi/3))  # Blue
            
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            # Fallback to blue if no points
            # Usar azul si no hay puntos
            pcd.paint_uniform_color([0.5, 0.5, 0.8])
        
        # Use standard visualization that doesn't affect main app DPI
        # Usar visualización estándar que no afecta el DPI de la aplicación principal
        print(f"Visualizing point cloud with {len(points_data):,} points in separate process (PID: {os.getpid()})")
        
        # Create a custom visualizer to control point size properly
        # Crear un visualizador personalizado para controlar el tamaño de punto correctamente
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_title, width=1024, height=768)
        vis.add_geometry(pcd)
        
        # Set rendering options including point size
        # Establecer opciones de renderizado incluyendo tamaño de punto
        opt = vis.get_render_option()
        opt.point_size = float(point_size)
        opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark gray background
        
        # Reset view to fit geometry
        # Resetear vista para ajustar geometría
        vis.reset_view_point(True)
        
        # Run the visualizer
        # Ejecutar el visualizador
        vis.run()
        vis.destroy_window()
        
    except Exception as e:
        print(f"Error in visualization process: {e}")
        traceback.print_exc()


def visualize_point_cloud(points, title="LiDAR PlotSafe - Point Cloud Preview", point_size=1):
    """
    Visualize a point cloud in an Open3D window.
    
    Visualiza una nube de puntos en una ventana Open3D.
    
    Args:
        points (numpy.ndarray): Points array with shape (n, 3) for XYZ coordinates
        title (str): Window title
        point_size (float): Size of points to display
    """
    # Start visualization in separate process
    # Iniciar visualización en proceso separado
    try:
        # Convert points to numpy array for passing between processes
        # Convertir puntos a array numpy para pasar entre procesos
        points_array = np.asarray(points)
        
        # Initialize multiprocessing if needed
        # Inicializar multiprocessing si es necesario
        if multiprocessing.get_start_method() != 'spawn':
            multiprocessing.set_start_method('spawn', force=True)
        
        # Create and start process
        # Crear e iniciar proceso
        process = multiprocessing.Process(
            target=_run_visualization_process, 
            args=(points_array, title, point_size)
        )
        process.daemon = True  # Process will terminate when main program exits
        process.start()
        
        logger.info(f"Started point cloud visualization in separate process with {len(points):,} points")
        return process
    except Exception as e:
        logger.error(f"Failed to start visualization process: {e}")
        traceback.print_exc()
        return None
