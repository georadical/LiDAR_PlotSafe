import laspy
import numpy as np

def load_point_cloud(filepath):
    """
    Load and summarize LiDAR point cloud data
    Args:
        filepath: Path to .laz file
    Returns:
        points: Nx3 array of point coordinates
        summary: Dictionary with point cloud statistics
    """
    # Load file
    las = laspy.read(filepath)
    
    # Extract points
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    # Calculate statistics
    summary = {
        'total_points': len(points),
        'x_range': (np.min(las.x), np.max(las.x)),
        'y_range': (np.min(las.y), np.max(las.y)),
        'z_range': (np.min(las.z), np.max(las.z)),
        'point_density': len(points)/((np.max(las.x)-np.min(las.x))*(np.max(las.y)-np.min(las.y)))
    }
    
    return points, summary

from multiprocessing import Pool, cpu_count
import numba
import cupy as cp

def _calculate_distances_chunk(args):
    """Helper function for parallel distance calculation"""
    points_chunk, center = args
    return np.sqrt((points_chunk[:,0] - center[0])**2 + (points_chunk[:,1] - center[1])**2)

@numba.jit(nopython=True, parallel=True)
def _numba_calculate_distances(points, center):
    """Optimized distance calculation using Numba"""
    distances = np.empty(points.shape[0])
    for i in numba.prange(points.shape[0]):
        distances[i] = np.sqrt((points[i,0] - center[0])**2 + (points[i,1] - center[1])**2)
    return distances

def segment_circular_area(points, center, radius, use_gpu=False):
    """
    Segment points within a circular area of given radius from center
    Args:
        points: Nx3 array of point coordinates
        center: (x,y) tuple of center coordinates
        radius: maximum distance from center (meters)
        use_gpu: whether to use GPU acceleration (requires CUDA)
    Returns:
        Filtered points within the circular area
    """
    if use_gpu:
        try:
            # Convert to CuPy arrays for GPU processing
            points_gpu = cp.asarray(points)
            center_gpu = cp.asarray(center)
            
            # Calculate distances on GPU
            distances = cp.sqrt((points_gpu[:,0] - center_gpu[0])**2 + 
                              (points_gpu[:,1] - center_gpu[1])**2)
            
            # Filter and convert back to NumPy
            return cp.asnumpy(points_gpu[distances <= radius])
        except ImportError:
            print("CuPy not available, falling back to CPU")
            use_gpu = False
    
    if not use_gpu:
        # Use multiprocessing for CPU parallelization
        if points.shape[0] > 1000000:  # Only parallelize for large datasets
            num_cores = cpu_count()
            chunk_size = points.shape[0] // num_cores
            chunks = [(points[i:i + chunk_size], center) 
                     for i in range(0, points.shape[0], chunk_size)]
            
            with Pool(num_cores) as pool:
                distances = np.concatenate(pool.map(_calculate_distances_chunk, chunks))
        else:
            # Use Numba for smaller datasets
            distances = _numba_calculate_distances(points, center)
    
    return points[distances <= radius]
import laspy
import numpy as np

def load_point_cloud(filepath):
    """
    Load and summarize LiDAR point cloud data
    Args:
        filepath: Path to .laz file
    Returns:
        points: Nx3 array of point coordinates
        summary: Dictionary with point cloud statistics
    """
    # Load file
    las = laspy.read(filepath)
    
    # Extract points
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    # Calculate statistics
    summary = {
        'total_points': len(points),
        'x_range': (np.min(las.x), np.max(las.x)),
        'y_range': (np.min(las.y), np.max(las.y)),
        'z_range': (np.min(las.z), np.max(las.z)),
        'point_density': len(points)/((np.max(las.x)-np.min(las.x))*(np.max(las.y)-np.min(las.y)))
    }
    
    return points, summary

def segment_circular_area(points, center, radius):
    """
    Segment points within a circular area of given radius from center
    Args:
        points: Nx3 array of point coordinates
        center: (x,y) tuple of center coordinates
        radius: maximum distance from center (meters)
    Returns:
        Filtered points within the circular area
    """
    # Calculate distances from center
    distances = np.sqrt((points[:,0] - center[0])**2 + (points[:,1] - center[1])**2)
    
    # Filter points within radius
    return points[distances <= radius]
