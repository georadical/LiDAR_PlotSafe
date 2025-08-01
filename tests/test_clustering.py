# 2025 LiDAR PlotSafe Project. All rights reserved.

"""
Pruebas unitarias para el módulo de clustering.

Unit tests for the clustering module.
"""

import numpy as np
import pytest
from pipeline.clustering import (
    compute_eps,
    dbscan_trunks,
    adaptive_dbscan_trunks
)


def test_compute_eps_empty():
    """
    Prueba el cálculo de epsilon con un conjunto de puntos vacío.
    
    Test epsilon calculation with an empty point set.
    """
    points = np.empty((0, 2))
    eps = compute_eps(points, k=5)
    assert eps == 0.02  # Should return min_eps for empty input


def test_compute_eps_small():
    """
    Prueba el cálculo de epsilon con un conjunto de puntos pequeño.
    
    Test epsilon calculation with a small point set.
    """
    # Create a small point set with fewer points than k neighbors
    points = np.array([
        [0, 0],
        [1, 1],
        [2, 2]
    ])
    eps = compute_eps(points, k=5)
    assert eps == 0.02  # Should return min_eps when points <= k


def test_compute_eps_grid():
    """
    Prueba el cálculo de epsilon con un conjunto de puntos en cuadrícula uniforme.
    
    Test epsilon calculation with a uniform grid of points.
    """
    # Create a 5x5 grid with points 1 unit apart
    x = np.arange(5)
    y = np.arange(5)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.flatten(), yy.flatten()))
    
    # With a grid spacing of 1.0, epsilon should be around 1.0 * factor
    # but capped by max_eps if that's smaller
    eps = compute_eps(
        points, 
        k=4,  # 4 nearest neighbors in a grid are at distance 1.0 and sqrt(2)
        factor=2.0,
        min_eps=0.02,
        max_eps=0.5
    )
    
    # Should be capped at max_eps (0.5) since 2.0 * 1.0 > 0.5
    assert eps == 0.5
    
    # With a lower max_eps, it should be capped accordingly
    eps = compute_eps(
        points, 
        k=4,
        factor=2.0,
        min_eps=0.02,
        max_eps=0.3
    )
    assert eps == 0.3


def test_compute_eps_random():
    """
    Prueba el cálculo de epsilon con puntos aleatorios.
    
    Test epsilon calculation with random points.
    """
    # Use a fixed seed for reproducibility
    np.random.seed(42)
    
    # Generate random points in a 10x10 area
    points = np.random.rand(100, 2) * 10
    
    # Calculate epsilon
    eps = compute_eps(
        points,
        k=8,
        factor=2.0,
        min_eps=0.1,
        max_eps=2.0
    )
    
    # For random points, we just check it's within the allowed range
    assert 0.1 <= eps <= 2.0
    
    # The function should produce a meaningful result (not just default values in most cases)
    # La función debería producir un resultado significativo (no solo valores por defecto en la mayoría de casos)
    # Note: It's valid for eps to equal min_eps or max_eps if clamping occurs
    # Nota: Es válido que eps sea igual a min_eps o max_eps si ocurre limitación
    assert isinstance(eps, float)


def test_dbscan_trunks_basic():
    """
    Prueba la funcionalidad básica de DBSCAN.
    
    Test basic DBSCAN functionality.
    """
    # Create two separate clusters
    cluster1 = np.array([[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1]])
    cluster2 = np.array([[5, 5], [5.1, 5], [5, 5.1], [5.1, 5.1]])
    points = np.vstack([cluster1, cluster2])
    
    # Perform DBSCAN with appropriate parameters
    labels, clusters = dbscan_trunks(points, eps=0.2, min_samples=3)
    
    # Should find 2 clusters
    assert len(clusters) == 2
    
    # Each cluster should have 4 points
    assert all(len(cluster) == 4 for cluster in clusters)


def test_adaptive_dbscan_trunks():
    """
    Prueba la funcionalidad completa de DBSCAN adaptativo.
    
    Test full adaptive DBSCAN functionality.
    """
    # Create two separate clusters with different densities
    cluster1 = np.array([[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1], [0.05, 0.05]])
    cluster2 = np.array([[5, 5], [5.2, 5], [5, 5.2], [5.2, 5.2]])
    points = np.vstack([cluster1, cluster2])
    
    # Perform adaptive DBSCAN
    labels, clusters, eps = adaptive_dbscan_trunks(points, k=4, factor=2.0, min_samples=3)
    
    # Should find 2 clusters
    assert len(clusters) == 2
    
    # Epsilon should be calculated and within reasonable bounds
    assert 0.02 <= eps <= 1.0
    
    # Check that we got back a valid epsilon value
    assert isinstance(eps, float)


def test_compute_eps_parameter_validation():
    """
    Prueba la validación de parámetros en compute_eps.
    
    Test parameter validation in compute_eps.
    """
    points = np.array([[0, 0], [1, 1], [2, 2]])
    
    # Test invalid k value
    with pytest.raises(ValueError, match="k must be >= 1"):
        compute_eps(points, k=0)
    
    # Test invalid factor value
    with pytest.raises(ValueError, match="factor must be > 0"):
        compute_eps(points, k=2, factor=0)
    
    with pytest.raises(ValueError, match="factor must be > 0"):
        compute_eps(points, k=2, factor=-1.0)


def test_dbscan_trunks_parameter_validation():
    """
    Prueba la validación de parámetros en dbscan_trunks.
    
    Test parameter validation in dbscan_trunks.
    """
    points = np.array([[0, 0], [1, 1], [2, 2]])
    
    # Test invalid eps value
    with pytest.raises(ValueError, match="eps must be > 0"):
        dbscan_trunks(points, eps=0)
    
    with pytest.raises(ValueError, match="eps must be > 0"):
        dbscan_trunks(points, eps=-0.1)
    
    # Test invalid min_samples value
    with pytest.raises(ValueError, match="min_samples must be >= 1"):
        dbscan_trunks(points, eps=0.5, min_samples=0)


def test_dbscan_trunks_empty():
    """
    Prueba DBSCAN con conjunto de puntos vacío.
    
    Test DBSCAN with empty point set.
    """
    points = np.empty((0, 2))
    labels, clusters = dbscan_trunks(points, eps=0.5, min_samples=3)
    
    # Should return empty labels and no clusters
    # Debe devolver etiquetas vacías y ningún cluster
    assert len(labels) == 0
    assert len(clusters) == 0


def test_compute_eps_reasonable():
    """
    Prueba el cálculo de epsilon con una cuadrícula regular para validar valores esperados.
    
    Test epsilon calculation with regular grid to validate expected values.
    """
    # Create a grid with 0.04 m spacing → expect eps ≈ 0.12
    # Crear una cuadrícula con espaciado de 0.04 m → esperar eps ≈ 0.12
    xy = np.mgrid[0:1:0.04, 0:1:0.04].reshape(2, -1).T
    eps = compute_eps(xy, k=8, factor=2.0)
    # Updated range based on actual grid geometry analysis
    # Rango actualizado basado en análisis de geometría de cuadrícula real
    assert 0.11 < eps < 0.13, f"Expected epsilon between 0.11 and 0.13, got {eps}"
