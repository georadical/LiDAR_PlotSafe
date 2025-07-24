#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
© 2025 LiDAR PlotSafe Project. All rights reserved.

Tests for trunk expansion functionality.
"""

import numpy as np
import pytest
from src.pipeline.segmentation import expand_cluster_to_trunk


def test_expand_cluster():
    """
    Test the expand_cluster_to_trunk function.
    
    Prueba la función expand_cluster_to_trunk.
    """
    # Create a full cloud: flat grid 0..2 m
    # Crear una nube completa: cuadrícula plana 0..2 m
    full = np.array([[x, y, 0] for x in range(5) for y in range(5)], dtype=float)
    
    # Create a cluster with points around (2,2)
    # Crear un cluster con puntos alrededor de (2,2)
    cluster = full[[12, 13]]  # points at (2,2) and (2,3)
    
    # Expand the cluster with radius 1.1
    # Expandir el cluster con radio 1.1
    expanded = expand_cluster_to_trunk(cluster, full, radius=1.1)
    
    # Should include a 3x3 neighborhood around (2,2)
    # Debería incluir un vecindario 3x3 alrededor de (2,2)
    assert len(expanded) == 9
    
    # Check that the expanded cluster contains the expected points
    # Verificar que el cluster expandido contiene los puntos esperados
    expected_points = set()
    for x in range(1, 4):  # 1, 2, 3
        for y in range(1, 4):  # 1, 2, 3
            expected_points.add((x, y, 0))
    
    # Convert expanded points to set of tuples for comparison
    # Convertir puntos expandidos a conjunto de tuplas para comparación
    expanded_points = {tuple(point) for point in expanded}
    
    assert expanded_points == expected_points


def test_expand_cluster_empty():
    """
    Test expand_cluster_to_trunk with empty inputs.
    
    Prueba expand_cluster_to_trunk con entradas vacías.
    """
    # Empty cluster
    # Cluster vacío
    cluster = np.empty((0, 3))
    full = np.array([[1, 1, 0], [2, 2, 0]])
    
    # Should return empty array
    # Debería retornar un array vacío
    expanded = expand_cluster_to_trunk(cluster, full, radius=1.0)
    assert len(expanded) == 0
    
    # Empty full cloud
    # Nube completa vacía
    cluster = np.array([[1, 1, 0]])
    full = np.empty((0, 3))
    
    # Should return empty array
    # Debería retornar un array vacío
    expanded = expand_cluster_to_trunk(cluster, full, radius=1.0)
    assert len(expanded) == 0


def test_expand_cluster_radius():
    """
    Test expand_cluster_to_trunk with different radius values.
    
    Prueba expand_cluster_to_trunk con diferentes valores de radio.
    """
    # Create a full cloud with points in a line
    # Crear una nube completa con puntos en una línea
    full = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
        [3, 0, 0],
        [4, 0, 0]
    ])
    
    # Cluster at position 2
    # Cluster en la posición 2
    cluster = np.array([[2, 0, 0]])
    
    # With radius 0.5, should only include the center point
    # Con radio 0.5, solo debería incluir el punto central
    expanded = expand_cluster_to_trunk(cluster, full, radius=0.5)
    assert len(expanded) == 1
    assert np.array_equal(expanded[0], [2, 0, 0])
    
    # With radius 1.0, should include points at positions 1, 2, 3
    # Con radio 1.0, debería incluir puntos en las posiciones 1, 2, 3
    expanded = expand_cluster_to_trunk(cluster, full, radius=1.0)
    assert len(expanded) == 3
    assert set(tuple(p) for p in expanded) == {(1, 0, 0), (2, 0, 0), (3, 0, 0)}
    
    # With radius 2.0, should include all points
    # Con radio 2.0, debería incluir todos los puntos
    expanded = expand_cluster_to_trunk(cluster, full, radius=2.0)
    assert len(expanded) == 5
