# © 2025 LiDAR PlotSafe Project. All rights reserved.

"""
Unit tests for verticality filter module.
Pruebas unitarias para el módulo de filtro de verticalidad.

This module tests the PCA-based normal computation and vertical surface filtering
functionality used for LiDAR tree segmentation.

Este módulo prueba la computación de normales basada en PCA y la funcionalidad
de filtrado de superficies verticales usada para segmentación de árboles LiDAR.
"""

import numpy as np
import pytest
import logging
from unittest.mock import patch, MagicMock

# Import the verticality filter functions
# Importar las funciones del filtro de verticalidad
from pipeline.verticality import (
    compute_point_normals,
    vertical_mask,
    apply_verticality_filter
)


def test_vertical_mask_keeps_vertical():
    """
    Test that vertical_mask properly distinguishes vertical trunks from angled branches.
    Probar que vertical_mask distingue correctamente troncos verticales de ramas inclinadas.
    
    This is the key realistic test case for tree segmentation accuracy.
    Este es el caso de prueba realista clave para la precisión de segmentación de árboles.
    """
    # Create vertical cylinder (trunk) + 45° branch
    # Crear cilindro vertical (tronco) + rama de 45°
    z = np.linspace(0, 10, 200)
    trunk = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    branch = np.column_stack([z * 0.5, np.zeros_like(z), z])  # 45° branch / rama de 45°
    pts = np.vstack([trunk, branch])
    
    # Apply vertical mask with strict threshold
    # Aplicar máscara vertical con umbral estricto
    mask = vertical_mask(pts, radius=0.2, cos_thresh=0.9, min_neighbors=5)
    
    # Trunk should be kept, branch should be filtered
    # El tronco debe mantenerse, la rama debe filtrarse
    assert mask.sum() > len(branch), "Trunk points should be kept (more than branch count)"
    assert mask.sum() < len(pts), "Branch points should be filtered (less than total)"
    
    # Log results for inspection
    # Registrar resultados para inspección
    kept_ratio = mask.sum() / len(pts)
    print(f"Vertical filter: kept {mask.sum()}/{len(pts)} points ({kept_ratio:.1%})")
    print(f"Trunk points: {len(trunk)}, Branch points: {len(branch)}")


class TestComputePointNormals:
    """
    Test cases for compute_point_normals function.
    Casos de prueba para la función compute_point_normals.
    """
    
    def test_compute_normals_simple_plane(self):
        """
        Test normal computation for points on a simple horizontal plane.
        Probar cálculo de normales para puntos en un plano horizontal simple.
        """
        # Create a simple horizontal plane (Z=0)
        # Crear un plano horizontal simple (Z=0)
        x = np.linspace(-1, 1, 5)
        y = np.linspace(-1, 1, 5)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
        
        # Compute normals
        # Calcular normales
        normals, valid_mask = compute_point_normals(
            points, radius=0.5, min_neighbors=3
        )
        
        # Check that some normals are computed
        # Verificar que se calculan algunas normales
        assert valid_mask.sum() > 0, "Should compute normals for some points"
        assert normals.shape == (len(points), 3), "Normals should have shape (N, 3)"
        
        # For a horizontal plane, normals should point upward (positive Z)
        # Para un plano horizontal, las normales deben apuntar hacia arriba (Z positivo)
        valid_normals = normals[valid_mask]
        if len(valid_normals) > 0:
            # Check that computed normals have reasonable Z components
            # Verificar que las normales calculadas tienen componentes Z razonables
            assert np.all(valid_normals[:, 2] >= 0), "Normals should point upward for horizontal plane"
    
    def test_compute_normals_vertical_plane(self):
        """
        Test normal computation for points on a vertical plane.
        Probar cálculo de normales para puntos en un plano vertical.
        """
        # Create a vertical plane (X=0, varying Y and Z)
        # Crear un plano vertical (X=0, Y y Z variables)
        y = np.linspace(-1, 1, 5)
        z = np.linspace(0, 2, 5)
        Y, Z = np.meshgrid(y, z)
        X = np.zeros_like(Y)
        points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
        
        # Compute normals
        # Calcular normales
        normals, valid_mask = compute_point_normals(
            points, radius=0.5, min_neighbors=3
        )
        
        # Check that some normals are computed
        # Verificar que se calculan algunas normales
        assert valid_mask.sum() > 0, "Should compute normals for vertical plane points"
        
        # For a vertical plane (X=0), normals should be horizontal (X-direction)
        # Para un plano vertical (X=0), las normales deben ser horizontales (dirección X)
        valid_normals = normals[valid_mask]
        if len(valid_normals) > 0:
            # Check that normals are roughly perpendicular to the plane
            # Verificar que las normales son aproximadamente perpendiculares al plano
            assert np.all(np.abs(valid_normals[:, 0]) > 0.5), "Normals should be roughly in X direction"
    
    def test_compute_normals_insufficient_neighbors(self):
        """
        Test behavior when points have insufficient neighbors.
        Probar comportamiento cuando los puntos tienen vecinos insuficientes.
        """
        # Create very sparse points
        # Crear puntos muy dispersos
        points = np.array([
            [0, 0, 0],
            [10, 10, 10],
            [20, 20, 20]
        ])
        
        # Compute normals with small radius
        # Calcular normales con radio pequeño
        normals, valid_mask = compute_point_normals(
            points, radius=1.0, min_neighbors=5
        )
        
        # Should have no valid normals due to insufficient neighbors
        # No debería tener normales válidas debido a vecinos insuficientes
        assert valid_mask.sum() == 0, "Should have no valid normals with insufficient neighbors"
        assert np.allclose(normals, 0), "Normals should be zero vectors for invalid points"
    
    def test_compute_normals_empty_input(self):
        """
        Test behavior with empty input.
        Probar comportamiento con entrada vacía.
        """
        points = np.empty((0, 3))
        
        normals, valid_mask = compute_point_normals(points, radius=0.1, min_neighbors=3)
        
        assert normals.shape == (0, 3), "Should return empty normals array"
        assert valid_mask.shape == (0,), "Should return empty valid mask"


class TestVerticalMask:
    """
    Test cases for vertical_mask function.
    Casos de prueba para la función vertical_mask.
    """
    
    def test_vertical_mask_horizontal_surface(self):
        """
        Test verticality mask for horizontal surface points.
        Probar máscara de verticalidad para puntos de superficie horizontal.
        """
        # Create horizontal plane points
        # Crear puntos de plano horizontal
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
        
        # Apply vertical mask with high threshold
        # Aplicar máscara vertical con umbral alto
        mask = vertical_mask(
            points, radius=0.5, cos_thresh=0.85, min_neighbors=3
        )
        
        # Horizontal surfaces should not pass vertical filter
        # Las superficies horizontales no deben pasar el filtro vertical
        assert mask.sum() == 0, "Horizontal surfaces should not pass vertical filter"
    
    def test_vertical_mask_vertical_surface(self):
        """
        Test verticality mask for vertical surface points.
        Probar máscara de verticalidad para puntos de superficie vertical.
        """
        # Create vertical plane points (tree trunk-like)
        # Crear puntos de plano vertical (como tronco de árbol)
        theta = np.linspace(0, 2*np.pi, 20)
        z = np.linspace(0, 3, 15)
        radius = 0.2
        
        # Create cylindrical surface (tree trunk)
        # Crear superficie cilíndrica (tronco de árbol)
        points = []
        for z_val in z:
            for theta_val in theta:
                x_val = radius * np.cos(theta_val)
                y_val = radius * np.sin(theta_val)
                points.append([x_val, y_val, z_val])
        
        points = np.array(points)
        
        # Apply vertical mask
        # Aplicar máscara vertical
        mask = vertical_mask(
            points, radius=0.3, cos_thresh=0.7, min_neighbors=5
        )
        
        # Should detect many vertical points
        # Debería detectar muchos puntos verticales
        assert mask.sum() > 0, "Should detect vertical points on cylindrical surface"
        ratio = mask.sum() / len(points)
        assert ratio > 0.1, f"Should detect reasonable ratio of vertical points, got {ratio:.3f}"
    
    def test_vertical_mask_precomputed_normals(self):
        """
        Test vertical mask with precomputed normals.
        Probar máscara vertical con normales precomputadas.
        """
        # Create test points
        # Crear puntos de prueba
        points = np.array([
            [0, 0, 0],
            [1, 0, 1],
            [0, 1, 2]
        ])
        
        # Create mock precomputed normals (vertical)
        # Crear normales precomputadas simuladas (verticales)
        normals = np.array([
            [1, 0, 0],  # Horizontal normal (vertical surface)
            [0, 1, 0],  # Horizontal normal (vertical surface)  
            [0, 0, 1]   # Vertical normal (horizontal surface)
        ])
        valid_mask = np.array([True, True, True])
        
        # Apply vertical mask with precomputed normals
        # Aplicar máscara vertical con normales precomputadas
        mask = vertical_mask(
            points, 
            cos_thresh=0.5,
            precomputed_normals=normals,
            precomputed_valid=valid_mask
        )
        
        # First two points should pass (horizontal normals = vertical surfaces)
        # Los primeros dos puntos deben pasar (normales horizontales = superficies verticales)
        assert mask[0] == True, "Point with horizontal normal should pass vertical filter"
        assert mask[1] == True, "Point with horizontal normal should pass vertical filter"
        assert mask[2] == False, "Point with vertical normal should not pass vertical filter"

    def test_vertical_mask_realistic_tree_scenario(self):
        """
        Test vertical mask with realistic tree trunk and foliage scenario.
        Probar máscara vertical con escenario realista de tronco de árbol y follaje.
        """
        # Create tree trunk (vertical cylinder)
        # Crear tronco de árbol (cilindro vertical)
        trunk_height = 5.0
        trunk_radius = 0.2
        n_trunk_points = 100
        
        z_trunk = np.random.uniform(0, trunk_height, n_trunk_points)
        theta_trunk = np.random.uniform(0, 2*np.pi, n_trunk_points)
        x_trunk = trunk_radius * np.cos(theta_trunk)
        y_trunk = trunk_radius * np.sin(theta_trunk)
        trunk_points = np.column_stack([x_trunk, y_trunk, z_trunk])
        
        # Create horizontal foliage points
        # Crear puntos horizontales de follaje
        n_foliage_points = 50
        x_foliage = np.random.uniform(-2, 2, n_foliage_points)
        y_foliage = np.random.uniform(-2, 2, n_foliage_points)
        z_foliage = np.random.uniform(3, 4, n_foliage_points)  # Crown level / Nivel de copa
        foliage_points = np.column_stack([x_foliage, y_foliage, z_foliage])
        
        # Combine all points
        # Combinar todos los puntos
        all_points = np.vstack([trunk_points, foliage_points])
        
        # Apply vertical mask
        # Aplicar máscara vertical
        mask = vertical_mask(all_points, radius=0.3, cos_thresh=0.8, min_neighbors=5)
        
        # Should keep more trunk points than foliage points
        # Debería mantener más puntos de tronco que de follaje
        trunk_mask = mask[:len(trunk_points)]
        foliage_mask = mask[len(trunk_points):]
        
        trunk_kept_ratio = trunk_mask.sum() / len(trunk_points)
        foliage_kept_ratio = foliage_mask.sum() / len(foliage_points)
        
        # Trunk should have higher retention than foliage
        # El tronco debería tener mayor retención que el follaje
        assert trunk_kept_ratio > foliage_kept_ratio, "Trunk should have higher retention than foliage"
        print(f"Trunk retention: {trunk_kept_ratio:.1%}, Foliage retention: {foliage_kept_ratio:.1%}")


class TestApplyVerticalityFilter:
    """
    Test cases for apply_verticality_filter function.
    Casos de prueba para la función apply_verticality_filter.
    """
    
    def test_apply_filter_basic(self):
        """
        Test basic functionality of apply_verticality_filter.
        Probar funcionalidad básica de apply_verticality_filter.
        """
        # Create mixed vertical and horizontal surfaces
        # Crear superficies verticales y horizontales mixtas
        
        # Horizontal points (should be filtered out)
        # Puntos horizontales (deben ser filtrados)
        horizontal = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]
        ])
        
        # Vertical points (should be kept) - simulating trunk
        # Puntos verticales (deben mantenerse) - simulando tronco
        vertical = np.array([
            [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]
        ])
        
        all_points = np.vstack([horizontal, vertical])
        
        # Apply filter
        # Aplicar filtro
        filtered_points, kept_indices, normals = apply_verticality_filter(
            all_points, 
            radius=1.0, 
            cos_thresh=0.5, 
            min_neighbors=3,
            return_normals=True
        )
        
        # Check that some points are kept
        # Verificar que se mantienen algunos puntos
        assert len(filtered_points) > 0, "Should keep some points after filtering"
        assert len(kept_indices) == len(filtered_points), "Indices should match filtered points"
        
        if normals is not None:
            assert normals.shape == (len(filtered_points), 3), "Normals should match filtered points"
    
    def test_apply_filter_empty_result(self):
        """
        Test filter behavior when all points are filtered out.
        Probar comportamiento del filtro cuando todos los puntos son filtrados.
        """
        # Create only horizontal surface points
        # Crear solo puntos de superficie horizontal
        points = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]
        ])
        
        # Apply very strict filter
        # Aplicar filtro muy estricto
        filtered_points, kept_indices, normals = apply_verticality_filter(
            points, 
            radius=0.5, 
            cos_thresh=0.95, 
            min_neighbors=3,
            return_normals=False
        )
        
        # Should filter out all points
        # Debe filtrar todos los puntos
        assert len(filtered_points) == 0, "Should filter out all horizontal points"
        assert len(kept_indices) == 0, "Should have no kept indices"
        assert normals is None, "Should not return normals when return_normals=False"
    
    @patch('pipeline.verticality.logger')
    def test_apply_filter_logging(self, mock_logger):
        """
        Test that apply_verticality_filter logs appropriate messages.
        Probar que apply_verticality_filter registra mensajes apropiados.
        """
        # Create test points
        # Crear puntos de prueba
        points = np.array([
            [0, 0, 0], [1, 0, 1], [2, 0, 2]
        ])
        
        # Apply filter
        # Aplicar filtro
        apply_verticality_filter(points, radius=1.0, cos_thresh=0.5, min_neighbors=2)
        
        # Check that debug and info logging occurred
        # Verificar que ocurrió el registro de debug e info
        mock_logger.debug.assert_called()
        mock_logger.info.assert_called()


class TestVerticalityFilterIntegration:
    """
    Integration tests for verticality filter components.
    Pruebas de integración para componentes del filtro de verticalidad.
    """
    
    def test_end_to_end_tree_trunk_simulation(self):
        """
        Test end-to-end filtering with simulated tree trunk data.
        Probar filtrado de extremo a extremo con datos simulados de tronco de árbol.
        """
        # Simulate a tree trunk (cylindrical) and ground points
        # Simular un tronco de árbol (cilíndrico) y puntos de suelo
        
        # Tree trunk points (vertical cylinder)
        # Puntos de tronco de árbol (cilindro vertical)
        trunk_points = []
        n_levels = 20
        n_points_per_level = 12
        trunk_radius = 0.15
        
        for i in range(n_levels):
            z = i * 0.1  # Height levels / Niveles de altura
            for j in range(n_points_per_level):
                angle = 2 * np.pi * j / n_points_per_level
                x = trunk_radius * np.cos(angle)
                y = trunk_radius * np.sin(angle)
                trunk_points.append([x, y, z])
        
        # Ground points (horizontal)
        # Puntos de suelo (horizontales)
        ground_points = []
        for x in np.linspace(-1, 1, 10):
            for y in np.linspace(-1, 1, 10):
                ground_points.append([x, y, 0])
        
        # Combine all points
        # Combinar todos los puntos
        all_points = np.array(trunk_points + ground_points)
        initial_count = len(all_points)
        trunk_count = len(trunk_points)
        ground_count = len(ground_points)
        
        # Apply verticality filter
        # Aplicar filtro de verticalidad
        filtered_points, kept_indices, _ = apply_verticality_filter(
            all_points,
            radius=0.2,
            cos_thresh=0.7,
            min_neighbors=5,
            return_normals=False
        )
        
        # Analyze results
        # Analizar resultados
        kept_count = len(filtered_points)
        kept_ratio = kept_count / initial_count if initial_count > 0 else 0
        
        # Should keep more trunk points than ground points
        # Debería mantener más puntos de tronco que de suelo
        assert kept_count > 0, "Should keep some points (presumably trunk)"
        assert kept_ratio < 1.0, "Should filter out some points (presumably ground)"
        
        # Log results for inspection
        # Registrar resultados para inspección
        print(f"Simulation results: {initial_count} total -> {kept_count} kept ({kept_ratio:.1%})")
        print(f"Original: {trunk_count} trunk + {ground_count} ground")

    def test_parameter_sensitivity(self):
        """
        Test sensitivity to different parameter values.
        Probar sensibilidad a diferentes valores de parámetros.
        """
        # Create test data with mixed orientations
        # Crear datos de prueba con orientaciones mixtas
        vertical_line = np.column_stack([
            np.zeros(50), np.zeros(50), np.linspace(0, 5, 50)
        ])
        horizontal_plane = np.column_stack([
            np.random.uniform(-1, 1, 50), 
            np.random.uniform(-1, 1, 50), 
            np.ones(50)
        ])
        all_points = np.vstack([vertical_line, horizontal_plane])
        
        # Test different cos_thresh values
        # Probar diferentes valores de cos_thresh
        thresholds = [0.5, 0.7, 0.85, 0.95]
        results = {}
        
        for thresh in thresholds:
            mask = vertical_mask(all_points, radius=0.3, cos_thresh=thresh, min_neighbors=3)
            results[thresh] = mask.sum()
        
        # Higher thresholds should filter more aggressively
        # Umbrales más altos deben filtrar más agresivamente
        for i in range(len(thresholds) - 1):
            assert results[thresholds[i]] >= results[thresholds[i+1]], \
                f"Higher threshold should keep fewer or equal points: {thresholds[i]} vs {thresholds[i+1]}"
        
        print(f"Threshold sensitivity: {results}")


class TestEdgeCases:
    """
    Test edge cases and error conditions.
    Probar casos límite y condiciones de error.
    """
    
    def test_single_point(self):
        """
        Test behavior with single point.
        Probar comportamiento con un solo punto.
        """
        points = np.array([[0, 0, 1]])
        mask = vertical_mask(points, radius=1.0, cos_thresh=0.5, min_neighbors=1)
        
        # Single point should not have valid normal
        # Un solo punto no debería tener normal válida
        assert mask.sum() == 0, "Single point should not pass filter"
    
    def test_identical_points(self):
        """
        Test behavior with identical points.
        Probar comportamiento con puntos idénticos.
        """
        points = np.array([[0, 0, 1]] * 10)  # 10 identical points / 10 puntos idénticos
        mask = vertical_mask(points, radius=1.0, cos_thresh=0.5, min_neighbors=3)
        
        # Identical points should not form valid surface
        # Puntos idénticos no deberían formar superficie válida
        assert mask.sum() == 0, "Identical points should not pass filter"
    
    def test_very_large_coordinates(self):
        """
        Test behavior with very large coordinate values.
        Probar comportamiento con valores de coordenadas muy grandes.
        """
        # Create points with large coordinates
        # Crear puntos con coordenadas grandes
        scale = 1e6
        points = np.array([
            [0, 0, 0],
            [0, 0, scale],
            [0, 0, 2*scale]
        ]) 
        
        # Should handle large coordinates gracefully
        # Debería manejar coordenadas grandes con gracia
        mask = vertical_mask(points, radius=scale*0.1, cos_thresh=0.5, min_neighbors=2)
        
        # Should not crash and return valid result
        # No debería fallar y devolver resultado válido
        assert isinstance(mask, np.ndarray), "Should return numpy array"
        assert len(mask) == len(points), "Mask should have same length as input"


if __name__ == "__main__":
    # Run tests with pytest
    # Ejecutar pruebas con pytest
    pytest.main([__file__, "-v"])
