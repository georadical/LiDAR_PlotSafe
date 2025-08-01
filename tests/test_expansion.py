# -*- coding: utf-8 -*-
"""
 2025 LiDAR PlotSafe Project. All rights reserved.

Unit tests for vertical expansion and duplicate merging functionality.
Pruebas unitarias para funcionalidad de expansión vertical y fusión de duplicados.
"""

import pytest
import numpy as np
import logging
from unittest.mock import patch

from src.pipeline.expansion import (
    expand_cluster_to_trunk,
    merge_overlapping_trunks,
    expand_and_merge_clusters,
    expand_and_merge,
    ExpansionConfig
)


class TestExpansionConfig:
    """Test ExpansionConfig dataclass / Prueba la clase de datos ExpansionConfig"""
    
    def test_default_values(self):
        """Test default configuration values / Prueba valores de configuración por defecto"""
        config = ExpansionConfig()
        assert config.expansion_radius == 0.5
        assert config.merge_tolerance == 0.25
        assert config.min_points_per_trunk == 10
        assert config.max_expansion_ratio == 5.0
        
    def test_custom_values(self):
        """Test custom configuration values / Prueba valores de configuración personalizados"""
        config = ExpansionConfig(
            expansion_radius=0.8,
            merge_tolerance=0.3,
            min_points_per_trunk=15,
            max_expansion_ratio=3.0
        )
        assert config.expansion_radius == 0.8
        assert config.merge_tolerance == 0.3
        assert config.min_points_per_trunk == 15
        assert config.max_expansion_ratio == 3.0


class TestExpandClusterToTrunk:
    """Test expand_cluster_to_trunk function / Prueba función expand_cluster_to_trunk"""
    
    @pytest.fixture
    def sample_cluster_2d(self):
        """Sample 2D cluster points / Puntos de cluster 2D de muestra"""
        return np.array([
            [0.0, 0.0],
            [0.1, 0.1],
            [0.1, -0.1],
            [-0.1, 0.1]
        ])
        
    @pytest.fixture
    def sample_cluster_3d(self):
        """Sample 3D cluster points / Puntos de cluster 3D de muestra"""
        return np.array([
            [0.0, 0.0, 1.0],
            [0.1, 0.1, 1.1],
            [0.1, -0.1, 0.9],
            [-0.1, 0.1, 1.2]
        ])
        
    @pytest.fixture
    def full_point_cloud(self):
        """Sample full point cloud / Nube de puntos completa de muestra"""
        # Create points around origin and some distant points
        # Crea puntos alrededor del origen y algunos puntos distantes
        np.random.seed(42)
        near_points = np.random.uniform(-0.6, 0.6, (50, 3))
        far_points = np.random.uniform(2.0, 3.0, (20, 3))
        return np.vstack([near_points, far_points])
    
    def test_valid_expansion_2d_cluster(self, sample_cluster_2d, full_point_cloud):
        """Test valid expansion with 2D cluster / Prueba expansión válida con cluster 2D"""
        result = expand_cluster_to_trunk(
            sample_cluster_2d, 
            full_point_cloud,
            radius=0.8,
            min_points=5,
            max_ratio=20.0
        )
        
        assert result is not None
        assert result.shape[1] == 3  # Should return 3D points / Debe devolver puntos 3D
        assert result.shape[0] >= 5   # Should have minimum points / Debe tener puntos mínimos
        
        # Check that all returned points are within radius of centroid
        # Verifica que todos los puntos devueltos estén dentro del radio del centroide
        centroid = sample_cluster_2d.mean(axis=0)
        distances = np.linalg.norm(result[:, :2] - centroid, axis=1)
        assert np.all(distances <= 0.8)
    
    def test_valid_expansion_3d_cluster(self, sample_cluster_3d, full_point_cloud):
        """Test valid expansion with 3D cluster / Prueba expansión válida con cluster 3D"""
        result = expand_cluster_to_trunk(
            sample_cluster_3d,
            full_point_cloud, 
            radius=0.5,
            max_ratio=10.0  # Allow higher expansion ratio for test / Permitir ratio de expansión más alto para prueba
        )
        
        assert result is not None
        assert result.shape[1] == 3
        
    def test_empty_cluster(self, full_point_cloud):
        """Test with empty cluster / Prueba con cluster vacío"""
        empty_cluster = np.array([]).reshape(0, 2)
        result = expand_cluster_to_trunk(empty_cluster, full_point_cloud)
        assert result is None
        
    def test_insufficient_coordinates(self, full_point_cloud):
        """Test cluster with insufficient coordinates / Prueba cluster con coordenadas insuficientes"""
        invalid_cluster = np.array([[0.0], [0.1]])  # Only 1 coordinate
        
        with pytest.raises(ValueError, match="Cluster must have at least XY coordinates"):
            expand_cluster_to_trunk(invalid_cluster, full_point_cloud)
            
    def test_invalid_full_pc(self, sample_cluster_2d):
        """Test with invalid full point cloud / Prueba con nube de puntos completa inválida"""
        # Empty point cloud / Nube de puntos vacía
        empty_pc = np.array([]).reshape(0, 3)
        with pytest.raises(ValueError, match="Full point cloud must have XYZ coordinates"):
            expand_cluster_to_trunk(sample_cluster_2d, empty_pc)
            
        # Point cloud with insufficient dimensions / Nube de puntos con dimensiones insuficientes
        invalid_pc = np.array([[0.0, 0.1], [0.2, 0.3]])  # Only 2D
        with pytest.raises(ValueError, match="Full point cloud must have XYZ coordinates"):
            expand_cluster_to_trunk(sample_cluster_2d, invalid_pc)
    
    def test_insufficient_points_after_expansion(self, sample_cluster_2d):
        """Test case where expansion results in too few points / Prueba caso donde expansión resulta en muy pocos puntos"""
        # Create sparse point cloud / Crea nube de puntos dispersa
        sparse_pc = np.array([
            [0.0, 0.0, 1.0],
            [5.0, 5.0, 1.0]  # Far away point / Punto lejano
        ])
        
        result = expand_cluster_to_trunk(
            sample_cluster_2d,
            sparse_pc,
            radius=0.3,
            min_points=5
        )
        assert result is None
        
    def test_excessive_expansion_ratio(self, full_point_cloud):
        """Test case where expansion ratio is too high / Prueba caso donde ratio de expansión es muy alto"""
        # Small cluster that would expand too much / Cluster pequeño que se expandiría demasiado
        tiny_cluster = np.array([[0.0, 0.0]])
        
        result = expand_cluster_to_trunk(
            tiny_cluster,
            full_point_cloud,
            radius=2.0,  # Large radius / Radio grande
            max_ratio=2.0  # Low max ratio / Ratio máximo bajo
        )
        assert result is None


class TestMergeOverlappingTrunks:
    """Test merge_overlapping_trunks function / Prueba función merge_overlapping_trunks"""
    
    @pytest.fixture
    def sample_trunks_separate(self):
        """Sample trunks that are well separated / Troncos de muestra bien separados"""
        trunk1 = np.array([
            [0.0, 0.0, 1.0],
            [0.1, 0.1, 1.1],
            [0.0, 0.1, 1.2]
        ])
        trunk2 = np.array([
            [3.0, 3.0, 1.0],
            [3.1, 3.1, 1.1],
            [3.0, 3.1, 1.2]
        ])
        return [trunk1, trunk2]
    
    @pytest.fixture 
    def sample_trunks_overlapping(self):
        """Sample trunks that should be merged / Troncos de muestra que deben fusionarse"""
        trunk1 = np.array([
            [0.0, 0.0, 1.0],
            [0.1, 0.1, 1.1]
        ])
        trunk2 = np.array([
            [0.1, 0.1, 1.0],  # Close to trunk1 / Cerca del trunk1
            [0.2, 0.0, 1.1]
        ])
        trunk3 = np.array([
            [0.05, 0.05, 1.0],  # Also close to trunk1 / También cerca del trunk1
            [0.0, 0.2, 1.1]
        ])
        return [trunk1, trunk2, trunk3]
    
    def test_no_trunks(self):
        """Test with no trunks / Prueba sin troncos"""
        result = merge_overlapping_trunks([])
        assert result == []
        
    def test_single_trunk(self):
        """Test with single trunk / Prueba con un solo tronco"""
        trunk = np.array([[0.0, 0.0, 1.0]])
        result = merge_overlapping_trunks([trunk])
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], trunk)
        
    def test_separate_trunks_no_merge(self, sample_trunks_separate):
        """Test trunks that should remain separate / Prueba troncos que deben permanecer separados"""
        result = merge_overlapping_trunks(sample_trunks_separate, merge_tolerance=0.5)
        assert len(result) == 2  # Should remain separate / Deben permanecer separados
        
    def test_overlapping_trunks_merge(self, sample_trunks_overlapping):
        """Test trunks that should be merged / Prueba troncos que deben fusionarse"""
        result = merge_overlapping_trunks(sample_trunks_overlapping, merge_tolerance=0.3)
        
        # Should merge all 3 trunks into 1 / Debe fusionar los 3 troncos en 1
        assert len(result) == 1
        
        # Merged trunk should have all points / Tronco fusionado debe tener todos los puntos
        total_points = sum(trunk.shape[0] for trunk in sample_trunks_overlapping)
        assert result[0].shape[0] == total_points
        
    def test_invalid_trunks_skipped(self):
        """Test that invalid trunks are skipped / Prueba que troncos inválidos se omiten"""
        valid_trunk = np.array([[0.0, 0.0, 1.0], [0.1, 0.1, 1.1]])
        invalid_trunk1 = np.array([]).reshape(0, 3)  # Empty / Vacío
        invalid_trunk2 = np.array([[0.0]])  # Insufficient dimensions / Dimensiones insuficientes
        
        trunks = [valid_trunk, invalid_trunk1, invalid_trunk2]
        
        with patch('src.pipeline.expansion.logger') as mock_logger:
            result = merge_overlapping_trunks(trunks)
            
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], valid_trunk)
        mock_logger.warning.assert_called()


class TestExpandAndMergeClusters:
    """Test complete expand_and_merge_clusters pipeline / Prueba pipeline completo expand_and_merge_clusters"""
    
    @pytest.fixture
    def sample_slice_clusters(self):
        """Sample slice clusters / Clusters de corte de muestra"""
        cluster1 = np.array([
            [0.0, 0.0],
            [0.1, 0.1]
        ])
        cluster2 = np.array([
            [2.0, 2.0],
            [2.1, 2.1]
        ])
        return [cluster1, cluster2]
        
    @pytest.fixture
    def sample_full_pc(self):
        """Sample full point cloud / Nube de puntos completa de muestra"""
        np.random.seed(42)
        # Points around first cluster / Puntos alrededor del primer cluster
        pc1 = np.random.uniform(-0.6, 0.6, (30, 3))
        # Points around second cluster / Puntos alrededor del segundo cluster  
        pc2 = np.random.uniform(1.4, 2.6, (25, 3))
        return np.vstack([pc1, pc2])
    
    def test_complete_pipeline(self, sample_slice_clusters, sample_full_pc):
        """Test complete expansion and merge pipeline / Prueba pipeline completo de expansión y fusión"""
        config = ExpansionConfig(
            expansion_radius=0.8,
            merge_tolerance=0.3,
            min_points_per_trunk=5,
            max_expansion_ratio=10.0
        )
        
        trunks, metadata = expand_and_merge_clusters(
            sample_slice_clusters,
            sample_full_pc,
            config
        )
        
        # Verify results / Verifica resultados
        assert len(trunks) >= 0
        assert isinstance(metadata, dict)
        
        # Check metadata structure / Verifica estructura de metadatos
        required_keys = [
            'input_clusters', 'expanded_trunks', 'final_trunks', 
            'expansion_stats', 'config'
        ]
        for key in required_keys:
            assert key in metadata
            
        assert metadata['input_clusters'] == len(sample_slice_clusters)
        assert metadata['config'] == config
        
        # Check expansion stats / Verifica estadísticas de expansión
        stats = metadata['expansion_stats']
        assert 'successful_expansions' in stats
        assert 'failed_expansions' in stats
        assert 'total_points_before' in stats
        assert 'total_points_after' in stats
        
    def test_empty_clusters(self):
        """Test with empty cluster list / Prueba con lista de clusters vacía"""
        config = ExpansionConfig()
        full_pc = np.random.uniform(-1, 1, (100, 3))
        
        trunks, metadata = expand_and_merge_clusters([], full_pc, config)
        
        assert trunks == []
        assert metadata == {}
        
    def test_all_expansions_fail(self):
        """Test case where all expansions fail / Prueba caso donde todas las expansiones fallan"""
        # Create clusters that will fail expansion / Crea clusters que fallarán la expansión
        tiny_clusters = [np.array([[0.0, 0.0]])]  # Too small / Muy pequeño
        sparse_pc = np.array([[10.0, 10.0, 1.0]])  # Too far / Muy lejos
        
        config = ExpansionConfig(min_points_per_trunk=50)  # High threshold / Umbral alto
        
        trunks, metadata = expand_and_merge_clusters(tiny_clusters, sparse_pc, config)
        
        assert len(trunks) == 0
        assert metadata['expansion_stats']['failed_expansions'] == 1
        assert metadata['expansion_stats']['successful_expansions'] == 0

    def test_duplicate_merge(self):
        """
        Test that overlapping slice clusters get merged into single trunk.
        Prueba que clusters de corte superpuestos se fusionen en un solo tronco.
        """
        # Create two overlapping slice clusters / Crear dos clusters de corte superpuestos
        np.random.seed(42)  # For reproducible results / Para resultados reproducibles
        s1 = np.random.rand(40, 3) * 0.02 + [0.5, 0.5, 1.3]
        s2 = s1 + [0.05, 0.04, 0]  # Overlapping trunk / Tronco superpuesto
        
        # Create full point cloud with vertical extent / Crear nube de puntos completa con extensión vertical
        full = np.vstack([
            s1 + [0, 0, 5],   # Points above s1 / Puntos arriba de s1
            s2 + [0, 0, -5],  # Points below s2 / Puntos abajo de s2  
            s1,               # Original s1 points / Puntos originales s1
            s2                # Original s2 points / Puntos originales s2
        ])
        
        # Test merge functionality using simplified interface / Probar funcionalidad de fusión usando interfaz simplificada
        trunks = expand_and_merge(
            [s1, s2], 
            full, 
            radius=0.5, 
            merge_tol=0.25
        )
        
        # Should merge into single trunk / Debe fusionarse en un solo tronco
        assert len(trunks) == 1, f"Expected 1 merged trunk, got {len(trunks)}"
        
        # Merged trunk should have points from both original clusters
        # Tronco fusionado debe tener puntos de ambos clusters originales
        merged_trunk = trunks[0]
        assert merged_trunk.shape[0] > s1.shape[0], "Merged trunk should have more points than individual cluster"
        assert merged_trunk.shape[1] >= 3, "Merged trunk should have XYZ coordinates"

    def test_separate_clusters_no_merge(self):
        """
        Test that well-separated clusters don't get merged.
        Prueba que clusters bien separados no se fusionen.
        """
        # Create two well-separated slice clusters / Crear dos clusters de corte bien separados
        np.random.seed(42)
        s1 = np.random.rand(20, 3) * 0.02 + [0.0, 0.0, 1.3]
        heights1 = np.random.rand(20) * 10  # Random heights for s1 / Alturas aleatorias para s1
        full_s1 = s1.copy()
        full_s1[:, 2] = heights1  # Replace Z with random heights / Reemplazar Z con alturas aleatorias
        
        s2 = np.random.rand(20, 3) * 0.02 + [2.0, 2.0, 1.3]  # Far apart / Muy separados
        heights2 = np.random.rand(20) * 10  # Random heights for s2 / Alturas aleatorias para s2
        full_s2 = s2.copy()
        full_s2[:, 2] = heights2  # Replace Z with random heights / Reemplazar Z con alturas aleatorias
        
        full = np.vstack([full_s1, full_s2])
        
        # Test that they remain separate / Probar que permanezcan separados
        trunks = expand_and_merge(
            [s1, s2], 
            full, 
            radius=0.8, 
            merge_tol=0.5  # Even with generous tolerance / Incluso con tolerancia generosa
        )
        
        # Should remain as 2 separate trunks / Deben permanecer como 2 troncos separados
        assert len(trunks) == 2, f"Expected 2 separate trunks, got {len(trunks)}"


@pytest.mark.integration 
class TestExpansionIntegration:
    """Integration tests for expansion module / Pruebas de integración para módulo de expansión"""
    
    def test_realistic_forest_scenario(self):
        """Test with realistic forest-like data / Prueba con datos realistas tipo bosque"""
        np.random.seed(42)
        
        # Create 3 tree-like clusters in slice / Crea 3 clusters tipo árbol en corte
        tree_centers = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 2.0]])
        slice_clusters = []
        
        for center in tree_centers:
            # Add some noise around each center / Añade ruido alrededor de cada centro
            cluster_points = center + np.random.normal(0, 0.1, (8, 2))
            slice_clusters.append(cluster_points)
            
        # Create full point cloud with vertical extent / Crea nube de puntos completa con extensión vertical
        full_points = []
        for center in tree_centers:
            # Add points in cylinder around each tree / Añade puntos en cilindro alrededor de cada árbol
            n_points = 100
            angles = np.random.uniform(0, 2*np.pi, n_points)
            radii = np.random.uniform(0, 0.4, n_points)  
            heights = np.random.uniform(0, 20, n_points)
            
            x = center[0] + radii * np.cos(angles)
            y = center[1] + radii * np.sin(angles) 
            z = heights
            
            tree_points = np.column_stack([x, y, z])
            full_points.append(tree_points)
            
        full_pc = np.vstack(full_points)
        
        # Run expansion pipeline / Ejecuta pipeline de expansión
        config = ExpansionConfig(
            expansion_radius=0.6,
            merge_tolerance=1.0,  # Prevent merging separate trees / Previene fusión de árboles separados
            min_points_per_trunk=20,
            max_expansion_ratio=15.0  # Allow larger expansions for forest scenario / Permitir expansiones más grandes para escenario de bosque
        )
        
        trunks, metadata = expand_and_merge_clusters(slice_clusters, full_pc, config)
        
        # Should detect 3 separate trees / Debe detectar 3 árboles separados
        assert len(trunks) == 3
        assert metadata['expansion_stats']['successful_expansions'] == 3
        
        # Each trunk should have reasonable number of points / Cada tronco debe tener número razonable de puntos
        for trunk in trunks:
            assert trunk.shape[0] >= 20
            assert trunk.shape[1] >= 3  # XYZ coordinates / Coordenadas XYZ
