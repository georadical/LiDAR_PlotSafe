#  2025 LiDAR PlotSafe Project. All rights reserved.

import unittest
import numpy as np
import os
import sys
import logging

# Configurar logging para pruebas
# Configure logging for tests
logging.basicConfig(level=logging.ERROR)

# Aadir directorio src al path para importaciones
# Add src directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.segmentation import (
    downsample_point_cloud,
    extract_horizontal_slice,
    cluster_trunks,
    filter_tree_clusters,
    segment_trees,
    assign_tree_ids
)


class TestSegmentation(unittest.TestCase):
    """Tests for the segmentation module."""

    def setUp(self):
        """
        Prepara los datos para las pruebas.
        
        Prepares data for tests.
        """
        # Crear nube de puntos sinttica para pruebas
        # Create synthetic point cloud for testing
        np.random.seed(42)  # Para resultados reproducibles / For reproducible results
        
        # Generar 3 rboles sintticos
        # Generate 3 synthetic trees
        self.tree1 = self._create_synthetic_tree(center=(0, 0), height=10, radius=0.2)
        self.tree2 = self._create_synthetic_tree(center=(5, 0), height=8, radius=0.15)
        self.tree3 = self._create_synthetic_tree(center=(2, 4), height=12, radius=0.25)
        
        # Combinar los rboles en una sola nube
        # Combine trees into a single cloud
        self.points = np.vstack([self.tree1, self.tree2, self.tree3])
        
        # Aadir algo de ruido aleatorio
        # Add some random noise
        noise = np.random.uniform(-0.05, 0.05, size=(100, 3))
        self.points = np.vstack([self.points, noise])
        
        # Mezclar los puntos
        # Shuffle the points
        np.random.shuffle(self.points)

    def _create_synthetic_tree(self, center=(0, 0), height=10, radius=0.2, num_points=500):
        """
        Crea un rbol sinttico para pruebas.
        
        Creates a synthetic tree for testing.
        
        Args:
            center: (x, y) coordinates of the tree center
                   Coordenadas (x, y) del centro del rbol
            height: Height of the tree in meters
                    Altura del rbol en metros
            radius: Radius of the trunk in meters
                    Radio del tronco en metros
            num_points: Number of points to generate
                       Nmero de puntos a generar
        
        Returns:
            Nx3 numpy array of point coordinates
            Array numpy Nx3 de coordenadas de puntos
        """
        # Generar puntos a lo largo del tronco
        # Generate points along the trunk
        z_values = np.random.uniform(0, height, num_points)
        
        # Radio variable (ms ancho en la base)
        # Variable radius (wider at the base)
        radii = radius * (1 - 0.7 * (z_values / height))
        
        # Generar ngulos aleatorios alrededor del tronco
        # Generate random angles around the trunk
        angles = np.random.uniform(0, 2 * np.pi, num_points)
        
        # Calcular coordenadas x, y
        # Calculate x, y coordinates
        x = center[0] + radii * np.cos(angles)
        y = center[1] + radii * np.sin(angles)
        
        # Pequea variacin en z
        # Small variation in z
        z = z_values + np.random.uniform(-0.05, 0.05, num_points)
        
        # Crear array de coordenadas
        # Create coordinate array
        return np.column_stack((x, y, z))

    def _create_synthetic_tree_with_dbh(self, center=(0, 0), height=10, trunk_radius=0.2, dbh_height=1.3, n_points=500):
        """
        Crea un árbol sintético asegurando que haya puntos en la altura DBH.
        
        Creates a synthetic tree ensuring points at DBH height.
        
        Args:
            center: (x, y) coordinates of the tree center
                   Coordenadas (x, y) del centro del árbol
            height: Height of the tree in meters
                    Altura del árbol en metros
            trunk_radius: Radius of the trunk in meters
                         Radio del tronco en metros
            dbh_height: Height at which to ensure points (typically 1.3m)
                       Altura a la que garantizar puntos (típicamente 1.3m)
            n_points: Number of points to generate
                     Número de puntos a generar
        
        Returns:
            Nx3 numpy array of point coordinates
            Array numpy Nx3 de coordenadas de puntos
        """
        # Asignar algunos puntos específicamente en la altura DBH
        # Assign some points specifically at DBH height
        n_dbh_points = max(20, n_points // 10)  # Al menos 20 puntos o 10% del total
        n_other_points = n_points - n_dbh_points
        
        # Crear puntos para DBH
        # Create DBH points
        dbh_angles = np.random.uniform(0, 2 * np.pi, n_dbh_points)
        # Radio variable en el DBH (pequeña variación)
        # Variable radius at DBH (small variation)
        dbh_radii = trunk_radius * (0.8 + 0.4 * np.random.random(n_dbh_points))
        
        dbh_x = center[0] + dbh_radii * np.cos(dbh_angles)
        dbh_y = center[1] + dbh_radii * np.sin(dbh_angles)
        # Pequeña variación en la altura DBH
        # Small variation at DBH height
        dbh_z = dbh_height + np.random.uniform(-0.02, 0.02, n_dbh_points)
        
        # Crear el resto de puntos distribuidos por todo el árbol
        # Create the rest of the points distributed throughout the tree
        
        # Generar alturas para el resto de puntos (excluyendo la zona DBH)
        # Generate heights for the rest of the points (excluding the DBH zone)
        z_values = np.concatenate([
            np.random.uniform(0, dbh_height - 0.2, n_other_points // 2),  # Puntos por debajo del DBH / Points below DBH
            np.random.uniform(dbh_height + 0.2, height, n_other_points - n_other_points // 2)  # Puntos por encima del DBH / Points above DBH
        ])
        
        # Radio variable (más ancho en la base, más estrecho en la copa)
        # Variable radius (wider at base, narrower at crown)
        crown_factor = 1.5  # La copa puede ser más ancha que la base / Crown can be wider than base
        trunk_factor = 0.7  # El tronco se estrecha hacia arriba / Trunk narrows upward
        
        # Para puntos por debajo de DBH (tronco)
        # For points below DBH (trunk)
        below_dbh = z_values < dbh_height
        radii_below = trunk_radius * (1.0 - trunk_factor * (z_values[below_dbh] / dbh_height))
        
        # Para puntos por encima de DBH (copa y tronco superior)
        # For points above DBH (crown and upper trunk)
        above_dbh = ~below_dbh
        # El radio aumenta y luego disminuye hacia la punta
        # Radius increases and then decreases towards the tip
        height_ratio = (z_values[above_dbh] - dbh_height) / (height - dbh_height)
        crown_shape = 1.0 - np.power(height_ratio - 0.5, 2) * 4  # Forma parabólica con máximo en el medio / Parabolic shape with maximum in the middle
        radii_above = trunk_radius * (0.8 + crown_factor * crown_shape)
        
        # Combinar radios
        # Combine radii
        radii = np.zeros_like(z_values)
        radii[below_dbh] = radii_below
        radii[above_dbh] = radii_above
        
        # Generar ángulos aleatorios alrededor del tronco
        # Generate random angles around the trunk
        angles = np.random.uniform(0, 2 * np.pi, n_other_points)
        
        # Calcular coordenadas x, y
        # Calculate x, y coordinates
        x = center[0] + radii * np.cos(angles)
        y = center[1] + radii * np.sin(angles)
        
        # Pequeña variación en z
        # Small variation in z
        z = z_values + np.random.uniform(-0.05, 0.05, n_other_points)
        
        # Combinar los puntos DBH con el resto
        # Combine DBH points with the rest
        tree_points = np.column_stack((
            np.concatenate([dbh_x, x]),
            np.concatenate([dbh_y, y]),
            np.concatenate([dbh_z, z])
        ))
        
        return tree_points

    def test_downsample_point_cloud(self):
        """
        Prueba la funcin de submuestreo de la nube de puntos.
        
        Tests the point cloud downsampling function.
        """
        original_count = len(self.points)
        voxel_size = 0.1
        
        downsampled = downsample_point_cloud(self.points, voxel_size)
        
        # Verificar que el nmero de puntos se redujo
        # Verify that the number of points was reduced
        self.assertLess(len(downsampled), original_count)
        
        # Verificar que la forma es correcta
        # Verify that the shape is correct
        self.assertEqual(downsampled.shape[1], 3)

    def test_extract_horizontal_slice(self):
        """
        Prueba la extraccin de una rebanada horizontal de la nube de puntos.
        
        Tests the extraction of a horizontal slice from the point cloud.
        """
        slice_height = 1.3  # Altura DBH / DBH height
        slice_thickness = 0.2
        
        slice_points = extract_horizontal_slice(
            self.points, 
            slice_height=slice_height, 
            slice_thickness=slice_thickness
        )
        
        # Verificar que todos los puntos estn dentro del rango de altura
        # Verify that all points are within the height range
        z_min = slice_height - slice_thickness / 2
        z_max = slice_height + slice_thickness / 2
        
        self.assertTrue(np.all(slice_points[:, 2] >= z_min))
        self.assertTrue(np.all(slice_points[:, 2] <= z_max))

    def test_cluster_trunks(self):
        """
        Prueba la agrupacin de puntos para identificar troncos individuales.
        
        Tests clustering points to identify individual trunks.
        """
        # Crear una rebanada horizontal
        # Create a horizontal slice
        slice_height = 1.3
        slice_thickness = 0.2
        slice_points = extract_horizontal_slice(
            self.points, 
            slice_height=slice_height, 
            slice_thickness=slice_thickness
        )
        
        # Aplicar clustering
        # Apply clustering
        eps = 0.5  # Mayor que el radio del tronco / Larger than trunk radius
        min_samples = 5
        
        labels, clusters = cluster_trunks(slice_points, eps, min_samples)
        
        # Verificar que se encontraron 3 clusters (rboles)
        # Verify that 3 clusters (trees) were found
        self.assertEqual(len(clusters), 3)
        
        # Verificar que los labels tienen el formato correcto
        # Verify that labels have the correct format
        self.assertEqual(len(labels), len(slice_points))
        
        # Verificar que cada cluster tiene puntos
        # Verify that each cluster has points
        for cluster in clusters:
            self.assertGreater(len(cluster), 0)

    def test_filter_tree_clusters(self):
        """
        Prueba el filtrado de clusters para eliminar falsos positivos.
        
        Tests filtering clusters to remove false positives.
        """
        # Crear clusters de prueba
        # Create test clusters
        valid_cluster = np.vstack([
            np.array([[0, 0, 0]]), 
            np.array([[0, 0, 1]]),
            np.array([[0, 0, 2]])
        ])  # Altura 2m / Height 2m y 3 puntos / 3 points
        
        small_cluster = np.array([[1, 1, 0], [1, 1, 0.5]])  # Altura 0.5m / Height 0.5m
        few_points = np.array([[2, 2, 0], [2, 2, 3]])  # Solo 2 puntos / Only 2 points
        
        clusters = [valid_cluster, small_cluster, few_points]
        
        # Aplicar filtrado
        # Apply filtering
        filtered = filter_tree_clusters(
            clusters, 
            min_tree_height=1.0, 
            min_points=3
        )
        
        # Verificar que solo el cluster válido pasa el filtro
        # Verify that only the valid cluster passes the filter
        self.assertEqual(len(filtered), 1)

    def test_segment_trees(self):
        """
        Prueba la segmentación completa de árboles en la nube de puntos.
        
        Tests complete tree segmentation in the point cloud.
        """
        # Crear una nube de puntos sintética con tres árboles
        # Create a synthetic point cloud with three trees
        n_points = 1000
        point_cloud = np.zeros((n_points, 3))
        
        # Crear tres árboles completos con puntos a lo largo de toda su altura
        # Create three complete trees with points along their entire height
        n_points_per_tree = n_points // 3
        
        # Tree 1: tall tree (5m)
        tree1 = self._create_synthetic_tree_with_dbh(
            center=(0, 0),
            height=5.0,
            trunk_radius=0.3,
            dbh_height=1.3,
            n_points=n_points_per_tree
        )
        
        # Tree 2: medium tree (3m)
        tree2 = self._create_synthetic_tree_with_dbh(
            center=(3, 3),
            height=3.0,
            trunk_radius=0.2,
            dbh_height=1.3,
            n_points=n_points_per_tree
        )
        
        # Tree 3: small tree (2m)
        tree3 = self._create_synthetic_tree_with_dbh(
            center=(6, 0),
            height=2.0,
            trunk_radius=0.15,
            dbh_height=1.3,
            n_points=n_points_per_tree
        )
        
        # Combinar árboles en una nube de puntos
        # Combine trees into a point cloud
        point_cloud = np.vstack([tree1, tree2, tree3])
        
        # Debuggear punto por punto la segmentación
        # Debug segmentation step by step
        
        # 1. Verificar que hay suficientes puntos a la altura DBH para cada árbol
        # Verify there are enough points at DBH height for each tree
        dbh_slice = extract_horizontal_slice(point_cloud, slice_height=1.3, slice_thickness=0.3)
        print(f"Points at DBH slice: {len(dbh_slice)}")
        
        # Verificar que cada árbol tiene puntos en la rebanada
        # Verify that each tree has points in the slice
        tree1_slice = dbh_slice[np.sqrt((dbh_slice[:, 0] - 0)**2 + (dbh_slice[:, 1] - 0)**2) < 0.5]
        tree2_slice = dbh_slice[np.sqrt((dbh_slice[:, 0] - 3)**2 + (dbh_slice[:, 1] - 3)**2) < 0.5]
        tree3_slice = dbh_slice[np.sqrt((dbh_slice[:, 0] - 6)**2 + (dbh_slice[:, 1] - 0)**2) < 0.5]
        
        print(f"Tree 1 points at DBH slice: {len(tree1_slice)}")
        print(f"Tree 2 points at DBH slice: {len(tree2_slice)}")
        print(f"Tree 3 points at DBH slice: {len(tree3_slice)}")
        
        # 2. Realizar segmentación con parámetros adecuados para los datos sintéticos
        # Perform segmentation with appropriate parameters for synthetic data
        tree_clusters, metadata = segment_trees(
            points=point_cloud,
            voxel_size=0.05,
            eps=0.3,  # Radio de búsqueda suficiente para nuestros árboles sintéticos
            min_samples=5,
            slice_height=1.3,
            slice_thickness=0.3,
            min_tree_height=0.1,  # Usar un valor pequeño ya que la altura real se calculará con la nube completa
            min_points=5
        )
        
        # 3. Verificar resultados
        # Verify results
        assert len(tree_clusters) == 3, f"Se esperaban 3 árboles, se encontraron {len(tree_clusters)}"
        
        # 4. Verificar que cada cluster está cerca del centro de cada árbol
        # Verify that each cluster is close to the center of each tree
        centers = []
        for cluster in tree_clusters:
            center_xy = np.mean(cluster[:, :2], axis=0)
            centers.append(center_xy)
            
        # Ordenar centros por coordenada X para facilitar la comparación
        # Sort centers by X coordinate for easier comparison
        centers = sorted(centers, key=lambda c: c[0])
        
        # Verificar que los centros están cerca de los centros originales
        # Verify that the centers are close to the original centers
        assert np.linalg.norm(centers[0] - np.array([0, 0])) < 0.5, "El primer árbol debería estar cerca de (0,0)"
        assert np.linalg.norm(centers[2] - np.array([6, 0])) < 0.5, "El tercer árbol debería estar cerca de (6,0)"
        # El árbol del medio debería estar cerca de (3,3)
        assert np.linalg.norm(centers[1] - np.array([3, 3])) < 0.5, "El segundo árbol debería estar cerca de (3,3)"
        
        # 5. Verificar metadatos
        # Verify metadata
        assert "n_trees" in metadata
        assert metadata["n_trees"] == 3
        
        print("Prueba de segmentación exitosa: se encontraron los 3 árboles sintéticos")
        print("Segmentation test successful: all 3 synthetic trees were found")

    def test_assign_tree_ids(self):
        """
        Prueba la asignacin de IDs a rboles.
        
        Tests assigning IDs to trees.
        """
        # Crear clusters de prueba
        # Create test clusters
        clusters = [
            np.array([[0, 0, 0], [0, 0, 1]]),  # 2 puntos / 2 points
            np.array([[1, 1, 0], [1, 1, 1], [1, 1, 2]]),  # 3 puntos / 3 points
        ]
        
        # Asignar IDs
        # Assign IDs
        points, labels = assign_tree_ids(clusters)
        
        # Verificar la forma de los resultados
        # Verify the shape of results
        self.assertEqual(len(points), 5)  # 2 + 3 puntos / 2 + 3 points
        self.assertEqual(len(labels), 5)
        
        # Verificar que hay dos IDs nicos (0 y 1)
        # Verify that there are two unique IDs (0 and 1)
        unique_labels = np.unique(labels)
        self.assertEqual(len(unique_labels), 2)
        self.assertTrue(0 in unique_labels)
        self.assertTrue(1 in unique_labels)
        
        # Verificar la coherencia de las etiquetas
        # Verify label consistency
        self.assertEqual(np.sum(labels == 0), 2)  # 2 puntos con ID 0 / 2 points with ID 0
        self.assertEqual(np.sum(labels == 1), 3)  # 3 puntos con ID 1 / 3 points with ID 1

    def _create_synthetic_tree_with_dbh(self, center=(0, 0), height=10, trunk_radius=0.2, dbh_height=1.3, n_points=500):
        """
        Crea un árbol sintético asegurando que haya puntos en la altura DBH.
        
        Creates a synthetic tree ensuring points at DBH height.
        
        Args:
            center: (x, y) coordinates of the tree center
                   Coordenadas (x, y) del centro del árbol
            height: Height of the tree in meters
                    Altura del árbol en metros
            trunk_radius: Radius of the trunk in meters
                         Radio del tronco en metros
            dbh_height: Height at which to ensure points (typically 1.3m)
                       Altura a la que garantizar puntos (típicamente 1.3m)
            n_points: Number of points to generate
                     Número de puntos a generar
        
        Returns:
            Nx3 numpy array of point coordinates
            Array numpy Nx3 de coordenadas de puntos
        """
        # Asignar algunos puntos específicamente en la altura DBH
        # Assign some points specifically at DBH height
        n_dbh_points = max(20, n_points // 10)  # Al menos 20 puntos o 10% del total
        n_other_points = n_points - n_dbh_points
        
        # Crear puntos para DBH
        # Create DBH points
        dbh_angles = np.random.uniform(0, 2 * np.pi, n_dbh_points)
        # Radio variable en el DBH (pequeña variación)
        # Variable radius at DBH (small variation)
        dbh_radii = trunk_radius * (0.8 + 0.4 * np.random.random(n_dbh_points))
        
        dbh_x = center[0] + dbh_radii * np.cos(dbh_angles)
        dbh_y = center[1] + dbh_radii * np.sin(dbh_angles)
        # Pequeña variación en la altura DBH
        # Small variation at DBH height
        dbh_z = dbh_height + np.random.uniform(-0.02, 0.02, n_dbh_points)
        
        # Crear el resto de puntos distribuidos por todo el árbol
        # Create the rest of the points distributed throughout the tree
        
        # Generar alturas para el resto de puntos (excluyendo la zona DBH)
        # Generate heights for the rest of the points (excluding the DBH zone)
        z_values = np.concatenate([
            np.random.uniform(0, dbh_height - 0.2, n_other_points // 2),  # Puntos por debajo del DBH / Points below DBH
            np.random.uniform(dbh_height + 0.2, height, n_other_points - n_other_points // 2)  # Puntos por encima del DBH / Points above DBH
        ])
        
        # Radio variable (más ancho en la base, más estrecho en la copa)
        # Variable radius (wider at base, narrower at crown)
        crown_factor = 1.5  # La copa puede ser más ancha que la base / Crown can be wider than base
        trunk_factor = 0.7  # El tronco se estrecha hacia arriba / Trunk narrows upward
        
        # Para puntos por debajo de DBH (tronco)
        # For points below DBH (trunk)
        below_dbh = z_values < dbh_height
        radii_below = trunk_radius * (1.0 - trunk_factor * (z_values[below_dbh] / dbh_height))
        
        # Para puntos por encima de DBH (copa y tronco superior)
        # For points above DBH (crown and upper trunk)
        above_dbh = ~below_dbh
        # El radio aumenta y luego disminuye hacia la punta
        # Radius increases and then decreases towards the tip
        height_ratio = (z_values[above_dbh] - dbh_height) / (height - dbh_height)
        crown_shape = 1.0 - np.power(height_ratio - 0.5, 2) * 4  # Forma parabólica con máximo en el medio / Parabolic shape with maximum in the middle
        radii_above = trunk_radius * (0.8 + crown_factor * crown_shape)
        
        # Combinar radios
        # Combine radii
        radii = np.zeros_like(z_values)
        radii[below_dbh] = radii_below
        radii[above_dbh] = radii_above
        
        # Generar ángulos aleatorios alrededor del tronco
        # Generate random angles around the trunk
        angles = np.random.uniform(0, 2 * np.pi, n_other_points)
        
        # Calcular coordenadas x, y
        # Calculate x, y coordinates
        x = center[0] + radii * np.cos(angles)
        y = center[1] + radii * np.sin(angles)
        
        # Pequeña variación en z
        # Small variation in z
        z = z_values + np.random.uniform(-0.05, 0.05, n_other_points)
        
        # Combinar los puntos DBH con el resto
        # Combine DBH points with the rest
        tree_points = np.column_stack((
            np.concatenate([dbh_x, x]),
            np.concatenate([dbh_y, y]),
            np.concatenate([dbh_z, z])
        ))
        
        return tree_points


if __name__ == '__main__':
    unittest.main()
