#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
© 2025 LiDAR PlotSafe Project. All rights reserved.

Tests for the io module.
"""

import os
import pytest
import tempfile
import numpy as np
from unittest import mock

# Import the module to test
from src import io


class TestIO:
    """Tests for the io.py module."""
    
    def test_get_supported_extensions(self):
        """
        Test that the get_supported_extensions function returns the expected extensions.
        
        Prueba que la función get_supported_extensions devuelve las extensiones esperadas.
        """
        extensions = io.get_supported_extensions()
        assert isinstance(extensions, list)
        assert 'las' in extensions
        assert 'laz' in extensions
        
    def test_is_valid_point_cloud_file_with_nonexistent_file(self):
        """
        Test that is_valid_point_cloud_file returns False for a non-existent file.
        
        Prueba que is_valid_point_cloud_file devuelve False para un archivo inexistente.
        """
        assert not io.is_valid_point_cloud_file('nonexistent_file.las')
    
    def test_is_valid_point_cloud_file_with_unsupported_extension(self):
        """
        Test that is_valid_point_cloud_file returns False for a file with unsupported extension.
        
        Prueba que is_valid_point_cloud_file devuelve False para un archivo con extensión no soportada.
        """
        # Create a temporary file with an unsupported extension
        with tempfile.NamedTemporaryFile(suffix='.txt') as temp_file:
            assert not io.is_valid_point_cloud_file(temp_file.name)
    
    @mock.patch('laspy.open')
    @mock.patch('os.path.exists')
    def test_get_file_info(self, mock_exists, mock_laspy_open):
        """
        Test that get_file_info returns the expected information.
        
        Prueba que get_file_info devuelve la información esperada.
        """
        # Setup mocks
        mock_exists.return_value = True
        
        # Mock the laspy.open context manager
        mock_context = mock.MagicMock()
        mock_header = mock.MagicMock()
        mock_header.point_count = 1000
        mock_header.major_version = 1
        mock_header.minor_version = 4
        mock_header.point_format.id = 7
        mock_context.__enter__.return_value = mock.MagicMock(header=mock_header)
        mock_laspy_open.return_value = mock_context
        
        # Test with a mock file path
        test_file = 'test.las'
        
        # Mock os.path.getsize
        with mock.patch('os.path.getsize') as mock_getsize:
            mock_getsize.return_value = 1024 * 1024  # 1 MB
            
            result = io.get_file_info(test_file)
            
            # Assert the result has the expected keys
            assert 'filepath' in result
            assert 'filename' in result
            assert 'file_size_bytes' in result
            assert 'file_size_mb' in result
            assert 'point_count' in result
            assert 'version' in result
            assert 'point_format' in result
            
            # Assert the values
            assert result['filepath'] == test_file
            assert result['filename'] == 'test.las'
            assert result['file_size_bytes'] == 1024 * 1024
            assert result['file_size_mb'] == 1.0
            assert result['point_count'] == 1000
            assert result['version'] == '1.4'
            assert result['point_format'] == 7
    
    def test_get_file_info_file_not_found(self):
        """
        Test that get_file_info raises FileNotFoundError for a non-existent file.
        
        Prueba que get_file_info lanza FileNotFoundError para un archivo inexistente.
        """
        with pytest.raises(FileNotFoundError):
            io.get_file_info('nonexistent_file.las')
    
    def test_get_file_info_unsupported_format(self):
        """
        Test that get_file_info raises ValueError for an unsupported file format.
        
        Prueba que get_file_info lanza ValueError para un formato de archivo no soportado.
        """
        # Create a temporary file with an unsupported extension
        with tempfile.NamedTemporaryFile(suffix='.txt') as temp_file:
            with pytest.raises(ValueError):
                io.get_file_info(temp_file.name)
    
    @mock.patch('laspy.read')
    @mock.patch('os.path.exists')
    def test_load_point_cloud(self, mock_exists, mock_laspy_read):
        """
        Test that load_point_cloud returns the expected points and summary.
        
        Prueba que load_point_cloud devuelve los puntos y resumen esperados.
        """
        # Setup mocks
        mock_exists.return_value = True
        
        # Mock las data
        mock_las = mock.MagicMock()
        mock_las.x = np.array([0, 1, 2])
        mock_las.y = np.array([3, 4, 5])
        mock_las.z = np.array([6, 7, 8])
        mock_laspy_read.return_value = mock_las
        
        # Test with a mock file path
        test_file = 'test.las'
        
        # Mock os.path.getsize
        with mock.patch('os.path.getsize') as mock_getsize:
            mock_getsize.return_value = 1024 * 1024  # 1 MB
            
            points, summary = io.load_point_cloud(test_file)
            
            # Assert points shape and content
            assert points.shape == (3, 3)
            np.testing.assert_array_equal(points[:, 0], np.array([0, 1, 2]))
            np.testing.assert_array_equal(points[:, 1], np.array([3, 4, 5]))
            np.testing.assert_array_equal(points[:, 2], np.array([6, 7, 8]))
            
            # Assert the summary has the expected keys
            assert 'total_points' in summary
            assert 'x_range' in summary
            assert 'y_range' in summary
            assert 'z_range' in summary
            assert 'point_density' in summary
            assert 'file_size_mb' in summary
            
            # Assert the values
            assert summary['total_points'] == 3
            assert summary['x_range'] == (0.0, 2.0)
            assert summary['y_range'] == (3.0, 5.0)
            assert summary['z_range'] == (6.0, 8.0)
    
    def test_load_point_cloud_file_not_found(self):
        """
        Test that load_point_cloud raises FileNotFoundError for a non-existent file.
        
        Prueba que load_point_cloud lanza FileNotFoundError para un archivo inexistente.
        """
        with pytest.raises(FileNotFoundError):
            io.load_point_cloud('nonexistent_file.las')
    
    def test_load_point_cloud_unsupported_format(self):
        """
        Test that load_point_cloud raises ValueError for an unsupported file format.
        
        Prueba que load_point_cloud lanza ValueError para un formato de archivo no soportado.
        """
        # Create a temporary file with an unsupported extension
        with tempfile.NamedTemporaryFile(suffix='.txt') as temp_file:
            with pytest.raises(ValueError):
                io.load_point_cloud(temp_file.name)
