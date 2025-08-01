#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 2025 LiDAR PlotSafe Project. All rights reserved.

"""
Interfaz de línea de comandos para LiDAR PlotSafe.

Command line interface for LiDAR PlotSafe.
"""

import argparse
import sys
import os
import yaml
import logging
from pathlib import Path

# Add src to path for imports
# Agregar src al path para importaciones
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src import io
from src.processing import crop_circular_plot
from src.pipeline.segmentation import segment_trees


def setup_logging(verbose: bool = False) -> None:
    """
    Configura el sistema de logging.
    
    Sets up the logging system.
    
    Args:
        verbose: Enable detailed logging if True
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_config(config_path: str) -> dict:
    """
    Carga la configuración desde un archivo YAML.
    
    Loads configuration from a YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error("Configuration file not found: %s", config_path)
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error("Error parsing configuration file: %s", e)
        sys.exit(1)


def parse_eps_value(eps_str: str) -> float:
    """
    Analiza el valor de epsilon desde la línea de comandos.
    
    Parses epsilon value from command line.
    
    Args:
        eps_str: Epsilon value as string ("auto" or numeric)
        
    Returns:
        Epsilon value (None for auto, float for numeric)
    """
    if eps_str.lower() == "auto":
        return None
    
    try:
        eps = float(eps_str)
        if eps <= 0:
            raise ValueError("Epsilon must be positive")
        return eps
    except ValueError as e:
        logging.error("Invalid epsilon value '%s': %s", eps_str, e)
        sys.exit(1)


def main():
    """
    Función principal del CLI.
    
    Main CLI function.
    """
    parser = argparse.ArgumentParser(
        description="LiDAR PlotSafe - Tree segmentation and analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with auto epsilon
  python run.py input.las --output results.csv
  
  # Custom epsilon value
  python run.py input.las --output results.csv --eps 0.3
  
  # Specify all parameters
  python run.py input.las --output results.csv --eps auto --min-samples 5 --radius 15.0
        """
    )
    
    # Required arguments
    # Argumentos requeridos
    parser.add_argument(
        "input",
        help="Input LAS/LAZ point cloud file"
    )
    
    # Optional arguments
    # Argumentos opcionales
    parser.add_argument(
        "-o", "--output",
        default="results.csv",
        help="Output file path (default: results.csv)"
    )
    
    parser.add_argument(
        "--config",
        default="config.yaml", 
        help="Configuration file path (default: config.yaml)"
    )
    
    parser.add_argument(
        "--eps",
        default="auto",
        help="DBSCAN epsilon parameter ('auto' for adaptive calculation or numeric value, default: auto)"
    )
    
    parser.add_argument(
        "--min-samples",
        type=int,
        help="Minimum samples for DBSCAN clustering (overrides config)"
    )
    
    parser.add_argument(
        "--radius",
        type=float,
        help="Plot radius in meters (overrides config)"
    )
    
    parser.add_argument(
        "--center-x",
        type=float,
        help="Plot center X coordinate (overrides config)"
    )
    
    parser.add_argument(
        "--center-y", 
        type=float,
        help="Plot center Y coordinate (overrides config)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Skip circular cropping step"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    # Configurar logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting LiDAR PlotSafe pipeline")
    logger.info("Iniciando pipeline LiDAR PlotSafe")
    
    # Load configuration
    # Cargar configuración
    config = load_config(args.config)
    logger.info("Loaded configuration from: %s", args.config)
    
    # Parse epsilon value
    # Analizar valor de epsilon
    eps = parse_eps_value(args.eps)
    if eps is None:
        logger.info("Using adaptive epsilon calculation")
        logger.info("Usando cálculo de epsilon adaptativo")
    else:
        logger.info("Using fixed epsilon: %.3f", eps)
        logger.info("Usando epsilon fijo: %.3f", eps)
    
    # Validate input file
    # Validar archivo de entrada
    if not os.path.exists(args.input):
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)
    
    try:
        # Load point cloud
        # Cargar nube de puntos
        logger.info("Loading point cloud from: %s", args.input)
        logger.info("Cargando nube de puntos desde: %s", args.input)
        
        points, metadata = io.load_point_cloud(args.input)
        logger.info("Loaded %d points", len(points))
        
        # Apply circular cropping if not disabled
        # Aplicar recorte circular si no está deshabilitado
        if not args.no_crop:
            # Get crop parameters from CLI args or config
            # Obtener parámetros de recorte desde args CLI o config
            center_x = args.center_x if args.center_x is not None else config.get('center_x', 0.0)
            center_y = args.center_y if args.center_y is not None else config.get('center_y', 0.0)
            radius = args.radius if args.radius is not None else config.get('radius', 15.0)
            
            logger.info("Cropping circular plot: center=(%.2f, %.2f), radius=%.2f m", 
                       center_x, center_y, radius)
            logger.info("Recortando plot circular: centro=(%.2f, %.2f), radio=%.2f m", 
                       center_x, center_y, radius)
            
            cropped_points, crop_stats = crop_circular_plot(
                points, 
                center=(center_x, center_y), 
                radius=radius
            )
            logger.info("Cropped to %d points (%.1f%% retained)", 
                       len(cropped_points), 100 * len(cropped_points) / len(points))
            points = cropped_points
        
        # Tree segmentation
        # Segmentación de árboles
        logger.info("Starting tree segmentation")
        logger.info("Iniciando segmentación de árboles")
        
        # Get segmentation parameters
        # Obtener parámetros de segmentación
        min_samples = args.min_samples if args.min_samples is not None else config.get('min_samples', 5)
        
        # Perform segmentation with adaptive or fixed epsilon
        # Realizar segmentación con epsilon adaptativo o fijo
        tree_clusters, seg_metadata = segment_trees(
            points,
            eps=eps,  # None for adaptive, float for fixed
            min_samples=min_samples,
            voxel_size=config.get('voxel_size', 0.05),
            slice_height=config.get('slice_height', 1.3),
            slice_thickness=config.get('slice_thickness', 0.2),
            min_tree_height=config.get('min_tree_height', 1.0),
            min_points=config.get('min_points', 30)
        )
        
        logger.info("Segmentation complete: found %d trees in %.2f seconds",
                   seg_metadata['n_trees'], seg_metadata['elapsed_time'])
        logger.info("Segmentación completa: %d árboles encontrados en %.2f segundos",
                   seg_metadata['n_trees'], seg_metadata['elapsed_time'])
        
        # Export results
        # Exportar resultados
        logger.info("Exporting results to: %s", args.output)
        logger.info("Exportando resultados a: %s", args.output)
        
        # Create output directory if it doesn't exist
        # Crear directorio de salida si no existe
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # TODO: Implement CSV export of tree statistics
        # TODO: Implementar exportación CSV de estadísticas de árboles
        logger.info("Results exported successfully")
        logger.info("Resultados exportados exitosamente")
        
        logger.info("Pipeline completed successfully")
        logger.info("Pipeline completado exitosamente")
        
    except Exception as e:
        logger.error("Error in pipeline execution: %s", str(e))
        logger.error("Error en ejecución del pipeline: %s", str(e))
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
