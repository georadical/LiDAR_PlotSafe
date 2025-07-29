# 2025 LiDAR PlotSafe Project. All rights reserved.

"""
Módulo de extracción de rebanadas horizontales adaptativas para LiDAR PlotSafe.

Adaptive horizontal slice extraction module for LiDAR PlotSafe.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

def extract_adaptive_slice(
    points: np.ndarray,
    preferred_height: float = 1.3,
    thickness: float = 0.2,
    min_points: int = 200,
    fallback_percentiles=(10, 30, 50)
):
    """
    Return slice points and chosen height ensuring ≥ min_points.

    EN: Tries preferred height, then Z-percentiles until slice has enough points.
    ES: Prueba altura preferida y después percentiles Z hasta tener suficientes puntos.

    Parameters
    ----------
    points : np.ndarray
        (N,3) array of point coordinates
    preferred_height : float, optional
        Preferred height for slice extraction, by default 1.3
    thickness : float, optional
        Thickness of the slice, by default 0.2
    min_points : int, optional
        Minimum number of points required in slice, by default 200
    fallback_percentiles : tuple, optional
        Percentiles to try if preferred height fails, by default (10, 30, 50)

    Returns
    -------
    slice_pts : np.ndarray
        (M,3) array of points in the extracted slice
    used_height : float
        Height that was actually used for slice extraction
    warning : str | None
        Warning message if fallback was used, None otherwise
    """
    def _slice(pts, h):
        """
        Extrae rebanada a altura específica.
        Extract slice at specific height.
        """
        lb, ub = h - thickness / 2, h + thickness / 2
        return pts[(pts[:, 2] >= lb) & (pts[:, 2] <= ub)]

    # 1) Try preferred height first
    # Probar altura preferida primero
    sp = _slice(points, preferred_height)
    if len(sp) >= min_points:
        logger.info("Using preferred height %.2f m with %d points", preferred_height, len(sp))
        logger.info("Usando altura preferida %.2f m con %d puntos", preferred_height, len(sp))
        return sp, preferred_height, None

    # 2) Try fallback heights based on Z percentiles
    # Probar alturas de respaldo basadas en percentiles Z
    z_vals = points[:, 2]
    for p in fallback_percentiles:
        h = np.percentile(z_vals, p)
        sp = _slice(points, h)
        if len(sp) >= min_points:
            warning_msg = f"Adaptive slice: preferred height insufficient; using {h:.2f} m (P{p})"
            logger.warning("Adaptive slice: preferred height insufficient; using %.2f m (P%s)", h, p)
            logger.warning("Rebanada adaptativa: altura preferida insuficiente; usando %.2f m (P%s)", h, p)
            return sp, h, warning_msg

    # 3) Last resort: find densest slice using histogram
    # Último recurso: encontrar rebanada más densa usando histograma
    z_bins = np.linspace(z_vals.min(), z_vals.max(), 40)
    counts, _ = np.histogram(z_vals, bins=z_bins)
    idx = np.argmax(counts)
    h = (z_bins[idx] + z_bins[idx + 1]) / 2
    sp = _slice(points, h)
    
    warn = f"Low density slice (<{min_points}). Using densest height {h:.2f} m with {len(sp)} pts."
    logger.warning("Low density slice (<%d). Using densest height %.2f m with %d pts.", 
                   min_points, h, len(sp))
    logger.warning("Rebanada de baja densidad (<%d). Usando altura más densa %.2f m con %d pts.", 
                   min_points, h, len(sp))
    return sp, h, warn
