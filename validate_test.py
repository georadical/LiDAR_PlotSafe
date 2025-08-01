#!/usr/bin/env python3
import sys, os
sys.path.insert(0, 'src')

import numpy as np
from pipeline.clustering import compute_eps

# Your exact test
# Tu prueba exacta
def test_compute_eps_reasonable():
    # grid 0.04 m spacing → expect eps ≈ 0.08
    xy = np.mgrid[0:1:0.04, 0:1:0.04].reshape(2, -1).T
    eps = compute_eps(xy, k=8, factor=2.0)
    assert 0.07 < eps < 0.10, f"Expected 0.07 < eps < 0.10, got {eps}"
    return eps

if __name__ == "__main__":
    try:
        print("Running your test_compute_eps_reasonable()...")
        eps_result = test_compute_eps_reasonable()
        print(f"✅ TEST PASSED! Computed epsilon: {eps_result:.4f}")
        print(f"✅ PRUEBA APROBADA! Epsilon calculado: {eps_result:.4f}")
    except AssertionError as e:
        print(f"❌ TEST FAILED: {e}")
        print(f"❌ PRUEBA FALLIDA: {e}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
