from __future__ import annotations

import unittest

import numpy as np

from HQPINN.lib.TAF import generate_aerofoil_training_sets as taf_sampling


class TAFSamplingTests(unittest.TestCase):
    def test_sample_domain_points_biases_points_toward_airfoil_box(self) -> None:
        total_points = 1000
        near_fraction = 0.5

        points, sampling_info = taf_sampling.sample_domain_points(
            total_points=total_points,
            wall_points=taf_sampling.X_wall,
            poly_x=taf_sampling.poly_x,
            poly_y=taf_sampling.poly_y,
            near_airfoil_fraction=near_fraction,
            near_airfoil_pad_x=0.25,
            near_airfoil_pad_y=0.25,
            rng=np.random.default_rng(123),
        )

        self.assertEqual(points.shape, (total_points, 2))
        self.assertFalse(
            np.any(taf_sampling.point_in_polygon(points, taf_sampling.poly_x, taf_sampling.poly_y))
        )
        self.assertEqual(
            sampling_info["n_near_target"], int(round(total_points * near_fraction))
        )
        self.assertGreaterEqual(
            sampling_info["actual_near_count"], sampling_info["n_near_target"]
        )

    def test_sample_domain_points_rejects_invalid_fraction(self) -> None:
        with self.assertRaises(ValueError):
            taf_sampling.sample_domain_points(
                total_points=10,
                wall_points=taf_sampling.X_wall,
                poly_x=taf_sampling.poly_x,
                poly_y=taf_sampling.poly_y,
                near_airfoil_fraction=1.2,
                near_airfoil_pad_x=0.25,
                near_airfoil_pad_y=0.25,
                rng=np.random.default_rng(0),
            )
