from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from HQPINN.lib.TAF import core_taf


class TAFPDELossTests(unittest.TestCase):
    def test_loss_pde_combines_near_and_far_subsets_separately(self) -> None:
        near_points = torch.tensor([[0.0, 0.0], [0.1, 0.0]], dtype=core_taf.DTYPE)
        far_points = torch.tensor(
            [[2.0, 0.0], [2.1, 0.0], [2.2, 0.0]],
            dtype=core_taf.DTYPE,
        )
        data = {
            "X_f": torch.cat([near_points, far_points], dim=0),
            "X_f_near": near_points,
            "X_f_far": far_points,
        }
        model = torch.nn.Identity()

        def _fake_mean_weighted_pde_residual(_model, X_f, eps_lambda):
            if torch.equal(X_f, near_points):
                return torch.tensor(2.0, dtype=core_taf.DTYPE)
            if torch.equal(X_f, far_points):
                return torch.tensor(6.0, dtype=core_taf.DTYPE)
            raise AssertionError(f"Unexpected collocation subset with shape {tuple(X_f.shape)}")

        with patch.object(
            core_taf,
            "_mean_weighted_pde_residual",
            side_effect=_fake_mean_weighted_pde_residual,
        ):
            loss = core_taf.loss_pde(model, data, n_f_batch=None)

        self.assertAlmostEqual(loss.item(), 4.0)

    def test_split_pde_points_by_airfoil_box_creates_near_and_far_sets(self) -> None:
        wall = torch.tensor(
            [[0.0, -0.05], [1.0, -0.05], [1.0, 0.05], [0.0, 0.05]],
            dtype=core_taf.DTYPE,
        )
        X_f = torch.tensor(
            [[0.5, 0.0], [1.2, 0.0], [2.0, 1.0]],
            dtype=core_taf.DTYPE,
        )

        near, far, near_box_low, near_box_high = core_taf.split_pde_points_by_airfoil_box(
            X_f,
            wall,
        )

        self.assertEqual(near.shape[0], 2)
        self.assertEqual(far.shape[0], 1)
        self.assertTrue(torch.all(near_box_low <= near_box_high))
