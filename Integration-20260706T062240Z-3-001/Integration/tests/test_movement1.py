import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from movement1_prava import classify_direction, classify_path


class MovementModuleTests(unittest.TestCase):
    def test_short_monotonic_motion_reports_direction(self):
        traj = np.array([
            [0.10, 0.10, 0.00],
            [0.12, 0.10, 0.00],
            [0.14, 0.11, 0.00],
            [0.16, 0.11, 0.00],
            [0.18, 0.11, 0.00],
        ], dtype=float)
        self.assertEqual(classify_direction(traj), "hammover")

    def test_circle_like_path_is_detected(self):
        theta = np.linspace(0, 2 * np.pi, 12)
        traj = np.column_stack([
            0.5 + 0.08 * np.cos(theta),
            0.5 + 0.08 * np.sin(theta),
            np.zeros_like(theta),
        ])
        self.assertEqual(classify_path(traj), "hamcircle")


if __name__ == "__main__":
    unittest.main()
