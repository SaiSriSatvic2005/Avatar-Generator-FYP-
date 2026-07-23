import os
import sys
import types
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Head_and_face_location import classify_face_region
from ori_model2 import calculate_orientation_from_landmarks


class ModuleRobustnessTests(unittest.TestCase):
    def test_classify_face_region_returns_mouth_label_for_center_contact(self):
        face_bbox = (100, 120, 220, 260)
        label = classify_face_region(face_bbox, (150, 180))
        self.assertEqual(label, "hamteeth")

    def test_orientation_helper_accepts_pose_context(self):
        hand_landmarks = [
            types.SimpleNamespace(x=0.52, y=0.58, z=-0.02),
            types.SimpleNamespace(x=0.52, y=0.58, z=-0.02),
            types.SimpleNamespace(x=0.52, y=0.58, z=-0.02),
            types.SimpleNamespace(x=0.52, y=0.58, z=-0.02),
            types.SimpleNamespace(x=0.52, y=0.58, z=-0.02),
            types.SimpleNamespace(x=0.56, y=0.54, z=-0.01),
            types.SimpleNamespace(x=0.52, y=0.58, z=-0.02),
            types.SimpleNamespace(x=0.52, y=0.58, z=-0.02),
            types.SimpleNamespace(x=0.68, y=0.50, z=-0.03),
            types.SimpleNamespace(x=0.52, y=0.58, z=-0.02),
            types.SimpleNamespace(x=0.52, y=0.58, z=-0.02),
            types.SimpleNamespace(x=0.78, y=0.48, z=-0.05),
            types.SimpleNamespace(x=0.52, y=0.58, z=-0.02),
            types.SimpleNamespace(x=0.52, y=0.58, z=-0.02),
            types.SimpleNamespace(x=0.52, y=0.58, z=-0.02),
            types.SimpleNamespace(x=0.52, y=0.58, z=-0.02),
            types.SimpleNamespace(x=0.50, y=0.45, z=-0.02),
            types.SimpleNamespace(x=0.48, y=0.45, z=-0.02),
            types.SimpleNamespace(x=0.52, y=0.58, z=-0.02),
            types.SimpleNamespace(x=0.52, y=0.58, z=-0.02),
        ]
        pose_landmarks = {
            "left_eye": (0.46, 0.22),
            "right_eye": (0.54, 0.22),
            "right_shoulder": (0.62, 0.40),
        }

        view, finger, palm = calculate_orientation_from_landmarks(
            hand_landmarks,
            handedness="Right",
            pose_landmarks=pose_landmarks,
        )

        self.assertIn(view, {"signer", "bird", "right"})
        self.assertTrue(finger.startswith("hamextfinger"))
        self.assertTrue(palm.startswith("hampalm"))


if __name__ == "__main__":
    unittest.main()
