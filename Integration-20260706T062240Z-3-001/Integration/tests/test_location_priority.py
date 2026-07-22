import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from integration_pipeline import combine_hamnosys


class LocationPriorityTests(unittest.TestCase):
    def test_arm_space_location_wins_over_body_location(self):
        handshape = {"final": "hamflathand", "per_frame": ["hamflathand"] * 10}
        orientation = {"final": ("hamextfingeru", "hampalmd"), "per_frame": [("hamextfingeru", "hampalmd")] * 10}
        upper_body = {"final": "hamshoulders", "per_frame": ["hamshoulders"] * 10}
        head_face = {"final": None, "per_frame": []}
        hand_location = {"final": "hampalm", "per_frame": ["hampalm"] * 10}
        finger_location = {"final": "none", "per_frame": ["none"] * 10}
        contact = {"final": "no-contact", "per_frame": ["no-contact"] * 10}
        arm_space = {"final": "hamneutralspace", "per_frame": ["hamneutralspace"] * 10}
        movement1 = {"final": "hamnomotion", "per_frame": ["hamnomotion"] * 10}
        movement2 = {"final": None, "trajectory": []}

        result = combine_hamnosys(
            handshape,
            orientation,
            upper_body,
            head_face,
            hand_location,
            finger_location,
            contact,
            arm_space,
            movement1,
            movement2,
        )

        self.assertIn("hamneutralspace", result)
        self.assertNotIn("hamshoulders", result)


if __name__ == "__main__":
    unittest.main()
