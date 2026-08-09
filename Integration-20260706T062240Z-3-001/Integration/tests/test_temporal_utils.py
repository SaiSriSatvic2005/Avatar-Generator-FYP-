import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from temporal_utils import smooth_frame_sequence, build_structured_sign_descriptor


class TemporalUtilsTests(unittest.TestCase):
    def test_smooth_frame_sequence_reduces_flicker(self):
        labels = ["a", "b", "a", "b", "a"]
        smoothed = smooth_frame_sequence(labels, window=3)
        self.assertEqual(len(smoothed), len(labels))
        self.assertEqual(smoothed[0], "a")
        self.assertEqual(smoothed[1], "a")
        self.assertEqual(smoothed[2], "a")

    def test_build_structured_sign_descriptor_creates_summary(self):
        modules = {
            "handshape": {"per_frame": ["hamfist", "hamfist", "hamflathand", "hamflathand"]},
            "orientation": {"per_frame": ["hampalmu", "hampalmu", "hampalmd", "hampalmd"]},
            "location": {"per_frame": ["hamchest", "hamchest", "hamhead", "hamhead"]},
            "movement1": {"per_frame": ["hammover", "hammover", "hammovel", "hammovel"]},
            "contact": {"per_frame": ["none", "none", "hamtouch", "hamtouch"]},
        }
        descriptor = build_structured_sign_descriptor(modules)
        self.assertIn("summary", descriptor)
        self.assertEqual(descriptor["summary"]["handshape"], "hamfist")
        self.assertEqual(descriptor["summary"]["orientation"], "hampalmu")


if __name__ == "__main__":
    unittest.main()
