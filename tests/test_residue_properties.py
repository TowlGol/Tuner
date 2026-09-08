import unittest

import numpy as np

from TopoTunnel_UI.core.residue_properties import (
    MATERIALIZED_PROPERTY_KEYS,
    decode_materialized_properties,
    derive_path_dynamic_properties,
    derive_residue_properties,
    encode_materialized_properties,
    normalize_residue_name,
)
try:
    from TopoTunnel_UI.vis.charts import _profile_length, _series_for_profile
except ImportError:  # Qt runtime is optional for the pure data-property tests.
    _profile_length = None
    _series_for_profile = None


class ResiduePropertyTests(unittest.TestCase):
    def test_aliases_and_chemical_features(self):
        self.assertEqual(normalize_residue_name("HSD"), "HIS")
        values = derive_residue_properties(
            [[1, 2], [3, 4], [0, 0], [0, 0]],
            {1: "LYS", 2: "ASP", 3: "SER", 4: "HEM"},
            length=2,
        )
        self.assertEqual(values["residue_charge"][0], 0.5)
        self.assertEqual(values["residue_charge"][1], -0.5)
        self.assertGreater(values["residue_hbond_donor"][0], 0.0)
        self.assertEqual(values["residue_hbond_acceptor"][1], 0.5)

    def test_unknown_residue_is_neutral(self):
        values = derive_residue_properties([[7], [0], [0], [0]], {}, length=1)
        self.assertEqual(values["residue_hydrophobicity"], [0.0])
        self.assertEqual(values["residue_charge"], [0.0])
        self.assertEqual(values["hbond"], [0.0])

    def test_materialized_property_round_trip(self):
        point_count = 3
        source = {
            key: np.linspace(index, index + 1.0, point_count, dtype=np.float32)
            for index, key in enumerate(MATERIALIZED_PROPERTY_KEYS)
        }
        payload = encode_materialized_properties(source, point_count)
        decoded = decode_materialized_properties(payload, point_count)
        for key in MATERIALIZED_PROPERTY_KEYS:
            np.testing.assert_allclose(decoded[key], source[key])
        self.assertEqual(decoded["residue_charge"], decoded["charge"])

    def test_dynamic_properties_are_stable_and_point_aligned(self):
        values = derive_path_dynamic_properties(
            [3.0, 2.0, 1.0],
            [0.0, 2.0, 4.0],
            [0.2, 0.4, 0.6],
            [[1, 1, 2], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            frame_id=12,
        )
        self.assertEqual(values["frame"], [12.0, 12.0, 12.0])
        self.assertEqual(values["path_progress"], [0.0, 0.5, 1.0])
        self.assertEqual(values["residue_turnover"], [0.0, 0.0, 0.25])
        np.testing.assert_allclose(values["throughput"], [0.2, 0.4, 0.6])

    def test_old_profile_remains_valid_for_chart_helpers(self):
        if _profile_length is None or _series_for_profile is None:
            self.skipTest("PySide6 Qt runtime is unavailable")
        profile = {
            "resSeq": [0.0, 1.0],
            "radius": [1.0, 1.2],
            "hydrophobicity": [0.1, 0.2],
            "numPoints": 2,
        }
        self.assertEqual(_profile_length(profile), 2)
        self.assertEqual(_series_for_profile(profile, "residue_charge").size, 0)
        np.testing.assert_allclose(_series_for_profile(profile, "radius"), [1.0, 1.2])


if __name__ == "__main__":
    unittest.main()
