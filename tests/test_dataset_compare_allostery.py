import unittest

import numpy as np

from TopoTunnel_UI.core.dataset_compare import (
    annotate_remote_hotspots,
    detect_residue_substitutions,
)


class DatasetCompareAllosteryTests(unittest.TestCase):
    def test_detects_only_explicit_identity_substitutions(self):
        rows = detect_residue_substitutions(
            {116: "ARG", 126: "ILE", 200: "UNK"},
            {116: "ALA", 126: "ILE", 200: "GLY"},
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sequence_id"], 117)
        self.assertEqual(rows[0]["label"], "S117 ARG→ALA")

    def test_remote_candidate_uses_active_site_distance_and_retains_region_distance(self):
        profile = {
            "hotspots": [{
                "region_id": "R1",
                "peak_fraction": 0.5,
                "max_score": 0.67,
                "reference_residue_ids": (126, 309),
                "target_residue_ids": (375, 379),
            }],
            "reference_bottleneck": {"fraction": 0.713, "radius": 1.25},
            "target_bottleneck": {"fraction": 0.661, "radius": 1.05},
        }
        centers = np.asarray([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        result = annotate_remote_hotspots(
            profile,
            [{
                "residue_id": 116,
                "sequence_id": 117,
                "reference_name": "ARG",
                "target_name": "ALA",
                "label": "S117 ARG→ALA",
            }],
            reference_center_points=centers,
            target_center_points=centers,
            reference_points_by_sequence={117: (10.0, 12.0, 0.0)},
            target_points_by_sequence={117: (10.0, 14.0, 0.0)},
            remote_distance_threshold=10.0,
        )
        hotspot = result["hotspots"][0]
        self.assertTrue(hotspot["remote_candidate"])
        self.assertAlmostEqual(hotspot["mutation_distance"], 12.0)
        self.assertAlmostEqual(hotspot["mutation_region_distance"], 12.0)
        self.assertAlmostEqual(hotspot["active_site_distance"], np.sqrt(244.0))
        self.assertEqual(
            hotspot["evidence_status"],
            "distal perturbation / path-response association candidate",
        )
        self.assertAlmostEqual(result["bottleneck_fraction_shift"], -0.052)
        self.assertAlmostEqual(result["bottleneck_radius_shift"], -0.20)

    def test_large_region_distance_does_not_replace_active_site_remote_criterion(self):
        profile = {
            "hotspots": [{
                "region_id": "R1",
                "peak_fraction": 1.0,
                "max_score": 0.7,
                "reference_residue_ids": (22,),
                "target_residue_ids": (22,),
            }],
        }
        centers = np.asarray([[0.0, 0.0, 0.0], [25.0, 0.0, 0.0]])
        result = annotate_remote_hotspots(
            profile,
            [{"residue_id": 10, "sequence_id": 11, "label": "S11 ALA→VAL"}],
            reference_center_points=centers,
            target_center_points=centers,
            reference_points_by_sequence={11: (5.0, 0.0, 0.0)},
            target_points_by_sequence={11: (5.0, 0.0, 0.0)},
            remote_distance_threshold=10.0,
        )
        hotspot = result["hotspots"][0]
        self.assertAlmostEqual(hotspot["mutation_region_distance"], 20.0)
        self.assertAlmostEqual(hotspot["active_site_distance"], 5.0)
        self.assertFalse(hotspot["remote_candidate"])
        self.assertEqual(hotspot["evidence_status"], "active-site distal criterion not met")

    def test_hotspot_containing_mutation_is_not_called_remote(self):
        profile = {
            "hotspots": [{
                "region_id": "R1",
                "peak_fraction": 0.5,
                "max_score": 0.8,
                "reference_residue_ids": (116,),
                "target_residue_ids": (116,),
            }],
        }
        result = annotate_remote_hotspots(
            profile,
            [{"residue_id": 116, "sequence_id": 117, "label": "S117 ARG→ALA"}],
            reference_center_points=((0, 0, 0), (20, 0, 0)),
            reference_points_by_sequence={117: (10, 30, 0)},
        )
        self.assertFalse(result["hotspots"][0]["remote_candidate"])


if __name__ == "__main__":
    unittest.main()
