import unittest

from TopoTunnel_UI.app.main_window import MainWindow


class _MappingOwner:
    def __init__(self):
        points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        self._centers = {
            "source": {"clusters": {"1": {"points": points}, "2": {"points": points}, "3": {"points": points}}},
            "target": {"clusters": {"10": {"points": points}, "20": {"points": points}}},
        }
        self._dataset_compare_mapping_locks = {("source", "target"): {1: (10, 20)}}
        self._dataset_compare_residue_similarity_threshold = 0.5
        self._dataset_compare_rmsd_distance_threshold = 0.79
        self._dataset_compare_rmsd_max_threshold = 3.0

    def _dataset_compare_center_data(self, key):
        return self._centers[key]

    def _dataset_compare_pair_metrics(self, reference_key, target_key):
        return {
            (1, 10): {"center_rmsd": 0.1, "relation_cost": 0.1, "residue_similarity": 0.8},
            (1, 20): {"center_rmsd": 0.2, "relation_cost": 0.2, "residue_similarity": 0.9},
            (2, 10): {"center_rmsd": 0.3, "relation_cost": 0.3, "residue_similarity": 0.7},
            (2, 20): {"center_rmsd": 0.05, "relation_cost": 0.05, "residue_similarity": 0.85},
            (3, 10): {"center_rmsd": 0.04, "relation_cost": 0.04, "residue_similarity": 0.3},
            (3, 20): {"center_rmsd": 0.06, "relation_cost": 0.06, "residue_similarity": 0.4},
        }


class DatasetCompareMappingTests(unittest.TestCase):
    def test_pure_rmsd_mapping_ignores_residue_filter_and_preserves_locks(self):
        owner = _MappingOwner()

        assignment = MainWindow._dataset_compare_global_assignment(
            owner,
            "source",
            "target",
        )

        self.assertEqual(assignment[1]["target_cluster_id"], 10)
        self.assertTrue(assignment[1]["locked"])
        self.assertEqual(assignment[1]["target_cluster_ids"], (10, 20))
        self.assertEqual(assignment[2]["target_cluster_id"], 20)
        self.assertFalse(assignment[2]["locked"])
        self.assertEqual(assignment[2]["target_cluster_ids"], (20, 10))
        self.assertEqual(assignment[3]["target_cluster_id"], 10)
        self.assertEqual(assignment[3]["matching_method"], "RMSD family")
        self.assertAlmostEqual(assignment[3]["rmsd_family_margin"], 0.79)
        self.assertEqual(assignment[3]["rmsd_family_cluster_ids"], (10, 20))

    def test_clusters_beyond_distance_threshold_are_excluded(self):
        owner = _MappingOwner()
        owner._dataset_compare_rmsd_distance_threshold = 0.10

        assignment = MainWindow._dataset_compare_global_assignment(owner, "source", "target")

        self.assertEqual(assignment[2]["target_cluster_ids"], (20,))
        self.assertEqual(assignment[3]["target_cluster_ids"], (10, 20))

    def test_source_is_unmatched_when_best_candidate_exceeds_absolute_maximum(self):
        owner = _MappingOwner()
        owner._dataset_compare_rmsd_max_threshold = 0.03

        assignment = MainWindow._dataset_compare_global_assignment(owner, "source", "target")

        self.assertEqual(assignment[1]["target_cluster_ids"], ())
        self.assertIsNone(assignment[1]["target_cluster_id"])
        self.assertIn("No match", assignment[1]["matching_method"])
        self.assertEqual(assignment[1]["rmsd_family_size"], 0)
        self.assertEqual(owner._dataset_compare_mapping_locks[("source", "target")], {})

    def test_absolute_maximum_is_applied_before_family_margin(self):
        owner = _MappingOwner()
        owner._dataset_compare_rmsd_distance_threshold = 0.79
        owner._dataset_compare_rmsd_max_threshold = 0.15

        assignment = MainWindow._dataset_compare_global_assignment(owner, "source", "target")

        self.assertEqual(assignment[1]["target_cluster_ids"], (10,))
        self.assertEqual(assignment[2]["target_cluster_ids"], (20,))
        self.assertEqual(assignment[3]["target_cluster_ids"], (10, 20))


if __name__ == "__main__":
    unittest.main()
