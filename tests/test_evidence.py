import unittest
import tempfile
from pathlib import Path

from TopoTunnel_UI.core.evidence import (
    ensemble_consistency_points,
    hotspot_residue_deltas,
    mapping_sensitivity,
    split_path_ids_by_frame,
    summarize_ca_distance_frames,
)


class EvidenceHelpersTest(unittest.TestCase):
    def test_mapping_sensitivity_reports_top_and_ineligible_cells(self):
        result = mapping_sensitivity([
            {"target_cluster_id": 4, "center_rmsd": 1.0, "residue_similarity": 0.62, "path_count_similarity": 0.8},
            {"target_cluster_id": 7, "center_rmsd": 0.8, "residue_similarity": 0.30, "path_count_similarity": 0.8},
        ], 4, thresholds=(0.25, 0.60), scenarios=(("Geometry", 0.0, 0.0),))
        self.assertEqual(result["total_count"], 2)
        self.assertEqual(result["eligible_count"], 2)
        self.assertEqual(result["top_count"], 1)

    def test_ensemble_points_keep_one_selected_relation_per_dataset(self):
        points = ensemble_consistency_points([
            {"reference_cluster_id": 2, "target_dataset": "a", "target_cluster_id": 4, "center_rmsd": 2.2, "residue_similarity": 0.63},
            {"reference_cluster_id": 3, "target_dataset": "b", "target_cluster_id": 1, "center_rmsd": 1.0, "residue_similarity": 0.8},
        ], 2)
        self.assertEqual([row["dataset_key"] for row in points], ["a"])

    def test_frame_blocks_are_sequential(self):
        blocks = split_path_ids_by_frame([
            {"pathIndex": 3, "frameId": 30},
            {"pathIndex": 1, "frameId": 10},
            {"pathIndex": 2, "frameId": 20},
            {"pathIndex": 4, "frameId": 40},
        ], 2)
        self.assertEqual(blocks, [[1, 2], [3, 4]])

    def test_hotspot_deltas_use_target_minus_reference(self):
        deltas = hotspot_residue_deltas({"position_rows": [{
            "start_fraction": 0.4,
            "end_fraction": 0.5,
            "reference_frequency": {10: 0.8, 11: 0.2},
            "target_frequency": {10: 0.3, 12: 0.7},
        }]}, {"start_fraction": 0.42, "end_fraction": 0.48})
        self.assertAlmostEqual(deltas[10], -0.5)
        self.assertAlmostEqual(deltas[12], 0.7)

    def test_ca_distance_summary_uses_framewise_median_and_iqr(self):
        def atom(serial, residue_id, x):
            return (
                f"ATOM  {serial:5d}  CA  ALA A{residue_id:4d}    "
                f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C\n"
            )

        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for frame, distance in enumerate((8.0, 10.0, 12.0, 14.0), start=1):
                path = Path(folder) / f"frame_{frame}.pdb"
                path.write_text(atom(1, 117, 0.0) + atom(2, 222, distance), encoding="utf-8")
                paths.append(str(path))
            result = summarize_ca_distance_frames(paths, 117, (222,))
        row = result["rows"][0]
        self.assertEqual(row["frame_count"], 4)
        self.assertAlmostEqual(row["median"], 11.0)
        self.assertAlmostEqual(row["q25"], 9.5)
        self.assertAlmostEqual(row["q75"], 12.5)


if __name__ == "__main__":
    unittest.main()
