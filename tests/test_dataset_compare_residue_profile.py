import sqlite3
import unittest
from types import SimpleNamespace

from TopoTunnel_UI.app.main_window import MainWindow


def build_cluster_database(first_half_residues):
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE paths (id INTEGER PRIMARY KEY, cluster_id INTEGER);
        CREATE TABLE path_points (
            path_id INTEGER,
            seq INTEGER,
            x REAL,
            y REAL,
            z REAL,
            radius REAL,
            res_1 INTEGER,
            res_2 INTEGER,
            res_3 INTEGER,
            res_4 INTEGER
        );
        """
    )
    for path_id in range(1, 6):
        connection.execute("INSERT INTO paths(id, cluster_id) VALUES (?, 10)", (path_id,))
        for seq in range(65):
            residues = first_half_residues if seq < 32 else (300, 301)
            connection.execute(
                "INSERT INTO path_points VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)",
                (path_id, seq, float(seq), 0.0, 0.0, 1.0 + abs(seq - 32), int(residues[0]), int(residues[1])),
            )
    connection.commit()
    return connection


class DatasetCompareResidueProfileTests(unittest.TestCase):
    def test_full_path_profile_finds_replaced_residue_region(self):
        reference = build_cluster_database((145, 212))
        target = build_cluster_database((107, 271))
        reference_db = SimpleNamespace(
            conn=reference,
            _residue_name_map={145: "ALA", 212: "VAL", 300: "GLY", 301: "SER"},
        )
        target_db = SimpleNamespace(
            conn=target,
            _residue_name_map={107: "THR", 271: "LEU", 300: "GLY", 301: "SER"},
        )
        owner = SimpleNamespace(
            db=SimpleNamespace(
                _datasets_by_key={
                    "reference": SimpleNamespace(db=reference_db),
                    "target": SimpleNamespace(db=target_db),
                }
            )
        )

        profile = MainWindow._dataset_compare_residue_change_profile(
            owner,
            "reference",
            10,
            "target",
            10,
        )

        self.assertTrue(profile["hotspots"])
        strongest = profile["hotspots"][0]
        self.assertTrue({145, 212}.issubset(set(strongest["lost_residue_ids"])))
        self.assertTrue({107, 271}.issubset(set(strongest["gained_residue_ids"])))
        self.assertEqual(strongest["region_id"], "R1")
        self.assertEqual(profile["position_basis"], "physical_arc_length")
        self.assertLess(strongest["start_fraction"], 0.5)

        events = MainWindow._dataset_compare_path_position_events(
            owner,
            "reference",
            "target",
            profile,
        )
        self.assertTrue(events)
        first_half_event = next(event for event in events if float(event["path_fraction"]) < 0.5)
        self.assertEqual(set(first_half_event["reference_residue_ids"]), {145, 212})
        self.assertEqual(set(first_half_event["target_residue_ids"]), {107, 271})
        self.assertIn("same physical arc-length position", first_half_event["impact_text"])
        self.assertIn("S146 ALA (MD145)", first_half_event["display_label"])
        self.assertIn("S272 LEU (MD271)", first_half_event["display_label"])
        reference.close()
        target.close()


if __name__ == "__main__":
    unittest.main()
