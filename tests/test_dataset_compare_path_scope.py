import sqlite3
import unittest

from TopoTunnel_UI.core.dataset_compare import (
    build_residue_change_profile,
    match_reference_cluster_paths,
)


def build_database():
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE paths (
            id INTEGER PRIMARY KEY,
            cluster_id INTEGER,
            path_length REAL,
            end_x REAL,
            end_y REAL,
            end_z REAL
        );
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
        CREATE TABLE residue_paths (residue_id INTEGER, path_id INTEGER);
        """
    )
    return connection


def add_path(connection, path_id, cluster_id, coords, residues):
    length = sum(
        ((coords[index][0] - coords[index - 1][0]) ** 2 +
         (coords[index][1] - coords[index - 1][1]) ** 2 +
         (coords[index][2] - coords[index - 1][2]) ** 2) ** 0.5
        for index in range(1, len(coords))
    )
    endpoint = coords[-1]
    connection.execute(
        "INSERT INTO paths VALUES (?, ?, ?, ?, ?, ?)",
        (path_id, cluster_id, length, endpoint[0], endpoint[1], endpoint[2]),
    )
    seen = set()
    for seq, (coord, point_residues) in enumerate(zip(coords, residues)):
        padded = list(point_residues)[:4] + [None] * (4 - len(point_residues))
        connection.execute(
            "INSERT INTO path_points VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (path_id, seq, coord[0], coord[1], coord[2], 1.0 + abs(seq - 1), *padded),
        )
        seen.update(int(value) for value in point_residues)
    for residue_id in sorted(seen):
        connection.execute("INSERT INTO residue_paths VALUES (?, ?)", (residue_id, path_id))
    connection.commit()


class DatasetComparePathScopeTests(unittest.TestCase):
    def test_reference_constraint_matches_paths_across_target_clusters(self):
        reference = build_database()
        target = build_database()
        coords = [(0, 0, 0), (5, 0, 0), (10, 0, 0)]
        add_path(reference, 1, 10, coords, [(10, 20), (10, 20), (30,)])
        add_path(reference, 2, 10, coords, [(10, 20), (20,), (30,)])
        add_path(target, 101, 8, coords, [(10,), (20,), (30,)])
        add_path(target, 102, 13, coords, [(10, 20), (30,), (30,)])
        add_path(target, 103, 8, coords, [(10,), (99,), (30,)])
        add_path(target, 104, 11, [(0, 0, 0), (20, 0, 0)], [(10,), (30,)])

        result = match_reference_cluster_paths(reference, target, 10, keep_ratio=1.0)

        self.assertEqual(result["target_path_ids"], [101, 102])
        self.assertEqual(result["target_cluster_counts"], {8: 1, 13: 1})
        self.assertEqual(result["target_path_ids_by_cluster"], {8: [101], 13: [102]})
        self.assertEqual(result["prefiltered_count"], 3)
        reference.close()
        target.close()

    def test_profile_uses_arc_length_and_individual_residue_regions(self):
        reference = build_database()
        target = build_database()
        # Unequal point spacing ensures seq/point-index normalization would put
        # the middle samples at the wrong physical location.
        add_path(
            reference, 1, 10,
            [(0, 0, 0), (8, 0, 0), (10, 0, 0)],
            [(145, 212), (145, 212), (300,)],
        )
        add_path(
            target, 101, 8,
            [(0, 0, 0), (2, 0, 0), (10, 0, 0)],
            [(107, 271), (107, 271), (300,)],
        )

        profile = build_residue_change_profile(
            reference,
            target,
            10,
            8,
            reference_path_ids=[1],
            target_path_ids=[101],
        )

        self.assertEqual(profile["position_basis"], "physical_arc_length")
        self.assertTrue(profile["hotspots"])
        self.assertEqual(profile["hotspots"][0]["region_id"], "R1")
        all_lost = {rid for row in profile["hotspots"] for rid in row["lost_residue_ids"]}
        all_gained = {rid for row in profile["hotspots"] for rid in row["gained_residue_ids"]}
        self.assertTrue({145, 212}.issubset(all_lost))
        self.assertTrue({107, 271}.issubset(all_gained))
        self.assertAlmostEqual(profile["reference_bottleneck"]["fraction"], 0.8)
        self.assertAlmostEqual(profile["target_bottleneck"]["fraction"], 0.2)
        reference.close()
        target.close()

    def test_profile_bottleneck_ignores_caver_seed_minimum(self):
        reference = build_database()
        target = build_database()
        coords = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)]
        residues = [(10,), (10,), (20,), (30,), (30,)]
        add_path(reference, 1, 10, coords, residues)
        add_path(target, 101, 8, coords, residues)
        for connection, path_id in ((reference, 1), (target, 101)):
            connection.execute(
                "UPDATE path_points SET radius = CASE seq "
                "WHEN 0 THEN 0.4 WHEN 2 THEN 0.8 ELSE 2.0 END WHERE path_id = ?",
                (path_id,),
            )
            connection.commit()

        profile = build_residue_change_profile(reference, target, 10, 8)

        self.assertAlmostEqual(profile["reference_bottleneck"]["fraction"], 0.5)
        self.assertAlmostEqual(profile["target_bottleneck"]["fraction"], 0.5)
        self.assertIn("seed zone", profile["reference_bottleneck"]["definition"])
        reference.close()
        target.close()

    def test_profile_is_invariant_to_point_density_for_same_environment(self):
        reference = build_database()
        target = build_database()
        sparse_coords = [(0, 0, 0), (5, 0, 0), (10, 0, 0)]
        dense_coords = [(float(index) / 10.0, 0, 0) for index in range(101)]
        add_path(reference, 1, 10, sparse_coords, [(10, 20)] * len(sparse_coords))
        add_path(target, 101, 8, dense_coords, [(10, 20)] * len(dense_coords))

        profile = build_residue_change_profile(reference, target, 10, 8)

        self.assertEqual(profile["position_basis"], "physical_arc_length")
        self.assertFalse(profile["hotspots"])
        self.assertTrue(all(float(row["score"]) == 0.0 for row in profile["bins"]))
        reference.close()
        target.close()


if __name__ == "__main__":
    unittest.main()
