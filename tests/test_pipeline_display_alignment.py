import unittest

import numpy as np

from TopoTunnel_UI.vis.protein_model import (
    compose_pipeline_row_transforms,
    kabsch_pipeline_row_transform,
)


class PipelineDisplayAlignmentTests(unittest.TestCase):
    def test_kabsch_matches_pipeline_row_vector_convention(self):
        mobile = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ],
            dtype=np.float64,
        )
        expected_rotation = np.asarray(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        expected_translation = np.asarray([4.0, -2.0, 7.0], dtype=np.float64)
        fixed = mobile @ expected_rotation + expected_translation
        keys = [("A", index, "", "CA") for index in range(len(mobile))]
        mobile_map = {key: point for key, point in zip(keys, mobile)}
        fixed_map = {key: point for key, point in zip(keys, fixed)}

        rotation, translation, rmsd, matched_count = kabsch_pipeline_row_transform(
            mobile_map, fixed_map
        )

        np.testing.assert_allclose(mobile @ rotation + translation, fixed, atol=1e-10)
        self.assertLess(rmsd, 1e-10)
        self.assertEqual(matched_count, len(mobile))

    def test_composed_transform_matches_two_sequential_transforms(self):
        points = np.asarray([[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]], dtype=np.float64)
        first_rotation = np.asarray(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        first_translation = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
        second_rotation = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            dtype=np.float64,
        )
        second_translation = np.asarray([-4.0, 5.0, 2.0], dtype=np.float64)

        rotation, translation = compose_pipeline_row_transforms(
            first_rotation,
            first_translation,
            second_rotation,
            second_translation,
        )

        sequential = (
            points @ first_rotation + first_translation
        ) @ second_rotation + second_translation
        np.testing.assert_allclose(points @ rotation + translation, sequential, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
