import unittest

from TopoTunnel_UI.app.main_window import MainWindow
from TopoTunnel_UI.app.widgets.residue_combination_panel import ResidueCombinationPanel


class _FakeCombinationPanel:
    def __init__(self):
        self.path_keys = None
        self.path_rows = None
        self.bottleneck_keys = None
        self.bottleneck_rows = None
        self.refresh_count = 0

    def begin_view_update_batch(self):
        pass

    def end_view_update_batch(self):
        pass

    def current_bottleneck_filter_active(self):
        return False

    def set_current_path_combination_keys(self, value):
        self.path_keys = value

    def set_current_path_combination_rows(self, rows, totals):
        self.path_rows = (rows, totals)

    def set_current_bottleneck_combination_keys(self, value):
        self.bottleneck_keys = value

    def set_current_bottleneck_combination_rows(self, rows, totals):
        self.bottleneck_rows = (rows, totals)

    def _refresh_view(self):
        self.refresh_count += 1


class _FakeMainWindow:
    def __init__(self):
        self._residue_combination_panel = _FakeCombinationPanel()
        self.path_refresh_calls = []
        self.distance_profiles = None

    def _begin_loading(self, _message):
        pass

    def _end_loading(self, *_args):
        pass

    def _refresh_residue_combination_current_paths(self, *, prefer_explicit_selection=False):
        self.path_refresh_calls.append(bool(prefer_explicit_selection))
        return {"dataset_1": [{"pathIndex": 1}]}

    def _refresh_residue_combination_current_bottlenecks(self, **_kwargs):
        raise AssertionError("Bottleneck refresh should be skipped when the filter is inactive")

    def _refresh_residue_combination_contained_match_status(self):
        pass

    def _sync_residue_combination_observer_frame_distances(self, profiles):
        self.distance_profiles = profiles

    def _sync_residue_combination_context(self):
        pass


class _SignalRecorder:
    def __init__(self):
        self.values = []

    def emit(self, value):
        self.values.append(value)


class _FakeScopePanel:
    def __init__(self):
        self.actions = []
        self._view_refresh_pending = False
        self.manual_update_requested = _SignalRecorder()

    def _set_loaded_scope_mode(self, mode):
        self.actions.append(("mode", mode))
        return True

    def _reload_csvs(self, *, refresh=True):
        self.actions.append(("reload", refresh))

    def _set_current_path_scope_active(self, active):
        self.actions.append(("path", active))
        return True

    def _set_current_bottleneck_scope_active(self, active):
        self.actions.append(("bottleneck", active))
        return True

    def _flush_pending_view_refresh(self):
        self.actions.append(("flush", True))


class ResidueCombinationScopeTests(unittest.TestCase):
    def test_load_dataset_turns_off_all_path_scope_filters(self):
        panel = _FakeScopePanel()
        ResidueCombinationPanel._on_load_dataset_clicked(panel)

        self.assertIn(("path", False), panel.actions)
        self.assertIn(("bottleneck", False), panel.actions)
        self.assertEqual(panel.manual_update_requested.values, ["dataset"])

    def test_load_dataset_ignores_current_path_context(self):
        window = _FakeMainWindow()
        MainWindow._manual_update_residue_combination_compare(window, "dataset")

        panel = window._residue_combination_panel
        self.assertEqual(window.path_refresh_calls, [])
        self.assertEqual(window.distance_profiles, {})
        self.assertEqual(panel.path_keys, set())
        self.assertEqual(panel.path_rows, ({}, {}))
        self.assertEqual(panel.bottleneck_keys, set())
        self.assertEqual(panel.bottleneck_rows, ({}, {}))

    def test_load_path_builds_current_path_context(self):
        window = _FakeMainWindow()
        MainWindow._manual_update_residue_combination_compare(window, "path")

        self.assertEqual(window.path_refresh_calls, [True])
        self.assertEqual(
            window.distance_profiles,
            {"dataset_1": [{"pathIndex": 1}]},
        )


if __name__ == "__main__":
    unittest.main()
