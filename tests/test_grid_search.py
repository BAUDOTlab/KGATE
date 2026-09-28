"""
Tests for kgate.grid_search (suggest_value, run_grid_search).

The search-space rules are unit-tested with a MockTrial; the
`run_grid_search` pipeline is tested end to end with a stub Architect
(full training is not unit-testable, see conftest.py).
"""

import optuna
import pytest

from kgate.grid_search import (
    _collect_search_space,
    _range_spec,
    run_grid_search,
    suggest_value,
)


class MockTrial:
    """Minimal stand-in for optuna.trial.Trial recording all suggestions."""

    def __init__(self):
        self.calls = []

    def suggest_float(self, name, low, high, step=None, log=False):
        self.calls.append(("float", name, low, high, step, log))
        return "suggested_float"

    def suggest_int(self, name, low, high, step=None, log=False):
        self.calls.append(("int", name, low, high, step, log))
        return "suggested_int"

    def suggest_categorical(self, name, choices):
        self.calls.append(("categorical", name, list(choices)))
        return choices[0]


class TestRangeSpec:
    def test_two_element_range(self):
        assert _range_spec([0.01, 0.1]) == (0.01, 0.1, None, False)
        assert _range_spec([1, 10]) == (1, 10, None, False)

    def test_step_range(self):
        assert _range_spec([1, 4, 1]) == (1, 4, 1, False)
        assert _range_spec([0.01, 0.1, 0.01]) == (0.01, 0.1, 0.01, False)

    def test_log_range(self):
        assert _range_spec([0.001, 0.1, True]) == (0.001, 0.1, None, True)

    def test_malformed_ranges_rejected(self):
        assert _range_spec([0.1, 0.01]) is None      # low >= high
        assert _range_spec([0.8, 0.1, 0.1]) is None  # data list (split proportions)
        assert _range_spec([1, 10, 0]) is None       # step not > 0
        assert _range_spec([1, 10, -1]) is None      # step not > 0
        assert _range_spec([0, 1, True]) is None     # log requires low > 0
        assert _range_spec([1, 10, "step"]) is None  # step not numeric
        assert _range_spec([1, 10, False]) is None   # not a documented form
        assert _range_spec(["a", "b"]) is None       # categorical choices
        assert _range_spec([1]) is None              # too short
        assert _range_spec([1, 2, 3, 4]) is None     # too long
        assert _range_spec([True, False]) is None    # bools are not numbers


class TestSuggestValue:
    def test_scalar_passthrough(self):
        trial = MockTrial()
        assert suggest_value(trial, "seed", 42) == 42
        assert trial.calls == []

    def test_evaluation_name_is_untouched(self):
        trial = MockTrial()
        assert suggest_value(trial, "evaluation", [1, 2, 3]) == [1, 2, 3]
        assert trial.calls == []

    def test_empty_list_passthrough(self):
        trial = MockTrial()
        assert suggest_value(trial, "target_edges", []) == []
        assert trial.calls == []

    def test_dict_recursion(self):
        trial = MockTrial()
        result = suggest_value(trial, "model", {"seed": 42, "name": ["A", "B"]})
        assert result == {"seed": 42, "name": "A"}
        assert ("categorical", "name", ["A", "B"]) in trial.calls

    def test_categorical_list(self):
        trial = MockTrial()
        result = suggest_value(trial, "name", ["TransE", "TransH"])
        assert result == "TransE"
        assert trial.calls == [("categorical", "name", ["TransE", "TransH"])]

    def test_string_passthrough(self):
        trial = MockTrial()
        assert suggest_value(trial, "name", "TransE") == "TransE"
        assert trial.calls == []

    def test_float_range_list(self):
        trial = MockTrial()
        result = suggest_value(trial, "learning_rate", [0.01, 0.1, 0.01])
        assert result == "suggested_float"
        assert trial.calls == [("float", "learning_rate", 0.01, 0.1, 0.01, False)]

    def test_int_range_list(self):
        trial = MockTrial()
        result = suggest_value(trial, "gnn_layer_number", [1, 4, 1])
        assert result == "suggested_int"
        assert trial.calls == [("int", "gnn_layer_number", 1, 4, 1, False)]

    def test_log_float_range_list(self):
        trial = MockTrial()
        result = suggest_value(trial, "learning_rate", [0.001, 0.1, True])
        assert result == "suggested_float"
        assert trial.calls == [("float", "learning_rate", 0.001, 0.1, None, True)]

    def test_two_element_range_list(self):
        trial = MockTrial()
        result = suggest_value(trial, "learning_rate", [0.01, 0.1])
        assert result == "suggested_float"
        assert trial.calls == [("float", "learning_rate", 0.01, 0.1, None, False)]

    def test_numeric_data_list_is_left_as_is(self):
        # e.g. the preprocessing.split proportions: a purely numeric list
        # that is not a well-formed range must not be sampled.
        trial = MockTrial()
        assert suggest_value(trial, "split", [0.8, 0.1, 0.1]) == [0.8, 0.1, 0.1]
        assert trial.calls == []


class TestSearchSpaceValidation:
    def test_name_collision_raises(self):
        config = {
            "model": {
                "decoder": {"name": ["TransE", "ComplEx"]},
                "loss": {"name": ["Margin", "BCE"]},
            }
        }
        with pytest.raises(ValueError, match="unique"):
            _collect_search_space(config)

    def test_same_name_same_space_is_allowed(self):
        config = {
            "a": {"name": ["x", "y"]},
            "b": {"name": ["x", "y"]},
        }
        assert _collect_search_space(config) == {"name": ("categorical", ("x", "y"))}

    def test_evaluation_section_is_excluded(self):
        config = {"evaluation": {"objective": ["Link Prediction"]}}
        assert _collect_search_space(config) == {}


class StubArchitect:
    """Stand-in for kgate.Architect recording the received configuration."""

    instances = []

    def __init__(self, config_path="", knowledge_graph=None, dataframe=None, **config):
        self.config_path = config_path
        self.knowledge_graph = knowledge_graph
        self.dataframe = dataframe
        self.config = config
        self.trained = False
        StubArchitect.instances.append(self)

    def evaluate(self, engine=None):  # referenced by _attach_pruning
        pass

    def train_model(self, *args, **kwargs):
        self.trained = True

    def test(self):
        # Deterministic score derived from the sampled configuration.
        dimensions = self.config["model"]["node_embedding_dimensions"]
        return {"Global_metrics": float(dimensions)}


class TestRunGridSearch:
    @pytest.fixture(autouse=True)
    def stub_architect(self, monkeypatch):
        StubArchitect.instances = []
        monkeypatch.setattr("kgate.grid_search.Architect", StubArchitect)
        yield

    def test_pipeline_resolves_lists_and_maximizes(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            "output_directory = 'out'\n"
            "[model]\n"
            "node_embedding_dimensions = [4, 8]\n"
            "edge_embedding_dimensions = 4\n"
        )
        (tmp_path / "out").mkdir()

        best = run_grid_search(
            config_path=str(config_file),
            number_of_trials=3,
            kg="a kg placeholder",
            dataframe=None,
        )

        # Every trial got a full, resolved (scalar) configuration.
        assert len(StubArchitect.instances) == 3
        for architect in StubArchitect.instances:
            assert architect.trained
            assert architect.knowledge_graph == "a kg placeholder"
            dimensions = architect.config["model"]["node_embedding_dimensions"]
            # [4, 8] is a range: any int between 4 and 8 (inclusive).
            assert isinstance(dimensions, int) and 4 <= dimensions <= 8
            # Non-listed hyperparameters are left untouched.
            assert architect.config["model"]["edge_embedding_dimensions"] == 4

        # The objective maximizes test()["Global_metrics"]: the best trial
        # is the trial with the highest score among the completed ones.
        sampled = [a.config["model"]["node_embedding_dimensions"]
                   for a in StubArchitect.instances]
        assert best is not None
        assert best.value == float(max(sampled))
        assert best.params["node_embedding_dimensions"] in sampled

    def test_no_list_config_runs_every_trial(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            "output_directory = 'out'\n"
            "[model]\n"
            "node_embedding_dimensions = 4\n"
            "edge_embedding_dimensions = 4\n"
        )
        (tmp_path / "out").mkdir()

        best = run_grid_search(config_path=str(config_file), number_of_trials=2)

        assert len(StubArchitect.instances) == 2
        assert best is not None
        assert best.value == 4.0
        assert best.params == {}

    def test_name_collision_fails_before_training(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            "output_directory = 'out'\n"
            "[model]\n"
            "node_embedding_dimensions = 4\n"
            "[model.decoder]\n"
            "name = ['TransE', 'ComplEx']\n"
            "[model.loss]\n"
            "name = ['Margin', 'BCE']\n"
        )
        (tmp_path / "out").mkdir()

        with pytest.raises(ValueError, match="unique"):
            run_grid_search(config_path=str(config_file), number_of_trials=2)

        # No Architect was ever constructed.
        assert StubArchitect.instances == []
