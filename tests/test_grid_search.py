"""
Tests for kgate.grid_search (suggest_value, run_grid_search).

Known bugs (documented in fixes/grid_search.py.txt and exposed with
xfail(strict=True)):
- ``suggest_value`` matches ``type(value[0])`` (a class) against the
  strings ``"float"`` / ``"int"``, which never matches, so numeric
  ``[low, high, step]`` / ``[low, high, log]`` lists fall through the
  ``match`` and return ``None`` instead of calling ``trial.suggest_*``.
- ``run_grid_search`` iterates over the ``Configuration`` object
  (``for key in config``) which is not iterable -> TypeError, and passes
  ``kg=`` / ``df=`` keyword arguments that ``Architect.__init__`` does
  not accept (they are silently swallowed by ``**kwargs``).
"""

import pytest

from kgate.grid_search import suggest_value


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

    @pytest.mark.xfail(
        strict=True,
        reason="`match type(value[0])` compares a type object against the "
               "strings 'float'/'int', which never matches, so numeric "
               "[low, high, step] lists fall through and return None "
               "instead of calling trial.suggest_float. "
               "See fixes/grid_search.py.txt.",
    )
    def test_float_range_list(self):
        trial = MockTrial()
        result = suggest_value(trial, "learning_rate", [0.01, 0.1, 0.01])
        assert result == "suggested_float"
        assert trial.calls == [("float", "learning_rate", 0.01, 0.1, 0.01, False)]

    @pytest.mark.xfail(
        strict=True,
        reason="Same `match type(value[0])` bug as test_float_range_list: "
               "the 'int' case never matches and the function returns None. "
               "See fixes/grid_search.py.txt.",
    )
    def test_int_range_list(self):
        trial = MockTrial()
        result = suggest_value(trial, "gnn_layer_number", [1, 4, 1])
        assert result == "suggested_int"
        assert trial.calls == [("int", "gnn_layer_number", 1, 4, 1, False)]

    @pytest.mark.xfail(
        strict=True,
        reason="Same `match type(value[0])` bug: the log-scale variant "
               "[low, high, True] also returns None. "
               "See fixes/grid_search.py.txt.",
    )
    def test_log_float_range_list(self):
        trial = MockTrial()
        result = suggest_value(trial, "learning_rate", [0.001, 0.1, True])
        assert result == "suggested_float"
        assert trial.calls == [("float", "learning_rate", 0.001, 0.1, None, True)]


class TestRunGridSearch:
    @pytest.mark.xfail(
        strict=True,
        reason="`run_grid_search` iterates over the Configuration object "
               "(not iterable -> TypeError) and passes `kg=`/`df=` keyword "
               "arguments that Architect.__init__ does not accept. "
               "See fixes/grid_search.py.txt.",
    )
    def test_run_grid_search_minimal(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            "output_directory = 'out'\n"
            "[model]\n"
            "node_embedding_dimensions = 4\n"
            "edge_embedding_dimensions = 4\n"
        )
        (tmp_path / "out").mkdir()

        from kgate.grid_search import run_grid_search

        run_grid_search(config_path=str(config_file), number_of_trials=1)
