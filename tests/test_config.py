"""
Tests for kgate.config.
"""

import tomllib

import pytest

from kgate.config import Configuration


class TestParsing:
    def test_defaults(self):
        config = Configuration()
        assert config.seed == 42
        assert config.verbose is True
        assert config.node_embedding_dimensions == 256
        assert config.edge_embedding_dimensions == -1
        assert config.preprocessing.run is True
        assert tuple(config.preprocessing.split_proportions) == (0.8, 0.1, 0.1)
        assert config.decoder.name == "TransE"
        assert config.loss.name == "Margin"
        assert config.encoder.name == "None"
        assert config.initializer.name == "Random"
        assert config.negative_sampler.name == "Positional"

    def test_inline_override(self):
        config = Configuration(
            config_dict={
                "seed": 7,
                "model": {"node_embedding_dimensions": 8},
            }
        )
        assert config.seed == 7
        assert config.node_embedding_dimensions == 8
        # Untouched defaults are kept
        assert config.verbose is True
        assert config.edge_embedding_dimensions == -1

    def test_file_override(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text("seed = 123\n")
        config = Configuration(config_path=str(config_file))
        assert config.seed == 123

    def test_inline_wins_over_file(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text("seed = 123\n")
        config = Configuration(config_path=str(config_file),
                               config_dict={"seed": 7})
        assert config.seed == 7

    def test_file_wins_over_default(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text("model = { node_embedding_dimensions = 16 }")
        config = Configuration(config_path=str(config_file))
        assert config.node_embedding_dimensions == 16
        assert config.seed == 42  # untouched default

    def test_missing_config_file_raises(self):
        with pytest.raises(FileNotFoundError):
            Configuration(config_path="/nonexistent/path/config.toml")

    def test_subconfiguration_objects(self):
        config = Configuration()
        assert config.preprocessing.run is True
        assert config.preprocessing.remove_duplicate_triplets is True
        assert isinstance(config.decoder.name, str)
        assert isinstance(config.loss.margin, (int, float))
        assert config.optimizer.name == "Adam"
        assert config.training.train_batch_size > 0
        assert config.evaluation.target_edges is not None

    def test_repr(self):
        config = Configuration()
        assert "seed" in repr(config)

    @pytest.mark.xfail(
        strict=True,
        reason="The `flag_cartesian_edges` getter only has a docstring "
               "(no return statement), so it always returns None instead of "
               "the configured value. See fixes/config.py.txt.",
    )
    def test_flag_cartesian_edges_default(self):
        config = Configuration()
        assert config.preprocessing.flag_cartesian_edges is True


class TestSetConfigKey:
    def test_priority_inline_config_default(self):
        default = {"a": 1, "b": 2}
        assert Configuration.set_config_key("a", default, None, {"a": 5}) == 5
        assert Configuration.set_config_key("a", default, {"a": 3}, None) == 3
        assert Configuration.set_config_key("a", default, {"a": 3}, {"a": 5}) == 5
        assert Configuration.set_config_key("a", default, None, None) == 1

    def test_recursive_merge(self):
        default = {"sub": {"x": 1, "y": 2}}
        assert Configuration.set_config_key("sub", default,
                                            {"sub": {"y": 3}}, None) == {"x": 1, "y": 3}
        assert Configuration.set_config_key("sub", default, None,
                                            {"sub": {"z": 4}}) == {"x": 1, "y": 2, "z": 4}

    def test_required_key_without_default_raises(self):
        default = {"a": None}
        with pytest.raises(ValueError):
            Configuration.set_config_key("a", default)

    def test_none_default_overridden_by_inline(self):
        default = {"a": None}
        assert Configuration.set_config_key("a", default, None, {"a": 5}) == 5


class TestSave:
    def test_save_to_file(self, tmp_path):
        config = Configuration(config_dict={"seed": 99})
        out = tmp_path / "saved.toml"
        config.save(filename=out)
        with open(out, "rb") as f:
            saved = tomllib.load(f)
        assert saved["seed"] == 99

    def test_save_default_filename(self, tmp_path):
        config = Configuration(
            config_dict={"output_directory": str(tmp_path), "seed": 5}
        )
        config.save()
        assert (tmp_path / "kgate_config.toml").exists()


class TestPropertySetters:
    def test_seed_setter(self):
        config = Configuration()
        config.seed = 3
        assert config.seed == 3

    def test_node_embedding_dimensions_setter(self):
        config = Configuration()
        config.node_embedding_dimensions = 8
        assert config.node_embedding_dimensions == 8
        with pytest.raises(AssertionError):
            config.node_embedding_dimensions = 0
        with pytest.raises(AssertionError):
            config.node_embedding_dimensions = 2.5  # not an int

    def test_edge_embedding_dimensions_setter(self):
        config = Configuration()
        config.edge_embedding_dimensions = 8
        assert config.edge_embedding_dimensions == 8
        with pytest.raises(AssertionError):
            config.edge_embedding_dimensions = 0

    def test_output_directory_setter_creates_dir(self, tmp_path):
        target = tmp_path / "nested" / "out"
        config = Configuration()
        config.output_directory = str(target)
        assert str(config.output_directory) == str(target)
        assert target.is_dir()

    def test_knowledge_graph_csv_file_setter(self, tmp_path):
        csv_file = tmp_path / "kg.csv"
        csv_file.write_text("head,tail,edge\n")
        config = Configuration()
        config.knowledge_graph_csv_file = str(csv_file)
        assert config.knowledge_graph_csv_file == csv_file
        with pytest.raises(FileNotFoundError):
            config.knowledge_graph_csv_file = str(tmp_path / "missing.csv")

    def test_knowledge_graph_pickle_file_setter(self, tmp_path):
        pkl_file = tmp_path / "kg.pkl"
        pkl_file.write_bytes(b"")
        config = Configuration()
        config.knowledge_graph_pickle_file = str(pkl_file)
        assert config.knowledge_graph_pickle_file == pkl_file

    def test_split_proportions_setter(self):
        config = Configuration()
        config.preprocessing.split_proportions = (0.5, 0.3, 0.2)
        assert tuple(config.preprocessing.split_proportions) == (0.5, 0.3, 0.2)
        with pytest.raises(AssertionError):
            config.preprocessing.split_proportions = (0.5, 0.3, 0.1)

    def test_preprocessing_run_setter(self):
        config = Configuration()
        config.preprocessing.run = False
        assert config.preprocessing.run is False

    def test_preprocessing_make_directed_setter(self):
        config = Configuration()
        config.preprocessing.make_directed = "all"
        assert config.preprocessing.make_directed == "all"
        config.preprocessing.make_directed = ["E1"]
        assert config.preprocessing.make_directed == ["E1"]
        with pytest.raises(AssertionError):
            config.preprocessing.make_directed = "invalid"
