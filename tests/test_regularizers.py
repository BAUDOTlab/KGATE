"""
Tests for kgate.regularizers and its Configuration / Architect wiring.

The Regularizer is the module that gathers what the decoders used to do in
their own `normalize_parameters` method (e.g. TransE, RESCAL and DistMult
L2-normalizing their node embeddings). It is initialized by the Architect
after the decoder, is given a set of parameters to regularize and the function
to apply to them, and is applied through the trainer hooks.
"""

import pytest
import torch
import torch.nn as nn

from kgate.architect import Architect
from kgate.config import Configuration
from kgate.regularizers import REGULARIZER_FUNCTIONS, Regularizer, l1_normalize, l2_normalize


class TestRegularizer:
    def test_l2_applies_in_place(self):
        param = nn.Parameter(torch.randn(5, 8))
        original = param
        regularizer = Regularizer(params=[param], func=l2_normalize)
        regularizer()
        # In place: the parameter object is preserved, so the optimizer keeps
        # its references to the same parameters
        assert param is original
        # Every row has unit L2 norm
        torch.testing.assert_close(torch.norm(param.data, dim=1), torch.ones(5))

    def test_l1_applies_in_place(self):
        param = nn.Parameter(torch.tensor([[3.0, 4.0], [1.0, 1.0]]))
        regularizer = Regularizer(params=[param], func=l1_normalize)
        regularizer()
        torch.testing.assert_close(torch.norm(param.data, p=1, dim=1), torch.ones(2))
        torch.testing.assert_close(param.data[0], torch.tensor([3.0 / 7.0, 4.0 / 7.0]))

    def test_custom_function(self):
        param = nn.Parameter(torch.ones(3, 4))
        regularizer = Regularizer(params=[param], func=lambda x: x * 0.5)
        regularizer()
        torch.testing.assert_close(param.data, torch.full_like(param.data, 0.5))

    def test_multiple_params(self):
        params = [nn.Parameter(torch.randn(3, 4)), nn.Parameter(torch.randn(2, 4))]
        regularizer = Regularizer(params=params, func=l2_normalize)
        regularizer()
        for param in params:
            torch.testing.assert_close(torch.norm(param.data, dim=1), torch.ones(param.data.shape[0]))

    def test_empty_params(self):
        regularizer = Regularizer(params=[], func=l2_normalize)
        regularizer()  # no error

    def test_non_callable_function_raises(self):
        with pytest.raises(TypeError):
            Regularizer(params=[nn.Parameter(torch.ones(1, 2))], func="L2")

    def test_non_parameter_raises(self):
        with pytest.raises(TypeError):
            Regularizer(params=[torch.ones(3, 4)], func=l2_normalize)

    def test_properties(self):
        param = nn.Parameter(torch.randn(2, 3))
        func = l2_normalize
        regularizer = Regularizer(params=[param], func=func)
        assert regularizer.params == [param]
        assert regularizer.func is func

    def test_repr(self):
        regularizer = Regularizer(params=[nn.Parameter(torch.randn(2, 3))], func=l2_normalize)
        assert "Regularizer" in repr(regularizer)
        assert "l2_normalize" in repr(regularizer)


class TestBuiltinFunctions:
    def test_registered_functions(self):
        assert set(REGULARIZER_FUNCTIONS) == {"L1", "L2"}

    def test_l2_matches_decoder_behavior(self):
        # This must be the same operation the TransE, RESCAL and DistMult
        # decoders applied in their `normalize_parameters` method
        x = torch.randn(6, 8)
        expected = torch.nn.functional.normalize(x, p=2, dim=1)
        torch.testing.assert_close(l2_normalize(x), expected)


class TestConfiguration:
    def test_defaults(self):
        config = Configuration()
        assert config.regularizer.name == "None"
        assert config.regularizer.params == "node"

    def test_inline_override(self):
        config = Configuration(config_dict={"model": {"regularizer": {"name": "L2", "params": "all"}}})
        assert config.regularizer.name == "L2"
        assert config.regularizer.params == "all"

    def test_invalid_name_raises(self):
        config = Configuration()
        with pytest.raises(AssertionError):
            config.regularizer.name = "L9"

    def test_invalid_params_raises(self):
        config = Configuration()
        with pytest.raises(AssertionError):
            config.regularizer.params = "banana"

    def test_register_name(self):
        config = Configuration()
        config.regularizer.register_name("MyCustom")
        assert config.regularizer.name == "MyCustom"
        assert "MyCustom" in config.regularizer.supported_regularizers

    def test_repr(self):
        assert "Regularizer_Configuration" in repr(Configuration().regularizer)


class TestArchitectWiring:
    def test_default_no_regularizer(self, embedded_kg, tmp_path):
        architect = Architect(
            knowledge_graph=embedded_kg,
            output_directory=str(tmp_path / "out"),
            preprocessing={"run_preprocessing": False},
        )
        assert architect.initialize_regularizer() is None
        architect.regularizer = None
        architect.apply_regularizer()  # no-op, no error

    def test_initialize_regularizer_node(self, embedded_kg, tmp_path):
        architect = Architect(
            knowledge_graph=embedded_kg,
            output_directory=str(tmp_path / "out"),
            preprocessing={"run_preprocessing": False},
            model={"regularizer": {"name": "L2", "params": "node"}},
        )
        regularizer = architect.initialize_regularizer()
        assert isinstance(regularizer, Regularizer)
        # Only the node embeddings are regularized
        assert len(regularizer.params) == 1
        assert regularizer.params[0] is embedded_kg.node_embeddings[0]
        regularizer()
        for node_embedding in embedded_kg.node_embeddings:
            torch.testing.assert_close(torch.norm(node_embedding.data, dim=1), torch.ones(node_embedding.data.shape[0]))

    def test_initialize_regularizer_edge(self, embedded_kg, tmp_path):
        architect = Architect(
            knowledge_graph=embedded_kg,
            output_directory=str(tmp_path / "out"),
            preprocessing={"run_preprocessing": False},
            model={"regularizer": {"name": "L1", "params": "edge"}},
        )
        regularizer = architect.initialize_regularizer()
        assert len(regularizer.params) == 1
        assert regularizer.params[0] is embedded_kg.edge_embeddings
        regularizer()
        torch.testing.assert_close(torch.norm(embedded_kg.edge_embeddings.data, p=1, dim=1), torch.ones(embedded_kg.edge_embeddings.data.shape[0]))

    def test_normalize_parameters_delegates(self, embedded_kg, tmp_path):
        architect = Architect(
            knowledge_graph=embedded_kg,
            output_directory=str(tmp_path / "out"),
            preprocessing={"run_preprocessing": False},
            model={"regularizer": {"name": "L2", "params": "all"}},
        )
        architect.initialize_model()
        assert isinstance(architect.regularizer, Regularizer)
        assert len(architect.regularizer.params) == 2  # node + edge
        architect.normalize_parameters()
        for param in architect.regularizer.params:
            expected = torch.ones(param.data.shape[0], device = param.data.device)
            torch.testing.assert_close(torch.norm(param.data, dim = 1), expected)
