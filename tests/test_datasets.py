"""
Tests for kgate.datasets.

Only ``get_data_root_directory`` is unit-tested: the dataset loaders
(``load_FB15k_237``, ``load_WN18RR``, ``load_PrimeKG``) download
multi-hundred-MB archives from the internet and are not suitable for a
unit test suite.

Note: the docstring says the fallback is "the current working directory",
but the implementation creates/uses a ``KGATE_DATA`` subfolder inside the
current working directory.
"""

from pathlib import Path

from kgate.datasets import get_data_root_directory


class TestGetDataRootDirectory:
    def test_default_creates_kgate_data_in_cwd(self, tmp_path, monkeypatch):
        monkeypatch.delenv("KGATE_DATA_ROOT", raising=False)
        monkeypatch.chdir(tmp_path)

        root = get_data_root_directory()

        assert root == tmp_path / "KGATE_DATA"
        assert root.is_dir()

    def test_environment_variable_overrides_default(self, tmp_path, monkeypatch):
        custom = tmp_path / "my_kgate_data"
        monkeypatch.setenv("KGATE_DATA_ROOT", str(custom))

        root = get_data_root_directory()

        assert root == custom
        assert root.is_dir()

    def test_returns_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KGATE_DATA_ROOT", str(tmp_path))
        assert isinstance(get_data_root_directory(), Path)
