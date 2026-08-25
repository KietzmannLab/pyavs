"""
Basic tests for pyAVS package.
"""

import pytest
import pyavs


def test_import():
    """Test that pyAVS can be imported."""
    assert pyavs.__version__ is not None


def test_main_functions_exist():
    """Test that main API functions are available."""
    assert hasattr(pyavs, 'set_data_path')
    assert hasattr(pyavs, 'load_and_preprocess')
    assert hasattr(pyavs, 'get_epochs')
    assert hasattr(pyavs, 'check_data_availability')


def test_set_data_path_visible_to_internal_utils_config(tmp_path):
    """pyavs.set_data_path() and pyavs.utils.config.get_data_path() must agree.

    Regression test: these used to read from two independent global config
    stores (pyavs.config vs pyavs.utils.config), so calling the documented
    top-level pyavs.set_data_path() never updated what ~20 internal call
    sites (io/, source/, preprocessing/, dataloader/, scenes/, ...) actually
    read via pyavs.utils.config.get_data_path().
    """
    import pyavs.utils.config

    pyavs.set_data_path(str(tmp_path))

    assert pyavs.get_data_path() == str(tmp_path)
    assert pyavs.utils.config.get_data_path() == str(tmp_path)


def test_set_data_path_updates_derivatives_path(tmp_path):
    """A second set_data_path() call must not stay pinned to the first path.

    Regression test: setup_paths() used to write the resolved derivatives
    default (<data_path>/derivatives/pyavs) back into config.derivatives_path
    itself, the same field reserved for an explicit user override. Once set
    once, it never went back to None, so every later set_data_path() call saw
    a "genuine override" that was really just the previous call's computed
    default — permanently pinning derivatives_path to the first data_path.
    """
    from pyavs.config.manager import get_config

    path_a = tmp_path / "a"
    path_b = tmp_path / "b"

    pyavs.set_data_path(str(path_a))
    assert get_config().config.get_derivatives_path() == str(path_a / "derivatives" / "pyavs")

    pyavs.set_data_path(str(path_b))
    assert get_config().config.get_derivatives_path() == str(path_b / "derivatives" / "pyavs")


def test_modules_importable():
    """Test that all main modules can be imported."""
    import pyavs.dataloader
    import pyavs.preprocessing
    import pyavs.scenes
    import pyavs.source
    import pyavs.utils
    
    assert pyavs.dataloader is not None
    assert pyavs.preprocessing is not None
    assert pyavs.scenes is not None
    assert pyavs.source is not None
    assert pyavs.utils is not None


def test_cli_importable():
    """Test that CLI module can be imported."""
    import pyavs.cli
    assert pyavs.cli is not None


@pytest.mark.parametrize("module_name", [
    "pyavs.dataloader.loaders",
    "pyavs.dataloader.eye", 
    "pyavs.dataloader.meg",
    "pyavs.preprocessing.eye",
    "pyavs.preprocessing.meg",
    "pyavs.preprocessing.ica",
    "pyavs.preprocessing.alignment",
    "pyavs.scenes.objects",
    "pyavs.scenes.crops",
    "pyavs.source.forward",
    "pyavs.source.reconstruction",
    "pyavs.source.spaces",
    "pyavs.utils.config",
    "pyavs.utils.paths",
    "pyavs.utils.validation",
])
def test_submodule_imports(module_name):
    """Test that all submodules can be imported."""
    import importlib
    module = importlib.import_module(module_name)
    assert module is not None