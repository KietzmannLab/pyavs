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