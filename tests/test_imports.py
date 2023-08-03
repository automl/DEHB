import importlib

import pytest


class TestImports():
    """This class bundles all import tests for the DEHB project."""
    def test_dehb_import(self):
        """Test if DEHB can be imported properly."""
        try:
            importlib.import_module("src.dehb.optimizers.dehb")
        except ImportError as e:
            pytest.fail(f"Failed to import dehb: {e}")

    def test_de_import(self):
        """Test if DE can be imported properly."""
        try:
            importlib.import_module("src.dehb.optimizers.de")
        except ImportError as e:
            pytest.fail(f"Failed to import de: {e}")

    def test_utils_import(self):
        """Test if BracketManager can be imported properly."""
        try:
            importlib.import_module("src.dehb.utils")
        except ImportError as e:
            pytest.fail(f"Failed to import utils: {e}")