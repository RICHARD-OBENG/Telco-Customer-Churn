"""Compatibility wrapper for the extractor module."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_module(module_name: str, file_name: str):
    module_path = Path(__file__).resolve().with_name(file_name)
    spec = spec_from_file_location(module_name, module_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module '{module_name}'")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_extractor_module = _load_module("src.data._extractor", "02_extractor.py")

Extractor = _extractor_module.Extractor
ExtractError = _extractor_module.ExtractionError

__all__ = ["Extractor", "ExtractError"]
