"""Compatibility wrapper for the validator module."""

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


_validator_module = _load_module("src.data._validator", "03_validator.py")

Validator = _validator_module.Validator
ValidationError = _validator_module.ValidationError

__all__ = ["Validator", "ValidationError"]
