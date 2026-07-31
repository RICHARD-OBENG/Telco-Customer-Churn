"""Compatibility wrapper for the downloader module."""

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


_downloader_module = _load_module("src.data._downloader", "01_downloader.py")

Downloader = _downloader_module.Downloader
DownloadError = _downloader_module.DownloadError

__all__ = ["Downloader", "DownloadError"]
