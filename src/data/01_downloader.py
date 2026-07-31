"""
Download datasets from remote sources.

Author: Richard Obeng
"""

from pathlib import Path
from typing import Optional, Dict, Any
import logging
import time

import yaml
import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)


class DownloadError(Exception):
    """Raised when downloading a file fails."""


class ConfigError(Exception):
    """Raised when configuration file cannot be loaded."""


class Downloader:
    """
    Downloads files from remote URLs.

    Parameters
    ----------
    retries : int, default=3
        Maximum number of download attempts.

    timeout : int, default=30
        Request timeout in seconds.

    chunk_size : int, default=8192
        Number of bytes per downloaded chunk.

    config_path : Path, optional
        Path to data_url.yaml configuration file.
    """

    def __init__(
        self,
        retries: int = 3,
        timeout: int = 30,
        chunk_size: int = 8192,
        config_path: Optional[Path] = None,
    ) -> None:

        self.retries = retries
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.config_path = config_path
        self._config: Dict[str, Any] = {}

        if config_path:
            self._load_config(config_path)

    def _load_config(self, config_path: Path) -> None:
        """
        Load configuration from YAML file.

        Parameters
        ----------
        config_path : Path
            Path to the YAML configuration file.

        Raises
        ------
        ConfigError
            If config file cannot be loaded or parsed.
        """
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                self._config = yaml.safe_load(file) or {}
            logger.info("Configuration loaded from %s", config_path)
        except FileNotFoundError as error:
            raise ConfigError(
                f"Configuration file not found: {config_path}"
            ) from error
        except yaml.YAMLError as error:
            raise ConfigError(
                f"Failed to parse YAML configuration: {error}"
            ) from error

    def get_url(self, key: str) -> str:
        """
        Get a URL from the configuration by key.

        Supports nested keys using dot notation, e.g. "datasets.telco_churn".

        Parameters
        ----------
        key : str
            Configuration key for the dataset URL.

        Returns
        -------
        str
            The URL string.

        Raises
        ------
        ConfigError
            If key is not found in configuration.
        """
        parts = key.split(".")
        value: Any = self._config
        for part in parts:
            if not isinstance(value, dict) or part not in value:
                raise ConfigError(
                    f"URL key '{key}' not found in configuration"
                )
            value = value[part]
        if not isinstance(value, str):
            raise ConfigError(
                f"URL key '{key}' does not resolve to a string URL"
            )
        return value

    def download(
        self,
        url: str,
        destination: Path,
    ) -> Path:
        """
        Download a file.

        Parameters
        ----------
        url : str
            URL of the remote file.

        destination : Path
            Local path where the file will be saved.

        Returns
        -------
        Path
            Path to the downloaded file.

        Raises
        ------
        DownloadError
            If download fails after all retries.
        """

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        for attempt in range(1, self.retries + 1):

            try:

                logger.info(
                    "Downloading %s (Attempt %d/%d)",
                    url,
                    attempt,
                    self.retries,
                )

                response = requests.get(
                    url,
                    stream=True,
                    timeout=self.timeout,
                )

                response.raise_for_status()

                with open(destination, "wb") as file:

                    for chunk in response.iter_content(
                        chunk_size=self.chunk_size
                    ):

                        if chunk:
                            file.write(chunk)

                logger.info(
                    "Download completed: %s",
                    destination,
                )

                return destination

            except RequestException as error:

                logger.warning(
                    "Attempt %d failed: %s",
                    attempt,
                    error,
                )

                if attempt == self.retries:

                    logger.error(
                        "Download failed after %d attempts.",
                        self.retries,
                    )

                    raise DownloadError(
                        f"Unable to download {url}"
                    ) from error

                wait_time = 2 ** attempt

                logger.info(
                    "Retrying in %d seconds...",
                    wait_time,
                )

                time.sleep(wait_time)

        raise DownloadError("Unexpected download failure.")

    def download_from_config(
        self,
        config_key: str,
        destination: Optional[Path] = None,
        raw_dir: Path = Path("data/raw"),
        overwrite: bool = False,
    ) -> Path:
        """
        Download a file using URL from configuration.

        Parameters
        ----------
        config_key : str
            Key in the YAML configuration for the dataset URL.
            Supports dot notation for nested keys, e.g. "datasets.telco_churn".

        destination : Path, optional
            Local path where the file will be saved.
            If not provided, the file is saved to ``raw_dir`` using the
            basename from the URL.

        raw_dir : Path, default=Path("data/raw")
            Directory where raw datasets are stored.

        overwrite : bool
            Whether to overwrite an existing file.

        Returns
        -------
        Path
            Path to the downloaded file.

        Raises
        ------
        ConfigError
            If config key not found.
        """
        url = self.get_url(config_key)
        if destination is None:
            filename = Path(url).name or "dataset.zip"
            destination = raw_dir / filename
        return self.download_if_needed(
            url=url,
            destination=destination,
            overwrite=overwrite,
        )

    def file_exists(
        self,
        destination: Path,
    ) -> bool:
        """
        Check whether a file already exists.

        Parameters
        ----------
        destination : Path

        Returns
        -------
        bool
        """
        return destination.exists()

    def download_if_needed(
        self,
        url: str,
        destination: Path,
        overwrite: bool = False,
    ) -> Path:
        """
        Download only if the file does not exist.

        Parameters
        ----------
        url : str

        destination : Path

        overwrite : bool
            Whether to overwrite an existing file.

        Returns
        -------
        Path
        """

        if destination.exists() and not overwrite:

            logger.info(
                "File already exists: %s",
                destination,
            )

            return destination

        return self.download(
            url=url,
            destination=destination,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "config" / "data_url.yaml"

    downloader = Downloader(config_path=config_path)

    downloader.download_from_config(
        config_key="datasets.telco_churn",
        raw_dir=project_root / "data" / "raw",
        overwrite=False,
    )