"""
Download datasets from remote sources.

Author: Richard Obeng
"""

from pathlib import Path
from typing import Optional
import logging
import time

import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)


class DownloadError(Exception):
    """Raised when downloading a file fails."""


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
    """

    def __init__(
        self,
        retries: int = 3,
        timeout: int = 30,
        chunk_size: int = 8192,
    ) -> None:

        self.retries = retries
        self.timeout = timeout
        self.chunk_size = chunk_size

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