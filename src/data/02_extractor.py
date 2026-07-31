"""
Dataset extraction utilities.

Handles extracting compressed dataset archives.

Author: Richard Obeng
"""

from pathlib import Path
import logging
import zipfile

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Raised when dataset extraction fails."""


class Extractor:
    """
    Extract files from compressed archives.

    Currently supports:
    - ZIP files

    Parameters
    ----------
    overwrite : bool, default=False
        Whether to overwrite existing extracted files.
    """

    def __init__(
        self,
        overwrite: bool = False,
    ) -> None:

        self.overwrite = overwrite


    def extract_zip(
        self,
        zip_path: Path,
        extract_to: Path,
    ) -> Path:
        """
        Extract a ZIP archive.

        Parameters
        ----------
        zip_path : Path
            Path to ZIP file.

        extract_to : Path
            Destination extraction directory.

        Returns
        -------
        Path
            Extraction directory.

        Raises
        ------
        ExtractionError
            If extraction fails.
        """

        try:

            logger.info(
                "Starting extraction: %s",
                zip_path,
            )


            if not zip_path.exists():

                raise ExtractionError(
                    f"ZIP file not found: {zip_path}"
                )


            if not zipfile.is_zipfile(zip_path):

                raise ExtractionError(
                    f"Invalid ZIP archive: {zip_path}"
                )


            extract_to.mkdir(
                parents=True,
                exist_ok=True,
            )


            with zipfile.ZipFile(
                zip_path,
                "r",
            ) as archive:


                self._validate_archive(
                    archive,
                    extract_to,
                )


                archive.extractall(
                    extract_to
                )


            logger.info(
                "Extraction completed: %s",
                extract_to,
            )


            return extract_to


        except zipfile.BadZipFile as error:

            logger.error(
                "Corrupted ZIP file: %s",
                zip_path,
            )

            raise ExtractionError(
                "ZIP archive is corrupted."
            ) from error


        except Exception as error:

            logger.exception(
                "Extraction failed."
            )

            raise ExtractionError(
                f"Unable to extract {zip_path}"
            ) from error



    def _validate_archive(
        self,
        archive: zipfile.ZipFile,
        extract_to: Path,
    ) -> None:
        """
        Prevent unsafe ZIP extraction.

        Protects against path traversal attacks such as:

        ../../important_file.txt

        Parameters
        ----------
        archive : ZipFile

        extract_to : Path

        Raises
        ------
        ExtractionError
        """

        extraction_path = extract_to.resolve()


        for member in archive.namelist():

            member_path = (
                extract_to / member
            ).resolve()


            if not str(member_path).startswith(
                str(extraction_path)
            ):

                raise ExtractionError(
                    f"Unsafe file path detected: {member}"
                )



    def remove_archive(
        self,
        zip_path: Path,
    ) -> None:
        """
        Delete ZIP archive after extraction.

        Parameters
        ----------
        zip_path : Path
        """

        if zip_path.exists():

            zip_path.unlink()

            logger.info(
                "Removed archive: %s",
                zip_path,
            )



    def extract(
        self,
        archive_path: Path,
        destination: Path,
    ) -> Path:
        """
        Extract supported archive formats.

        Parameters
        ----------
        archive_path : Path

        destination : Path

        Returns
        -------
        Path
        """

        extension = archive_path.suffix.lower()


        if extension == ".zip":

            return self.extract_zip(
                archive_path,
                destination,
            )


        raise ExtractionError(
            f"Unsupported archive format: {extension}"
        )