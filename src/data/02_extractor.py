"""
Dataset extraction utilities.

Handles extracting compressed dataset archives and moving CSV files
into the raw data folder.

Author: Richard Obeng
"""

import shutil
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


    def find_csv_files(
        self,
        directory: Path,
    ) -> list[Path]:
        """
        Find all CSV files in a directory recursively.

        Parameters
        ----------
        directory : Path
            Directory to search.

        Returns
        -------
        list[Path]
            Sorted list of CSV file paths.

        Raises
        ------
        ExtractionError
            If no CSV file is found.
        """

        csv_files = sorted(
            path for path in directory.rglob("*.csv") if path.is_file()
        )


        if not csv_files:

            raise ExtractionError(
                f"No CSV file found in {directory}"
            )


        logger.info(
            "Found %d CSV file(s) in %s",
            len(csv_files),
            directory,
        )


        return csv_files



    def move_csv_to_raw(
        self,
        csv_path: Path,
        raw_dir: Path,
        new_name: str | None = None,
    ) -> Path:
        """
        Move or copy a CSV file into the raw data folder.

        Parameters
        ----------
        csv_path : Path
            Path to the extracted CSV file.

        raw_dir : Path
            Destination raw data folder.

        new_name : str, optional
            New filename for the CSV. If None, the original name is kept.

        Returns
        -------
        Path
            Path to the CSV in the raw folder.

        Raises
        ------
        ExtractionError
            If destination already exists and overwrite is False.
        """

        raw_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        destination_name = new_name or csv_path.name
        destination = raw_dir / destination_name


        if destination.exists() and not self.overwrite:

            logger.info(
                "CSV already exists: %s",
                destination,
            )

            return destination


        shutil.copy2(
            csv_path,
            destination,
        )


        logger.info(
            "CSV saved to raw folder: %s",
            destination,
        )


        return destination



    def extract_zip_to_raw(
        self,
        zip_path: Path,
        raw_dir: Path,
        csv_name: str | None = None,
        cleanup: bool = False,
    ) -> Path:
        """
        Extract a ZIP archive, locate the CSV, and store it in data/raw.

        Parameters
        ----------
        zip_path : Path
            Path to the ZIP archive.

        raw_dir : Path
            Destination raw data folder.

        csv_name : str, optional
            Desired filename for the CSV in the raw folder.

        cleanup : bool, default=False
            Whether to delete the ZIP archive after extraction.

        Returns
        -------
        Path
            Path to the CSV file in the raw folder.

        Raises
        ------
        ExtractionError
            If extraction or CSV handling fails.
        """

        extraction_dir = raw_dir / "extracted"
        extract_to = self.extract_zip(
            zip_path,
            extraction_dir,
        )


        csv_files = self.find_csv_files(extract_to)


        # Use the first CSV found. Datasets typically contain a single CSV.
        csv_path = csv_files[0]


        raw_csv_path = self.move_csv_to_raw(
            csv_path,
            raw_dir,
            new_name=csv_name,
        )


        if cleanup:

            self.remove_archive(zip_path)


        return raw_csv_path



if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"
    zip_path = raw_dir / "WA_Fn-UseC_-Telco-Customer-Churn.zip"

    extractor = Extractor(overwrite=False)

    extractor.extract_zip_to_raw(
        zip_path=zip_path,
        raw_dir=raw_dir,
        csv_name="telco_customer_churn.csv",
        cleanup=False,
    )