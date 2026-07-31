"""
Dataset validation utilities.

Validates downloaded datasets before
they are used in ML pipelines.

Author: Richard Obeng
"""

from pathlib import Path
from typing import List, Optional
import hashlib
import logging
import zipfile

import pandas as pd


logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when dataset validation fails."""


class Validator:
    """
    Validate dataset files.

    Provides:
    - File checks
    - Size checks
    - Hash checks
    - ZIP integrity checks
    - CSV validation
    """


    def validate_exists(
        self,
        file_path: Path,
    ) -> bool:
        """
        Check if file exists.

        Parameters
        ----------
        file_path : Path

        Returns
        -------
        bool
        """

        exists = file_path.exists()

        if exists:
            logger.info(
                "File exists: %s",
                file_path,
            )

        else:
            logger.error(
                "File missing: %s",
                file_path,
            )

        return exists



    def validate_size(
        self,
        file_path: Path,
        minimum_bytes: int = 1,
    ) -> bool:
        """
        Validate minimum file size.

        Parameters
        ----------
        file_path : Path

        minimum_bytes : int
            Minimum acceptable size.

        Returns
        -------
        bool
        """

        if not self.validate_exists(file_path):

            return False


        size = file_path.stat().st_size


        if size < minimum_bytes:

            logger.error(
                "File too small: %s bytes",
                size,
            )

            return False


        logger.info(
            "File size valid: %s bytes",
            size,
        )

        return True



    def calculate_checksum(
        self,
        file_path: Path,
        algorithm: str = "sha256",
    ) -> str:
        """
        Calculate file checksum.

        Parameters
        ----------
        file_path : Path

        algorithm : str
            Hash algorithm.

        Returns
        -------
        str
            Hexadecimal checksum.
        """

        hash_function = hashlib.new(
            algorithm
        )


        with open(
            file_path,
            "rb",
        ) as file:

            for chunk in iter(
                lambda: file.read(8192),
                b"",
            ):

                hash_function.update(chunk)


        checksum = hash_function.hexdigest()


        logger.info(
            "Checksum generated: %s",
            checksum,
        )


        return checksum



    def validate_checksum(
        self,
        file_path: Path,
        expected_checksum: str,
    ) -> bool:
        """
        Compare file checksum.

        Parameters
        ----------
        file_path : Path

        expected_checksum : str

        Returns
        -------
        bool
        """

        actual_checksum = self.calculate_checksum(
            file_path
        )


        if actual_checksum != expected_checksum:

            logger.error(
                "Checksum mismatch."
            )

            return False


        logger.info(
            "Checksum validation passed."
        )

        return True



    def validate_zip(
        self,
        zip_path: Path,
    ) -> bool:
        """
        Validate ZIP archive integrity.

        Parameters
        ----------
        zip_path : Path

        Returns
        -------
        bool
        """

        if not zipfile.is_zipfile(zip_path):

            logger.error(
                "Invalid ZIP file: %s",
                zip_path,
            )

            return False


        try:

            with zipfile.ZipFile(
                zip_path,
                "r",
            ) as archive:

                corrupt_file = (
                    archive.testzip()
                )


                if corrupt_file:

                    logger.error(
                        "Corrupted file inside ZIP: %s",
                        corrupt_file,
                    )

                    return False


            logger.info(
                "ZIP validation passed."
            )

            return True


        except zipfile.BadZipFile:

            logger.error(
                "Bad ZIP archive."
            )

            return False



    def validate_directory(
        self,
        directory: Path,
        required_files: List[str],
    ) -> bool:
        """
        Validate extracted dataset structure.

        Example:

        required_files=[
            "train.csv",
            "test.csv"
        ]

        Parameters
        ----------
        directory : Path

        required_files : List[str]

        Returns
        -------
        bool
        """

        missing_files = []


        for file in required_files:

            file_path = directory / file


            if not file_path.exists():

                missing_files.append(file)



        if missing_files:

            logger.error(
                "Missing files: %s",
                missing_files,
            )

            return False



        logger.info(
            "Dataset structure validated."
        )

        return True



    def validate_csv(
        self,
        csv_path: Path,
        required_columns: Optional[List[str]] = None,
    ) -> bool:
        """
        Validate CSV dataset.

        Parameters
        ----------
        csv_path : Path

        required_columns : List[str], optional

        Returns
        -------
        bool
        """

        try:

            dataframe = pd.read_csv(
                csv_path
            )


            if dataframe.empty:

                logger.error(
                    "CSV file is empty."
                )

                return False



            if required_columns:

                missing = set(
                    required_columns
                ) - set(
                    dataframe.columns
                )


                if missing:

                    logger.error(
                        "Missing columns: %s",
                        missing,
                    )

                    return False



            logger.info(
                "CSV validation passed."
            )

            return True



        except Exception as error:

            logger.exception(
                "CSV validation failed."
            )

            return False



    def validate_dataset(
        self,
        dataset_path: Path,
        required_files: Optional[List[str]] = None,
    ) -> bool:
        """
        Run complete dataset validation.

        Parameters
        ----------
        dataset_path : Path

        required_files : List[str]

        Returns
        -------
        bool
        """

        logger.info(
            "Starting dataset validation..."
        )


        if not dataset_path.exists():

            raise ValidationError(
                f"Dataset not found: {dataset_path}"
            )


        if dataset_path.is_file():

            return self.validate_size(
                dataset_path
            )


        if dataset_path.is_dir() and required_files:

            return self.validate_directory(
                dataset_path,
                required_files,
            )


        return True