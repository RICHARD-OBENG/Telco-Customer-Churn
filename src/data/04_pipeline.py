"""
Data ingestion pipeline orchestration.

Coordinates:
- Dataset downloading
- Archive extraction
- Dataset validation

Author: Richard Obeng
"""

from pathlib import Path
import logging

from src.data.downloader import Downloader
from src.data.extractor import Extractor
from src.data.validator import Validator
from ...config import config


logger = logging.getLogger(__name__)


class DataPipeline:
    """
    End-to-end data ingestion pipeline.

    Parameters
    ----------
    downloader : Downloader
        Handles downloading datasets.

    extractor : Extractor
        Handles archive extraction.

    validator : Validator
        Handles dataset validation.
    """

    def __init__(
        self,
        downloader: Downloader,
        extractor: Extractor,
        validator: Validator,
    ) -> None:

        self.downloader = downloader
        self.extractor = extractor
        self.validator = validator



    def build_url(
        self,
        config: dict,
    ) -> str:
        """
        Build GitHub raw download URL.

        Parameters
        ----------
        config : dict
            Dataset configuration.

        Returns
        -------
        str
            Dataset URL.
        """

        owner = config["github"]["owner"]
        repo = config["github"]["repository"]
        branch = config["github"]["branch"]

        zip_file = config["dataset"]["zip_file"]


        url = (
            "https://raw.githubusercontent.com/"
            f"{owner}/{repo}/{branch}/{zip_file}"
        )


        return url



    def run(
        self,
        config: dict,
    ) -> Path:
        """
        Execute complete data pipeline.

        Parameters
        ----------
        config : dict
            Pipeline configuration.

        Returns
        -------
        Path
            Extracted dataset location.

        Raises
        ------
        Exception
            If pipeline fails.
        """

        try:

            logger.info(
                "Starting data pipeline"
            )


            # --------------------------------
            # Build URL
            # --------------------------------

            url = self.build_url(
                config
            )


            dataset_name = (
                config["dataset"]["name"]
            )

            zip_file = (
                config["dataset"]["zip_file"]
            )


            download_dir = Path(
                config["download"]["destination"]
            )


            zip_path = (
                download_dir / zip_file
            )


            extract_path = (
                download_dir / dataset_name
            )


            logger.info(
                "Dataset URL created: %s",
                url,
            )


            # --------------------------------
            # Download dataset
            # --------------------------------

            self.downloader.download_if_needed(
                url=url,
                destination=zip_path,
            )


            logger.info(
                "Dataset downloaded successfully"
            )


            # --------------------------------
            # Extract dataset
            # --------------------------------

            if config["download"]["extract"]:

                self.extractor.extract(
                    archive_path=zip_path,
                    destination=extract_path,
                )


                logger.info(
                    "Dataset extracted successfully"
                )


            # --------------------------------
            # Validate dataset
            # --------------------------------

            if config["validation"]["enabled"]:

                self.validator.validate(
                    dataset_path=extract_path,
                    validation_config=config["validation"],
                )


                logger.info(
                    "Dataset validation successful"
                )


            # --------------------------------
            # Remove ZIP
            # --------------------------------

            if config["download"]["remove_zip"]:

                self.extractor.remove_archive(
                    zip_path
                )


                logger.info(
                    "ZIP archive removed"
                )


            logger.info(
                "Data pipeline completed successfully"
            )


            return extract_path



        except Exception as error:

            logger.exception(
                "Data pipeline failed"
            )

            raise error