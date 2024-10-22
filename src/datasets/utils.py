import os
import gdown
import zipfile

from src.path_manager import PathManager
from src.logger_config import logger

def download_dataset():
    fname = os.path.join(PathManager.preprocessed_path)
    link = os.getenv("GOOGLE_LINK", None)
    assert link is not None, "Please provide a link to download the data. Set environment variable 'GOOGLE_LINK'"
    archive_path = os.path.join(PathManager.preprocessed_path, "preprocessed.zip")
    logger.info(f"download_preprocessed_corpus() -- Checking {archive_path}")
    if not os.path.exists(archive_path):
        logger.info("download_preprocessed_corpus() -- Does not exist. Downloading")
        gdown.download(id=link, output=archive_path, quiet=False)
    else:
        logger.info("download_preprocessed_corpus() -- Archive exists.")

    logger.info("download_preprocessed_corpus() -- Unzipping")
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        [zip_ref.extract(member, fname) for member in zip_ref.namelist() if '__MACOSX' not in member]