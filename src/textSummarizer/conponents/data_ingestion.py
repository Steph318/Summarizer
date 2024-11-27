import os
import urllib.request as request
import zipfile
from textSummarizer.logging import logger
from textSummarizer.utils.common import get_size
from pathlib import Path
from textSummarizer.entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")  

        
    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        # root_dir = self.config.root_dir

        # Liste tous les fichiers dans le répertoire
        # for dirpath, dirnames, filenames in os.walk(root_dir):
        #     for file in filenames:
        #         print(os.path.join(dirpath, file))


        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

        # if zipfile.is_zipfile(self.config.local_data_file):
        #     with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
        #         zip_ref.extractall("unzip_path")
        # else:
        #     print("The file is not a valid ZIP archive.")

            