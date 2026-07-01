import os
import urllib.request
import zipfile
import io
from app.utils.logger import logger
from app.utils.config_loader import load_config

# Main dataset URLs
URLS = [
    "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"
]

def download_dataset() -> str:
    """Downloads the Diabetes 130-US hospitals dataset, extracts diabetic_data.csv, and saves it."""
    config = load_config()
    raw_dir = config["paths"]["raw_data_dir"]
    target_path = os.path.join(raw_dir, config["data"]["raw_filename"])
    
    # Check if file already exists
    if os.path.exists(target_path):
        logger.info(f"Dataset already exists at {target_path}. Skipping download.")
        return target_path
        
    os.makedirs(raw_dir, exist_ok=True)
    logger.info("Starting dataset download...")
    
    download_success = False
    for url in URLS:
        try:
            logger.info(f"Attempting to download from {url}...")
            # Set a user-agent to bypass basic scraping blocks
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            
            with urllib.request.urlopen(req, timeout=30) as response:
                zip_data = response.read()
                
            logger.info("Download completed. Extracting CSV file...")
            
            # Read zip in memory and extract diabetic_data.csv
            with zipfile.ZipFile(io.BytesIO(zip_data)) as zip_ref:
                # Find the file ending with diabetic_data.csv
                csv_filename = None
                for file_in_zip in zip_ref.namelist():
                    if file_in_zip.endswith("diabetic_data.csv"):
                        csv_filename = file_in_zip
                        break
                        
                if csv_filename is None:
                    raise FileNotFoundError("diabetic_data.csv not found inside the downloaded ZIP file.")
                    
                # Extract file contents and write to target path
                with zip_ref.open(csv_filename) as source_file:
                    with open(target_path, "wb") as dest_file:
                        dest_file.write(source_file.read())
                        
            logger.info(f"Dataset successfully saved to {target_path}.")
            download_success = True
            break
        except Exception as e:
            logger.warning(f"Failed to download or extract from {url}: {e}")
            
    if not download_success:
        logger.error(
            "All download attempts failed.\n"
            "Please manually download the dataset from:\n"
            "https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008\n"
            "Extract 'diabetic_data.csv' and place it in the project directory at data/raw/diabetic_data.csv"
        )
        raise IOError("Failed to download dataset. See logs for manual installation instructions.")
        
    return target_path

if __name__ == "__main__":
    download_dataset()
