import os
import gdown
import zipfile
import tarfile

# Google Drive File URL
FILE_URL = "https://drive.google.com/uc?id=16iETdmk1J2jJ1FBMY7MC43FvQHL7q_FN"

# Set Paths
BASE_DIR = "/workspaces/KannadaHandwritingRecognition/dataset"
OUTPUT_FILE = os.path.join(BASE_DIR, "dataset.zip")  # Change to dataset.tar.gz if needed
EXTRACT_DIR = os.path.join(BASE_DIR, "dataset")

# Ensure the target directory exists
os.makedirs(BASE_DIR, exist_ok=True)

def download_file():
    """Download the dataset file from Google Drive."""
    print(f"[INFO] Downloading dataset to {OUTPUT_FILE}...")
    gdown.download(FILE_URL, OUTPUT_FILE, quiet=False, fuzzy=True)
    print("[INFO] Download complete.")

def verify_and_extract():
    """Verify and extract the downloaded file."""
    if zipfile.is_zipfile(OUTPUT_FILE):
        print("[INFO] ZIP file detected. Extracting...")
        with zipfile.ZipFile(OUTPUT_FILE, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
    elif tarfile.is_tarfile(OUTPUT_FILE):
        print("[INFO] TAR file detected. Extracting...")
        with tarfile.open(OUTPUT_FILE, "r") as tar_ref:
            tar_ref.extractall(EXTRACT_DIR)
    else:
        print("[ERROR] The downloaded file is not a valid ZIP or TAR file. Deleting...")
        os.remove(OUTPUT_FILE)
        return False
    print(f"[INFO] Extraction complete. Files saved in '{EXTRACT_DIR}/'.")
    return True

def main():
    download_file()
    if not verify_and_extract():
        print("[ERROR] Download failed. Please check the file ID or try again.")

if __name__ == "__main__":
    main()