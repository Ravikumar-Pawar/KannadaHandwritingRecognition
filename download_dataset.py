import os
import gdown
import zipfile
import tarfile
import re
import csv
from PIL import Image

# Google Drive File URL
FILE_URL = "https://drive.google.com/uc?id=16iETdmk1J2jJ1FBMY7MC43FvQHL7q_FN"

# Set Paths
BASE_DIR = "/workspaces/KannadaHandwritingRecognition/dataset"
OUTPUT_FILE = os.path.join(BASE_DIR, "dataset.zip")
EXTRACT_DIR = os.path.join(BASE_DIR, "dataset")
CSV_FILE_PNG = os.path.join(BASE_DIR, "dataset_png.csv")

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

def rename_sample_dirs(base_path):
    """Renames 'SampleXXX' directories to just numbers."""
    for folder in os.listdir(base_path):
        match = re.match(r"Sample0*(\d+)", folder)
        if match:
            new_name = match.group(1)  # Extract number without leading zeros
            old_path = os.path.join(base_path, folder)
            new_path = os.path.join(base_path, new_name)
            if os.path.isdir(old_path):
                os.rename(old_path, new_path)
                print(f"Renamed directory: {old_path} --> {new_path}")

def rename_images(base_path):
    """Renames images to match their parent directory and removes leading zeros."""
    for root, _, files in os.walk(base_path):
        label = os.path.basename(root)
        if not label.isdigit():
            continue  # Skip non-numeric directories
        
        for file in files:
            if file.lower().endswith(".png"):
                match = re.search(r"img\d+-(\d+).png", file)
                if match:
                    new_name = f"{int(match.group(1))}.png"  # Remove leading zeros
                    old_path = os.path.join(root, file)
                    new_path = os.path.join(root, new_name)
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} --> {new_path}")

def generate_png_csv(base_path, csv_file):
    """Generates a CSV file for all PNG images after renaming."""
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)  # Ensure directory exists
    
    data = []
    index = 0
    
    for root, _, files in os.walk(base_path):
        label = os.path.basename(root)
        if not label.isdigit():
            continue
        
        for file in sorted(files):  # Ensure files are sorted
            if file.lower().endswith(".png"):
                img_path = os.path.join(root, file)
                data.append([index, file, int(label)-1])
                index += 1
    
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["index", "img_name", "label"])
        writer.writerows(data)
    print(f"[INFO] PNG CSV file created at {csv_file}")

def main():
    download_file()
    if not verify_and_extract():
        print("[ERROR] Download failed. Please check the file ID or try again.")
        return
    
    dataset_path = os.path.join(EXTRACT_DIR, "Kannada/Hnd/Img")
    rename_sample_dirs(dataset_path)
    rename_images(dataset_path)
    generate_png_csv(dataset_path, CSV_FILE_PNG)
    
if __name__ == "__main__":
    main()
