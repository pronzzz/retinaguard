import os
import subprocess
import glob
import zipfile

def download_dataset():
    """
    Downloads the APTOS 2019 Blindness Detection dataset using Kaggle API.
    Requires kaggle.json to be set up in ~/.kaggle/
    """
    DATA_DIR = "data"
    DATASET_NAME = "aptos2019-blindness-detection"
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created {DATA_DIR} directory.")
    
    # Check if data already exists
    if os.path.exists(os.path.join(DATA_DIR, "train.csv")):
        print("Dataset appears to be already present.")
        return

    print(f"Downloading {DATASET_NAME}...")
    try:
        subprocess.run(["kaggle", "competitions", "download", "-c", DATASET_NAME, "-p", DATA_DIR], check=True)
    except FileNotFoundError:
        print("Error: 'kaggle' command not found. Please install it via 'pip install kaggle' and ensure it's in your PATH.")
        return
    except subprocess.CalledProcessError:
        print("Error during download. Please ensure you have accepted the competition rules and have a valid ~/.kaggle/kaggle.json")
        return

    # Unzip if downloaded as zip
    zip_files = glob.glob(os.path.join(DATA_DIR, "*.zip"))
    for zip_path in zip_files:
        print(f"Unzipping {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        os.remove(zip_path) # Clean up zip
    
    print("Download and extraction complete.")

if __name__ == "__main__":
    download_dataset()
