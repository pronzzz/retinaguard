import os
import cv2
import numpy as np
import pandas as pd
import subprocess
import shutil

def setup_dummy_data():
    os.makedirs("data_test/train_images", exist_ok=True)
    
    # Create 5 dummy images
    ids = []
    labels = []
    for i in range(5):
        img_id = f"test_img_{i}"
        # Create random image
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        cv2.imwrite(f"data_test/train_images/{img_id}.png", img)
        ids.append(img_id)
        labels.append(np.random.randint(0, 5))
        
    df = pd.DataFrame({'id_code': ids, 'diagnosis': labels})
    df.to_csv("data_test/train.csv", index=False)
    print("Dummy data created.")

def run_training_test():
    cmd = [
        "python3", "-m", "src.train",
        "--epochs", "1",
        "--batch_size", "2",
        "--data_dir", "data_test",
        "--csv_path", "data_test/train.csv",
        "--dry_run"
    ]
    print("Running training dry run...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    if result.returncode == 0:
        print("Training script ran successfully!")
    else:
        print("Training script failed.")

def cleanup():
    if os.path.exists("data_test"):
        shutil.rmtree("data_test")
        print("Cleanup done.")

if __name__ == "__main__":
    try:
        setup_dummy_data()
        run_training_test()
    finally:
        cleanup()
