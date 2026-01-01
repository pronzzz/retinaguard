import os
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np

def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):
            return img
        else:
            img1=img[:,:,0][np.ix_(mask.any(1), mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1), mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def ben_graham_process(image):
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (512, 512))
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image

def visualize(data_dir="data", n_images=5):
    # Find some png images
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found. Please run download_data.py first.")
        return

    # Assuming images are in a subdirectory like 'train_images' or directly in data
    # The structure depends on the unzip. Let's look recursively or in common folder name
    image_paths = glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True)
    
    if not image_paths:
        print("No .png images found in data folder.")
        return

    # Select random 5
    selected_paths = np.random.choice(image_paths, min(len(image_paths), n_images), replace=False)
    
    fig, axes = plt.subplots(n_images, 2, figsize=(10, 5 * n_images))
    if n_images == 1: axes = [axes] # Handle single case

    for idx, path in enumerate(selected_paths):
        # Original
        original = cv2.imread(path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Processed
        processed = ben_graham_process(original.copy())
        
        axes[idx][0].imshow(original)
        axes[idx][0].set_title("Original")
        axes[idx][0].axis('off')
        
        axes[idx][1].imshow(processed)
        axes[idx][1].set_title("Ben Graham Processed")
        axes[idx][1].axis('off')
        
    plt.tight_layout()
    output_path = "preprocessing_demo.png"
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    visualize()
