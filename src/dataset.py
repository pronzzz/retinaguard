import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class RetinaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, mode='train'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (string): 'train' or 'test'. If test, likely no labels.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0] + '.png')
        
        # Read image using OpenCV
        image = cv2.imread(img_name)
        if image is None:
             # Handle missing files if any, or error out
             raise FileNotFoundError(f"Image not found: {img_name}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply Ben Graham's Preprocessing
        image = self.ben_graham_preprocessing(image)
        
        # Convert to PIL for standard torchvision transforms if needed, or keep numpy for albumentations
        # Assuming standard transforms for now which usually take PIL or Tensor
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        if self.mode == 'train':
            label = int(self.data_frame.iloc[idx, 1])
            return image, label
        else:
            return image

    def crop_image_from_gray(self, img, tol=7):
        """
        Crops the black borders of the fundus image.
        """
        if img.ndim == 2:
            mask = img > tol
            return img[np.ix_(mask.any(1), mask.any(0))]
        elif img.ndim == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = gray_img > tol
            
            check_shape = img[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0]
            if (check_shape == 0): # image is too dark so that we crop out everything,
                return img # return original image
            else:
                img1=img[:,:,0][np.ix_(mask.any(1), mask.any(0))]
                img2=img[:,:,1][np.ix_(mask.any(1), mask.any(0))]
                img3=img[:,:,2][np.ix_(mask.any(1), mask.any(0))]
                img = np.stack([img1,img2,img3],axis=-1)
            return img

    def circle_crop(self, img, sigmaX=10):
        """
        Circular crop to remove irregular borders.
        """   
        height, width, depth = img.shape
        
        largest_side = np.max((height, width))
        img = cv2.resize(img, (largest_side, largest_side))

        height, width, depth = img.shape
        
        x = int(width/2)
        y = int(height/2)
        r = np.amin((x,y))
        
        circle_img = np.zeros((height, width), np.uint8)
        cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
        
        img = cv2.bitwise_and(img, img, mask=circle_img)
        return self.crop_image_from_gray(img)

    def ben_graham_preprocessing(self, image):
        """
        Full pipeline: Crop -> Resize -> Weighted Gaussian Blur subtraction (CLAHE alternative or specific Ben Graham style)
        Common Ben Graham: `image = image * 4 - 4 * GaussianBlur(image, 10)` + 128
        But Prompt asks for: "CLAHE + Gaussian Blur" specially. 
        So I will implement the specific pipeline requested:
        1. Crop black borders.
        2. Resize (we do this in transforms usually, but good to do here for uniform processing).
        3. CLAHE (Local contrast enhancement).
        """
        # 1. Crop
        image = self.crop_image_from_gray(image)
        
        # 2. Resize to prevent massive images
        image = cv2.resize(image, (512, 512))
        
        # 3. CLAHE on Green channel or LAB L-channel? 
        # Best practice for fundus: LAB -> CLAHE on L -> Merge
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # 4. Gaussian Blur / Smoothing if requested to reduce noise. 
        # "noise and poor lighting ... specific preprocessing (CLAHE + Gaussian Blur)"
        # Sometimes Gaussian blur is used *before* to estimate background, but here it says "CLAHE + Gaussian Blur".
        # I'll apply a mild Gaussian blur to denoise after CLAHE which enhances noise.
        image = cv2.GaussianBlur(image, (5, 5), 0)
        
        return image

if __name__ == '__main__':
    # Simple test
    print("Dataset class defined.")
