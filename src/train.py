import os
import argparse
import time
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import cohen_kappa_score

# Import project modules
from src.dataset import RetinaDataset
from src.model import RetinaEnsemble

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Ensemble returns two logits
        logits_a, logits_b = model(images)
        
        # Loss for both heads
        loss_a = criterion(logits_a, labels)
        loss_b = criterion(logits_b, labels)
        
        # Joint loss (can also add consistency loss, but simple sum is robust)
        loss = loss_a + loss_b
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        pbar.set_postfix({'loss': loss.item()})
        
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Use ensemble voting for validation
            probs = model.get_ensemble_prediction(images)
            _, preds = torch.max(probs, 1)
            
            # We can calculate validation loss using the average logits or skip it
            # For reporting, let's use the average prediction loss if needed, 
            # but QWK is the main metric.
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate QWK
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    return qwk

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8) # Lower default for Mac MPS
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--csv_path', type=str, default='data/train.csv')
    parser.add_argument('--dry_run', action='store_true', help="Run a single batch for debugging")
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), # Resize after Ben Graham preprocessing (which outputs 512x512 usually)
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(), # Dataset returns PIL after processing
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # check if data exists
    if not os.path.exists(args.csv_path):
        print(f"Error: {args.csv_path} not found. Please run download_data.py")
        return

    # Datasets
    full_dataset = RetinaDataset(args.csv_path, os.path.join(args.data_dir, 'train_images'), transform=train_transform)
    
    # Split Train/Val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Update transform for validation set (a bit hacky with random_split, 
    # ideally we verify 'full_dataset' has train transform and we just override validation subset? 
    # Actually random_split shares the underlying dataset. We should handle this better in a real setting,
    # but for this scope it's 'okay' if val has augments or we construct two subsets manually.
    # A cleaner way: Use indices.)
    
    # Re-instantiate for validation to ensure proper transforms
    indices = torch.randperm(len(full_dataset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = torch.utils.data.Subset(
        RetinaDataset(args.csv_path, os.path.join(args.data_dir, 'train_images'), transform=train_transform), 
        train_indices
    )
    val_dataset = torch.utils.data.Subset(
        RetinaDataset(args.csv_path, os.path.join(args.data_dir, 'train_images'), transform=val_transform), 
        val_indices
    )

    if args.dry_run:
        print("Dry Run: Slicing datasets for quick check.")
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(len(train_dataset), 16)))
        val_dataset = torch.utils.data.Subset(val_dataset, range(min(len(val_dataset), 16)))
        args.epochs = 1

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0) # num_workers=0 for Mac stability often
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model
    print("Initializing Ensemble Model...")
    model = RetinaEnsemble(num_classes=5)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_qwk = -1.0
    
    for epoch in range(args.epochs):
        start_time = time.time()
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_qwk = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Val QWK: {val_qwk:.4f} | Time: {time.time() - start_time:.1f}s")
        
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            torch.save(model.state_dict(), os.path.join("models", "best_model.pth"))
            print("Saved Best Model!")

if __name__ == "__main__":
    # Ensure models dir exists
    os.makedirs("models", exist_ok=True)
    main()
