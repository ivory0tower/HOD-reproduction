#!/usr/bin/env python3
"""
Improved Faster R-CNN Training Script
Based on official HOD implementation with longer training and better configuration
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models.detection as detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

class HODDataset(Dataset):
    """HOD Dataset class with improved data loading"""
    
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Class mapping (same as official implementation)
        self.class_names = ['alcohol', 'insulting_gesture', 'blood', 'cigarette', 'gun', 'knife']
        self.class_to_idx = {name: idx + 1 for idx, name in enumerate(self.class_names)}  # +1 for background
        self.idx_to_class = {idx + 1: name for idx, name in enumerate(self.class_names)}
        
        # Load annotations
        self.annotations = self._load_annotations()
        print(f"Loaded {len(self.annotations)} {split} samples")
        
        # Print class distribution
        self._print_class_distribution()
    
    def _load_annotations(self):
        """Load annotations from XML files"""
        annotations = []
        
        # Get image and annotation paths
        img_dir = os.path.join(self.root_dir, 'dataset', 'all', 'jpg')
        ann_dir = os.path.join(self.root_dir, 'dataset', 'all', 'xml')
        
        if not os.path.exists(img_dir) or not os.path.exists(ann_dir):
            print(f"Warning: Dataset directories not found at {self.root_dir}")
            return []
        
        # Get all XML files
        xml_files = [f for f in os.listdir(ann_dir) if f.endswith('.xml')]
        
        # Split data (simple split for now)
        total_files = len(xml_files)
        if self.split == 'train':
            xml_files = xml_files[:int(0.8 * total_files)]
        elif self.split == 'val':
            xml_files = xml_files[int(0.8 * total_files):int(0.9 * total_files)]
        else:  # test
            xml_files = xml_files[int(0.9 * total_files):]
        
        for xml_file in xml_files:
            xml_path = os.path.join(ann_dir, xml_file)
            img_name = xml_file.replace('.xml', '.jpg')
            img_path = os.path.join(img_dir, img_name)
            
            if os.path.exists(img_path):
                annotation = self._parse_xml(xml_path, img_path)
                if annotation:
                    annotations.append(annotation)
        
        return annotations
    
    def _parse_xml(self, xml_path, img_path):
        """Parse XML annotation file"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get image info
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            
            boxes = []
            labels = []
            
            # Parse objects
            for obj in root.findall('object'):
                class_name = obj.find('name').text.strip().lower()
                
                # Map class name to our classes
                if class_name in self.class_to_idx:
                    bbox = obj.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)
                    
                    # Ensure valid bounding box
                    if xmax > xmin and ymax > ymin:
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(self.class_to_idx[class_name])
            
            if len(boxes) > 0:
                return {
                    'image_path': img_path,
                    'boxes': np.array(boxes, dtype=np.float32),
                    'labels': np.array(labels, dtype=np.int64),
                    'image_size': (width, height)
                }
        
        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
        
        return None
    
    def _print_class_distribution(self):
        """Print class distribution in dataset"""
        class_counts = {name: 0 for name in self.class_names}
        
        for ann in self.annotations:
            for label in ann['labels']:
                if label in self.idx_to_class:
                    class_name = self.idx_to_class[label]
                    class_counts[class_name] += 1
        
        print(f"\nClass distribution in {self.split} set:")
        total_objects = sum(class_counts.values())
        for class_name, count in class_counts.items():
            percentage = (count / total_objects * 100) if total_objects > 0 else 0
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        print(f"Total objects: {total_objects}\n")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # Load image
        image = Image.open(ann['image_path']).convert('RGB')
        
        # Prepare target
        target = {
            'boxes': torch.tensor(ann['boxes'], dtype=torch.float32),
            'labels': torch.tensor(ann['labels'], dtype=torch.int64),
            'image_id': torch.tensor([idx], dtype=torch.int64)
        }
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            image = transform(image)
        
        return image, target

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    images, targets = zip(*batch)
    return list(images), list(targets)

def create_model(num_classes):
    """Create Faster R-CNN model with improved configuration"""
    # Load pretrained model
    model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Replace classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for images, targets in progress_bar:
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        
        # Calculate total loss
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        losses.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += losses.item()
        num_batches += 1
        
        # Update progress bar
        avg_loss = total_loss / num_batches
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return total_loss / num_batches

def validate_model(model, dataloader, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Validation'):
            # Move to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            total_loss += losses.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0

def main():
    """Main training function"""
    print("=== Improved Faster R-CNN Training (Based on Official Implementation) ===")
    
    # Configuration (based on official implementation)
    config = {
        'root_dir': './HOD-Benchmark-Dataset',
        'batch_size': 4,  # Smaller batch size for stability
        'num_epochs': 50,  # Much longer training (official uses 150)
        'learning_rate': 0.005,  # Lower learning rate
        'weight_decay': 0.0005,
        'momentum': 0.9,
        'step_size': 20,  # Learning rate decay
        'gamma': 0.1,
        'num_classes': 7,  # 6 classes + background
        'save_interval': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Configuration: {config}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = HODDataset(config['root_dir'], split='train')
    val_dataset = HODDataset(config['root_dir'], split='val')
    
    if len(train_dataset) == 0:
        print("Error: No training data found!")
        return
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Create model
    print("Creating model...")
    model = create_model(config['num_classes'])
    model.to(config['device'])
    
    # Create optimizer and scheduler
    optimizer = optim.SGD(
        model.parameters(), 
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config['step_size'], 
        gamma=config['gamma']
    )
    
    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(1, config['num_epochs'] + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, config['device'], epoch)
        train_losses.append(train_loss)
        
        # Validate
        if len(val_dataset) > 0:
            val_loss = validate_model(model, val_loader, config['device'])
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch}/{config['num_epochs']}: "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'config': config
                }, 'faster_rcnn_best.pth')
                print(f"Saved best model (val_loss: {val_loss:.4f})")
        else:
            print(f"Epoch {epoch}/{config['num_epochs']}: Train Loss: {train_loss:.4f}")
        
        # Save checkpoint
        if epoch % config['save_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_losses[-1] if val_losses else 0,
                'config': config
            }, f'faster_rcnn_epoch_{epoch}.pth')
            print(f"Saved checkpoint at epoch {epoch}")
        
        # Update learning rate
        scheduler.step()
    
    # Save final model
    torch.save({
        'epoch': config['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1] if val_losses else 0,
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses
    }, 'faster_rcnn_improved_final.pth')
    
    print("\n=== Training completed! ===")
    print(f"Final model saved as: faster_rcnn_improved_final.pth")
    print(f"Best model saved as: faster_rcnn_best.pth (val_loss: {best_val_loss:.4f})")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Training curves saved as: training_progress.png")

if __name__ == "__main__":
    main()