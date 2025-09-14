#!/usr/bin/env python3
"""
Improved Faster R-CNN Inference Script
Based on official HOD implementation logic but adapted for current environment
"""

import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models.detection as detection
import matplotlib.pyplot as plt
from PIL import Image
import json

class HODFasterRCNN:
    """Improved Faster R-CNN implementation based on official HOD code"""
    
    def __init__(self, checkpoint_path='./faster_rcnn_final.pth', device=None):
        self.class_names = ['alcohol', 'insulting_gesture', 'blood', 'cigarette', 'gun', 'knife']
        self.num_classes = len(self.class_names) + 1  # +1 for background
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        
    def _load_model(self, checkpoint_path):
        """Load Faster R-CNN model with proper configuration"""
        print("Loading Faster R-CNN model...")
        
        # Create model with correct number of classes
        model = detection.fasterrcnn_resnet50_fpn(
            pretrained=False,
            num_classes=self.num_classes,
            pretrained_backbone=True
        )
        
        # Load checkpoint if available
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Load state dict with error handling
                try:
                    model.load_state_dict(state_dict, strict=True)
                    print("Checkpoint loaded successfully (strict mode)")
                except RuntimeError as e:
                    print(f"Strict loading failed: {e}")
                    print("Trying flexible loading...")
                    model.load_state_dict(state_dict, strict=False)
                    print("Checkpoint loaded with flexible mode")
                    
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Using pretrained backbone only")
        else:
            print(f"Checkpoint {checkpoint_path} not found. Using pretrained backbone.")
        
        model.to(self.device)
        model.eval()
        return model
    
    def preprocess_image(self, image_path):
        """Preprocess image for inference"""
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Convert to tensor (following official implementation)
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        image_tensor = transform(image).to(self.device)
        return image_tensor.unsqueeze(0), np.array(image)
    
    def inference(self, image_path, confidence_threshold=0.3):
        """Perform inference on image"""
        print(f"Processing image: {image_path}")
        
        # Preprocess
        image_tensor, original_image = self.preprocess_image(image_path)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Process results
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        
        # Filter by confidence (following official implementation)
        confident_indices = scores >= confidence_threshold
        
        filtered_boxes = boxes[confident_indices]
        filtered_labels = labels[confident_indices]
        filtered_scores = scores[confident_indices]
        
        return {
            'boxes': filtered_boxes,
            'labels': filtered_labels,
            'scores': filtered_scores,
            'original_image': original_image
        }
    
    def visualize_results(self, results, output_path, confidence_threshold=0.3):
        """Visualize detection results"""
        image = results['original_image'].copy()
        boxes = results['boxes']
        labels = results['labels']
        scores = results['scores']
        
        detection_count = 0
        class_detections = {name: 0 for name in self.class_names}
        
        # Draw bounding boxes
        for box, label, score in zip(boxes, labels, scores):
            if score >= confidence_threshold:
                detection_count += 1
                x1, y1, x2, y2 = box.astype(int)
                
                # Get class name (adjust for background class)
                class_idx = label - 1  # Subtract 1 for background class
                if 0 <= class_idx < len(self.class_names):
                    class_name = self.class_names[class_idx]
                    class_detections[class_name] += 1
                else:
                    class_name = f'class_{label}'
                
                # Choose color based on class
                colors = {
                    'alcohol': (255, 0, 0),      # Red
                    'insulting_gesture': (0, 255, 0),  # Green
                    'blood': (0, 0, 255),        # Blue
                    'cigarette': (255, 255, 0),  # Yellow
                    'gun': (255, 0, 255),        # Magenta
                    'knife': (0, 255, 255)      # Cyan
                }
                color = colors.get(class_name, (128, 128, 128))
                
                # Draw rectangle
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with background
                label_text = f"{class_name}: {score:.2f}"
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Draw label background
                cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(image, label_text, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Save result
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Print detection summary
        print(f"\nDetection Summary (confidence >= {confidence_threshold}):")
        print(f"Total detections: {detection_count}")
        for class_name, count in class_detections.items():
            if count > 0:
                print(f"  {class_name}: {count}")
        
        return detection_count, class_detections
    
    def test_multiple_thresholds(self, image_path, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5]):
        """Test with multiple confidence thresholds (following official implementation)"""
        print(f"\n=== Testing multiple confidence thresholds ===")
        
        # Perform inference once
        results = self.inference(image_path, confidence_threshold=0.0)  # Get all detections
        
        best_threshold = 0.0
        best_detection_count = 0
        
        for threshold in thresholds:
            print(f"\n--- Confidence threshold: {threshold} ---")
            
            # Filter results by current threshold
            confident_indices = results['scores'] >= threshold
            filtered_results = {
                'boxes': results['boxes'][confident_indices],
                'labels': results['labels'][confident_indices],
                'scores': results['scores'][confident_indices],
                'original_image': results['original_image']
            }
            
            # Visualize
            output_path = f"faster_rcnn_improved_conf_{threshold:.1f}.jpg"
            detection_count, class_detections = self.visualize_results(
                filtered_results, output_path, threshold
            )
            
            # Track best threshold
            if detection_count > best_detection_count:
                best_detection_count = detection_count
                best_threshold = threshold
            
            print(f"Saved result to: {output_path}")
        
        print(f"\nBest threshold: {best_threshold} (detected {best_detection_count} objects)")
        return best_threshold, best_detection_count

def test_blood_detection():
    """Specifically test blood detection capability"""
    print("\n=== Testing Blood Detection Capability ===")
    
    # Find blood images from metadata
    blood_images = []
    metadata_path = "./HOD-Benchmark-Dataset/dataset/metadata.csv"
    
    if os.path.exists(metadata_path):
        import pandas as pd
        try:
            df = pd.read_csv(metadata_path)
            blood_df = df[df['Category'] == 'blood']
            blood_images = blood_df['Image Name'].tolist()[:3]  # Test first 3 blood images
            print(f"Found {len(blood_images)} blood images to test")
        except Exception as e:
            print(f"Error reading metadata: {e}")
    
    if not blood_images:
        # Fallback to known blood images
        blood_images = ['img_hod_000007.jpg', 'img_hod_000040.jpg']
        print("Using fallback blood images")
    
    # Test each blood image
    detector = HODFasterRCNN()
    
    for img_name in blood_images:
        img_path = f"./HOD-Benchmark-Dataset/dataset/all/jpg/{img_name}"
        if not os.path.exists(img_path):
            img_path = img_name  # Try current directory
            
        if os.path.exists(img_path):
            print(f"\nTesting blood detection on: {img_name}")
            results = detector.inference(img_path, confidence_threshold=0.2)
            
            # Check if blood was detected
            blood_detected = False
            for label in results['labels']:
                class_idx = label - 1
                if 0 <= class_idx < len(detector.class_names):
                    if detector.class_names[class_idx] == 'blood':
                        blood_detected = True
                        break
            
            print(f"Blood detected: {blood_detected}")
            
            # Visualize
            output_path = f"blood_test_{img_name}"
            detector.visualize_results(results, output_path, 0.2)
        else:
            print(f"Image not found: {img_path}")

def main():
    """Main function"""
    print("=== Improved Faster R-CNN Inference (Based on Official Logic) ===")
    
    # Initialize detector
    detector = HODFasterRCNN()
    
    # Test image
    test_image = "img_hod_000007.jpg"  # Known blood image
    
    if not os.path.exists(test_image):
        # Try in dataset directory
        test_image = "./HOD-Benchmark-Dataset/dataset/all/jpg/img_hod_000007.jpg"
        
    if not os.path.exists(test_image):
        print("Warning: Test image not found. Using alternative...")
        test_image = "img_hod_000040.jpg"
        
    if not os.path.exists(test_image):
        print("Error: No test images found!")
        return
    
    print(f"Testing with image: {test_image}")
    
    # Test with multiple thresholds (following official implementation)
    best_threshold, best_count = detector.test_multiple_thresholds(test_image)
    
    # Test blood detection specifically
    test_blood_detection()
    
    print("\n=== Inference completed! ===")
    print("Check the generated images with different confidence thresholds.")
    print("This implementation follows the official HOD code logic for better accuracy.")

if __name__ == "__main__":
    main()