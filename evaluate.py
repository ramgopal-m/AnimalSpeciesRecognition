import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from model import AnimalSpeciesClassifier
from data_loader import get_data_loaders
import os

def evaluate_model(model, test_loader, classes, device='cuda'):
    print("Starting model evaluation...")
    model.eval()
    all_preds = []
    all_labels = []
    
    print(f"Number of test batches: {len(test_loader)}")
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            print(f"Processing batch {batch_idx + 1}/{len(test_loader)}")
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print("Evaluation complete. Calculating metrics...")
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    
    print("\nClassification Report:")
    print(report)
    
    # Calculate per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    print("\nPer-class accuracy:")
    for i, acc in enumerate(per_class_acc):
        print(f"{classes[i]}: {acc:.4f}")
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix plot...")
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png')
    plt.close()
    
    print("Confusion matrix plot saved to models/confusion_matrix.png")
    
    # Plot per-class accuracy
    plt.figure(figsize=(15, 6))
    plt.bar(range(len(classes)), per_class_acc)
    plt.title('Per-class Accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('models/per_class_accuracy.png')
    plt.close()
    
    print("Per-class accuracy plot saved to models/per_class_accuracy.png")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = AnimalSpeciesClassifier.load_model('models/best_model.pth')
    model = model.to(device)
    
    # Get data loaders
    print("Loading test dataset...")
    train_loader, val_loader, test_loader, classes = get_data_loaders(batch_size=32)
    print(f"Found {len(classes)} classes: {classes}")
    
    # Evaluate
    evaluate_model(model, test_loader, classes, device)

if __name__ == '__main__':
    main() 