import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import os
import pickle
import argparse
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from model.utils.pytorch.train_loop import train_loop
from model.utils.pytorch.test_loop import test_loop
from model.utils.pytorch.cross_validate import cross_validate_model
from model.data.loaders.loader_pytorch import MethylationAlzheimerDataset
from model.models.pytorch.ConvNet import ConvNet
from model.models.pytorch.HybridCNN import HybridCNN
from model.models.pytorch.RegularizedMLP import RegularizedMLP
from model.models.pytorch.SimpleMLP import SimpleMLP

def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def train_with_validation(model, train_loader, val_loader, criterion, optimizer, 
                         scheduler, device, epochs=100, early_stopping_patience=15):
    """Enhanced training with validation and early stopping"""
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    model.to(device)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'Val Acc: {val_accuracy:.2f}%')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "./model/models/pytorch/best_model.pkl")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    # Load best model
    model.load_state_dict(torch.load("./model/models/pytorch/best_model.pkl"))
    
    return model, train_losses, val_losses, val_accuracies

def evaluate_model(model, test_loader, device, class_names=['Control', 'MCI', "Alzheimer's"]):
    """Comprehensive model evaluation"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_predictions))
    
    return all_predictions, all_labels

def save_model_with_metadata(model, model_name, input_dim, train_losses, val_losses, val_accuracies):
    """Save model with comprehensive metadata"""
    os.makedirs("./model/models/pytorch/", exist_ok=True)
    
    # Save model state dict
    model_path = f"./model/models/pytorch/{model_name}_model.pkl"
    torch.save(model.state_dict(), model_path)
    
    # Save complete model (architecture + weights)
    complete_model_path = f"./model/models/pytorch/{model_name}_complete.pkl"
    torch.save(model, complete_model_path)
    
    # Save training metadata
    metadata = {
        'model_name': model_name,
        'input_dim': input_dim,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'final_val_accuracy': val_accuracies[-1] if val_accuracies else 0,
        'model_architecture': str(model)
    }
    
    metadata_path = f"./model/models/pytorch/{model_name}_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Model saved to {model_path}")
    print(f"Complete model saved to {complete_model_path}")
    print(f"Metadata saved to {metadata_path}")

def main():
    parser = argparse.ArgumentParser(description='Train enhanced PyTorch models for Alzheimer\'s prediction')
    parser.add_argument('--model', choices=['convnet', 'hybrid', 'mlp'], default='hybrid',
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.4,
                       help='Dropout rate')
    args = parser.parse_args()
    
    device = setup_device()
    
    # Load data
    h5_path = "./model/data/train/methylation.csv"
    mapping_csv_path = "./model/data/train/idmap.csv"
    
    print("Loading dataset...")
    dataset = MethylationAlzheimerDataset(h5_path, mapping_csv_path)
    input_dim = dataset.data.shape[1]
    print(f"Dataset loaded: {len(dataset)} samples, {input_dim} features")
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Initialize model
    if args.model == 'convnet':
        model = ConvNet(input_dim, dropout_rate=args.dropout)
    elif args.model == 'hybrid':
        model = HybridCNN(input_dim, dropout_rate=args.dropout)
    else:  # mlp
        model = SimpleMLP(input_dim, dropout_rate=args.dropout)
    
    print(f"Initialized {args.model} model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training setup
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 2.0]).to(device))  # Handle class imbalance
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    print(f"Starting training for {args.epochs} epochs...")
    
    # Train model
    model, train_losses, val_losses, val_accuracies = train_with_validation(
        model, train_loader, val_loader, criterion, optimizer, scheduler, 
        device, epochs=args.epochs, early_stopping_patience=20
    )
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    predictions, true_labels = evaluate_model(model, test_loader, device)
    
    # Save model with metadata
    save_model_with_metadata(model, args.model, input_dim, train_losses, val_losses, val_accuracies)
    
    print(f"\nTraining completed successfully!")
    print(f"Final validation accuracy: {val_accuracies[-1]:.2f}%")

if __name__ == "__main__":
    main()