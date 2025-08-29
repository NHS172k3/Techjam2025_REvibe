import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, test_dataloader, device=None):
    """
    Evaluate model on test data
    
    Args:
        model: Trained model
        test_dataloader: DataLoader for test data
        device: Device to use (if None, will detect)
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    # Containers for predictions and true labels
    all_preds = []
    all_labels = []
    
    # Evaluation loop
    with torch.no_grad():
        for batch in test_dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    # Store metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_preds,
        'true_labels': all_labels
    }
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return metrics

def plot_confusion_matrix(metrics, class_names=None):
    """
    Plot confusion matrix
    
    Args:
        metrics: Metrics dictionary from evaluate_model
        class_names: Names of classes (if None, will use numbered classes)
    """
    # Get predictions and true labels
    y_pred = metrics['predictions']
    y_true = metrics['true_labels']
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Set class names
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()