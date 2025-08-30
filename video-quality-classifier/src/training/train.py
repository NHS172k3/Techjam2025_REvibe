import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from src.data.dataset import CommentDataset
from src.models.classifier import QualityClassifier
from src.utils.metrics import calculate_metrics
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_model(config):
    # Load dataset
    dataset = CommentDataset(config['data']['path'])
    comments, labels = dataset.load_data()

    # Preprocess data
    comments = dataset.preprocess_comments(comments)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(comments, labels, test_size=config['training']['val_size'], random_state=42)


    # Initialize classifier
    model = QualityClassifier(config['model'])

    # Train model
    logging.info("Training the model...")
    model.fit(X_train, y_train)

    # Validate model
    logging.info("Validating the model...")
    predictions = model.predict(X_val)
    metrics = calculate_metrics(y_val, predictions)

    logging.info(f"Validation Metrics: {metrics}")

    # Save the model
    model.save(config['model']['save_path'])

def train_model_transformers(model, train_dataloader, val_dataloader, config):
    """
    Train the quality classification model
    
    Args:
        model: Model to train
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        config: Training configuration
        
    Returns:
        trained_model: Trained model
        history: Training history
    """
    # Create model directory
    os.makedirs(config['model_dir'], exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Training parameters
    num_epochs = config.get('num_epochs', 5)
    learning_rate = config.get('learning_rate', 2e-5)
    warmup_steps = config.get('warmup_steps', 0)
    weight_decay = config.get('weight_decay', 0.01)
    
    # Set up optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Total steps
    total_steps = len(train_dataloader) * num_epochs
    
    # Create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    # Training loop
    best_val_accuracy = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_dataloader, desc="Training"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        # Average training loss
        avg_train_loss = train_loss / len(train_dataloader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store predictions and labels for metrics
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Average validation loss and accuracy
        avg_val_loss = val_loss / len(val_dataloader)
        val_accuracy = correct / total
        
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        report = classification_report(
            val_labels, 
            val_preds, 
            target_names=['Negative Quality', 'Not About Quality', 'Positive Quality'],
            digits=3
        )
        print(report)
        
        # Plot confusion matrix
        cm = confusion_matrix(val_labels, val_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Not Quality', 'Positive'],
                   yticklabels=['Negative', 'Not Quality', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - Epoch {epoch+1}')
        plt.savefig(os.path.join(config['model_dir'], f'confusion_matrix_epoch_{epoch+1}.png'))
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model_path = os.path.join(config['model_dir'], 'best_quality_classifier.pth')
            model.save(model_path)
            print(f"New best model saved with accuracy: {val_accuracy:.4f}")
    
    # Load best model
    best_model_path = os.path.join(config['model_dir'], 'best_quality_classifier.pth')
    model = QualityClassifier.load(best_model_path)
    
    return model, history

def train_integrated_model(model, train_loader, val_loader, num_epochs=3, learning_rate=2e-5, warmup_steps=0, model_dir='models'):
    """
    Train the integrated quality+sentiment model
    
    Args:
        model: IntegratedQualityModel to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        model_dir: Directory to save model
        
    Returns:
        model: Trained model
    """
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Set up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Set up loss function for regression
    criterion = nn.MSELoss()
    
    # Set up scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_mse = float('inf')
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['label'].to(device)  # This should be unified_score
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            train_loss += loss.item()
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        # Average training loss
        avg_train_loss = train_loss / len(train_loader)
        print(f"Training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['label'].to(device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                # Store predictions and targets
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        # Average validation loss and metrics
        avg_val_loss = val_loss / len(val_loader)
        val_mse = mean_squared_error(val_targets, val_preds)
        
        print(f"Validation loss: {avg_val_loss:.4f}")
        print(f"Validation MSE: {val_mse:.4f}")
        
        # Plot prediction vs actual
        plt.figure(figsize=(8, 6))
        plt.scatter(val_targets, val_preds, alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('Actual Unified Score')
        plt.ylabel('Predicted Unified Score')
        plt.title(f'Unified Quality-Sentiment Score - Epoch {epoch+1}')
        plt.savefig(os.path.join(model_dir, f'unified_score_epoch_{epoch+1}.png'))
        
        # Save best model
        if val_mse < best_mse:
            best_mse = val_mse
            model_path = os.path.join(model_dir, 'best_integrated_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved with MSE: {val_mse:.4f}")
    
    # Load best model
    best_model_path = os.path.join(model_dir, 'best_integrated_model.pth')
    model.load_state_dict(torch.load(best_model_path))
    
    return model

def train_multi_task_model(model, train_loader, val_loader, num_epochs=3, learning_rate=2e-5, warmup_steps=0, model_dir='models'):
    """
    Train the multi-task model that predicts both quality class and sentiment score
    
    Args:
        model: QualitySentimentModel to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        model_dir: Directory to save model
        
    Returns:
        model: Trained model
    """
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Set up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Set up loss functions
    quality_criterion = nn.CrossEntropyLoss()
    sentiment_criterion = nn.MSELoss()
    
    # Set up scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_combined_metric = float('inf')
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        train_quality_loss = 0.0
        train_sentiment_loss = 0.0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            quality_labels = batch['quality_label'].to(device)
            sentiment_scores = batch['sentiment_score'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            quality_logits = outputs['quality_logits']
            sentiment_pred = outputs['sentiment_score']
            
            # Calculate losses
            quality_loss = quality_criterion(quality_logits, quality_labels)
            sentiment_loss = sentiment_criterion(sentiment_pred, sentiment_scores)
            
            # Combined loss (weight can be adjusted)
            combined_loss = 0.6 * quality_loss + 0.4 * sentiment_loss
            
            train_quality_loss += quality_loss.item()
            train_sentiment_loss += sentiment_loss.item()
            
            # Backward pass and optimize
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        # Average training loss
        avg_train_quality_loss = train_quality_loss / len(train_loader)
        avg_train_sentiment_loss = train_sentiment_loss / len(train_loader)
        print(f"Training quality loss: {avg_train_quality_loss:.4f}")
        print(f"Training sentiment loss: {avg_train_sentiment_loss:.4f}")
        
        # Validation
        model.eval()
        val_quality_loss = 0.0
        val_sentiment_loss = 0.0
        val_quality_preds = []
        val_quality_labels = []
        val_sentiment_preds = []
        val_sentiment_scores = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                quality_labels = batch['quality_label'].to(device)
                sentiment_scores = batch['sentiment_score'].to(device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask)
                quality_logits = outputs['quality_logits']
                sentiment_pred = outputs['sentiment_score']
                
                # Calculate losses
                quality_loss = quality_criterion(quality_logits, quality_labels)
                sentiment_loss = sentiment_criterion(sentiment_pred, sentiment_scores)
                
                val_quality_loss += quality_loss.item()
                val_sentiment_loss += sentiment_loss.item()
                
                # Get quality predictions
                _, quality_preds = torch.max(quality_logits, 1)
                
                # Store predictions and labels
                val_quality_preds.extend(quality_preds.cpu().numpy())
                val_quality_labels.extend(quality_labels.cpu().numpy())
                val_sentiment_preds.extend(sentiment_pred.cpu().numpy())
                val_sentiment_scores.extend(sentiment_scores.cpu().numpy())
        
        # Average validation loss and metrics
        avg_val_quality_loss = val_quality_loss / len(val_loader)
        avg_val_sentiment_loss = val_sentiment_loss / len(val_loader)
        quality_accuracy = accuracy_score(val_quality_labels, val_quality_preds)
        sentiment_mse = mean_squared_error(val_sentiment_scores, val_sentiment_preds)
        
        # Combined metric (lower is better)
        combined_metric = 0.6 * (1 - quality_accuracy) + 0.4 * sentiment_mse
        
        print(f"Validation quality loss: {avg_val_quality_loss:.4f}")
        print(f"Validation sentiment loss: {avg_val_sentiment_loss:.4f}")
        print(f"Quality accuracy: {quality_accuracy:.4f}")
        print(f"Sentiment MSE: {sentiment_mse:.4f}")
        
        # Print classification report
        print("\nQuality Classification Report:")
        report = classification_report(
            val_quality_labels, 
            val_quality_preds, 
            target_names=['Negative Quality', 'Not About Quality', 'Positive Quality'],
            digits=3
        )
        print(report)
        
        # Plot confusion matrix
        cm = confusion_matrix(val_quality_labels, val_quality_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Not Quality', 'Positive'],
                   yticklabels=['Negative', 'Not Quality', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Quality Confusion Matrix - Epoch {epoch+1}')
        plt.savefig(os.path.join(model_dir, f'quality_confusion_matrix_epoch_{epoch+1}.png'))
        
        # Plot sentiment prediction vs actual
        plt.figure(figsize=(8, 6))
        plt.scatter(val_sentiment_scores, val_sentiment_preds, alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('Actual Sentiment Score')
        plt.ylabel('Predicted Sentiment Score')
        plt.title(f'Sentiment Score Prediction - Epoch {epoch+1}')
        plt.savefig(os.path.join(model_dir, f'sentiment_prediction_epoch_{epoch+1}.png'))
        
        # Save best model
        if combined_metric < best_combined_metric:
            best_combined_metric = combined_metric
            model_path = os.path.join(model_dir, 'best_multi_task_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved with combined metric: {combined_metric:.4f}")
    
    # Load best model
    best_model_path = os.path.join(model_dir, 'best_multi_task_model.pth')
    model.load_state_dict(torch.load(best_model_path))
    
    return model

# Original train_model function for backwards compatibility
def train_model(model, train_loader, val_loader, num_epochs=3, learning_rate=2e-5, warmup_steps=0, model_dir='models'):
    """
    Train the quality classifier model
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        model_dir: Directory to save model
        
    Returns:
        model: Trained model
    """
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Set up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Set up loss function
    criterion = nn.CrossEntropyLoss()
    
    # Set up scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        # Average training loss
        avg_train_loss = train_loss / len(train_loader)
        print(f"Training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Get predictions
                _, preds = torch.max(outputs, 1)
                
                # Store predictions and labels
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Average validation loss and metrics
        avg_val_loss = val_loss / len(val_loader)
        accuracy = accuracy_score(val_labels, val_preds)
        
        print(f"Validation loss: {avg_val_loss:.4f}")
        print(f"Validation accuracy: {accuracy:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        report = classification_report(
            val_labels, 
            val_preds, 
            target_names=['Negative Quality', 'Not About Quality', 'Positive Quality'],
            digits=3
        )
        print(report)
        
        # Plot confusion matrix
        cm = confusion_matrix(val_labels, val_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Not Quality', 'Positive'],
                   yticklabels=['Negative', 'Not Quality', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - Epoch {epoch+1}')
        plt.savefig(os.path.join(model_dir, f'confusion_matrix_epoch_{epoch+1}.png'))
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model_path = os.path.join(model_dir, 'best_quality_classifier.pth')
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved with accuracy: {accuracy:.4f}")
    
    # Load best model
    best_model_path = os.path.join(model_dir, 'best_quality_classifier.pth')
    model.load_state_dict(torch.load(best_model_path))
    
    return model

def train_multitask_model(model, train_loader, val_loader, device, epochs=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Loss functions
    quality_criterion = nn.BCELoss()  # Binary classification
    sentiment_criterion = nn.MSELoss()  # Regression
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # Progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            # Get inputs
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            is_quality = batch['is_quality'].to(device)
            sentiment_score = batch['sentiment_score'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            quality_pred, sentiment_pred = model(input_ids, attention_mask)
            
            # Calculate losses
            quality_loss = quality_criterion(quality_pred.squeeze(), is_quality)
            sentiment_loss = sentiment_criterion(sentiment_pred.squeeze(), sentiment_score)
            
            # Combined loss (can adjust weights between tasks)
            loss = 0.5 * quality_loss + 0.5 * sentiment_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': train_loss / (progress_bar.n + 1)})
        
        # Validation
        model.eval()
        val_quality_loss = 0.0
        val_sentiment_loss = 0.0
        val_unified_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                is_quality = batch['is_quality'].to(device)
                sentiment_score = batch['sentiment_score'].to(device)
                
                # Forward pass
                quality_pred, sentiment_pred = model(input_ids, attention_mask)
                
                # Calculate losses
                val_quality_loss += quality_criterion(quality_pred.squeeze(), is_quality).item()
                val_sentiment_loss += sentiment_criterion(sentiment_pred.squeeze(), sentiment_score).item()
                
                # Optional: calculate unified score loss for comparison
                unified_pred = model.get_unified_score(input_ids, attention_mask)
                unified_target = is_quality * sentiment_score + (1 - is_quality) * 0.5
                val_unified_loss += nn.MSELoss()(unified_pred.squeeze(), unified_target).item()
        
        val_quality_loss /= len(val_loader)
        val_sentiment_loss /= len(val_loader)
        val_unified_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss / len(train_loader):.4f}")
        print(f"  Val Quality Loss: {val_quality_loss:.4f}")
        print(f"  Val Sentiment Loss: {val_sentiment_loss:.4f}")
        print(f"  Val Unified Loss: {val_unified_loss:.4f}")
        
    return model