import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

# Add parent directories to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
project_dir = src_dir.parent
sys.path.extend([str(src_dir), str(project_dir)])

from data.dataset import SentimentDataset
from models.classifier import SentimentClassifier

def train_sentiment_model(num_epochs=5, batch_size=16, learning_rate=2e-5):
    """
    Train the sentiment classifier
    """
    # Change to project directory
    os.chdir(project_dir)
    
    print("Loading preprocessed data...")
    
    # Load preprocessed data
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Train columns: {train_df.columns.tolist()}")
    
    # Verify required columns
    required_cols = ['comment', 'sentiment_score']
    missing_cols = [col for col in required_cols if col not in train_df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print("Available columns:", train_df.columns.tolist())
        return None
    
    from transformers import AutoTokenizer
    
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = SentimentClassifier()
    
    # Create datasets
    train_dataset = SentimentDataset(train_df, tokenizer)
    test_dataset = SentimentDataset(test_df, tokenizer)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Device and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Training on: {device}")
    
    # Loss function for sentiment
    criterion = nn.MSELoss()
    
    # Training loop
    train_losses = []
    test_losses = []
    test_r2_scores = []
    best_r2 = -float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Ground truth sentiment
            sentiment_target = batch['sentiment_score'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Loss on sentiment prediction
            loss = criterion(outputs['sentiment'], sentiment_target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if batch_count % 10 == 0:
                print(f"  Batch {batch_count}: Loss = {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        print(f'Epoch {epoch+1} Average Loss: {avg_loss:.4f}')
        
        # Evaluation
        model.eval()
        test_predictions = []
        test_targets = []
        total_test_loss = 0
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                sentiment_target = batch['sentiment_score'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs['sentiment'], sentiment_target)
                
                total_test_loss += loss.item()
                test_predictions.extend(outputs['sentiment'].cpu().numpy())
                test_targets.extend(sentiment_target.cpu().numpy())
        
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        # Calculate metrics
        mse = mean_squared_error(test_targets, test_predictions)
        mae = mean_absolute_error(test_targets, test_predictions)
        r2 = r2_score(test_targets, test_predictions)
        test_r2_scores.append(r2)
        
        print(f'Test Loss: {avg_test_loss:.4f}')
        print(f'Test MSE: {mse:.4f}')
        print(f'Test MAE: {mae:.4f}')
        print(f'Test R²: {r2:.4f}')
        
        # Save best model based on R²
        if r2 > best_r2:
            best_r2 = r2
            torch.save(model.state_dict(), 'models/best_sentiment_model.pth')
            print(f'✅ Saved new best model (R² = {r2:.4f})')
        
        # Create visualization
        plt.figure(figsize=(8, 6))
        plt.scatter(test_targets, test_predictions, alpha=0.6)
        plt.plot([-1, 1], [-1, 1], 'r--', lw=2)
        plt.xlabel('True Sentiment Scores')
        plt.ylabel('Predicted Sentiment Scores')
        plt.title(f'Epoch {epoch+1} - R² = {r2:.3f}')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.savefig(f'models/sentiment_epoch_{epoch+1}.png')
        plt.close()
        
        # Show some example predictions
        if epoch == num_epochs - 1:  # Last epoch
            print(f"\nSample Predictions:")
            for i in range(min(5, len(test_predictions))):
                true_val = test_targets[i]
                pred_val = test_predictions[i]
                print(f"  True: {true_val:.3f}, Predicted: {pred_val:.3f}, Diff: {abs(true_val - pred_val):.3f}")
    
    # Final training plots
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(test_losses, label='Test Loss', marker='s')
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(test_r2_scores, label='Test R²', marker='d', color='green')
    plt.title('Model Performance')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.scatter(test_targets, test_predictions, alpha=0.6)
    plt.plot([-1, 1], [-1, 1], 'r--', lw=2)
    plt.xlabel('True Sentiment Scores')
    plt.ylabel('Predicted Sentiment Scores')
    plt.title(f'Final Results (R² = {best_r2:.3f})')
    
    plt.tight_layout()
    plt.savefig("models/sentiment_training_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'\n✅ Training completed!')
    print(f'Best R² Score: {best_r2:.4f}')
    print(f'Model saved to: models/best_sentiment_model.pth')
    
    return model, best_r2

# Legacy function name for compatibility
def train_multitask_model():
    """Legacy function - now trains sentiment only"""
    return train_sentiment_model()

if __name__ == "__main__":
    train_sentiment_model()