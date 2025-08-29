import os
import sys
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

# Import local modules
from src.data.quality_labeler import detect_quality_and_sentiment
from src.data.preprocessing import preprocess_dataset
from src.data.dataset import create_data_loaders
from src.models.classifier import QualityClassifier, QualitySentimentModel, IntegratedQualityModel
from src.training.train import train_model, train_multi_task_model, train_integrated_model

def prepare_data(input_path='data/raw/comments_dataset.csv'):
    """Prepare data for training"""
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)
    
    # Add quality and sentiment labels
    labeled_df = detect_quality_and_sentiment(df)
    labeled_path = 'data/processed/labeled_comments.csv'
    labeled_df.to_csv(labeled_path, index=False)
    
    # Preprocess data
    preprocessed_df = preprocess_dataset(labeled_path)
    preprocessed_path = 'data/processed/preprocessed_comments.csv'
    preprocessed_df.to_csv(preprocessed_path, index=False)
    
    return preprocessed_df

def train(model_type='classifier'):
    """
    Train the selected model type
    
    Args:
        model_type: Type of model to train
            'classifier': Original quality classifier
            'multi-task': Model that predicts both quality class and sentiment
            'integrated': Model that outputs unified quality+sentiment score
    """
    # Prepare directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Prepare data
    print("Preparing data...")
    preprocessed_df = prepare_data('data/raw/comments_dataset.csv')
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Create data loaders based on model type
    print("Creating data loaders...")
    
    if model_type == 'classifier':
        # Quality classification only
        target_column = 'quality_label'
        train_loader, val_loader, test_loader = create_data_loaders(
            preprocessed_df, 
            tokenizer=tokenizer,
            batch_size=16,
            target_column=target_column
        )
        
    elif model_type == 'multi-task':
        # For multi-task model, use the specialized data loader
        from src.data.dataset import create_multi_task_data_loaders
        
        train_loader, val_loader, test_loader = create_multi_task_data_loaders(
            preprocessed_df, 
            tokenizer=tokenizer,
            batch_size=16
        )
        
    elif model_type == 'integrated':
        # For integrated model, use unified_score
        target_column = 'unified_score'
        train_loader, val_loader, test_loader = create_data_loaders(
            preprocessed_df, 
            tokenizer=tokenizer,
            batch_size=16,
            target_column=target_column
        )
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Initialize model
    print(f"Initializing {model_type} model...")
    
    if model_type == 'classifier':
        model = QualityClassifier(num_labels=3)
    elif model_type == 'multi-task':
        model = QualitySentimentModel(quality_labels=3)
    elif model_type == 'integrated':
        model = IntegratedQualityModel()
    
    # Train model
    print(f"Training {model_type} model...")
    
    if model_type == 'classifier':
        trained_model = train_model(
            model, 
            train_loader, 
            val_loader, 
            num_epochs=3,
            learning_rate=2e-5,
            model_dir='models'
        )
    elif model_type == 'multi-task':
        trained_model = train_multi_task_model(
            model, 
            train_loader, 
            val_loader, 
            num_epochs=3,
            learning_rate=2e-5,
            model_dir='models'
        )
    elif model_type == 'integrated':
        trained_model = train_integrated_model(
            model, 
            train_loader, 
            val_loader, 
            num_epochs=3,
            learning_rate=2e-5,
            model_dir='models'
        )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    if model_type == 'classifier':
        evaluate_classifier(trained_model, test_loader)
    elif model_type == 'multi-task':
        evaluate_multi_task(trained_model, test_loader)
    elif model_type == 'integrated':
        evaluate_integrated(trained_model, test_loader)
    
    print("Training complete!")
    return trained_model

def evaluate_classifier(model, test_loader):
    """Evaluate classifier model on test set"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Classification predictions
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Classification metrics
    from sklearn.metrics import accuracy_score, classification_report
    
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=['Negative Quality', 'Not About Quality', 'Positive Quality'],
        digits=3
    )
    
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nTest Classification Report:")
    print(report)

def evaluate_multi_task(model, test_loader):
    """Evaluate multi-task model on test set"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    quality_preds = []
    quality_labels = []
    sentiment_preds = []
    sentiment_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            quality_label = batch['quality_label'].to(device)
            sentiment_score = batch['sentiment_score'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Get quality predictions
            _, quality_pred = torch.max(outputs['quality_logits'], 1)
            sentiment_pred = outputs['sentiment_score']
            
            quality_preds.extend(quality_pred.cpu().numpy())
            quality_labels.extend(quality_label.cpu().numpy())
            sentiment_preds.extend(sentiment_pred.cpu().numpy())
            sentiment_labels.extend(sentiment_score.cpu().numpy())
    
    # Quality metrics
    from sklearn.metrics import accuracy_score, classification_report
    
    quality_accuracy = accuracy_score(quality_labels, quality_preds)
    quality_report = classification_report(
        quality_labels, 
        quality_preds, 
        target_names=['Negative Quality', 'Not About Quality', 'Positive Quality'],
        digits=3
    )
    
    # Sentiment metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    sentiment_mse = mean_squared_error(sentiment_labels, sentiment_preds)
    sentiment_mae = mean_absolute_error(sentiment_labels, sentiment_preds)
    sentiment_r2 = r2_score(sentiment_labels, sentiment_preds)
    
    print(f"Quality Test Accuracy: {quality_accuracy:.4f}")
    print("\nQuality Classification Report:")
    print(quality_report)
    
    print(f"\nSentiment Test MSE: {sentiment_mse:.4f}")
    print(f"Sentiment Test MAE: {sentiment_mae:.4f}")
    print(f"Sentiment Test R²: {sentiment_r2:.4f}")

def evaluate_integrated(model, test_loader):
    """Evaluate integrated model on test set"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Regression metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")
    
    # Plot prediction vs actual
    plt.figure(figsize=(8, 6))
    plt.scatter(all_labels, all_preds, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Actual Unified Score')
    plt.ylabel('Predicted Unified Score')
    plt.title('Unified Quality-Sentiment Score')
    plt.savefig('models/unified_score_test.png')

def predict_quality_sentiment(comment, model_path='models/best_integrated_model.pth', model_type='integrated'):
    """
    Predict quality and sentiment for a single comment
    
    Args:
        comment: Comment text
        model_path: Path to trained model
        model_type: Type of model to use ('classifier', 'multi-task', or 'integrated')
        
    Returns:
        prediction: Dictionary with prediction results
    """
    # Initialize model
    if model_type == 'classifier':
        model = QualityClassifier()
        model.load_state_dict(torch.load(model_path))
    elif model_type == 'multi-task':
        model = QualitySentimentModel()
        model.load_state_dict(torch.load(model_path))
    elif model_type == 'integrated':
        model = IntegratedQualityModel()
        model.load_state_dict(torch.load(model_path))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Set to evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Tokenize comment
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    encoding = tokenizer(
        comment,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Get prediction
    with torch.no_grad():
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        outputs = model(input_ids, attention_mask)
        
        if model_type == 'classifier':
            # Classification prediction
            probs = torch.softmax(outputs, dim=1)
            _, pred = torch.max(outputs, 1)
            
            pred_class = pred.item()
            probabilities = probs[0].cpu().numpy()
            
            # Map prediction to label
            label_map = {0: 'Negative Quality', 1: 'Not About Quality', 2: 'Positive Quality'}
            prediction = {
                'quality_class': pred_class,
                'quality_label': label_map[pred_class],
                'probabilities': {label_map[i]: float(prob) for i, prob in enumerate(probabilities)},
                'is_quality_related': pred_class != 1,  # True if it's about quality (positive or negative)
                'sentiment_score': None  # Not available in this model
            }
        
        elif model_type == 'multi-task':
            # Multi-task prediction
            quality_logits = outputs['quality_logits']
            sentiment_score = outputs['sentiment_score'].item()
            
            quality_probs = torch.softmax(quality_logits, dim=1)
            _, quality_pred = torch.max(quality_logits, 1)
            
            quality_class = quality_pred.item()
            quality_probabilities = quality_probs[0].cpu().numpy()
            
            # Map prediction to label
            label_map = {0: 'Negative Quality', 1: 'Not About Quality', 2: 'Positive Quality'}
            prediction = {
                'quality_class': quality_class,
                'quality_label': label_map[quality_class],
                'quality_probabilities': {label_map[i]: float(prob) for i, prob in enumerate(quality_probabilities)},
                'is_quality_related': quality_class != 1,  # True if it's about quality (positive or negative)
                'sentiment_score': sentiment_score
            }
        
        elif model_type == 'integrated':
            # Integrated prediction (single unified score)
            unified_score = outputs.item()
            
            # Interpret the unified score
            prediction = {
                'unified_score': unified_score,
                'interpretation': interpret_unified_score(unified_score)
            }
    
    return prediction

def interpret_unified_score(score):
    """Interpret the unified quality+sentiment score"""
    if score >= 0.8:
        return "Excellent quality with very positive sentiment"
    elif score >= 0.6:
        return "Good quality with positive sentiment"
    elif score >= 0.5:
        return "Average quality with neutral to positive sentiment"
    elif score >= 0.4:
        return "Average quality with neutral to negative sentiment"
    elif score >= 0.2:
        return "Poor quality with negative sentiment"
    else:
        return "Very poor quality with very negative sentiment"

def main():
    parser = argparse.ArgumentParser(description='Video Quality and Sentiment Analysis')
    parser.add_argument('--mode', choices=['train', 'predict'], default='train', help='Mode (train or predict)')
    parser.add_argument('--model-type', choices=['classifier', 'multi-task', 'integrated'], default='integrated', 
                        help='Type of model to train/use')
    parser.add_argument('--comment', type=str, help='Comment to predict (for predict mode)')
    parser.add_argument('--model', type=str, help='Model path (for predict mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(model_type=args.model_type)
    elif args.mode == 'predict':
        if args.comment:
            model_path = args.model or f'models/best_{args.model_type}_model.pth'
            prediction = predict_quality_sentiment(args.comment, model_path, args.model_type)
            print(f"Comment: {args.comment}")
            print(f"Prediction: {prediction}")
        else:
            print("Please provide a comment using --comment")

if __name__ == "__main__":
    main()