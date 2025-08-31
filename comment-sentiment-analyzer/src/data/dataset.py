import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

class CommentDataset(Dataset):
    """Dataset for video quality classification or regression"""
    
    def __init__(self, data, tokenizer=None, max_length=128, target_column='quality_label'):
        """
        Initialize dataset
        
        Args:
            data: DataFrame with comments and labels
            tokenizer: Tokenizer for text encoding (or name of pretrained model)
            max_length: Maximum sequence length
            target_column: Target column for classification or regression
        """
        self.data = data
        self.max_length = max_length
        self.target_column = target_column
        
        # Initialize tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        elif isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get comment
        comment = str(self.data.iloc[idx]['cleaned_comment'])
        
        # Tokenize comment
        encoding = self.tokenizer(
            comment,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get quality label (1=not quality, 0=negative quality, 2=positive quality)
        quality_label = self.data.iloc[idx]['quality_label']
        
        # Convert quality_label to binary (is quality related)
        is_quality = torch.tensor(1.0 if quality_label != 1 else 0.0, dtype=torch.float)
        
        # Get sentiment score
        sentiment_score = torch.tensor(
            float(self.data.iloc[idx]['sentiment_score']), 
            dtype=torch.float
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'is_quality': is_quality,
            'sentiment_score': sentiment_score
        }

class MultiTaskDataset(Dataset):
    """Dataset for multi-task learning (quality classification + sentiment regression)"""
    
    def __init__(self, data, tokenizer=None, max_length=128):
        """
        Initialize dataset for multi-task learning
        
        Args:
            data: DataFrame with comments and labels
            tokenizer: Tokenizer for text encoding
            max_length: Maximum sequence length
        """
        self.data = data
        self.max_length = max_length
        
        # Initialize tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        elif isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get comment
        comment = str(self.data.iloc[idx]['cleaned_comment'])
        
        # Tokenize comment
        encoding = self.tokenizer(
            comment,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get labels
        quality_label = torch.tensor(int(self.data.iloc[idx]['quality_label']), dtype=torch.long)
        sentiment_score = torch.tensor(float(self.data.iloc[idx]['sentiment_score']), dtype=torch.float32)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'quality_label': quality_label,
            'sentiment_score': sentiment_score
        }

def create_data_loaders(df, tokenizer=None, batch_size=16, max_length=128, 
                        test_size=0.2, val_size=0.1, target_column='quality_label', stratify=True):
    """Create train, validation and test data loaders"""
    
    # Handle stratification for regression vs classification
    if target_column == 'unified_score' or 'regression' in target_column:
        # For regression, don't stratify
        stratify_col = None
    else:
        # For classification, stratify by target column if requested
        stratify_col = df[target_column] if stratify else None
    
    # First split: train and temp (test + validation)
    train_df, temp_df = train_test_split(
        df, 
        test_size=(test_size + val_size), 
        random_state=42,
        stratify=stratify_col
    )
    
    # Second split: test and validation
    val_size_adjusted = val_size / (test_size + val_size)
    
    if stratify_col is not None:
        stratify_temp = temp_df[target_column]
    else:
        stratify_temp = None
        
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=(1-val_size_adjusted), 
        random_state=42,
        stratify=stratify_temp
    )
    
    # Create datasets
    train_dataset = CommentDataset(train_df, tokenizer, max_length, target_column)
    val_dataset = CommentDataset(val_df, tokenizer, max_length, target_column)
    test_dataset = CommentDataset(test_df, tokenizer, max_length, target_column)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

def create_multi_task_data_loaders(df, tokenizer=None, batch_size=16, max_length=128, 
                                   test_size=0.2, val_size=0.1):
    """Create data loaders for multi-task learning"""
    
    # Split data - stratify by quality_label
    train_df, temp_df = train_test_split(
        df, 
        test_size=(test_size + val_size), 
        random_state=42,
        stratify=df['quality_label']
    )
    
    val_size_adjusted = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=(1-val_size_adjusted), 
        random_state=42,
        stratify=temp_df['quality_label']
    )
    
    # Create datasets
    train_dataset = MultiTaskDataset(train_df, tokenizer, max_length)
    val_dataset = MultiTaskDataset(val_df, tokenizer, max_length)
    test_dataset = MultiTaskDataset(test_df, tokenizer, max_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f"Multi-task Train samples: {len(train_dataset)}")
    print(f"Multi-task Validation samples: {len(val_dataset)}")
    print(f"Multi-task Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

import torch
from torch.utils.data import Dataset
import pandas as pd

class SentimentDataset(Dataset):
    """Dataset class for sentiment analysis only"""
    
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"Dataset initialized with {len(df)} samples")
        print(f"Available columns: {df.columns.tolist()}")
        
        # Check for required columns
        required_cols = ['comment', 'sentiment_score']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing essential columns: {missing_cols}")
        
        print(f"Sentiment score range: {df['sentiment_score'].min():.3f} to {df['sentiment_score'].max():.3f}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        comment = str(row['comment'])
        
        # Tokenize comment
        encoding = self.tokenizer(
            comment,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'comment': comment,
            'sentiment_score': torch.tensor(float(row['sentiment_score']), dtype=torch.float32)
        }
        
        return result