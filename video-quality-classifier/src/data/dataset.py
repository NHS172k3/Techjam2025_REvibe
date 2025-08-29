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
        
        # Get label - fix data type based on target column
        target_value = self.data.iloc[idx][self.target_column]
        
        # Check if this is for regression (unified_score) or classification
        if self.target_column == 'unified_score' or 'regression' in self.target_column:
            # For regression, use float32
            label = torch.tensor(float(target_value), dtype=torch.float32)
        else:
            # For classification, use long
            label = torch.tensor(int(target_value), dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': label
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