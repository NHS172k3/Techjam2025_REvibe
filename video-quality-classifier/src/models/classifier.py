import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class QualityClassifier(nn.Module):
    """Transformer-based classifier for video quality detection"""
    def __init__(self, model_name='distilbert-base-uncased', num_labels=3):
        super(QualityClassifier, self).__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output)
        
        return logits

class QualitySentimentModel(nn.Module):
    """
    Multi-task model that predicts both quality class AND sentiment score
    
    This model outputs:
    1. Quality classification (negative quality, not about quality, positive quality)
    2. Sentiment score (continuous 0-1 value)
    """
    def __init__(self, model_name='distilbert-base-uncased', quality_labels=3):
        super(QualitySentimentModel, self).__init__()
        
        # Load pretrained model
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Shared layers
        self.dropout = nn.Dropout(0.1)
        
        # Task-specific heads
        self.quality_classifier = nn.Linear(self.config.hidden_size, quality_labels)
        self.sentiment_regressor = nn.Linear(self.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        """Forward pass"""
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use CLS token for classification
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        # Task 1: Quality classification
        quality_logits = self.quality_classifier(pooled_output)
        
        # Task 2: Sentiment regression
        sentiment_score = torch.sigmoid(self.sentiment_regressor(pooled_output)).squeeze(-1)
        
        return {
            'quality_logits': quality_logits,
            'sentiment_score': sentiment_score
        }

class IntegratedQualityModel(nn.Module):
    """
    Integrated model that predicts a single quality score that combines
    both technical quality assessment and sentiment analysis.
    
    This model directly outputs a single unified score (0-1) that represents:
    - How good the video quality is
    - How positive the viewer sentiment is
    """
    def __init__(self, model_name='distilbert-base-uncased'):
        super(IntegratedQualityModel, self).__init__()
        
        # Load pretrained model
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Intermediate layers
        self.dropout = nn.Dropout(0.1)
        self.intermediate = nn.Linear(self.config.hidden_size, 256)
        self.activation = nn.ReLU()
        
        # Output layer - single score
        self.unified_score = nn.Linear(256, 1)
        
    def forward(self, input_ids, attention_mask):
        """Forward pass"""
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use CLS token
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        # Intermediate representation
        intermediate = self.activation(self.intermediate(pooled_output))
        
        # Unified quality + sentiment score
        score = torch.sigmoid(self.unified_score(intermediate)).squeeze(-1)
        
        return score

class MultiTaskQualityModel(nn.Module):
    def __init__(self, pretrained_model="distilbert-base-uncased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model)
        
        # Shared features dimension
        hidden_size = self.bert.config.hidden_size
        
        # Quality detection head (binary classification)
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)  # Binary output: mentions quality or not
        )
        
        # Sentiment analysis head (regression)
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)  # Continuous output: sentiment score
        )
    
    def forward(self, input_ids, attention_mask):
        # Shared encoder
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # Task-specific heads
        quality_logits = self.quality_head(pooled_output)
        sentiment_logits = self.sentiment_head(pooled_output)
        
        # Apply appropriate activations
        quality_output = torch.sigmoid(quality_logits)      # 0-1 probability
        sentiment_output = torch.sigmoid(sentiment_logits)  # 0-1 score
        
        return quality_output, sentiment_output
    
    def get_unified_score(self, input_ids, attention_mask):
        """Optional method to get a combined score for backward compatibility"""
        quality_prob, sentiment_score = self(input_ids, attention_mask)
        
        # If quality is mentioned, use sentiment score
        # If not, use a neutral or slightly lower score
        unified = quality_prob * sentiment_score + (1 - quality_prob) * 0.5
        
        return unified