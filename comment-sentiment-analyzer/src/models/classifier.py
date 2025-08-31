import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer

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
    Model that predicts sentiment and usefulness scores, then integrates them
    Compatible with DistilBERT (no pooler_output)
    """
    def __init__(self, model_name='distilbert-base-uncased', hidden_size=128, dropout=0.3):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze early layers for efficiency
        for param in list(self.bert.parameters())[:-4]:
            param.requires_grad = False
            
        bert_hidden = self.bert.config.hidden_size
        
        # Shared representation
        self.shared_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bert_hidden, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Component prediction heads
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Tanh()  # -1 to 1 (sentiment)
        )
        
        self.usefulness_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Tanh()  # -1 to 1 (usefulness when mentioned)
        )
        
        # Integration layer - learns how to combine sentiment + usefulness
        self.integration_layer = nn.Sequential(
            nn.Linear(hidden_size + 2, 32),  # shared_features + sentiment + usefulness
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 0 to 1 (final unified score)
        )
        
    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # DistilBERT doesn't have pooler_output, use mean pooling of last_hidden_state
        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Mean pooling: average over sequence length, weighted by attention mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        # Shared features
        shared_features = self.shared_layer(pooled_output)
        
        # Component predictions
        sentiment = self.sentiment_head(shared_features).squeeze(-1)
        usefulness_score = self.usefulness_head(shared_features).squeeze(-1)
        
        # Combine for integration (model learns the combination)
        combined_features = torch.cat([
            shared_features,
            sentiment.unsqueeze(-1),
            usefulness_score.unsqueeze(-1)
        ], dim=-1)
        
        # Learn unified score
        unified_score = self.integration_layer(combined_features).squeeze(-1)
        
        return {
            'sentiment': sentiment,
            'usefulness_score': usefulness_score,
            'unified_score': unified_score
        }

    def predict_comment(self, comment, tokenizer, device='cpu'):
        """Predict scores for a single comment"""
        self.eval()
        
        inputs = tokenizer(
            comment,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            outputs = self.forward(input_ids, attention_mask)
            
        return {
            'unified_score': outputs['unified_score'].item(),
            'sentiment': outputs['sentiment'].item(),
            'usefulness_score': outputs['usefulness_score'].item()
        }

class SentimentClassifier(nn.Module):
    """
    Pure sentiment classifier - no quality/usefulness prediction
    """
    def __init__(self, model_name='distilbert-base-uncased', hidden_size=128, dropout=0.3):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze early layers for efficiency
        for param in list(self.bert.parameters())[:-6]:
            param.requires_grad = False
            
        bert_hidden = self.bert.config.hidden_size
        
        # Sentiment prediction network
        self.sentiment_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bert_hidden, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # Output range: -1 to 1
        )
        
    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # DistilBERT doesn't have pooler_output, use mean pooling
        last_hidden_state = outputs.last_hidden_state
        
        # Mean pooling with attention mask weighting
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        # Predict sentiment
        sentiment = self.sentiment_classifier(pooled_output).squeeze(-1)
        
        return {
            'sentiment': sentiment
        }

    def predict_comment(self, comment, tokenizer, device='cpu'):
        """Predict sentiment for a single comment"""
        self.eval()
        
        inputs = tokenizer(
            comment,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            outputs = self.forward(input_ids, attention_mask)
            
        return {
            'sentiment': outputs['sentiment'].item()
        }

    def get_sentiment_label(self, sentiment_score):
        """Convert sentiment score to label"""
        if sentiment_score > 0.3:
            return "Positive"
        elif sentiment_score < -0.3:
            return "Negative"
        else:
            return "Neutral"