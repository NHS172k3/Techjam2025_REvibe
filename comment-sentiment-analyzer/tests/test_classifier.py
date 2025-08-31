import unittest
import torch
from transformers import AutoTokenizer
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.classifier import SentimentClassifier

class TestSentimentClassifier(unittest.TestCase):
    
    def setUp(self):
        self.model = SentimentClassifier()
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    def test_model_forward(self):
        """Test model forward pass"""
        # Create dummy input
        inputs = self.tokenizer(
            "This is a test comment",
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Forward pass
        outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
        
        # Check output structure
        self.assertIn('sentiment', outputs)
        
        # Check output ranges
        sentiment = outputs['sentiment'].item()
        self.assertGreaterEqual(sentiment, -1.0)
        self.assertLessEqual(sentiment, 1.0)
    
    def test_predict_comment(self):
        """Test single comment prediction"""
        result = self.model.predict_comment(
            "This is amazing! I love it!",
            self.tokenizer
        )
        
        self.assertIn('sentiment', result)
        self.assertIsInstance(result['sentiment'], float)
    
    def test_sentiment_label(self):
        """Test sentiment label conversion"""
        self.assertEqual(self.model.get_sentiment_label(0.5), "Positive")
        self.assertEqual(self.model.get_sentiment_label(-0.5), "Negative")
        self.assertEqual(self.model.get_sentiment_label(0.0), "Neutral")

if __name__ == '__main__':
    unittest.main()