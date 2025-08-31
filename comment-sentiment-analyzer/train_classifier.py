#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.append(str(src_dir))

# Change to project directory
os.chdir(current_dir)

def main():
    print("🚀 Starting Sentiment Classifier Training")
    print("=" * 50)
    
    # Step 1: Preprocess data
    print("\n1. Preprocessing data...")
    from data.preprocessing import preprocess_dataset
    
    result = preprocess_dataset()
    if result is None:
        print("❌ Preprocessing failed!")
        return
    
    # Step 2: Train model
    print("\n2. Training sentiment model...")
    from training.train import train_sentiment_model
    
    model = train_sentiment_model()
    if model is None:
        print("❌ Training failed!")
        return
    
    print("\n✅ Training completed successfully!")
    print("Model saved to: models/best_sentiment_model.pth")

if __name__ == "__main__":
    main()