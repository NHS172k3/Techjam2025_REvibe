import os
import sys
import argparse

# Ensure the script can find the src directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create necessary directories with proper Windows path handling
try:
    # Use os.path.join for platform-independent paths
    data_dir = os.path.join('data')
    processed_dir = os.path.join(data_dir, 'processed')
    models_dir = os.path.join('models')
    
    # Create directories carefully
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
except Exception as e:
    print(f"Warning: Could not create directories: {e}")
    print("Will attempt to continue anyway...")

# Import from the correct location - prepare_data is in src.main, not src.data
from src.main import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Video Quality Analysis Model')
    parser.add_argument('--model-type', choices=['classifier', 'multi-task', 'integrated'], 
                        default='integrated', 
                        help='Type of model to train')
    args = parser.parse_args()
    
    print(f"Starting {args.model_type} model training...")
    
    if args.model_type == 'classifier':
        print("Training classifier for quality labels only")
    elif args.model_type == 'multi-task':
        print("Training multi-task model for both quality classification and sentiment regression")
    elif args.model_type == 'integrated':
        print("Training integrated model for unified quality+sentiment score")
    
    train(model_type=args.model_type)
    print("\nTraining complete! The model is ready to use.")