# Video Quality Classifier

This project aims to develop a video quality classifier that intelligently assesses the quality of videos based on comments. The classifier is designed to handle conflicting opinions within the comment section and uncover the true quality of a video.

## Project Structure

- **src/**: Contains the source code for the project.
  - **data/**: Includes scripts for data preprocessing and dataset management.
    - `preprocessing.py`: Functions for cleaning and preprocessing comments data.
    - `dataset.py`: Class for loading and managing the dataset.
  - **models/**: Implements the classifier and feature extraction methods.
    - `classifier.py`: Main classifier model with training and prediction methods.
    - `feature_extraction.py`: Functions for extracting features from comments.
    - `ensemble.py`: Ensemble methods for combining predictions from multiple models.
  - **training/**: Contains scripts for training and evaluating the model.
    - `train.py`: Training loop with hyperparameter tuning.
    - `evaluation.py`: Functions for model evaluation using various metrics.
  - **utils/**: Utility functions for text processing and metrics.
    - `text_processing.py`: Functions for tokenization and stemming.
    - `metrics.py`: Custom metrics for evaluating model performance.
  - `main.py`: Entry point for the application, orchestrating data loading, training, and evaluation.

- **data/**: Contains datasets.
  - **raw/**: Raw comments dataset used for training and evaluation.
    - `comments_dataset.csv`: The dataset containing comments.
  - **processed/**: Directory for storing processed data files.
  - **embeddings/**: Directory for storing precomputed embeddings.

- **notebooks/**: Jupyter notebooks for analysis and model comparison.
  - `exploratory_analysis.ipynb`: Exploratory data analysis notebook.
  - `model_comparison.ipynb`: Comparison of different models.

- **config/**: Configuration settings for the model.
  - `model_config.yaml`: Hyperparameters and model architecture settings.

- **tests/**: Unit tests for the project.
  - `test_preprocessing.py`: Tests for preprocessing functions.
  - `test_classifier.py`: Tests for the classifier model.

- **requirements.txt**: Lists dependencies required for the project.

- **setup.py**: Packaging script for the project.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd video-quality-classifier
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Preprocess the comments data:
   - Run the preprocessing script to clean and prepare the data for training.

2. Train the classifier:
   - Execute the training script to train the model on the processed dataset.

3. Evaluate the model:
   - Use the evaluation script to assess the model's performance on validation and test datasets.

4. Explore the notebooks for data analysis and model comparisons.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- This project utilizes various libraries for data processing, machine learning, and evaluation.