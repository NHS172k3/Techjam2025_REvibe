import unittest
from src.models.classifier import VideoQualityClassifier
from src.data.dataset import CommentsDataset

class TestVideoQualityClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = CommentsDataset('data/raw/comments_dataset.csv')
        cls.model = VideoQualityClassifier()

    def test_model_initialization(self):
        self.assertIsNotNone(self.model)

    def test_train_model(self):
        X_train, y_train = self.dataset.get_training_data()
        self.model.train(X_train, y_train)
        self.assertTrue(self.model.is_trained)

    def test_predict_quality(self):
        sample_comments = ["This video is amazing!", "I didn't like this video at all."]
        predictions = self.model.predict(sample_comments)
        self.assertEqual(len(predictions), len(sample_comments))

    def test_save_and_load_model(self):
        self.model.train(*self.dataset.get_training_data())
        model_path = 'test_model.pth'
        self.model.save(model_path)
        
        new_model = VideoQualityClassifier()
        new_model.load(model_path)
        self.assertTrue(new_model.is_trained)

    @classmethod
    def tearDownClass(cls):
        import os
        if os.path.exists('test_model.pth'):
            os.remove('test_model.pth')

if __name__ == '__main__':
    unittest.main()