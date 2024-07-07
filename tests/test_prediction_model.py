
# tests/test_prediction_model.py
import unittest
from src.models.prediction_model import train_model, evaluate_model

class TestPredictionModel(unittest.TestCase):
    def test_train_model(self):
        # Create mock data
        X = np.random.rand(100, 10)
        y = np.random.randint(2, size=100)
        
        predictor = train_model(X, y)
        self.assertIsNotNone(predictor)

    def test_evaluate_model(self):
        # Create mock data and predictor
        X_test = np.random.rand(50, 10)
        y_test = np.random.randint(2, size=50)
        mock_predictor = lambda x: np.random.randint(2, size=len(x))
        
        accuracy, precision, recall, f1 = evaluate_model(mock_predictor, X_test, y_test)
        self.assertTrue(0 <= accuracy <= 1)
        self.assertTrue(0 <= precision <= 1)
        self.assertTrue(0 <= recall <= 1)
        self.assertTrue(0 <= f1 <= 1)