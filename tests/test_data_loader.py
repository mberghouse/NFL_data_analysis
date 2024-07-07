
# tests/test_data_loader.py
import unittest
from src.data.data_loader import load_nfl_data, preprocess_data

class TestDataLoader(unittest.TestCase):
    def test_load_nfl_data(self):
        # Mock S3 bucket and key
        s3_bucket = 'mock-bucket'
        s3_key = 'mock-key'
        
        df = load_nfl_data(s3_bucket, s3_key)
        self.assertIsNotNone(df)
        self.assertTrue(len(df) > 0)

    def test_preprocess_data(self):
        # Create a mock dataframe
        mock_df = pd.DataFrame({
            'date': ['2021-01-01', '2021-01-02'],
            'home_team': ['Team A', 'Team B'],
            'away_team': ['Team C', 'Team D'],
            'home_score': [21, 28],
            'away_score': [17, 24]
        })
        
        processed_df = preprocess_data(mock_df)
        self.assertIsNotNone(processed_df)
        self.assertTrue('date' in processed_df.columns)
        self.assertTrue('home_team_Team A' in processed_df.columns)
        self.assertTrue('away_team_Team C' in processed_df.columns)
