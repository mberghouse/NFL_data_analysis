# src/features/feature_engineering.py
import pandas as pd
import numpy as np

def create_features(df):
    """
    Create features for NFL game prediction
    """
    # Calculate rolling averages for key stats
    stats = ['points', 'total_yards', 'passing_yards', 'rushing_yards', 'turnovers']
    for stat in stats:
        df[f'home_{stat}_rolling_avg'] = df.groupby('home_team')[f'home_{stat}'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        df[f'away_{stat}_rolling_avg'] = df.groupby('away_team')[f'away_{stat}'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    
    # Create features for team's recent performance
    df['home_win_streak'] = df.groupby('home_team')['home_score'].transform(lambda x: (x > x.shift(1)).cumsum())
    df['away_win_streak'] = df.groupby('away_team')['away_score'].transform(lambda x: (x > x.shift(1)).cumsum())
    
    # Create features for head-to-head history
    df['h2h_home_wins'] = df.groupby(['home_team', 'away_team'])['home_score'].transform(lambda x: (x > x.shift(1)).cumsum())
    
    return df