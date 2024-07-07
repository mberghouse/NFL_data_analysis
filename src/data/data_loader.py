import pandas as pd
import boto3
from sagemaker.session import Session

def load_nfl_data(s3_bucket, s3_key):
    """
    Load NFL data from S3 bucket
    """
    sagemaker_session = Session()
    s3_client = boto3.client('s3')
    
    obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
    df = pd.read_csv(obj['Body'])
    
    return df

def preprocess_data(df):
    """
    Preprocess the NFL data
    """
    # Handle missing values
    df = df.dropna()
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Encode categorical variables
    df = pd.get_dummies(df, columns=['home_team', 'away_team'])
    
    return df