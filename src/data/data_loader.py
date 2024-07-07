import pandas as pd
import boto3
#from sagemaker.session import Session

def load_nfl_data(source, local_path=None, s3_bucket=None, s3_key=None):
    """
    Load NFL data from local file or S3 bucket
    """
    if source == 'local':
        if local_path is None:
            raise ValueError("local_path must be provided for local data loading")
        df = pd.read_csv(local_path)
    elif source == 'aws':
        if s3_bucket is None or s3_key is None:
            raise ValueError("s3_bucket and s3_key must be provided for AWS data loading")
        sagemaker_session = Session()
        s3_client = boto3.client('s3')
        obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
        df = pd.read_csv(obj['Body'])
    else:
        raise ValueError("Invalid source. Choose 'local' or 'aws'")
    
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
