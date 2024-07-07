
import argparse
import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import joblib

def train_model(X, y, mode='local'):
    """
    Train a machine learning model locally or using SageMaker
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if mode == 'local':
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, 'nfl_prediction_model.joblib')
        return model
    elif mode == 'aws':
        sagemaker_session = sagemaker.Session()
        role = sagemaker.get_execution_role()

        sklearn = SKLearn(
            entry_point='sklearn_model_script.py',
            role=role,
            instance_count=1,
            instance_type='ml.m5.large',
            framework_version='0.23-1',
            py_version='py3',
            sagemaker_session=sagemaker_session
        )

        sklearn.fit({'train': X_train, 'test': X_test})
        predictor = sklearn.deploy(initial_instance_count=1, instance_type='ml.t2.medium')
        return predictor
    else:
        raise ValueError("Invalid mode. Choose 'local' or 'aws'")

def evaluate_model(model, X_test, y_test, mode='local'):
    """
    Evaluate the trained model
    """
    if mode == 'local':
        predictions = model.predict(X_test)
    elif mode == 'aws':
        predictions = model.predict(X_test)
    else:
        raise ValueError("Invalid mode. Choose 'local' or 'aws'")

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    return accuracy, precision, recall, f1

def perform_ab_testing(model_a, model_b, X_test, y_test, mode='local'):
    """
    Perform A/B testing between two models
    """
    if mode == 'local':
        predictions_a = model_a.predict(X_test)
        predictions_b = model_b.predict(X_test)
    elif mode == 'aws':
        predictions_a = model_a.predict(X_test)
        predictions_b = model_b.predict(X_test)
    else:
        raise ValueError("Invalid mode. Choose 'local' or 'aws'")

    accuracy_a = accuracy_score(y_test, predictions_a)
    accuracy_b = accuracy_score(y_test, predictions_b)

    print(f"Model A Accuracy: {accuracy_a}")
    print(f"Model B Accuracy: {accuracy_b}")

    if accuracy_a > accuracy_b:
        print("Model A performs better")
    elif accuracy_b > accuracy_a:
        print("Model B performs better")
    else:
        print("Both models perform equally")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NFL prediction model")
    parser.add_argument("--mode", type=str, choices=['local', 'aws'], default='local',
                        help="Mode to run the script (local or aws)")
    args = parser.parse_args()

    # Load and preprocess data
    from src.data.data_loader import load_nfl_data, preprocess_data
    if args.mode == 'local':
        df = load_nfl_data('local', local_path='path/to/your/local/data.csv')
    else:
        df = load_nfl_data('aws', s3_bucket='your-bucket-name', s3_key='path/to/your/data.csv')
    
    df = preprocess_data(df)

    # Feature engineering
    from src.features.feature_engineering import create_features
    df = create_features(df)

    # Prepare data for modeling
    X = df.drop('target_column', axis=1)
    y = df['target_column']

    # Train and evaluate model
    model = train_model(X, y, mode=args.mode)
    evaluate_model(model, X, y, mode=args.mode)
