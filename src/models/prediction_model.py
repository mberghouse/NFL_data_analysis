

# src/models/prediction_model.py
import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

def train_model(X, y):
    """
    Train a machine learning model using SageMaker
    """
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a SageMaker estimator
    sklearn = SKLearn(
        entry_point='sklearn_model_script.py',
        role=role,
        instance_count=1,
        instance_type='ml.m5.large',
        framework_version='0.23-1',
        py_version='py3',
        sagemaker_session=sagemaker_session
    )


    # Train the model
    sklearn.fit({'train': X_train, 'test': X_test})

    # Deploy the model
    predictor = sklearn.deploy(initial_instance_count=1, instance_type='ml.t2.medium')

    return predictor

def evaluate_model(predictor, X_test, y_test):
    """
    Evaluate the trained model
    """
    predictions = predictor.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    return accuracy, precision, recall, f1


def perform_ab_testing(model_a, model_b, X_test, y_test):
    """
    Perform A/B testing between two models
    """
    predictions_a = model_a.predict(X_test)
    predictions_b = model_b.predict(X_test)

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