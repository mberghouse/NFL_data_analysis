![cover](https://github.com/mberghouse/NFL_data_analysis/assets/55556564/d664205e-d4e6-4979-b1ff-04248854f00d)
# NFL Game Prediction Project

This project uses machine learning to analyze NFL data and predict game outcomes. It can run both locally and on AWS SageMaker.

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. For AWS execution:
   - Configure AWS credentials for SageMaker access
   - Ensure you have the necessary IAM roles and permissions set up
4. Run exploratory data analysis: `jupyter notebook notebooks/exploratory_data_analysis.ipynb`
5. Train and deploy model:
   - Locally: `python src/models/prediction_model.py --mode local`
   - On AWS: `python src/models/prediction_model.py --mode aws`

## Project Structure

- `src/`: Source code for data loading, feature engineering, and model training
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model evaluation
- `tests/`: Unit tests for key components
- `config/`: Configuration files

## Key Features

- Data loading and preprocessing (local and S3)
- Feature engineering specific to NFL game prediction
- Machine learning model implementation (local and SageMaker)
- Model performance monitoring and A/B testing
- Visualization of results and insights
- Exploratory Data Analysis notebook

