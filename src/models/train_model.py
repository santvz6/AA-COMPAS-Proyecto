import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from fairlearn.postprocessing import ThresholdOptimizer

def train_baseline(X_train: pd.DataFrame, y_train: pd.Series, preprocessor, random_state: int = 42) -> Pipeline:
    """
    Trains a baseline Logistic Regression model using the provided preprocessor.
    """
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=random_state, max_iter=1000))
    ])
    
    # Fit the pipeline
    # The preprocessor will ONLY learn from the train data (means, stds, categories)
    clf.fit(X_train, y_train)
    
    return clf

def train_fair_model(
    baseline_model: Pipeline, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    A_train: pd.Series,
    constraint: str = "demographic_parity",
    random_state: int = 42
) -> ThresholdOptimizer:
    """
    Trains a mitigated model using ThresholdOptimizer on top of the baseline model.
    Constraints can be: 'demographic_parity', 'equalized_odds', 'false_positive_rate_parity', 'false_negative_rate_parity'
    """
    
    # ThresholdOptimizer is a post-processing technique. 
    # It takes the PRE-TRAINED model and optimizes thresholds for each group.
    
    optimizer = ThresholdOptimizer(
        estimator=baseline_model,
        constraints=constraint,
        predict_method='predict_proba', # Needs probabilities to adjust thresholds
        prefit=True # We pass an already fitted baseline
    )
    
    # We must fit the optimizer with the sensitive features
    optimizer.fit(X_train, y_train, sensitive_features=A_train)
    
    return optimizer
