import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple, List

def get_data_splits(
    df: pd.DataFrame, 
    target_col: str, 
    sensitive_col: str, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits the data into train and test sets BEFORE any feature engineering 
    to prevent data leakage.
    
    Returns:
        X_train, X_test, y_train, y_test, A_train, A_test
        Where A is the sensitive attribute (race)
    """
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # We also need to keep track of the sensitive attribute for fairness evaluation
    A = df[sensitive_col]
    
    # Split the data
    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X, y, A, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, A_train, A_test

def build_preprocessor(
    numeric_features: List[str], 
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer for preprocessing.
    This ensures that fitting (e.g. mean, standard dev, distinct categories) 
    happens ONLY on the training data.
    """
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # handle_unknown='ignore' prevents errors if test set has new categories
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Drop columns not explicitly listed (like 'c_charge_desc' which has too many categories)
    )
    
    return preprocessor
