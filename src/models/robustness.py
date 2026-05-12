import numpy as np
import pandas as pd
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import FastGradientMethod
from sklearn.pipeline import Pipeline

def evaluate_robustness(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluates the robustness of the model using an evasion attack (FGM).
    """
    # X_test needs to be transformed by the pipeline's preprocessor
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']
    
    X_test_transformed = preprocessor.transform(X_test)
    
    # Wrap the sklearn classifier in an ART classifier
    art_classifier = SklearnClassifier(model=classifier)
    
    # Baseline accuracy
    predictions = art_classifier.predict(X_test_transformed)
    accuracy_baseline = np.sum(np.argmax(predictions, axis=1) == y_test.values) / len(y_test)
    
    # Craft adversarial samples
    attack = FastGradientMethod(estimator=art_classifier, eps=0.1)
    X_test_adv = attack.generate(x=X_test_transformed)
    
    # Adversarial accuracy
    predictions_adv = art_classifier.predict(X_test_adv)
    accuracy_adv = np.sum(np.argmax(predictions_adv, axis=1) == y_test.values) / len(y_test)
    
    return {
        'accuracy_baseline': accuracy_baseline,
        'accuracy_adversarial': accuracy_adv,
        'robustness_drop': accuracy_baseline - accuracy_adv
    }
