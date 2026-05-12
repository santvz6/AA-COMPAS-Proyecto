import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference, false_positive_rate, false_negative_rate


def evaluate_model(y_true, y_pred, sensitive_features) -> pd.DataFrame:
    """
    Evaluates model performance overall and by sensitive group.
    
    Returns a DataFrame with metrics by group.
    """
    
    # Custom metric functions for FPR and FNR to be used with MetricFrame
    def fpr(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else 0

    def fnr(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        return fn / (fn + tp) if (fn + tp) > 0 else 0
        
    metrics_dict = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
        'fpr': fpr,
        'fnr': fnr
    }
    
    metric_frame = MetricFrame(
        metrics=metrics_dict,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
    
    # Get by group
    df_metrics = metric_frame.by_group
    
    # Add overall metrics as a new row
    df_metrics.loc['OVERALL'] = metric_frame.overall
    
    return df_metrics

def get_fairness_summary(y_true, y_pred, sensitive_features) -> dict:
    """
    Calculates scalar fairness metrics.
    """
    dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
    eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)
    
    # Para la diferencia de FPR, usamos MetricFrame y calculamos la diferencia entre grupos
    def fpr_metric(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else 0

    mf = MetricFrame(
        metrics=fpr_metric,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
    fpr_diff = mf.difference(method='between_groups')
    
    return {
        'demographic_parity_difference': dp_diff,
        'equalized_odds_difference': eo_diff,
        'fpr_difference': fpr_diff
    }

def generate_comparative_report(results_list: list) -> pd.DataFrame:
    """
    Generates a consolidated table comparing different models/interventions.
    results_list should contain dicts with 'model_name', 'accuracy', 'dp_diff', 'eo_diff'.
    """
    df_report = pd.DataFrame(results_list)
    df_report = df_report.set_index('model_name')
    return df_report


def cross_validate_baseline(pipeline, X_train: pd.DataFrame,
                             y_train: pd.Series,
                             n_splits: int = 5,
                             random_state: int = 80) -> pd.DataFrame:
    """
    Stratified K-Fold cross-validation of the baseline pipeline.

    Justification for CV:
    - Provides unbiased estimate of generalization error.
    - Stratified split preserves class imbalance in each fold.
    - Reports mean ± std, required for academic comparisons.

    Returns
    -------
    DataFrame with columns ['metric', 'mean', 'std'] for each scoring metric.
    """
    scoring = {
        'accuracy':  'accuracy',
        'f1':        'f1',
        'roc_auc':   'roc_auc',
        'precision': 'precision',
        'recall':    'recall',
    }
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=random_state)
    results = cross_validate(pipeline, X_train, y_train,
                             cv=cv, scoring=scoring,
                             return_train_score=False)

    rows = []
    for metric, key in scoring.items():
        scores = results[f'test_{metric}']
        rows.append({
            'metric': metric.upper().replace('_', '-'),
            'mean':   round(scores.mean(), 4),
            'std':    round(scores.std(), 4),
        })
    return pd.DataFrame(rows)
