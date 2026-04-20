import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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
    
    return {
        'demographic_parity_difference': dp_diff,
        'equalized_odds_difference': eo_diff
    }
