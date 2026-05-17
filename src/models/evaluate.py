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


def cross_validate_all_models(X_train: pd.DataFrame, y_train: pd.Series, A_train: pd.Series, preprocessor, seed: int = 80) -> pd.DataFrame:
    """
    Evaluates Baseline and Fair Models using 5-fold Stratified CV.
    Returns a DataFrame with AUC-ROC and F1 means and stds for each model.
    """
    from sklearn.metrics import f1_score, roc_auc_score
    from .train_model import train_baseline, train_fair_model
    import numpy as np

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    results_cv = {'Baseline': {'f1': [], 'auc': []}, 
                  'Demographic Parity': {'f1': [], 'auc': []},
                  'FPR Parity': {'f1': [], 'auc': []},
                  'Equalized Odds': {'f1': [], 'auc': []}}

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        A_tr, A_val = A_train.iloc[train_idx], A_train.iloc[val_idx]
        
        base_model = train_baseline(X_tr, y_tr, preprocessor, seed)
        y_pred_proba_base = base_model.predict_proba(X_val)[:, 1]
        y_pred_base = base_model.predict(X_val)
        results_cv['Baseline']['f1'].append(f1_score(y_val, y_pred_base))
        results_cv['Baseline']['auc'].append(roc_auc_score(y_val, y_pred_proba_base))
        
        fm_dp = train_fair_model(base_model, X_tr, y_tr, A_tr, constraint='demographic_parity')
        fm_fpr = train_fair_model(base_model, X_tr, y_tr, A_tr, constraint='false_positive_rate_parity')
        fm_eo = train_fair_model(base_model, X_tr, y_tr, A_tr, constraint='equalized_odds')
        
        y_pred_dp_val = fm_dp.predict(X_val, sensitive_features=A_val)
        y_pred_fpr_val = fm_fpr.predict(X_val, sensitive_features=A_val)
        y_pred_eo_val = fm_eo.predict(X_val, sensitive_features=A_val)
        
        results_cv['Demographic Parity']['f1'].append(f1_score(y_val, y_pred_dp_val))
        results_cv['Demographic Parity']['auc'].append(roc_auc_score(y_val, y_pred_dp_val))
        results_cv['FPR Parity']['f1'].append(f1_score(y_val, y_pred_fpr_val))
        results_cv['FPR Parity']['auc'].append(roc_auc_score(y_val, y_pred_fpr_val))
        results_cv['Equalized Odds']['f1'].append(f1_score(y_val, y_pred_eo_val))
        results_cv['Equalized Odds']['auc'].append(roc_auc_score(y_val, y_pred_eo_val))

    rows = []
    for model_name, metrics in results_cv.items():
        rows.append({
            'Model': model_name,
            'AUC-ROC': f"{np.mean(metrics['auc']):.4f} ± {np.std(metrics['auc']):.4f}",
            'F1': f"{np.mean(metrics['f1']):.4f} ± {np.std(metrics['f1']):.4f}"
        })
    return pd.DataFrame(rows).set_index('Model')
