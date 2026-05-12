"""
Visualization module for COMPAS Fairness Analysis.
All plots use Seaborn with a consistent, publication-quality style.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# ── Global style ──────────────────────────────────────────────────────────────
PALETTE = {
    'Baseline (LR)':             '#4C72B0',
    'Demographic Parity':        '#DD8452',
    'FPR Parity':                '#55A868',
    'FNR Parity':                '#C44E52',
}
RACE_PALETTE = {'African-American': '#C44E52', 'Caucasian': '#4C72B0'}

def set_style():
    """Apply a consistent, publication-ready Seaborn style."""
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.15)
    plt.rcParams.update({
        'figure.dpi': 150,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


# ── 1. Cross-Validation Results ───────────────────────────────────────────────
def plot_cv_results(cv_df: pd.DataFrame, save_path: str = None):
    """
    Bar chart of CV metric means ± std for the baseline model.

    Parameters
    ----------
    cv_df : DataFrame with columns ['metric', 'mean', 'std']
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(
        cv_df['metric'], cv_df['mean'],
        yerr=cv_df['std'], capsize=5,
        color='#4C72B0', alpha=0.85, error_kw={'elinewidth': 1.5}
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title(
        'Baseline — Validación Cruzada Estratificada (5 folds)\n'
        'Regresión Logística con preprocesamiento en pipeline',
        fontsize=11
    )
    for bar, (_, row) in zip(bars, cv_df.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + row['std'] + 0.01,
            f"{row['mean']:.3f}", ha='center', va='bottom', fontsize=9
        )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# ── 2. Fairness Comparison (4 approaches) ────────────────────────────────────
def plot_fairness_comparison(df_comparison: pd.DataFrame, save_path: str = None):
    """
    Grouped bar chart comparing accuracy, DP difference, FPR difference and EO difference
    across the four fairness approaches.

    Parameters
    ----------
    df_comparison : DataFrame indexed by model_name with columns
                    ['accuracy', 'dp_difference', 'fpr_difference', 'eo_difference']
    """
    set_style()
    df_plot = df_comparison.reset_index().melt(
        id_vars='model_name',
        value_vars=['accuracy', 'dp_difference', 'fpr_difference', 'eo_difference'],
        var_name='Métrica', value_name='Valor'
    )
    metric_labels = {
        'accuracy':      'Accuracy',
        'dp_difference': 'DP Difference',
        'fpr_difference': 'FPR Difference',
        'eo_difference': 'EO Difference',
    }
    df_plot['Métrica'] = df_plot['Métrica'].map(metric_labels)

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(
        data=df_plot, x='model_name', y='Valor',
        hue='Métrica',
        palette=['#4C72B0', '#DD8452', '#55A868', '#C44E52'],
        ax=ax, alpha=0.88
    )
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('')
    ax.set_ylabel('Valor de la métrica')
    ax.set_title(
        'Comparativa de Accuracy y Equidad (DP, FPR, EO) entre los Cuatro Enfoques',
        fontsize=12
    )
    ax.tick_params(axis='x', rotation=10)
    ax.legend(title='Métrica', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# ── 3. Per-Group Metrics Heatmap ─────────────────────────────────────────────
def plot_group_metrics_heatmap(metrics_dict: dict, save_path: str = None):
    """
    Heatmap of FPR and FNR per racial group for each fairness approach.

    Parameters
    ----------
    metrics_dict : {model_name: DataFrame from evaluate_model()}
    """
    set_style()
    rows = []
    for model_name, df in metrics_dict.items():
        for group in df.index:
            if group == 'OVERALL':
                continue
            rows.append({
                'Modelo': model_name,
                'Grupo':  group,
                'FPR':    df.loc[group, 'fpr'],
                'FNR':    df.loc[group, 'fnr'],
            })
    df_long = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, metric in zip(axes, ['FPR', 'FNR']):
        pivot = df_long.pivot(index='Grupo', columns='Modelo', values=metric)
        sns.heatmap(
            pivot, annot=True, fmt='.3f', cmap='RdYlGn_r',
            linewidths=0.5, ax=ax, vmin=0, vmax=0.7,
            cbar_kws={'label': metric}
        )
        ax.set_title(f'{metric} por Grupo Racial y Enfoque')
        ax.set_xlabel('')
        ax.set_ylabel('Grupo Racial')
        ax.tick_params(axis='x', rotation=15)
    plt.suptitle(
        'Disparidad Racial: Falsos Positivos y Negativos por Estrategia de Equidad',
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# ── 4. Accuracy–Fairness Trade-off ───────────────────────────────────────────
def plot_accuracy_fairness_tradeoff(df_comparison: pd.DataFrame,
                                    fairness_metric: str = 'fpr_difference',
                                    save_path: str = None):
    """
    Scatter plot: accuracy (x) vs EO difference (y) per approach.
    Ideal point is top-left (high accuracy, low disparity).
    """
    set_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = list(PALETTE.values())[:len(df_comparison)]

    for (name, row), color in zip(df_comparison.iterrows(), colors):
        ax.scatter(row['accuracy'], abs(row[fairness_metric]),
                   s=140, color=color, zorder=3, label=name)
        ax.annotate(name, (row['accuracy'], abs(row[fairness_metric])),
                    textcoords='offset points', xytext=(6, 4), fontsize=8)

    ax.set_xlabel('Accuracy Global')
    ax.set_ylabel(f'|{fairness_metric} Difference|')
    ax.set_title(
        f'Trade-off: Accuracy vs. Disparidad de Equidad\n'
        '(esquina inferior-derecha = óptimo)',
        fontsize=11
    )
    ax.legend(title='Enfoque', bbox_to_anchor=(1.3, 1), loc='upper left',
              fontsize=8)
    ax.annotate('Óptimo', xy=(ax.get_xlim()[1], 0),
                xytext=(ax.get_xlim()[1] - 0.04, 0.03),
                arrowprops=dict(arrowstyle='->', color='green'),
                color='green', fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# ── 5. Confusion Matrices (side by side) ─────────────────────────────────────
def plot_confusion_matrices(y_true, predictions_dict: dict,
                             save_path: str = None):
    """
    Side-by-side normalized confusion matrices for each model.

    Parameters
    ----------
    predictions_dict : {model_name: y_pred}
    """
    set_style()
    n = len(predictions_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, predictions_dict.items()):
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        sns.heatmap(
            cm, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=['No reincide', 'Reincide'],
            yticklabels=['No reincide', 'Reincide'],
            ax=ax, linewidths=0.5, cbar=False
        )
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Verdad')

    plt.suptitle('Matrices de Confusión Normalizadas por Enfoque',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# ── 6. Robustness Summary ────────────────────────────────────────────────────
def plot_robustness_summary(robustness_results: dict, save_path: str = None):
    """
    Horizontal bar chart comparing clean vs adversarial accuracy.

    Parameters
    ----------
    robustness_results : dict from evaluate_robustness()
    """
    set_style()
    labels = ['Accuracy limpia', 'Accuracy adversaria']
    values = [robustness_results['accuracy_baseline'],
              robustness_results['accuracy_adversarial']]
    colors = ['#4C72B0', '#C44E52']

    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(labels, values, color=colors, alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Accuracy')
    ax.set_title(
        f'Robustez Adversaria — Ataque FGM (ε=0.1)\n'
        f'Caída: {robustness_results["robustness_drop"]:.3f}',
        fontsize=11
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
