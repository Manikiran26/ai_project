"""
evaluate.py - Model Evaluation Module
======================================
This module evaluates all trained models using:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1 Score (weighted)
- Classification Report
- Confusion Matrix

Results are saved to CSV and confusion matrix plots are generated.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(y_test, y_pred, model_name, feature_method):
    """
    Evaluate a single model's predictions.

    Args:
        y_test: True labels
        y_pred: Predicted labels
        model_name (str): Name of the model
        feature_method (str): Feature extraction method used

    Returns:
        dict: Evaluation metrics
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    return {
        'Feature Method': feature_method,
        'Model': model_name,
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1 Score': round(f1, 4),
    }


def print_classification_report(y_test, y_pred, model_name, feature_method):
    """
    Print detailed classification report for a model.
    """
    print(f"\n{'-' * 50}")
    print(f"Classification Report: {model_name} ({feature_method})")
    print(f"{'-' * 50}")
    print(classification_report(y_test, y_pred, zero_division=0))


def plot_confusion_matrix(y_test, y_pred, model_name, feature_method, save_dir):
    """
    Generate and save a confusion matrix heatmap.

    Args:
        y_test: True labels
        y_pred: Predicted labels
        model_name (str): Name of the model
        feature_method (str): Feature method used
        save_dir (str): Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)

    # Get unique labels sorted
    labels = sorted(y_test.unique())

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix: {model_name} ({feature_method})', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the plot
    filename = f"cm_{feature_method.lower().replace('-', '')}_{model_name.lower().replace(' ', '_')}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Confusion matrix saved: {filename}")


def evaluate_all_models(results, results_dir):
    """
    Evaluate all trained models and save results.

    Args:
        results (dict): Dictionary with model results from training
        results_dir (str): Directory to save evaluation results

    Returns:
        pd.DataFrame: Comparison table of all models
    """
    os.makedirs(results_dir, exist_ok=True)
    cm_dir = os.path.join(results_dir, 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("STEP 3: MODEL EVALUATION")
    print("=" * 60)

    all_metrics = []

    for key, result in results.items():
        y_test = result['y_test']
        y_pred = result['y_pred']
        model_name = result['model_name']
        feature_method = result['feature_method']

        # Calculate metrics
        metrics = evaluate_model(y_test, y_pred, model_name, feature_method)
        all_metrics.append(metrics)

        # Print classification report
        print_classification_report(y_test, y_pred, model_name, feature_method)

        # Generate confusion matrix plot
        plot_confusion_matrix(y_test, y_pred, model_name, feature_method, cm_dir)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(all_metrics)
    comparison_df = comparison_df.sort_values('F1 Score', ascending=False).reset_index(drop=True)

    # Save to CSV
    csv_path = os.path.join(results_dir, 'model_comparison.csv')
    comparison_df.to_csv(csv_path, index=False)
    print(f"\nModel comparison saved to: {csv_path}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("MODEL COMPARISON TABLE")
    print("=" * 60)
    print(comparison_df.to_string(index=False))

    return comparison_df


def find_best_model(comparison_df):
    """
    Find the best model based on F1 Score.

    Args:
        comparison_df (pd.DataFrame): Comparison table

    Returns:
        tuple: (best_feature_method, best_model_name, best_f1_score)
    """
    best_row = comparison_df.loc[comparison_df['F1 Score'].idxmax()]

    best_feature = best_row['Feature Method']
    best_model = best_row['Model']
    best_f1 = best_row['F1 Score']
    best_accuracy = best_row['Accuracy']

    print("\n" + "=" * 60)
    print("BEST MODEL SELECTION")
    print("=" * 60)
    print(f"  Best Model:                  {best_model}")
    print(f"  Best Feature Representation: {best_feature}")
    print(f"  F1 Score:                    {best_f1}")
    print(f"  Accuracy:                    {best_accuracy}")

    return best_feature, best_model, best_f1


def plot_model_comparison(comparison_df, results_dir):
    """
    Create comparison bar charts for all models.

    Args:
        comparison_df (pd.DataFrame): Comparison table
        results_dir (str): Directory to save plots
    """
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Create a combined label for plotting
    comparison_df = comparison_df.copy()
    comparison_df['Label'] = comparison_df['Model'] + '\n(' + comparison_df['Feature Method'] + ')'

    # ----- Plot 1: Accuracy Comparison -----
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette('viridis', len(comparison_df))
    bars = ax.barh(comparison_df['Label'], comparison_df['Accuracy'], color=colors)
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title('Model Comparison: Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.0)

    # Add value labels on bars
    for bar, val in zip(bars, comparison_df['Accuracy']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'accuracy_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ----- Plot 2: F1 Score Comparison -----
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(comparison_df['Label'], comparison_df['F1 Score'], color=colors)
    ax.set_xlabel('F1 Score', fontsize=12)
    ax.set_title('Model Comparison: F1 Score', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.0)

    for bar, val in zip(bars, comparison_df['F1 Score']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'f1_score_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ----- Plot 3: All Metrics Grouped Bar Chart -----
    fig, ax = plt.subplots(figsize=(14, 7))
    metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    x = np.arange(len(comparison_df))
    width = 0.2

    for i, metric in enumerate(metrics_cols):
        bars = ax.bar(x + i * width, comparison_df[metric], width, label=metric)

    ax.set_xlabel('Model (Feature Method)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison: All Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(comparison_df['Label'], rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'all_metrics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ----- Plot 4: BoW vs TF-IDF Comparison -----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, method in enumerate(['BoW', 'TF-IDF']):
        subset = comparison_df[comparison_df['Feature Method'] == method]
        ax = axes[idx]
        x = np.arange(len(subset))
        width = 0.35

        ax.bar(x - width/2, subset['Accuracy'], width, label='Accuracy', color='steelblue')
        ax.bar(x + width/2, subset['F1 Score'], width, label='F1 Score', color='coral')

        ax.set_xlabel('Model', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(f'{method} Feature Extraction', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(subset['Model'], rotation=30, ha='right', fontsize=9)
        ax.legend()
        ax.set_ylim(0, 1.1)

    plt.suptitle('BoW vs TF-IDF: Model Performance', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'bow_vs_tfidf.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nComparison plots saved to: {plots_dir}")


def plot_class_distribution(df, results_dir):
    """
    Plot class distribution of the dataset.

    Args:
        df (pd.DataFrame): The dataset
        results_dir (str): Directory to save the plot
    """
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    target_col = 'Ticket Subject' if 'Ticket Subject' in df.columns else 'category'
    category_counts = df[target_col].value_counts()

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = sns.color_palette('viridis', len(category_counts))
    bars = ax.bar(category_counts.index, category_counts.values, color=colors)

    ax.set_xlabel('Ticket Subject Category', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Dataset: Class Distribution (16 Support Categories)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)

    # Add value labels on top of bars
    for bar, val in zip(bars, category_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'class_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Class distribution plot saved to: {plots_dir}")

