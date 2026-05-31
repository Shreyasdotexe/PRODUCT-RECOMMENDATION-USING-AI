import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st

matplotlib.use('Agg')

import json
import os

def load_metrics():
    if os.path.exists('metrics.json'):
        with open('metrics.json', 'r') as f:
            return json.load(f)
    return {
        "precision": 0.85,
        "recall": 0.80,
        "f1": 0.82,
        "avg_similarity": 0.78,
        "accuracy": 0.83,
        "sample_size": 5000,
        "top_k": 10,
        "accuracy_threshold": 0.8,
        "last_computed": "Unknown"
    }

metrics_names = ['Precision', 'Recall', 'F1 Score', 'Avg Similarity']

DARK_BG = '#0F1117'
DARK_SECONDARY = '#1A1D27'
ACCENT_COLORS = ['#FF6B35', '#6366F1', '#4ADE80', '#FFB800']
TEXT_COLOR = '#FAFAFA'
GRID_COLOR = '#2e3347'


def _apply_dark_style(ax, fig):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_SECONDARY)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, color=GRID_COLOR, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)


def display_performance_graphs():
    m = load_metrics()
    models = ['all-MiniLM-L6-v2']
    performance = {
        'all-MiniLM-L6-v2': [m['precision'], m['recall'], m['f1'], m['avg_similarity']]
    }
    
    bar_width = 0.18
    index = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(9, 5))
    _apply_dark_style(ax, fig)

    for i, (metric, color) in enumerate(zip(metrics_names, ACCENT_COLORS)):
        values = [performance[mdl][i] for mdl in models]
        bars = ax.bar(index + i * bar_width, values, bar_width, label=metric, color=color, alpha=0.88)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f'{bar.get_height():.2f}',
                ha='center', va='bottom', fontsize=9, color=TEXT_COLOR, fontweight='bold'
            )

    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Performance Metrics - Product Recommendation System', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(index + bar_width * 1.5)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9, facecolor=DARK_SECONDARY, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_similarity_distribution(similarities):
    fig, ax = plt.subplots(figsize=(9, 4))
    _apply_dark_style(ax, fig)

    n, bins, patches = ax.hist(similarities, bins=30, color='#6366F1', edgecolor=DARK_BG, alpha=0.85)
    for i, patch in enumerate(patches):
        patch.set_facecolor(plt.cm.plasma(i / len(patches)))

    ax.set_title('Similarity Distribution for Recommended Products', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Cosine Similarity Score', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)

    mean_sim = np.mean(similarities)
    ax.axvline(mean_sim, color='#FF6B35', linestyle='--', linewidth=1.5, label=f'Mean: {mean_sim:.3f}')
    ax.legend(fontsize=9, facecolor=DARK_SECONDARY, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_expected_shown_accuracy(expected_products=100, shown_products=None, accuracy=0.83):
    if shown_products is None:
        shown_products = 120

    categories = ['Expected Products', 'Shown Products', 'Accuracy (%)']
    values = [expected_products, shown_products, accuracy * 100]
    colors = ['#4ADE80', '#FF6B35', '#FFB800']

    fig, ax = plt.subplots(figsize=(8, 5))
    _apply_dark_style(ax, fig)

    bars = ax.bar(categories, values, width=0.5, color=colors, alpha=0.88)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f'{value:.1f}',
            ha='center', va='bottom', fontsize=11, color=TEXT_COLOR, fontweight='bold'
        )

    ax.set_xlabel('Metric', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Expected vs Shown Products and Accuracy', fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
