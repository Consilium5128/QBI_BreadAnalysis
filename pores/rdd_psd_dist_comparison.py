import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp, entropy
from itertools import combinations

def kl_divergence(p, q):
    p = np.array(p)
    q = np.array(q)
    return entropy(p, q)

def ks_test(cdf1, cdf2):
    return ks_2samp(cdf1, cdf2)

def calculate_pairwise_metrics(samples, metric_fn):
    keys = list(samples.keys())
    n = len(keys)
    results = np.zeros((n, n))
    p_values = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        result, p_value = metric_fn(samples[keys[i]], samples[keys[j]])
        results[i, j] = result
        results[j, i] = result
        p_values[i, j] = p_value
        p_values[j, i] = p_value
    return results, p_values, keys

def plot_heatmaps(metrics, pvalues, metric_names, sample_names, title, cmap="viridis"):
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    for ax, metric, pvalue, name in zip(axes, metrics, pvalues, metric_names):
        sns.heatmap(metric, xticklabels=sample_names, yticklabels=sample_names, vmin=0.0, vmax=1.0, annot=True, fmt=".2f", cmap=cmap, ax=ax, cbar_kws={'label': 'Statistic'})
        for i in range(len(sample_names)):
            for j in range(len(sample_names)):
                if i != j:
                    ax.text(j+0.5, i+0.75, f'\n(p={pvalue[i, j]:.2e})', color='black', ha='center', va='center', fontsize=8)
        ax.set_title(name)
    #fig.colorbar(axes[0].collections[0], ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    plt.suptitle(title)
    plt.show()