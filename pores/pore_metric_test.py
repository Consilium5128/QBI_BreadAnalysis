import numpy as np
import pandas as pd

from scipy.stats import wilcoxon, ks_2samp, f_oneway, entropy
from scipy.spatial.distance import directed_hausdorff

def create_metric_df(stats, pvals, samples, pairs):
    '''
    Create 2 dataframes - one for statistic value and another for p-values for a given list of pvals and stats
    '''
    
    df_stat = pd.DataFrame(index=samples, columns=samples)
    df_p = pd.DataFrame(index=samples, columns=samples)
    
    for (i, j), ks, p in zip(pairs, stats, pvals):
        df_stat.loc[i, j] = ks
        df_stat.loc[j, i] = ks
        df_p.loc[i, j] = p
        df_p.loc[j, i] = p

    df_stat = df_stat.astype(float)
    df_p = df_p.astype(float)
    return df_stat, df_p

def run_pairwise_ks_test(sample_metrics_dict, metrics_to_compare, samples, sampling_size):
    '''
    Run pairwise KS tests
    Return a list of dataframes with each dataframe corresponding to the given metric
    '''
    
    ks_stats_list = []
    p_values_list = []
    pairs = []
    print(metrics_to_compare)

    for m, metric in enumerate(metrics_to_compare):
        ks_stats_list_metric = []
        p_values_list_metric = []

        for i,sample1 in enumerate(samples):
            for j,sample2 in enumerate(samples):
                if j>i:
                    if m == 0:
                        pairs.append((sample1, sample2))
                    w_stat, p_value = ks_2samp(sample_metrics_dict[sample1][metric].sample(sampling_size), sample_metrics_dict[sample2][metric].sample(sampling_size))
                    ks_stats_list_metric.append(w_stat)
                    p_values_list_metric.append(p_value)
                    #print(f"KS-Rank Test between {bread1} and {bread2}: KS-statistic = {w_stat:.3f}, p-value = {p_value:.4e}")
                    #print(f'p-value after Bonferonni correction (*{sampling_size}): {p_value*3000:.3e} (0.05)\n')
        ks_stats_list.append(ks_stats_list_metric)
        p_values_list.append(p_values_list_metric)
    
    dfs_ks_p = [create_metric_df(ks_stats, p_vals, samples, pairs) for ks_stats, p_vals in zip(ks_stats_list, p_values_list)]
    
    return dfs_ks_p, ks_stats_list, p_values_list

def run_pairwise_wilcoxon_test(sample_metrics_dict, metrics_to_compare, samples, sampling_size, normalize):
    '''
    Run pairwise KS tests
    Return a list of dataframes with each dataframe corresponding to the given metric
    '''
    
    ks_stats_list = []
    p_values_list = []
    pairs = []

    for m, metric in enumerate(metrics_to_compare):
        ks_stats_list_metric = []
        p_values_list_metric = []

        for i,sample1 in enumerate(samples):
            for j,sample2 in enumerate(samples):
                if j>i:
                    if m == 0:
                        pairs.append((sample1, sample2))
                    w_stat, p_value = wilcoxon(sample_metrics_dict[sample1][metric].sample(sampling_size), sample_metrics_dict[sample2][metric].sample(sampling_size))
                    if normalize:
                        w_stat /= sampling_size
                    ks_stats_list_metric.append(w_stat)
                    p_values_list_metric.append(p_value)
                    #print(f"KS-Rank Test between {bread1} and {bread2}: KS-statistic = {w_stat:.3f}, p-value = {p_value:.4e}")
                    #print(f'p-value after Bonferonni correction (*{sampling_size}): {p_value*3000:.3e} (0.05)\n')
        ks_stats_list.append(ks_stats_list_metric)
        p_values_list.append(p_values_list_metric)
    
    dfs_ks_p = [create_metric_df(ks_stats, p_vals, samples, pairs) for ks_stats, p_vals in zip(ks_stats_list, p_values_list)]
    
    return dfs_ks_p, ks_stats_list, p_values_list

def dice_coefficient(image1, image2):
    intersection = np.sum((image1 > 0) & (image2 > 0))
    sum_of_areas = np.sum(image1 > 0) + np.sum(image2 > 0)
    dice = 2 * intersection / sum_of_areas
    return dice

def jaccard_index(image1, image2):
    intersection = np.sum((image1 > 0) & (image2 > 0))
    union = np.sum((image1 > 0) | (image2 > 0))
    jaccard = intersection / union
    return jaccard

def hausdorff_distance(image1, image2):
    coords1 = np.argwhere(image1 > 0)
    coords2 = np.argwhere(image2 > 0)
    hausdorff = max(directed_hausdorff(coords1, coords2)[0], directed_hausdorff(coords2, coords1)[0])
    return hausdorff

def volume_comparison(image1, image2):
    volume1 = np.sum(image1 > 0)
    volume2 = np.sum(image2 > 0)
    return volume1, volume2