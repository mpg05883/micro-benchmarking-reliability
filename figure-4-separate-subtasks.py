import os
import json
import numpy as np
import pandas as pd
import math
import itertools
from scipy.stats import kendalltau

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pylab import savefig
import seaborn as sns
matplotlib.use('Agg')
sns.set_style("whitegrid")
plt.rcParams["font.family"] = 'DejaVu Sans'

from plot_utils import *

save_dir = 'graphs-separate-subtasks'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

mdad_correctness = 0.8

num_source_models = 300
benchmarks = ['mmlu', 'mmlu-pro', 'bbh', 'gpqa']

f, axs = plt.subplots(3, len(benchmarks), figsize=(6.0*len(benchmarks), 3.0 * 3))
plt.subplots_adjust(wspace=0.15, hspace=0.35)    

techniques = ['Random (uniform)', 'Random (subtask stratified, equal)', 'Stratified sampling (confidence)', 'Diversity', 'Anchor Points', 'tinyBenchmarks']

for col, benchmark in enumerate(benchmarks):

    all_fraction_sampled_points = [0.02, 0.04, 0.08, 0.16, 0.24, 0.32, 0.4]

    cached_mdad_results_fn = f'{save_dir}/{benchmark}-mdad-results.csv'
    cached_estimation_results_fn = f'{save_dir}/{benchmark}-estimation-results.csv'
    cached_correctness_results_fn = f'{save_dir}/{benchmark}-correctness-results.csv'

    if not os.path.exists(cached_mdad_results_fn) or not os.path.exists(cached_estimation_results_fn) \
        or not os.path.exists(cached_mdad_results_fn):
        print(f'Processed results files not generated for {benchmark}')
        continue
        
    df_mdad = pd.read_csv(cached_mdad_results_fn)
    full_estimation_df = pd.read_csv(cached_estimation_results_fn)
    df_correctness = pd.read_csv(cached_correctness_results_fn)

    xticklabels = []
    for k in all_fraction_sampled_points:
        xticklabels.append(f'{k*100:.0f}%')
    
    split = 'seen'

    ax = axs[0, col]
    plot_df = full_estimation_df.loc[(full_estimation_df['Number of source models'] == num_source_models) \
                                & (full_estimation_df['Technique'].isin(techniques))]
    sns.lineplot(ax=ax, data=plot_df, x='Fraction of sampled points', y=f"Mean estimation error against {split} accuracies", hue="Technique", palette=palette, marker='o', markersize=10, alpha=0.8, linewidth=4, zorder=10, clip_on=True)
    ax.set_ylim((0, 0.3))
    ax.set_xticks(all_fraction_sampled_points)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('')
    ax.set_ylabel('')
    yticks = ax.get_yticks()
    yticklabels = [f'{y * 100:.0f}' for y in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylim((0, 0.3))
    if col == 0:
        ax.set_ylabel(f'Estimation error', fontsize=14, labelpad=20, fontweight='bold')
        ax.text(-0.1, 0.5, '↓', fontsize=24, fontweight='bold', va='center', ha='center', transform=ax.transAxes)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.set_title(benchmark_to_display_name[benchmark], fontsize=24, fontweight='bold', pad=15)

    ax = axs[1, col]
    plot_df = full_estimation_df.loc[(full_estimation_df['Number of source models'] == num_source_models) \
                                & (full_estimation_df['Technique'].isin(techniques))]
    sns.lineplot(ax=ax, data=plot_df, x='Fraction of sampled points', y=f"Kendall tau correlation against {split} accuracies", hue="Technique", palette=palette, marker='o', markersize=10, alpha=0.8, linewidth=4, zorder=10, clip_on=True)
    ax.set_ylim((0, 1))
    ax.set_xticks(all_fraction_sampled_points)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('')
    ax.set_ylabel('')
    if col == 0:
        ax.set_ylabel(f'Rank correlation', fontsize=14, labelpad=20, fontweight='bold')
        ax.text(-0.11, 0.5, '↑', fontsize=24, fontweight='bold', va='center', ha='center', transform=ax.transAxes)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    ax = axs[2, col]
    plot_df = df_mdad.loc[(df_mdad['Split'] == split) \
                         & (df_mdad['Number of source models'] == num_source_models) \
                         & (df_mdad['Technique'].isin(techniques))]
    sns.lineplot(ax=ax, data=plot_df, x='Fraction of sampled points', y=f'MDAD', hue='Technique', palette=palette, marker='o', markersize=10, alpha=0.8, linewidth=4, zorder=10, clip_on=True)
    for technique in techniques:
        technique_df = plot_df.loc[(plot_df['Technique'] == technique)]
        data = sorted(zip(technique_df['Fraction of sampled points'].tolist(), technique_df['MDAD'].tolist(), technique_df['MDAD lower CI'].tolist(),  technique_df['MDAD upper CI'].tolist()))
        xs = [x for x, _, _, _ in data]
        y1s = [min(l, m-(u-m)) for _, m, l, u in data]
        y2s = [max(m+(m-l), u) for _, m, l, u in data]
        ax.fill_between(xs, y1s, y2s, color=palette[technique], alpha=0.2)
    ax.set_ylim((0,0.55))
    yticks = ax.get_yticks()
    yticklabels = [f'{y * 100:.0f}' for y in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xticks(all_fraction_sampled_points)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('')
    if col == 0:
        ax.set_ylabel(f'MDAD', fontsize=14, labelpad=20, fontweight='bold')
        ax.text(-0.1, 0.5, '↓', fontsize=24, fontweight='bold', va='center', ha='center', transform=ax.transAxes)
    else:
        ax.set_ylabel('')
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
f.text(0.5, 0.03, 'Fraction of examples selected for micro-benchmark from each subtask', ha='center', va='center', fontsize=18, fontweight='bold')
    

ls = [matplotlib.patches.Patch(facecolor=palette[t]) for t in techniques]
plt.legend(ls,techniques,bbox_to_anchor=(0.5, 0), loc='upper center', bbox_transform=f.transFigure, borderaxespad=0., fontsize=18, ncols=3)#len(techniques))#, prop={'weight':'bold'})
savefig(f'{save_dir}/figure-4_separate-subtasks.pdf', bbox_inches='tight')