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

save_dir = 'graphs-combine-subtasks'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

mdad_correctness = 0.8

num_source_models = 300
benchmarks = ['mmlu', 'mmlu-pro', 'bbh', 'gpqa']

f, axs = plt.subplots(3, len(benchmarks), figsize=(6.0*len(benchmarks), 3.0 * 3))#, gridspec_kw={'height_ratios': [0.5, 0.5, 1]})
plt.subplots_adjust(wspace=0.15, hspace=0.35)    

techniques = ['Random (uniform)', 'Random (subtask stratified, equal)', 'Stratified sampling (confidence)', 'Diversity', 'Anchor Points', 'tinyBenchmarks']

for ax_col, benchmark in enumerate(benchmarks):

    all_fraction_sampled_points = [10, 25, 50, 100, 250, 500, 1000]
    if benchmark == 'gpqa':
        all_fraction_sampled_points = [10, 25, 50, 100, 200]

    for ax_row, mdad_threshold in enumerate(['0.7', '0.8', '0.9']):
        if mdad_threshold in ['0.7', '0.9']:
            cached_mdad_results_fn = f'graphs-combine-subtasks_{mdad_threshold}-mdad-correctness/{benchmark}-mdad-results.csv'
        else:
            cached_mdad_results_fn = f'graphs-combine-subtasks/{benchmark}-mdad-results.csv'

        if not os.path.exists(cached_mdad_results_fn):
            print(f'Processed results files not generated for {benchmark}')
            continue
            
        df_mdad = pd.read_csv(cached_mdad_results_fn)
        
        xticklabels = [f'{k}\n({k/benchmark_to_full_seen_size[benchmark]*100:.1f}%)' for k in all_fraction_sampled_points]
        
        split = 'seen'

        ax = axs[ax_row, ax_col]
        plot_df = df_mdad.loc[(df_mdad['Split'] == split) \
                            & (df_mdad['Number of source models'] == num_source_models) \
                            & (df_mdad['Technique'].isin(techniques))]
        sns.lineplot(ax=ax, data=plot_df, x='Fraction of sampled points', y=f'MDAD', hue='Technique', palette=palette, marker='o', markersize=10, alpha=0.8, linewidth=4, zorder=10, clip_on=True)
        for technique in techniques:
            technique_df = plot_df.loc[(plot_df['Technique'] == technique)]
            data = sorted(zip(technique_df['Fraction of sampled points'].tolist(), technique_df['MDAD'].tolist(), technique_df['MDAD lower CI'].tolist(),  technique_df['MDAD upper CI'].tolist()))
            xs = [x for x, _, _, _ in data]
            y1s = [l for _, _, l, _ in data]
            y2s = [u for _, _, _, u in data]
            ax.fill_between(xs, y1s, y2s, color=palette[technique], alpha=0.2)
        ax.set_xscale('log')
        ax.set_ylim((0,0.35))
        yticks = ax.get_yticks()
        yticklabels = [f'{y * 100:.0f}' for y in yticks]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_ylim((0,0.35))
        ax.set_xticks(all_fraction_sampled_points)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel('')
        if ax_col == 0:
            ax.set_ylabel(f'MDAD, {mdad_threshold} threshold', fontsize=14, labelpad=20, fontweight='bold')
            ax.text(-0.1, 0.5, '↓', fontsize=24, fontweight='bold', va='center', ha='center', transform=ax.transAxes)
        else:
            ax.set_ylabel('')
        if ax_row == 0:
            ax.set_title(benchmark_to_display_name[benchmark], fontsize=24, fontweight='bold', pad=15)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
f.text(0.5, 0.03, 'Number of examples selected for micro-benchmark', ha='center', va='center', fontsize=18, fontweight='bold')
    

ls = [matplotlib.patches.Patch(facecolor=palette[t]) for t in techniques]
plt.legend(ls,techniques,bbox_to_anchor=(0.5, 0), loc='upper center', bbox_transform=f.transFigure, borderaxespad=0., fontsize=18, ncols=3)#len(techniques))#, prop={'weight':'bold'})
savefig(f'{save_dir}/figure-different-thresholds-mdad.pdf', bbox_inches='tight')