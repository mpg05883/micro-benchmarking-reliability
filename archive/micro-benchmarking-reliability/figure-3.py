import os
import json
import numpy as np
import pandas as pd
import math
import itertools
import shutil
from scipy.stats import kendalltau

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pylab import savefig
import seaborn as sns
matplotlib.use('Agg')
sns.set_style("whitegrid")
plt.rcParams["font.family"] = 'DejaVu Sans'

from plot_utils import *

mdad_threshold = 0.8

results_dir = './results-cached/results-combine-subtasks'

save_dir = 'graphs-combine-subtasks'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

techniques = ['Random (uniform)', 'Random (subtask stratified, equal)', 'Stratified sampling (confidence)', 'Diversity', 'Anchor Points', 'tinyBenchmarks']
all_num_source_models = [10, 50, 100, 150, 200, 250, 300]
benchmarks = ['mmlu', 'mmlu-pro', 'bbh', 'gpqa']
all_fraction_sampled_points = [10, 25, 50, 100, 250]
fig, axs = plt.subplots(len(benchmarks), len(all_fraction_sampled_points) + 1, figsize=(4.8*(len(all_fraction_sampled_points) + 1), 3.2 * len(benchmarks)))#, gridspec_kw={'height_ratios': [0.5, 0.5, 1]})
plt.subplots_adjust(wspace=0.25, hspace=0.4)

num_source_models_to_plot = 300

for benchmark in benchmarks:
    print(benchmark)
    all_fraction_sampled_points = [10, 25, 50, 100, 250]
    all_fraction_sampled_points_mdad = [10, 25, 50, 100, 250, 500, 1000]
    if benchmark == 'gpqa':
        all_fraction_sampled_points = [10, 25, 50, 100, 200]
        all_fraction_sampled_points_mdad = [10, 25, 50, 100, 200]

    ax_row = benchmarks.index(benchmark)

    cached_mdad_results_fn = f'{save_dir}/{benchmark}-mdad-results.csv'
    if not os.path.exists(cached_mdad_results_fn):
        print(f'Processed results files not generated for {benchmark}')
        continue
    

    df_mdad = pd.read_csv(cached_mdad_results_fn)

    tidy_data_correctness = []
    fn = f'{results_dir}/results_{benchmark}_{num_source_models_to_plot}-source-models_50-runs.json'

    with open(fn, 'r') as f:
        results = json.load(f)
    tidy_results = []
    for result in results:
        tidy_results.extend(make_tidy_results(*result + [1]))
    full_df = pd.DataFrame(tidy_results)
    full_df['Technique'] = full_df['Technique'].map(lambda t: technique_to_display_name[t])
    num_source_models = list(set(full_df['Number of source models']))[0]
    if num_source_models != num_source_models_to_plot:
        continue
    print(benchmark, num_source_models)
    for ax_col, fraction_sampled_points in enumerate(all_fraction_sampled_points):
        ax = axs[ax_row, ax_col]
        df = full_df.loc[(full_df['Fraction of sampled points'] == fraction_sampled_points) \
                            & (full_df['Seen full accuracy difference'] <= 0.2) \
                            & (full_df['Technique'].isin(techniques))]    
        sns.lineplot(ax=ax, data=df, x='Seen full accuracy difference', y='Seen correct', hue='Technique', palette=palette, linewidth=2, zorder=10, clip_on=False)
        ax.set_xlim((0, 0.2))
        ax.set_ylim((0.3, 1))
        xticks = np.arange(0, 0.21, 0.025)
        xticklabels = ['0', '2.5', '5', '7.5', '10', '12.5', '15', '17.5', '20']
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=12)
        if ax_row == len(benchmarks) - 1:
            ax.set_xlabel('$d$', fontsize=14)
        else:
            ax.set_xlabel('')
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        #if ax_row == 0:
        highlight_color = number_to_highlight_color[fraction_sampled_points]
        ax.set_title(r'$\mathbf{|S|=}$' + f'{fraction_sampled_points} examples ({fraction_sampled_points/benchmark_to_full_seen_size[benchmark]*100:.1f}%)', pad=10, fontsize=16, fontweight='bold', \
                        bbox=dict(boxstyle="round,pad=0.15", fc=highlight_color, lw=0, alpha=0.4))
        if ax_col == 0:
            ax.set_ylabel('Agreement', fontsize=18, fontweight='bold')
            ax.text(-0.3, 0.5, benchmark_to_display_name[benchmark], transform=ax.transAxes, rotation_mode='anchor', va='center', ha='center', fontsize=24, fontweight='bold', rotation=90)
        else:
            ax.set_ylabel('')
        
    df_mdad.loc[(df_mdad['Split'] == 'seen') & (df_mdad['Number of source models'] == num_source_models)].to_csv(f'{save_dir}/figure-3_{num_source_models}-source-models_{benchmark}-mdad.csv', index=False)

    #plot mdad now
    ax = axs[ax_row, -1]
    plot_df = df_mdad.loc[(df_mdad['Split'] == 'seen') \
                        & (df_mdad['Number of source models'] == num_source_models) \
                        & (df_mdad['Technique'].isin(techniques))]
    sns.lineplot(ax=ax, data=plot_df, x='Fraction of sampled points', y=f'MDAD', hue='Technique', palette=palette, marker='o', markersize=10, alpha=0.8, linewidth=4, zorder=10, clip_on=True)
    for technique in techniques:
        technique_df = plot_df.loc[(plot_df['Technique'] == technique)]
        data = sorted(zip(technique_df['Fraction of sampled points'].tolist(), technique_df['MDAD'].tolist(), technique_df['MDAD lower CI'].tolist(),  technique_df['MDAD upper CI'].tolist()))
        print(data)
        xs = [x for x, _, _, _ in data]
        y1s = [l for _, _, l, _ in data]
        y2s = [u for _, _, _, u in data]
        ax.fill_between(xs, y1s, y2s, color=palette[technique], alpha=0.2)
    ax.set_xscale('log')
    ax.set_ylim((0,0.26))
    yticks = ax.get_yticks()
    yticklabels = [f'{y * 100:.0f}' for y in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xticks(all_fraction_sampled_points_mdad)
    xticklabels = [f'{k}\n({k/benchmark_to_full_seen_size[benchmark]*100:.1f}%)' for k in all_fraction_sampled_points_mdad]
    ax.set_xticklabels(xticklabels, fontsize=7)

    # now highlight
    for xtick, xticklabel in zip(all_fraction_sampled_points_mdad, ax.get_xticklabels()):
        highlight_color = number_to_highlight_color[xtick]
        xticklabel.set_bbox(dict(boxstyle="round,pad=0.24", fc=highlight_color, lw=0, alpha=0.4))

    if ax_row == len(benchmarks) - 1:
        ax.set_xlabel('Number of examples\nselected for micro-benchmark', fontsize=12, fontweight='bold')
    else:
        ax.set_xlabel('')
    ax.set_ylabel(f'MDAD', fontsize=16, labelpad=12, fontweight='bold')
    ax.text(-0.12, 0.5, '↓', fontsize=24, fontweight='bold', va='center', ha='center', transform=ax.transAxes)
    if ax_row == 0:
        ax.set_title('MDAD', fontsize=16, fontweight='bold')
    #ax.set_title('Minimum accuracy difference for correct judgment')
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


axs[-1, 2].text(0.5, -0.4, 'Accuracy difference between models on full benchmark', transform=axs[-1, 2].transAxes, ha='center', va='center', fontsize=18, fontweight='bold')
ls = [matplotlib.patches.Patch(facecolor=palette[t]) for t in techniques]
plt.legend(ls,techniques,bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure, loc='upper center', borderaxespad=0., fontsize=18, ncols=len(techniques))#, prop={'weight':'bold'})
savefig(f'{save_dir}/figure-3.pdf', bbox_inches='tight')
plt.close()
