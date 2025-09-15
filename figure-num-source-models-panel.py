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

save_dir = 'graphs-combine-subtasks'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

mdad_correctness = 0.8


palette = {
    'Random (uniform)': '#80b1d3',
    'Random (subtask stratified, equal)': '#bebada',
    'Random (subtask stratified, proportional)': '#fccde5',
    'Anchor Points': '#fdb462',
    'Anchor Points (predictor)': '#b3de69',
    'Diversity': '#8dd3c7',
    'tinyBenchmarks': '#fb8072',
    'Stratified sampling (confidence)': '#bc80bd',
}
technique_to_display_name = {
    "Random": 'Random (uniform)',
    "Random_Subtask_Stratified_Equal": 'Random (subtask stratified, equal)',
    "Random_Subtask_Stratified_Proportional": 'Random (subtask stratified, proportional)',
    "Anchor_Points_Weighted": 'Anchor Points',
    "Anchor_Points_Predictor": 'Anchor Points (predictor)',
    "DPP": 'Diversity',
    "tinyBenchmarks": 'tinyBenchmarks',
    'Stratified_Random_Sampling': 'Stratified sampling (confidence)',
}


benchmark_to_ylim_estimation = {
    'gpqa': (0, 0.25),
    'bbh': (0, 0.25),
    'mmlu-pro': (0, 0.25),
    'mmlu': (0, 0.25),
}
benchmark_to_ylim_kendall_tau = {
    'gpqa': (0, 1),
    'bbh': (0, 0.8),
    'mmlu-pro': (0, 1),
    'mmlu': (0, 1),
}
benchmark_to_ylim_mdad = {
    'gpqa': (0, 0.3),
    'bbh': (0, 0.6),
    'mmlu-pro': (0, 0.3),
    'mmlu': (0, 0.5),
}
benchmark_to_display_name = {
    'gpqa': 'GPQA',
    'bbh': 'BBH',
    'mmlu-pro': 'MMLU-Pro',
    'mmlu': 'MMLU',
}
benchmark_to_full_seen_size = {
    'gpqa': 224,
    'bbh': 2880,
    'mmlu-pro': 6012,
    'mmlu': 5306,
}


benchmarks = ['mmlu', 'mmlu-pro', 'bbh', 'gpqa']

all_fraction_sampled_points = [10, 25, 50, 100, 250, 500, 1000]
f, axs = plt.subplots(len(benchmarks), len(all_fraction_sampled_points), figsize=(4.8*len(all_fraction_sampled_points), 3.2 * len(benchmarks)))#, gridspec_kw={'height_ratios': [0.5, 0.5, 1]})
plt.subplots_adjust(wspace=0.25, hspace=0.4)    

techniques = ['Random (uniform)', 'Random (subtask stratified, equal)', 'Stratified sampling (confidence)', 'Diversity', 'Anchor Points', 'tinyBenchmarks']
all_num_source_models = [10, 50, 100, 150, 200, 250, 300]

for row, benchmark in enumerate(benchmarks):

    all_fraction_sampled_points = [10, 25, 50, 100, 250, 500, 1000]
    if benchmark == 'gpqa':
        all_fraction_sampled_points = [10, 25, 50, 100, 200]

    cached_mdad_results_fn = f'{save_dir}/{benchmark}-mdad-results.csv'
    cached_estimation_results_fn = f'{save_dir}/{benchmark}-estimation-results.csv'

    if not os.path.exists(cached_mdad_results_fn) and not os.path.exists(cached_estimation_results_fn):
        print(f'Processed results files not generated for {benchmark}')
        continue
        
    df_mdad = pd.read_csv(cached_mdad_results_fn)
    full_estimation_df = pd.read_csv(cached_estimation_results_fn)

    split = 'seen'

    for i, (ax, fraction_sampled_points) in enumerate(zip(axs[row, :], all_fraction_sampled_points)):
        plot_df = df_mdad.loc[(df_mdad['Split'] == split) & (df_mdad['Fraction of sampled points'] == fraction_sampled_points)]
        sns.lineplot(ax=ax, data=plot_df, x='Number of source models', y=f'MDAD', hue='Technique', palette=palette, marker='o', markersize=10, alpha=0.8, linewidth=4, zorder=10, clip_on=True)
        for technique in techniques:
            technique_df = plot_df.loc[(plot_df['Technique'] == technique)]
            data = sorted(zip(technique_df['Number of source models'].tolist(), technique_df['MDAD'].tolist(), technique_df['MDAD lower CI'].tolist(),  technique_df['MDAD upper CI'].tolist()))
            xs = [x for x, _, _, _ in data]
            y1s = [l for _, _, l, _ in data]
            y2s = [u for _, _, _, u in data]
            ax.fill_between(xs, y1s, y2s, color=palette[technique], alpha=0.2)
        ax.set_ylim((0, 0.3))
        ax.set_xlim((0, max(all_num_source_models) + 10))
        ax.set_xticks(all_num_source_models)
        ax.set_xlabel('')
        
        if i == 0:
            ax.set_ylabel(f'MDAD', fontsize=18, labelpad=25, fontweight='bold')
            ax.text(-0.2, 0.5, '↓', fontsize=24, fontweight='bold', va='center', ha='center', transform=ax.transAxes)
        else:
            ax.set_ylabel('')
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.set_title(r'$\mathbf{|S|=}$' + f'{fraction_sampled_points} examples ({fraction_sampled_points/benchmark_to_full_seen_size[benchmark]*100:.1f}%)', fontsize=16, pad=10, fontweight='bold')
        if i == 0:
            ax.text(-0.45, 0.5, benchmark_to_display_name[benchmark], transform=ax.transAxes, rotation_mode='anchor', va='center', ha='center', fontsize=24, fontweight='bold', rotation=90)
f.text(0.5, 0.06, 'Number of source models', ha='center', va='center', fontsize=20, fontweight='bold')
# hide unused GPQA panels
axs[-1,-1].set_visible(False)
axs[-1,-2].set_visible(False)
ls = [matplotlib.patches.Patch(facecolor=palette[t]) for t in techniques]
axs[-1,3].legend(ls,techniques,bbox_to_anchor=(0.5, -0.55), loc='upper center', borderaxespad=0., fontsize=20, ncols=len(ls))#, prop={'weight':'bold'})
savefig(f'{save_dir}/entire_benchmarks_num-source-models-panel.pdf', bbox_inches='tight')