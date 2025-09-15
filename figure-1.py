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

save_dir = 'graphs-combine-subtasks'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

from plot_utils import *

benchmark = 'mmlu-pro'
results_fn = 'results-cached/results-combine-subtasks/figure-1-raw-results.json'

techniques = ["Random (uniform)", 'Stratified sampling (confidence)', "Anchor Points", "tinyBenchmarks"]
all_num_source_models = [10, 50, 100, 150, 200, 250, 300]

all_fraction_sampled_points = [10, 25, 50, 100, 250, 500, 1000]

fig = plt.figure(figsize=(6.4, 5.0))
spec = matplotlib.gridspec.GridSpec(ncols=2, nrows=2, figure=fig, height_ratios=[2, 3.5])
ax1 = fig.add_subplot(spec[0, 0])
ax4 = fig.add_subplot(spec[0, 1])
ax2 = fig.add_subplot(spec[1, 0])
ax3 = fig.add_subplot(spec[1, 1])
axs = [ax1, ax2, ax3]
plt.subplots_adjust(hspace=0.635)

all_tidy_results_estimation = []
tidy_data_correctness = []

with open(results_fn, 'r') as f:
    results = json.load(f)
tidy_results = []
for result in results:
    all_tidy_results_estimation.append(make_tidy_results_estimation(*result))
    tidy_results.extend(make_tidy_results(*result + [1]))
full_df = pd.DataFrame(tidy_results)
full_df['Technique'] = full_df['Technique'].map(lambda t: technique_to_display_name[t])
num_source_models = list(set(full_df['Number of source models']))[0]

# bottom left axis
fraction_sampled_points = 10
ax = axs[1]
df = full_df.loc[(full_df['Fraction of sampled points'] == fraction_sampled_points) \
                    & (full_df['Technique'].isin(techniques)) \
                    & (full_df['Seen full accuracy difference'] <= 0.2)]    
sns.lineplot(ax=ax, data=df, x='Seen full accuracy difference', y='Seen correct', hue='Technique', palette=palette, linewidth=2, zorder=10, clip_on=False, alpha=0.8)
ax.set_xlim((0, 0.2))
ax.set_ylim((0.3, 1))
xticks = [0, 0.04, 0.08, 0.12, 0.16, 0.20]
xticklabels = [f'{x * 100:.0f}' for x in xticks]
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
legend = ax.get_legend()
if legend is not None:
    legend.remove()
ax.set_title(f'{fraction_sampled_points} examples ({fraction_sampled_points/benchmark_to_full_seen_size[benchmark]*100:.1f}%)', fontsize=10, fontweight='bold')
ax.set_xlabel('')
ax.set_ylabel('Probability micro-benchmark\nagrees with full benchmark', fontsize=10)
#ax.text(-0.3, 0.5, f'{num_source_models} source models', fontsize=18, fontweight='bold', va='center', ha='right', transform=ax.transAxes)

# bottom right axis
fraction_sampled_points = 500
ax = axs[2]
df = full_df.loc[(full_df['Fraction of sampled points'] == fraction_sampled_points) \
                    & (full_df['Technique'].isin(techniques)) \
                    & (full_df['Seen full accuracy difference'] <= 0.2)]    
sns.lineplot(ax=ax, data=df, x='Seen full accuracy difference', y='Seen correct', hue='Technique', palette=palette, linewidth=2, zorder=10, clip_on=False, alpha=0.8)
ax.set_xlim((0, 0.2))
ax.set_ylim((0.3, 1))
xticks = [0, 0.04, 0.08, 0.12, 0.16, 0.20]
xticklabels = [f'{x * 100:.0f}' for x in xticks]
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
legend = ax.get_legend()
if legend is not None:
    legend.remove()
ax.set_title(f'{fraction_sampled_points} examples ({fraction_sampled_points/benchmark_to_full_seen_size[benchmark]*100:.1f}%)', fontsize=10, fontweight='bold')
ax.set_xlabel('')
ax.set_ylabel('')
fig.text(0.5, 0.03, 'Accuracy difference between models on full benchmark', ha='center', fontsize=10)
    
# top left axis
full_estimation_df = pd.DataFrame(all_tidy_results_estimation)
full_estimation_df['Technique'] = full_estimation_df['Technique'].map(lambda t: technique_to_display_name[t])
plot_df = full_estimation_df.loc[(full_estimation_df['Number of source models'] == num_source_models) \
                                & (full_estimation_df['Technique'].isin(techniques))
                                & (full_estimation_df['Fraction of sampled points'] == 10)]
fraction_sampled_points = 10
ax = axs[0]
split = 'seen'
sns.barplot(ax=ax, data=plot_df, x='Technique', y=f"Kendall tau correlation against {split} accuracies", hue="Technique", order=['Random (uniform)', 'Stratified sampling (confidence)', 'tinyBenchmarks', 'Anchor Points'], palette=palette, alpha=1, zorder=10, clip_on=True)
for bar in ax.containers:
    ax.bar_label(bar, fmt='%0.2f', fontsize=8, padding=-12, zorder=20, color='#000', \
                 bbox=dict(boxstyle="round,pad=0.18", fc='#fff', lw=0, alpha=0.2))
ax.set_ylim((0, 1.0))
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['0', '', '0.5', '', '1.0'])
ax.set_xlabel('Micro-benchmarking method', fontsize=10)
ax.set_ylabel(f'Kendall tau correlation\nwith full benchmark', fontsize=10)
ax.set_title(f'{fraction_sampled_points} examples ({fraction_sampled_points/benchmark_to_full_seen_size[benchmark]*100:.1f}%)', fontsize=10, fontweight='bold')
ax.set_xticklabels([])
legend = ax.get_legend()
if legend is not None:
    legend.remove()

# top right axis
full_estimation_df = pd.DataFrame(all_tidy_results_estimation)
full_estimation_df['Technique'] = full_estimation_df['Technique'].map(lambda t: technique_to_display_name[t])
plot_df = full_estimation_df.loc[(full_estimation_df['Number of source models'] == num_source_models) \
                                & (full_estimation_df['Technique'].isin(techniques))
                                & (full_estimation_df['Fraction of sampled points'] == 500)]
fraction_sampled_points = 500
ax = ax4
split = 'seen'
sns.barplot(ax=ax, data=plot_df, x='Technique', y=f"Kendall tau correlation against {split} accuracies", hue="Technique", order=['Random (uniform)', 'Stratified sampling (confidence)', 'tinyBenchmarks', 'Anchor Points'], palette=palette, alpha=1, zorder=10, clip_on=True)
for bar in ax.containers:
    ax.bar_label(bar, fmt='%0.2f', fontsize=8, padding=-12, zorder=20, color='#000', \
                 bbox=dict(boxstyle="round,pad=0.18", fc='#fff', lw=0, alpha=0.2))
ax.set_ylim((0, 1.0))
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['0', '', '0.5', '', '1.0'])
ax.set_xlabel('Micro-benchmarking method', fontsize=10)
ax.set_ylabel('')
ax.set_title(f'{fraction_sampled_points} examples ({fraction_sampled_points/benchmark_to_full_seen_size[benchmark]*100:.1f}%)', fontsize=10, fontweight='bold')
ax.set_xticklabels([])
legend = ax.get_legend()
if legend is not None:
    legend.remove()

fig.text(0.5, 0.95, f'Aggregate correlation with full benchmark', ha='center', fontsize=14, fontweight='bold')
fig.text(0.5, 0.545, f'Fine-grained probability of correct ranking', ha='center', fontsize=14, fontweight='bold')

# gray backgrounds and dashed lines
gray_bg = matplotlib.patches.Rectangle((0, 0), 0.04, 1, color='#f0f0f0', zorder=0)
axs[1].add_patch(gray_bg)
gray_bg = matplotlib.patches.Rectangle((0, 0), 0.04, 1, color='#f0f0f0', zorder=0)
axs[2].add_patch(gray_bg)
axs[1].axvline(x=0.02, color='#aaa', linestyle='--')
axs[2].axvline(x=0.02, color='#aaa', linestyle='--')


ls = [matplotlib.patches.Patch(facecolor=palette[t]) for t in techniques]
legend = axs[-1].legend(ls,techniques,bbox_to_anchor=(0.5, 0), loc='upper center', borderaxespad=0., fontsize=10, ncols=2, bbox_transform=fig.transFigure, title='Micro-benchmarking methods', title_fontproperties={'weight':'bold'})#, prop={'weight':'bold'})
savefig(f'{save_dir}/figure-1.pdf', bbox_inches='tight')
plt.close()
