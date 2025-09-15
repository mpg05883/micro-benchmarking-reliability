import os
import json
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pylab import savefig
import seaborn as sns
matplotlib.use('Agg')
sns.set_style("whitegrid")
plt.rcParams["font.family"] = 'DejaVu Sans'

from mdad import calculate_mdad
from plot_utils import *

mdad_threshold = 0.8

save_dir = 'graphs-combine-subtasks'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

num_source_models = 300

techniques = ['Random (uniform)', f'Anchor Points']
sampled_points_to_plot = [10, 50, 1000]

benchmark = 'mmlu-pro'
results_fn = 'results-cached/results-combine-subtasks/results_mmlu-pro_300-source-models_50-runs.json'

all_fraction_sampled_points = [10, 25, 50, 100, 250, 500, 1000]


gridspec = dict(wspace=0.25, width_ratios=[0.249, 0.249, 0.249, 0.004, 0.249])
fig, axs = plt.subplots(1, 5, figsize=(6.4*4, 4.8), gridspec_kw=gridspec)
axs[3].set_visible(False)

with open(results_fn, 'r') as f:
    results = json.load(f)

tidy_results = []
for result in results:
    tidy_results.extend(make_tidy_results(*result))
full_df = pd.DataFrame(tidy_results)

tidy_data_mdad = []
tidy_data_correctness = []
for fraction_sampled_points in all_fraction_sampled_points:
    # first get average correctness
    fraction_sampled_points_df = full_df.loc[(full_df['Fraction of sampled points'] == fraction_sampled_points)] 
    for technique in ['Random', f'Anchor_Points_Weighted']:
        print('MDAD calculation', fraction_sampled_points, technique)
        df = fraction_sampled_points_df.loc[(fraction_sampled_points_df['Technique'] == technique)]    
        tidy_data_correctness.append({'Technique': technique,
                                    'Number of source models': num_source_models,
                                    'Fraction of sampled points': fraction_sampled_points,
                                    'Average correctness seen (mean)': df['Seen correct'].mean(),
                                    'Average correctness unseen (mean)': df['Unseen correct'].mean()})
        mdad, mdad_lower, mdad_upper = calculate_mdad(df[f'Seen full accuracy difference'].tolist(),
                                                                   df[f'Seen correct'].tolist(),
                                                                   mdad_threshold=mdad_threshold)
        print(fraction_sampled_points, technique, mdad)
        tidy_data_mdad.append({'Technique': technique,
                            'Number of source models': num_source_models,
                            'Split': 'seen',
                            'Fraction of sampled points': fraction_sampled_points,
                            'MDAD': mdad,
                            'MDAD lower CI': mdad_lower,
                            'MDAD upper CI': mdad_upper,})
mdad_df = pd.DataFrame(tidy_data_mdad)

mdad_df['Technique'] = mdad_df['Technique'].map(lambda t: technique_to_display_name[t])
full_df['Technique'] = full_df['Technique'].map(lambda t: technique_to_display_name[t])
full_df = full_df.dropna()

title_colors = [number_to_highlight_color[k] for k in [10, 50, 1000]]

letters = ['B', 'A', 'D', 'C', 'E', 'F']
has = ['right', 'left', 'right', 'left', 'left', 'right']
offsets = [0.015, -0.015, 0.015, -0.015, -0.015, 0.015]
i = 0
annotateds = []

for ax_col, (ax, fraction_sampled_points, title_color) in enumerate(zip(axs[:4], sampled_points_to_plot, title_colors)):
    plot_df = full_df.loc[#(full_df['Dataset'] == dataset)
                        (full_df['Technique'].isin(techniques))
                        & (full_df['Fraction of sampled points'] == fraction_sampled_points)
                        #& (full_df['Number of source models'] == num_source_models)
                        ]
    sns.lineplot(ax=ax, data=plot_df, x='Seen full accuracy difference', y='Seen correct', hue='Technique', palette=palette, linewidth=4, alpha=0.8, zorder=10, clip_on=True)
    
    # points to plot
    ymin = 0.4
    for technique in techniques:
        row = mdad_df.loc[(mdad_df['Split'] == 'seen') & (mdad_df['Technique'] == technique) & (mdad_df['Fraction of sampled points'] == fraction_sampled_points)]
        x = row['MDAD'].iloc[0]
        if x < 0.2:
            ax.scatter(x, 0.8, color=palette[technique], edgecolors='#ffffff', s=200, zorder=20, clip_on=False)
            ax.scatter(x, ymin, color=palette[technique], edgecolors='#ffffff', s=150, zorder=20, clip_on=False)
            ax.vlines(x=x, ymin=ymin, ymax=0.8, linestyle='--', color=palette[technique], alpha=0.6, linewidth=2, zorder=5)
            ax.text(x + offsets[i], ymin + 0.03, f'{letters[i]}', ha=has[i], va='center', fontsize=20, fontweight='bold')
            annotateds.append((fraction_sampled_points, x, f'{letters[i]}'))
            i += 1
    ax.set_xlim((0, 0.2))
    ax.set_ylim((ymin, 1))
    ax.axhline(y=0.8, linestyle='--', color='#000000', linewidth=2, zorder=5)
    ax.set_xlabel(r'$d$', fontsize=12, labelpad=10)
    if ax_col == 0:
        ax.set_ylabel('Agreement', fontsize=16, labelpad=20, fontweight='bold')
        ax.text(-0.13, 0.5, '↑', fontsize=24, fontweight='bold', va='center', ha='center', transform=ax.transAxes)
    else:
        ax.set_ylabel('')
    xticks = ax.get_xticks()
    xticklabels = [f'{x * 100:.1f}' for x in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
    #ax.set_title(f'{fraction_sampled_points*100:.0f}% of dataset selected')
    ax.set_title(r'$\mathbf{|S|=}$' + f'{fraction_sampled_points} examples ({fraction_sampled_points/benchmark_to_full_seen_size[benchmark]*100:.1f}%)', fontsize=16, fontweight='bold', pad=10,
                    bbox=dict(boxstyle="round,pad=0.22", fc=title_color, lw=0, alpha=0.5))
    if ax_col == 1:
        ax.text(0.5, 1.15, 'Probability micro-benchmark agrees with full benchmark', ha='center', fontsize=18, fontweight='bold', transform=ax.transAxes)
    ax.get_legend().remove()
fig.text(0.5, -0.2, 'Accuracy difference between models on full benchmark', fontsize=14, fontweight='bold', ha='center', va='center', transform=axs[1].transAxes)
# now final plot
ax = axs[-1]
plot_df = mdad_df.loc[(mdad_df['Split'] == 'seen') & (mdad_df['Technique'].isin(techniques))]
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
has = ['left', 'left', 'left', 'left', 'left', 'left']
vas = ['bottom', 'top', 'bottom', 'top', 'center', 'center']
offsets = [(0,0.01), (0,-0.01), (0,0.01), (0,-0.01), (100,0), (100, 0)]
for (x, y, text), ha, va, (offsetx, offsety) in zip(annotateds, has, vas, offsets):
    ax.text(x + offsetx, y + offsety/2, text, ha=ha, va=va, fontsize=20, fontweight='bold')
#ax.set_xlim(())
ax.set_xticks(all_fraction_sampled_points)
ax.set_ylim((0,0.2))
# ax.set_xlabel('Fraction of sampled points per subtask')
xticklabels = [f'{k}\n({k/benchmark_to_full_seen_size[benchmark]*100:.1f}%)' for k in all_fraction_sampled_points]
ax.set_xticklabels(xticklabels)
yticks = ax.get_yticks()
yticklabels = [f'{y * 100:.0f}' for y in yticks]
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels, fontsize=12)
xticklabels = ax.get_xticklabels()
xticklabels[0].set_bbox(dict(boxstyle="round,pad=0.22", fc=title_colors[0], lw=0, alpha=0.5))
xticklabels[2].set_bbox(dict(boxstyle="round,pad=0.22", fc=title_colors[1], lw=0, alpha=0.5))
xticklabels[6].set_bbox(dict(boxstyle="round,pad=0.22", fc=title_colors[2], lw=0, alpha=0.5))

ax.set_xlabel('Number of examples selected for micro-benchmark', fontsize=11.1, labelpad=15, fontweight='bold')
ax.set_ylabel('MDAD', fontsize=16, labelpad=20, fontweight='bold')
ax.text(-0.11, 0.5, '↓', fontsize=24, fontweight='bold', va='center', ha='center', transform=ax.transAxes)
ax.set_title('MDAD', ha='center', fontsize=18, fontweight='bold', pad=25)

l = axs[-1].get_legend()
l.set_title('Micro-benchmarking technique')

savefig(f'{save_dir}/figure-2.pdf', bbox_inches='tight')
    
    

