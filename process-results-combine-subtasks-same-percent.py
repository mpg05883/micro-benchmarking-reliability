import os
import json
import numpy as np
import pandas as pd
import math
import itertools
import shutil
from scipy.stats import kendalltau
from tqdm import tqdm
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pylab import savefig
import seaborn as sns
matplotlib.use('Agg')
sns.set_style("whitegrid")
plt.rcParams["font.family"] = 'DejaVu Sans'

from mdad import calculate_mdad
from plot_utils import *

redo = True
mdad_threshold = 0.8

benchmark_to_ylim_estimation = {
    'gpqa': (0, 0.4),
    'bbh': (0, 0.18),
    'mmlu-pro': (0, 0.18),
    'mmlu': (0, 0.18),
}

benchmark_to_ylim_mdad = {
    'gpqa': (0, 0.55),
    'bbh': (0, 0.3),
    'mmlu-pro': (0, 0.3),
    'mmlu': (0, 0.3),
}

save_dir = 'graphs-combine-subtasks-same-percent'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

results_dir = './results-cached/results-combine-subtasks-same-percent'


def multiple_panel_figure_x_axis_fraction_sampled(benchmark, df_mdad, full_estimation_df, df_correctness):   
    all_num_source_models = [10, 50, 100, 150, 200, 250, 300]

    techniques_to_plot = [technique_to_display_name[t] for t in techniques]

    xticklabels = []
    for k in all_fraction_sampled_points:
        if k < 0.01:
            xticklabels.append(f'{math.floor(k*benchmark_to_full_seen_size[benchmark]):.0f}\n({k*100:.1f}%)')
        else:
            xticklabels.append(f'{math.floor(k*benchmark_to_full_seen_size[benchmark]):.0f}\n({k*100:.0f}%)')

    split = 'seen'
    f, axs = plt.subplots(3, len(all_num_source_models), figsize=(6.0*len(all_num_source_models), 3.0 * 3))#, gridspec_kw={'height_ratios': [0.5, 0.5, 1]})
    
    plt.subplots_adjust(wspace=0.25, hspace=0.4)    
    for i, (ax, num_source_models) in enumerate(zip(axs[0, :], all_num_source_models)):
        plot_df = full_estimation_df.loc[(full_estimation_df['Number of source models'] == num_source_models) \
                                        & (full_estimation_df['Technique'].isin(techniques_to_plot))]
        sns.lineplot(ax=ax, data=plot_df, x='Fraction of sampled points', y=f"Mean estimation error against {split} accuracies", hue="Technique", palette=palette, marker=split_to_marker[split], linestyle=split_to_linestyle[split], markersize=10, alpha=0.8, linewidth=4, zorder=10, clip_on=True)
        ax.set_xscale('log')
        ax.set_ylim(benchmark_to_ylim_estimation[benchmark])
        #ax.set_xlim((0, max(all_fraction_sampled_points) + 0.02))
        ax.set_xticks(all_fraction_sampled_points)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel('')
        yticks = ax.get_yticks()
        yticklabels = [f'{y * 100:.0f}' for y in yticks]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        if i == 0:
            ax.set_ylabel(f'Estimation error', fontsize=14, labelpad=20, fontweight='bold')
            ax.text(-0.1, 0.5, '↓', fontsize=24, fontweight='bold', va='center', ha='center', transform=ax.transAxes)
        else:
            ax.set_ylabel('')
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.set_title(f'{num_source_models} source models', fontsize=20, fontweight='bold')
    
    for i, (ax, num_source_models) in enumerate(zip(axs[1, :], all_num_source_models)):
        plot_df = full_estimation_df.loc[(full_estimation_df['Number of source models'] == num_source_models) \
                                        & (full_estimation_df['Technique'].isin(techniques_to_plot))]
        sns.lineplot(ax=ax, data=plot_df, x='Fraction of sampled points', y=f"Kendall tau correlation against {split} accuracies", hue="Technique", palette=palette, marker=split_to_marker[split], linestyle=split_to_linestyle[split], markersize=10, alpha=0.8, linewidth=4, zorder=10, clip_on=True)
        ax.set_xscale('log')
        ax.set_ylim(benchmark_to_ylim_kendall_tau[benchmark])
        # ax.set_xlim((0, max(all_fraction_sampled_points) + 0.02))
        ax.set_xticks(all_fraction_sampled_points)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel('')
        if i == 0:
            ax.set_ylabel(f'Rank correlation', fontsize=14, labelpad=20, fontweight='bold')
            ax.text(-0.11, 0.5, '↑', fontsize=24, fontweight='bold', va='center', ha='center', transform=ax.transAxes)
        else:
            ax.set_ylabel('')
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    for i, (ax, num_source_models) in enumerate(zip(axs[2, :], all_num_source_models)):
        plot_df = df_mdad.loc[(df_mdad['Split'] == split) & (df_mdad['Number of source models'] == num_source_models) \
                           & (df_mdad['Technique'].isin(techniques_to_plot))]
        sns.lineplot(ax=ax, data=plot_df, x='Fraction of sampled points', y=f'MDAD', hue='Technique', palette=palette, marker=split_to_marker[split], linestyle=split_to_linestyle[split], markersize=10, alpha=0.8, linewidth=4, zorder=10, clip_on=True)
        for technique in techniques_to_plot:
            technique_df = plot_df.loc[(plot_df['Technique'] == technique)]
            data = sorted(zip(technique_df['Fraction of sampled points'].tolist(), technique_df['MDAD'].tolist(), technique_df['MDAD lower CI'].tolist(),  technique_df['MDAD upper CI'].tolist()))
            xs = [x for x, _, _, _ in data]
            y1s = [min(l, m-(u-m)) for _, m, l, u in data]
            y2s = [max(m+(m-l), u) for _, m, l, u in data]
            ax.fill_between(xs, y1s, y2s, color=palette[technique], alpha=0.2)
        ax.set_xscale('log')
        ax.set_ylim(benchmark_to_ylim_mdad[benchmark])
        yticks = ax.get_yticks()
        yticklabels = [f'{y * 100:.0f}' for y in yticks]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_xticks(all_fraction_sampled_points)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel('')
        if i == 0:
            ax.set_ylabel(f'MDAD', fontsize=14, labelpad=20, fontweight='bold')
            ax.text(-0.1, 0.5, '↓', fontsize=24, fontweight='bold', va='center', ha='center', transform=ax.transAxes)
        else:
            ax.set_ylabel('')
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    f.text(0.5, 0.03, 'Number of examples selected for micro-benchmark', ha='center', va='center', fontsize=18, fontweight='bold')
    ls = [matplotlib.patches.Patch(facecolor=palette[t]) for t in techniques_to_plot]
    plt.legend(ls,techniques_to_plot,bbox_to_anchor=(0.5, -0.01), loc='upper center', bbox_transform=f.transFigure, borderaxespad=0., fontsize=20, ncols=len(techniques_to_plot))#len(techniques))#, prop={'weight':'bold'})
    f.text(0.5, 0.96, benchmark_to_display_name[benchmark], fontsize=36, va='center', ha='center', fontweight='bold')
    savefig(f'{save_dir}/{benchmark}_combine-subtasks-same-percent_full-results-panel.pdf', bbox_inches='tight')
    plt.close()

benchmarks = ['mmlu', 'mmlu-pro', 'bbh', 'gpqa']
all_num_source_models = [10, 50, 100, 150, 200, 250, 300]

techniques = ["Random", 'Random_Subtask_Stratified_Equal', 'DPP', 'Stratified_Random_Sampling', "Anchor_Points_Weighted", "tinyBenchmarks"]
for benchmark in benchmarks:
    all_fraction_sampled_points = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.24]
    cached_mdad_results_fn = f'{save_dir}/{benchmark}-mdad-results.csv'
    cached_correctness_results_fn = f'{save_dir}/{benchmark}-correctness-results.csv'
    cached_estimation_results_fn = f'{save_dir}/{benchmark}-estimation-results.csv'

    if not redo and os.path.exists(cached_mdad_results_fn) \
      and os.path.exists(cached_estimation_results_fn) \
      and os.path.exists(cached_mdad_results_fn):
        df_mdad = pd.read_csv(cached_mdad_results_fn)
        full_estimation_df = pd.read_csv(cached_estimation_results_fn)
        df_correctness = pd.read_csv(cached_correctness_results_fn)
    else:
        all_tidy_results_estimation = []
        tidy_data_mdad = []
        tidy_data_correctness = []
        for num_source_models in all_num_source_models:
            print(f'{benchmark}, {num_source_models} source models, processing results:')
            fn = f'{results_dir}/results_{benchmark}_{num_source_models}-source-models.json'
            with open(fn, 'r') as f:
                results = json.load(f)
            tidy_results = []
            for result in results:
                tidy_results.extend(make_tidy_results(*result))
                all_tidy_results_estimation.append(make_tidy_results_estimation(*result))            
            full_df = pd.DataFrame(tidy_results)
            datasets = sorted(list(set(full_df['Dataset'])))
            for fraction_sampled_points in tqdm(all_fraction_sampled_points):
                fraction_sampled_points_df = full_df.loc[(full_df['Number of source models'] == num_source_models) \
                            & (full_df['Fraction of sampled points'] == fraction_sampled_points)] 
                for technique in techniques:
                    df = fraction_sampled_points_df.loc[(fraction_sampled_points_df['Technique'] == technique)]    
                    if len(df) == 0:
                        continue
                    tidy_data_correctness.append({'Technique': technique,
                                            'Number of source models': num_source_models,
                                            'Fraction of sampled points': fraction_sampled_points,
                                            'Average correctness seen (mean)': df['Seen correct'].mean(),
                                            'Average correctness unseen (mean)': df['Unseen correct'].mean()})
                    for split in ['Seen', 'Unseen']:
                        mdad, mdad_lower, mdad_upper = calculate_mdad(df[f'{split} full accuracy difference'].tolist(),
                                                                   df[f'{split} correct'].tolist(),
                                                                   mdad_threshold=mdad_threshold)
                        #print(benchmark, num_source_models, fraction_sampled_points, technique, mdad_lower, mdad, mdad_upper)
                        tidy_data_mdad.append({'Technique': technique,
                                                'Number of source models': num_source_models,
                                                'Split': split.lower(),
                                                'Fraction of sampled points': fraction_sampled_points,
                                                'MDAD': mdad,
                                                'MDAD lower CI': mdad_lower,
                                                'MDAD upper CI': mdad_upper,})
        df_correctness = pd.DataFrame(tidy_data_correctness)
        df_mdad = pd.DataFrame(tidy_data_mdad)
        full_estimation_df = pd.DataFrame(all_tidy_results_estimation)
        full_estimation_df['Technique'] = full_estimation_df['Technique'].map(lambda t: technique_to_display_name[t])
        df_mdad['Technique'] = df_mdad['Technique'].map(lambda t: technique_to_display_name[t])
        df_correctness['Technique'] = df_correctness['Technique'].map(lambda t: technique_to_display_name[t])
        df_correctness.to_csv(cached_correctness_results_fn)
        df_mdad.to_csv(cached_mdad_results_fn)
        full_estimation_df.to_csv(cached_estimation_results_fn)
    multiple_panel_figure_x_axis_fraction_sampled(benchmark, df_mdad, full_estimation_df, df_correctness)
    