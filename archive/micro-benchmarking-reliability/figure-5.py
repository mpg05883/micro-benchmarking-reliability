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

save_dir = 'graphs-fixed-target-models'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

mdad_correctness = 0.8

all_data_to_plot = [
    ('mmlu-pro', 'mmlu-pro_8b_instruct'),
    ('mmlu-pro', 'mmlu-pro_70b_instruct'),
    ('bbh', 'bbh_7b_instruct'),
    ('bbh', 'bbh_70b_instruct'),
]

setting_to_model_names = {
    'mmlu-pro_7b_instruct': ['Intel__neural-chat-7b-v3', 'Deci__DeciLM-7B-instruct', 'togethercomputer__Llama-2-7B-32K-Instruct', 'internlm__internlm2_5-7b-chat', 'google__gemma-7b-it', 'mistralai__Mistral-7B-Instruct-v0.2', 'ibm-granite__granite-7b-instruct', 'deepseek-ai__deepseek-llm-7b-chat', 'tiiuae__Falcon3-Mamba-7B-Instruct', 'google__gemma-1.1-7b-it', 'Qwen__Qwen2-7B-Instruct', 'allenai__OLMo-2-1124-7B-Instruct', 'tiiuae__falcon-7b-instruct', 'tiiuae__Falcon3-7B-Instruct', 'Qwen__Qwen2.5-Coder-7B-Instruct', 'meta-llama__Llama-2-7b-chat-hf', 'Qwen__Qwen1.5-7B-Chat', 'allenai__OLMo-7B-Instruct-hf', 'Qwen__Qwen2.5-7B-Instruct', 'Qwen__Qwen2.5-7B-Instruct-1M', 'Qwen__Qwen2-VL-7B-Instruct', 'Qwen__Qwen2.5-Math-7B-Instruct', 'mistralai__Mistral-7B-Instruct-v0.3', 'togethercomputer__RedPajama-INCITE-7B-Instruct', 'nvidia__AceInstruct-7B', 'mistralai__Mistral-7B-Instruct-v0.1', 'nvidia__AceMath-7B-Instruct', 'argilla__notus-7b-v1', 'mlabonne__NeuralBeagle14-7B', 'HuggingFaceH4__zephyr-7b-gemma-v0.1', 'mlabonne__AlphaMonarch-7B', 'lmsys__vicuna-7b-v1.3', 'CohereForAI__c4ai-command-r7b-12-2024', 'NousResearch__Nous-Hermes-2-Mistral-7B-DPO', 'ibm__merlinite-7b', 'NousResearch__Nous-Hermes-llama-2-7b', 'togethercomputer__RedPajama-INCITE-7B-Chat', 'Intel__neural-chat-7b-v3-2', 'Intel__neural-chat-7b-v3-3', 'Intel__neural-chat-7b-v3-1', 'cognitivecomputations__dolphin-2.9.2-qwen2-7b', 'HuggingFaceH4__zephyr-7b-beta', 'teknium__OpenHermes-7B', 'Open-Orca__Mistral-7B-OpenOrca', 'cognitivecomputations__dolphin-2.9.3-mistral-7B-32k', 'berkeley-nest__Starling-LM-7B-alpha', 'HuggingFaceH4__zephyr-7b-alpha', 'databricks__dolly-v2-7b', 'lmsys__vicuna-7b-v1.5', 'teknium__CollectiveCognition-v1.1-Mistral-7B', 'nvidia__AceMath-7B-RM', 'togethercomputer__LLaMA-2-7B-32K', 'microsoft__Orca-2-7b', 'teknium__OpenHermes-2.5-Mistral-7B', 'deepseek-ai__DeepSeek-R1-Distill-Qwen-7B', 'NousResearch__Hermes-2-Pro-Mistral-7B'],
    'mmlu-pro_7b_base': ['togethercomputer__RedPajama-INCITE-7B-Base', 'google__gemma-7b', 'deepseek-ai__deepseek-llm-7b-base', 'tiiuae__Falcon3-Mamba-7B-Base', 'Qwen__Qwen1.5-7B', 'Qwen__Qwen2-7B', 'allenai__OLMo-1.7-7B-hf', 'ibm-granite__granite-7b-base', 'Qwen__Qwen2.5-7B', 'tiiuae__falcon-7b', 'tiiuae__Falcon3-7B-Base', 'mlabonne__Beyonder-4x7B-v3', 'mistralai__Mistral-7B-v0.3', 'mistralai__Mistral-7B-v0.1', 'Qwen__Qwen2.5-Coder-7B', 'mosaicml__mpt-7b', 'mistral-community__Mistral-7B-v0.2', 'Qwen__Qwen2-Math-7B', 'NousResearch__Yarn-Mistral-7b-128k', 'NousResearch__Yarn-Mistral-7b-64k', 'allenai__OLMo-7B-hf', 'meta-llama__Llama-2-7b-hf', 'Deci__DeciLM-7B', 'bigscience__bloom-7b1', 'tiiuae__falcon-mamba-7b', 'NousResearch__Yarn-Llama-2-7b-128k', 'bigcode__starcoder2-7b', 'Qwen__Qwen2.5-Math-7B'],
    'mmlu-pro_8b_instruct': ['meta-llama__Llama-3.1-8B', 'CohereForAI__aya-expanse-8b', 'openchat__openchat-3.6-8b-20240522', 'ibm-granite__granite-3.2-8b-instruct', 'nvidia__Mistral-NeMo-Minitron-8B-Instruct', 'mlabonne__OrpoLlama-3-8B', 'mistralai__Ministral-8B-Instruct-2410', 'gradientai__Llama-3-8B-Instruct-Gradient-1048k', 'ibm-granite__granite-3.1-8b-instruct', 'allenai__Llama-3.1-Tulu-3-8B', 'mlabonne__Meta-Llama-3.1-8B-Instruct-abliterated', 'meta-llama__Llama-3.1-8B-Instruct', 'NousResearch__Hermes-3-Llama-3.1-8B', 'NousResearch__Hermes-2-Pro-Llama-3-8B', 'argilla-warehouse__Llama-3.1-8B-MagPie-Ultra', 'Salesforce__LLaMA-3-8B-SFR-Iterative-DPO-R', 'allenai__Llama-3.1-Tulu-3-8B-DPO', 'TencentARC__LLaMA-Pro-8B-Instruct', 'nvidia__OpenMath2-Llama3.1-8B', 'cognitivecomputations__dolphin-2.9.4-llama3.1-8b', 'allenai__Llama-3.1-Tulu-3-8B-SFT', 'cognitivecomputations__Dolphin3.0-Llama3.1-8B', 'ibm-granite__granite-3.0-8b-instruct', 'meta-llama__Meta-Llama-3-8B-Instruct', 'CohereForAI__aya-23-8B', 'NousResearch__Hermes-2-Theta-Llama-3-8B', 'mlabonne__Daredevil-8B', 'mlabonne__NeuralDaredevil-8B-abliterated', 'mlabonne__Daredevil-8B-abliterated', 'mlabonne__ChimeraLlama-3-8B-v3', 'mlabonne__ChimeraLlama-3-8B-v2', 'abacusai__Llama-3-Smaug-8B'],
    'mmlu-pro_8b_base': ['nvidia__Mistral-NeMo-Minitron-8B-Base', 'ibm-granite__granite-3.1-8b-base', 'meta-llama__Meta-Llama-3-8B', 'cognitivecomputations__dolphin-2.9-llama3-8b', 'TencentARC__Mistral_Pro_8B_v0.1', 'nvidia__Minitron-8B-Base', 'TencentARC__LLaMA-Pro-8B', 'ibm-granite__granite-3.0-8b-base'],
    'mmlu-pro_70b_instruct': ['meta-llama__Llama-2-70b-chat-hf', 'nvidia__Llama-3.1-Nemotron-70B-Instruct-HF', 'NousResearch__Hermes-3-Llama-3.1-70B', 'meta-llama__Llama-3.3-70B-Instruct', 'meta-llama__Meta-Llama-3-70B-Instruct', 'allenai__Llama-3.1-Tulu-3-70B-SFT', 'abacusai__Smaug-Llama-3-70B-Instruct-32K', 'allenai__Llama-3.1-Tulu-3-70B-DPO', 'meta-llama__Llama-3.1-70B-Instruct', 'allenai__Llama-3.1-Tulu-3-70B', 'WizardLMTeam__WizardLM-70B-V1.0'],
    'mmlu-pro_70b_base': ['cognitivecomputations__dolphin-2.9.1-llama-3-70b', 'meta-llama__Llama-3.1-70B', 'meta-llama__Llama-2-70b-hf', 'meta-llama__Meta-Llama-3-70B', 'mlabonne__Hermes-3-Llama-3.1-70B-lorablated'],
    
    'bbh_7b_instruct': ['Qwen__Qwen2.5-7B-Instruct-1M', 'HuggingFaceH4__zephyr-7b-gemma-v0.1', 'ibm__merlinite-7b', 'nvidia__AceMath-7B-Instruct', 'meta-llama__Llama-2-7b-chat-hf', 'lmsys__vicuna-7b-v1.5', 'Qwen__Qwen2.5-7B-Instruct', 'Intel__neural-chat-7b-v3-3', 'berkeley-nest__Starling-LM-7B-alpha', 'Intel__neural-chat-7b-v3', 'mistralai__Mistral-7B-Instruct-v0.2', 'Intel__neural-chat-7b-v3-1', 'allenai__OLMo-7B-Instruct-hf', 'internlm__internlm2_5-7b-chat', 'NousResearch__Nous-Hermes-llama-2-7b', 'databricks__dolly-v2-7b', 'Qwen__Qwen2-7B-Instruct', 'google__gemma-1.1-7b-it', 'Open-Orca__Mistral-7B-OpenOrca', 'tiiuae__Falcon3-7B-Instruct', 'Intel__neural-chat-7b-v3-2', 'Qwen__Qwen1.5-7B-Chat', 'nvidia__AceMath-7B-RM', 'Qwen__Qwen2.5-Coder-7B-Instruct', 'lmsys__vicuna-7b-v1.3', 'tiiuae__falcon-7b-instruct', 'tiiuae__Falcon3-Mamba-7B-Instruct', 'google__gemma-7b-it', 'allenai__OLMo-2-1124-7B-Instruct', 'teknium__OpenHermes-7B', 'togethercomputer__Llama-2-7B-32K-Instruct', 'mistralai__Mistral-7B-Instruct-v0.3', 'HuggingFaceH4__zephyr-7b-alpha', 'Qwen__Qwen2.5-Math-7B-Instruct', 'NousResearch__Nous-Hermes-2-Mistral-7B-DPO', 'teknium__OpenHermes-2.5-Mistral-7B', 'togethercomputer__RedPajama-INCITE-7B-Instruct', 'deepseek-ai__deepseek-llm-7b-chat', 'cognitivecomputations__dolphin-2.9.3-mistral-7B-32k', 'mistralai__Mistral-7B-Instruct-v0.1', 'NousResearch__Hermes-2-Pro-Mistral-7B', 'nvidia__AceInstruct-7B', 'togethercomputer__RedPajama-INCITE-7B-Chat', 'HuggingFaceH4__zephyr-7b-beta', 'CohereForAI__c4ai-command-r7b-12-2024', 'Deci__DeciLM-7B-instruct', 'Qwen__Qwen2-VL-7B-Instruct', 'deepseek-ai__DeepSeek-R1-Distill-Qwen-7B', 'ibm-granite__granite-7b-instruct', 'teknium__CollectiveCognition-v1.1-Mistral-7B', 'togethercomputer__LLaMA-2-7B-32K', 'microsoft__Orca-2-7b'],
    'bbh_7b_base': ['tiiuae__Falcon3-Mamba-7B-Base', 'Qwen__Qwen2.5-Coder-7B', 'Qwen__Qwen2.5-7B', 'bigscience__bloom-7b1', 'allenai__OLMo-7B-hf', 'deepseek-ai__deepseek-llm-7b-base', 'mistralai__Mistral-7B-v0.1', 'mosaicml__mpt-7b', 'mistral-community__Mistral-7B-v0.2', 'Deci__DeciLM-7B', 'bigcode__starcoder2-7b', 'Qwen__Qwen1.5-7B', 'NousResearch__Yarn-Mistral-7b-128k', 'Qwen__Qwen2-7B', 'tiiuae__falcon-7b', 'NousResearch__Yarn-Mistral-7b-64k', 'togethercomputer__RedPajama-INCITE-7B-Base', 'tiiuae__Falcon3-7B-Base', 'NousResearch__Yarn-Llama-2-7b-128k', 'ibm-granite__granite-7b-base', 'Qwen__Qwen2-Math-7B', 'mistralai__Mistral-7B-v0.3', 'tiiuae__falcon-mamba-7b', 'Qwen__Qwen2.5-Math-7B', 'google__gemma-7b'],
    'bbh_70b_instruct': ['allenai__Llama-3.1-Tulu-3-70B-DPO', 'allenai__Llama-3.1-Tulu-3-70B-SFT', 'meta-llama__Llama-3.1-70B-Instruct', 'abacusai__Smaug-Llama-3-70B-Instruct-32K', 'NousResearch__Hermes-3-Llama-3.1-70B', 'WizardLMTeam__WizardLM-70B-V1.0', 'meta-llama__Llama-3.3-70B-Instruct', 'meta-llama__Meta-Llama-3-70B-Instruct', 'meta-llama__Llama-2-70b-chat-hf', 'allenai__Llama-3.1-Tulu-3-70B', 'nvidia__Llama-3.1-Nemotron-70B-Instruct-HF', 'deepseek-ai__DeepSeek-R1-Distill-Llama-70B'],
    'bbh_70b_base': ['meta-llama__Llama-2-70b-hf', 'meta-llama__Llama-3.1-70B', 'mlabonne__Hermes-3-Llama-3.1-70B-lorablated', 'meta-llama__Meta-Llama-3-70B', 'cognitivecomputations__dolphin-2.9.1-llama-3-70b'],

    'gpqa_7b_instruct': ['teknium__CollectiveCognition-v1.1-Mistral-7B', 'Intel__neural-chat-7b-v3-2', 'ibm__merlinite-7b', 'NousResearch__Nous-Hermes-llama-2-7b', 'togethercomputer__RedPajama-INCITE-7B-Chat', 'Open-Orca__Mistral-7B-OpenOrca', 'Qwen__Qwen2.5-7B-Instruct', 'CohereForAI__c4ai-command-r7b-12-2024', 'teknium__OpenHermes-2.5-Mistral-7B', 'NousResearch__Nous-Hermes-2-Mistral-7B-DPO', 'togethercomputer__Llama-2-7B-32K-Instruct', 'internlm__internlm2_5-7b-chat', 'togethercomputer__LLaMA-2-7B-32K', 'Qwen__Qwen2.5-Math-7B-Instruct', 'Qwen__Qwen2-7B-Instruct', 'Qwen__Qwen1.5-7B-Chat', 'mistralai__Mistral-7B-Instruct-v0.1', 'nvidia__AceMath-7B-Instruct', 'mistralai__Mistral-7B-Instruct-v0.2', 'HuggingFaceH4__zephyr-7b-gemma-v0.1', 'Deci__DeciLM-7B-instruct', 'google__gemma-7b-it', 'allenai__OLMo-7B-Instruct-hf', 'berkeley-nest__Starling-LM-7B-alpha', 'nvidia__AceInstruct-7B', 'HuggingFaceH4__zephyr-7b-alpha', 'lmsys__vicuna-7b-v1.5', 'cognitivecomputations__dolphin-2.9.2-qwen2-7b', 'databricks__dolly-v2-7b', 'teknium__OpenHermes-7B', 'tiiuae__Falcon3-7B-Instruct', 'meta-llama__Llama-2-7b-chat-hf', 'tiiuae__falcon-7b-instruct', 'cognitivecomputations__dolphin-2.9.3-mistral-7B-32k', 'google__gemma-1.1-7b-it', 'microsoft__Orca-2-7b', 'togethercomputer__RedPajama-INCITE-7B-Instruct', 'Qwen__Qwen2.5-Coder-7B-Instruct', 'ibm-granite__granite-7b-instruct', 'tiiuae__Falcon3-Mamba-7B-Instruct', 'deepseek-ai__deepseek-llm-7b-chat', 'NousResearch__Hermes-2-Pro-Mistral-7B', 'Intel__neural-chat-7b-v3-3', 'Qwen__Qwen2-VL-7B-Instruct', 'HuggingFaceH4__zephyr-7b-beta', 'argilla__notus-7b-v1', 'lmsys__vicuna-7b-v1.3', 'mistralai__Mistral-7B-Instruct-v0.3', 'allenai__OLMo-2-1124-7B-Instruct', 'teknium__OpenHermes-2-Mistral-7B'],
    'gpqa_7b_base': ['Deci__DeciLM-7B', 'tiiuae__Falcon3-7B-Base', 'mistralai__Mistral-7B-v0.1', 'meta-llama__Llama-2-7b-hf', 'Qwen__Qwen2.5-7B', 'Qwen__Qwen2.5-Math-7B', 'Qwen__Qwen2-7B', 'mlabonne__Beyonder-4x7B-v3', 'NousResearch__Yarn-Mistral-7b-128k', 'NousResearch__Yarn-Llama-2-7b-128k', 'tiiuae__falcon-mamba-7b', 'Qwen__Qwen1.5-7B', 'bigscience__bloom-7b1', 'Qwen__Qwen2.5-Coder-7B', 'tiiuae__falcon-7b', 'mistralai__Mistral-7B-v0.3', 'mistral-community__Mistral-7B-v0.2', 'allenai__OLMo-1.7-7B-hf', 'mosaicml__mpt-7b', 'deepseek-ai__deepseek-llm-7b-base', 'google__gemma-7b', 'NousResearch__Yarn-Mistral-7b-64k', 'bigcode__starcoder2-7b', 'ibm-granite__granite-7b-base', 'allenai__OLMo-7B-hf', 'tiiuae__Falcon3-Mamba-7B-Base', 'togethercomputer__RedPajama-INCITE-7B-Base', 'Qwen__Qwen2-Math-7B', 'NousResearch__Yarn-Llama-2-7b-64k'],
    'gpqa_70b_instruct': ['meta-llama__Llama-2-70b-chat-hf', 'meta-llama__Llama-3.3-70B-Instruct', 'NousResearch__Hermes-3-Llama-3.1-70B', 'allenai__Llama-3.1-Tulu-3-70B', 'allenai__Llama-3.1-Tulu-3-70B-SFT', 'nvidia__Llama-3.1-Nemotron-70B-Instruct-HF', 'meta-llama__Llama-3.1-70B-Instruct', 'WizardLMTeam__WizardLM-70B-V1.0', 'meta-llama__Meta-Llama-3-70B-Instruct', 'abacusai__Smaug-Llama-3-70B-Instruct-32K', 'allenai__Llama-3.1-Tulu-3-70B-DPO', 'deepseek-ai__DeepSeek-R1-Distill-Llama-70B'],
    'gpqa_70b_base': ['meta-llama__Meta-Llama-3-70B', 'cognitivecomputations__dolphin-2.9.1-llama-3-70b', 'mlabonne__Hermes-3-Llama-3.1-70B-lorablated', 'meta-llama__Llama-3.1-70B', 'meta-llama__Llama-2-70b-hf'],
}


for (benchmark, fixed_target_models_name) in all_data_to_plot:
    all_datasets_configs = []
    times = []
    # load all data
    models_by_confidence_fn = f'./open-llm-leaderboard-results-combined/{benchmark}_models-by-confidence_combined.npy'
    models_by_correctness_fn =f'./open-llm-leaderboard-results-combined/{benchmark}_models-by-correctness_combined.npy'
    labels_fn = f'./open-llm-leaderboard-results-combined/{benchmark}_labels_combined.npy'
    all_data = np.load(models_by_confidence_fn)
    models_by_correctness = np.load(models_by_correctness_fn)
    gt_labels = np.load(labels_fn)
    true_scores = np.mean(models_by_correctness, axis=1)
    print(all_data.shape)
    print(models_by_correctness.shape)
    print(gt_labels.shape)
    print(true_scores.shape, np.min(true_scores), np.max(true_scores))

    with open(f'./open-llm-leaderboard-results-combined/{benchmark}_model-names-in-order.json', 'r') as f:
        model_names = json.load(f)

    model_name_to_acc = {model_name: acc for model_name, acc in zip(model_names, true_scores)}

    models = setting_to_model_names[fixed_target_models_name]


    f = plt.figure(figsize=(3.2, 2.4 * 1))
    ax = plt.axes()
    diffs = []
    for name_1, name_2 in list(itertools.combinations(models, 2)):
        idx_1 = model_names.index(name_1)
        idx_2 = model_names.index(name_2)
        diff = abs(true_scores[idx_1] - true_scores[idx_2])
        diffs.append(diff * 100)
    weights = np.ones_like(diffs) / len(diffs)
    ax.hist(diffs, bins=np.arange(0,20,1), weights=weights, color='#aac8e1', edgecolor='#7eaed2')
    ax.set_ylim((0, 0.15))
    yticks = ax.get_yticks()
    yticklabels = [f'{y * 100:.1f}%' for y in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylim((0,0.15))
    ax.set_xlim((0, 20))
    ax.set_xlabel('Accuracy difference', fontweight='bold')
    ax.set_ylabel('Percent of\nmodel comparisons', fontweight='bold')
    ax.set_title('Accuracy differences', fontweight='bold')
    savefig(f'{save_dir}/{benchmark}_{fixed_target_models_name}_5b.pdf', bbox_inches='tight')
    plt.close()

    f = plt.figure(figsize=(3.2, 2.4 * 1))
    ax = plt.axes()
    accs = []
    for name in models:
        idx = model_names.index(name)
        accs.append(true_scores[idx] * 100)
    weights = np.ones_like(accs) / len(accs)
    ax.hist(accs, bins=np.arange(0,100,2), weights=weights, color='#aac8e1', edgecolor='#7eaed2')
    yticks = ax.get_yticks()
    yticklabels = [f'{y * 100:.1f}%' for y in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlim((24,76))
    ax.set_xlabel('Accuracy', fontweight='bold')
    ax.set_ylabel('Percent of models', fontweight='bold')
    ax.set_title('Model accuracies', fontweight='bold')
    savefig(f'{save_dir}/{benchmark}_{fixed_target_models_name}_5a.pdf', bbox_inches='tight')
    plt.close()

techniques = ["Random (uniform)", 'Stratified sampling (confidence)', "Anchor Points", "tinyBenchmarks"]

fig, axs = plt.subplots(4, 3, figsize=(6.4, 4.8))#, gridspec_kw={'height_ratios': [0.5, 0.5, 1]})
plt.subplots_adjust(wspace=0.25, hspace=0.25)

num_source_models = 300
for (benchmark, fixed_target_models_name) in all_data_to_plot:
    print('here')
    print(benchmark)

    model_names = setting_to_model_names[fixed_target_models_name]

    df_mdad = pd.read_csv(f'{save_dir}/{benchmark}-{fixed_target_models_name}-mdad-results.csv')

    all_fraction_sampled_points = [10, 25, 50, 100, 250, 500, 1000]
    if benchmark == 'gpqa':
        all_fraction_sampled_points = [10, 25, 50, 100, 200]


    # mdads only
    fig = plt.figure(figsize=(3.6, 1.6))
    ax = plt.axes()

    plot_df = df_mdad.loc[(df_mdad['Split'] == 'seen') \
                            & (df_mdad['Number of source models'] == num_source_models) \
                            & (df_mdad['Technique'].isin(techniques))]
    print(plot_df)
    sns.lineplot(ax=ax, data=plot_df, x='Fraction of sampled points', y=f'MDAD', hue='Technique', palette=palette, marker='o', markersize=5, alpha=0.8, linewidth=2, zorder=10, clip_on=True)
    ax.set_xscale('log')
    ymax = 0.2
    if fixed_target_models_name == 'bbh_7b_instruct':
        ymax = 0.3
    ax.set_ylim((0,ymax))
    yticks = ax.get_yticks()
    yticklabels = [f'{y * 100:.0f}' for y in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylim((0,ymax))
    ax.set_xticks(all_fraction_sampled_points)
    xticklabels = [f'{k}\n({k/benchmark_to_full_seen_size[benchmark]*100:.1f}%)' for k in all_fraction_sampled_points]
    ax.set_xticklabels(xticklabels, fontsize=7)
    for xtick, xticklabel in zip(all_fraction_sampled_points, ax.get_xticklabels()):
        highlight_color = number_to_highlight_color[xtick]
        xticklabel.set_bbox(dict(boxstyle="round,pad=0.1", fc=highlight_color, lw=0, alpha=0.4))
    ax.set_xlabel('Number of examples\nselected for micro-benchmark', fontsize=9, fontweight='bold')
    ax.set_ylabel(f'MDAD', fontsize=9, labelpad=10, fontweight='bold')
    ax.text(-0.12, 0.5, '↓', fontsize=14, fontweight='bold', va='center', ha='center', transform=ax.transAxes)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ls = [matplotlib.patches.Patch(facecolor=palette[t]) for t in techniques]
    plt.legend(ls,techniques,loc='upper right', bbox_transform=ax.transAxes, borderaxespad=0.02, fontsize=7, ncols=1)#len(techniques))#, prop={'weight':'bold'})
    savefig(f'{save_dir}/{benchmark}-{fixed_target_models_name}_5c_mdad.pdf', bbox_inches='tight')
    plt.close()