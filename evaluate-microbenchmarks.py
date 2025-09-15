import argparse
import numpy as np
import os
from tqdm import tqdm
import multiprocessing
from functools import partial
import logging

import math
import random
import json
import datetime
import time

from microbenchmarks import *

parser = argparse.ArgumentParser()

parser.add_argument(
    "--benchmark",
    type=str,
    required=True,
)

parser.add_argument(
    "--selection_techniques",
    nargs="+",
    help="List of point selection techniques to try",
    default=[
        "Random",
        "Anchor_Points_Weighted",
    ],
)

parser.add_argument(
    "--max_dataset_size",
    type=int,
    help="Maximum number of datapoints to keep in dataset",
    default=6000,
)

parser.add_argument(
    "--num_points",
    type=int,
    help="Number of points to forward pass the target model",
    default=5,
)
parser.add_argument(
    "--num_source_models",
    nargs="+",
    default=[10],
)
parser.add_argument(
    "--num_runs",
    type=int,
    help="Number of randomized runs to average results over",
    default=500,
)
parser.add_argument(
    "--point_counts",
    type=list,
    help="List of eval set sizes to evaluate",
    default=list(range(1, 31, 2)),  # + [30, 35, 40, 50, 100],
)
parser.add_argument(
    "--max_samples", type=int, help="maximum number of samples to use", default=5000
)
parser.add_argument(
    "--same_points",
    action='store_true',
    default=False,
)
parser.add_argument(
    "--reuse_configs",
    action='store_true',
    default=False,
)
parser.add_argument(
    "--combine_subtasks",
    action='store_true',
    default=False,
)
parser.add_argument(
    "--fixed_target_models", type=str, default=None
)

def separate_subtasks_loop(args, arg_string,
                           all_source_models, all_target_models,
                           dataset_to_all_splits_reused,
                           techniques):
    all_results = []
    all_datasets_configs = []
    times = []
    for ds in args.datasets_to_run:
        # load data
        all_data = np.load(f'./open-llm-leaderboard-results/{ds}_models-by-confidence.npy')
        models_by_correctness = np.load(f'./open-llm-leaderboard-results/{ds}_models-by-correctness.npy')
        true_scores = np.mean(models_by_correctness, axis=1)
        gt_labels = np.load(f'./open-llm-leaderboard-results/{ds}_labels.npy')

        # truncate dataset if necessary
        if all_data.shape[1] > args.max_samples:
            all_data = all_data[:, : args.max_samples, :]
            models_by_correctness = models_by_correctness[:, : args.max_samples]

        all_seen_idxs = []
        all_unseen_idxs = []
        if args.reuse_configs:
            all_seen_idxs, all_unseen_idxs = dataset_to_all_splits_reused[ds]
        else:
            # split the dataset into a seen half and an unseen half for each run
            # the seen half is used to construct the microbenchmark
            # the unseen half is used to evaluate generalization
            # these splits are the same across all techniques run on this dataset    
            for _ in range(args.num_runs):
                # take half of the data points for creating microbenchmark, half for generalization
                shuffled_point_indices = list(range(all_data.shape[1]))
                np.random.shuffle(shuffled_point_indices)
                n = all_data.shape[1]
                half_n = math.floor(n * 0.5)
                seen_idxs = shuffled_point_indices[:half_n]
                unseen_idxs = shuffled_point_indices[half_n:]
                all_seen_idxs.append(seen_idxs)
                all_unseen_idxs.append(unseen_idxs)
        all_datasets_configs.append((num_source_models, ds, all_seen_idxs, all_unseen_idxs))

        for technique_name, technique in techniques.items():
            logging.info(
                f"Running {technique_name} on {ds}"
            )

            for num_medoids_fraction in tqdm(args.point_counts):
                
                num_medoids = max(2, math.floor(num_medoids_fraction * len(all_seen_idxs[0])))
                if args.same_points:
                    num_medoids = num_medoids_fraction
                print(ds, technique_name, num_medoids, len(all_seen_idxs[0]))
                # run the different trials in parallel on CPU
                start_time = time.time()
                pool_obj = multiprocessing.Pool()
                helper = partial(
                    technique,
                    ds,
                    all_data,
                    gt_labels,
                    models_by_correctness,
                    num_medoids,
                    num_medoids_fraction,
                    num_source_models,
                    true_scores,
                )
                results = pool_obj.starmap(helper, zip(all_source_models,
                                                        all_target_models,
                                                        all_seen_idxs,
                                                        all_unseen_idxs,
                                                        [f'{i}_{timestamp}' for i in range(args.num_runs)]), 1)
                end_time = time.time()

                all_results.extend(results)

                times.append({'Dataset': ds,
                                'Number of sampled points': num_medoids,
                                'Fraction of sampled points': num_medoids_fraction,
                                'Technique': technique_name,
                                'Number of source models': num_source_models,
                                'Number of trials': args.num_runs,
                                'Elapsed time': end_time-start_time})

    with open(f'{results_dir}/results_{arg_string}_{timestamp}.json', 'w') as f:
        json.dump(all_results, f)
    with open(f'{results_dir}/configs-source-target-models_{arg_string}_{timestamp}.json', 'w') as f:
        json.dump([(args.benchmark, num_source_models, all_source_models, all_target_models)], f)
    with open(f'{results_dir}/configs-seen-unseen-idxs_{arg_string}_{timestamp}.json', 'w') as f:
        json.dump(all_datasets_configs, f)
    with open(f'{results_dir}/times_{arg_string}_{timestamp}.json', 'w') as f:
        json.dump(times, f)



def combine_subtasks_loop(args, arg_string,
                           all_source_models, all_target_models,
                           dataset_to_all_splits_reused,
                           techniques):
    all_results = []
    all_datasets_configs = []
    times = []
    # load all data
    all_data = []
    models_by_correctness = []
    gt_labels = []
    # each range is [start, end)
    subtask_idx_ranges = []
    start_idx = 0
    for ds in args.datasets_to_run:
        all_data.append(np.load(f'./open-llm-leaderboard-results/{ds}_models-by-confidence.npy'))
        models_by_correctness.append(np.load(f'./open-llm-leaderboard-results/{ds}_models-by-correctness.npy'))
        gt_labels.append(np.load(f'./open-llm-leaderboard-results/{ds}_labels.npy'))
        end_idx = start_idx + all_data[-1].shape[1]
        subtask_idx_ranges.append((start_idx, end_idx))
        start_idx = end_idx
    # if necessary, pad confidences with dummy answer options for concatenation
    task_num_answers = [confidences.shape[2] for confidences in all_data]
    if len(set(task_num_answers)) > 1:
        padded_confidences = []
        max_answers = max(task_num_answers)
        for confidences, num_answers in zip(all_data, task_num_answers):
            d = np.pad(confidences,
                       pad_width=((0, 0), (0, 0), (0, max_answers - num_answers)),
                       mode='constant', constant_values=0)
            padded_confidences.append(d)
        all_data = padded_confidences

    ds = f'{args.benchmark}_all'
    all_data = np.concatenate(all_data, axis=1)
    models_by_correctness = np.concatenate(models_by_correctness, axis=1)
    gt_labels = np.concatenate(gt_labels)
    true_scores = np.mean(models_by_correctness, axis=1)
    print(all_data.shape)
    print(models_by_correctness.shape)
    print(gt_labels.shape)
    print(true_scores.shape, np.min(true_scores), np.max(true_scores))

    all_seen_idxs = []
    all_unseen_idxs = []
    all_seen_subtask_idxs = []
    if args.reuse_configs:
        all_seen_idxs, all_unseen_idxs = dataset_to_all_splits_reused[ds]
    else:
        # split the dataset into a seen half and an unseen half for each run
        # the seen half is used to construct the microbenchmark
        # the unseen half is used to evaluate generalization
        # these splits are the same across all techniques run on this dataset    
        for _ in range(args.num_runs):
            # take half of each subtask for creating microbenchmark, half for generalization
            seen_idxs = []
            unseen_idxs = []
            seen_subtask_idxs = []
            in_order_start = 0
            for start, end in subtask_idx_ranges:
                shuffled_subtask_indices = list(range(start, end))
                np.random.shuffle(shuffled_subtask_indices)
                n = len(shuffled_subtask_indices)
                half_n = math.floor(n * 0.5)
                seen_idxs.extend(shuffled_subtask_indices[:half_n])
                unseen_idxs.extend(shuffled_subtask_indices[half_n:])
                #seen_subtask_idxs.append(shuffled_subtask_indices[:half_n])
                # map new indices to subtasks
                in_order_end = in_order_start + half_n
                seen_subtask_idxs.append(list(range(in_order_start, in_order_end)))
                in_order_start = in_order_end
            all_seen_idxs.append(seen_idxs)
            all_unseen_idxs.append(unseen_idxs)
            all_seen_subtask_idxs.append(seen_subtask_idxs)
    all_datasets_configs.append((num_source_models, ds, all_seen_idxs, all_unseen_idxs))

    for technique_name, technique in techniques.items():
        logging.info(
            f"Running {technique_name} on {ds}"
        )

        if technique_name == 'tinyBenchmarks':
            all_num_medoids = []
            for num_medoids_fraction in args.point_counts:
                if args.same_points:
                    all_num_medoids.append(num_medoids_fraction)
                else:
                    all_num_medoids.append(max(2, math.floor(num_medoids_fraction * len(all_seen_idxs[0]))))
            # run the different trials in parallel on CPU
            start_time = time.time()
            pool_obj = multiprocessing.Pool()
            helper = partial(
                technique,
                ds,
                all_data,
                gt_labels,
                models_by_correctness,
                all_num_medoids,
                args.point_counts,
                num_source_models,
                true_scores,
            )
            results = pool_obj.starmap(helper, zip(all_source_models,
                                                    all_target_models,
                                                    all_seen_idxs,
                                                    all_unseen_idxs,
                                                    [f'{i}_{timestamp}' for i in range(args.num_runs)],
                                                    all_seen_subtask_idxs), 1)
            end_time = time.time()
            print(len(results))
            all_results.extend([single_medoid_result for many_medoid_results in results \
                                                     for single_medoid_result in many_medoid_results])

            times.append({'Dataset': ds,
                        'Technique': f'{technique_name}_all-num_medoids',
                        'Number of source models': num_source_models,
                        'Number of trials': args.num_runs,
                        'Elapsed time': end_time-start_time})
        else:

            for num_medoids_fraction in tqdm(args.point_counts):
                
                num_medoids = max(2, math.floor(num_medoids_fraction * len(all_seen_idxs[0])))
                if args.same_points:
                    num_medoids = num_medoids_fraction
                print(ds, technique_name, num_medoids, len(all_seen_idxs[0]))
                # run the different trials in parallel on CPU
                start_time = time.time()
                pool_obj = multiprocessing.Pool()
                helper = partial(
                    technique,
                    ds,
                    all_data,
                    gt_labels,
                    models_by_correctness,
                    num_medoids,
                    num_medoids_fraction,
                    num_source_models,
                    true_scores,
                )
                results = pool_obj.starmap(helper, zip(all_source_models,
                                                        all_target_models,
                                                        all_seen_idxs,
                                                        all_unseen_idxs,
                                                        [f'{i}_{timestamp}' for i in range(args.num_runs)],
                                                        all_seen_subtask_idxs), 1)
                end_time = time.time()

                all_results.extend(results)

                times.append({'Dataset': ds,
                                'Number of sampled points': num_medoids,
                                'Fraction of sampled points': num_medoids_fraction,
                                'Technique': technique_name,
                                'Number of source models': num_source_models,
                                'Number of trials': args.num_runs,
                                'Elapsed time': end_time-start_time})

    with open(f'{results_dir}/results_{arg_string}_{timestamp}.json', 'w') as f:
        json.dump(all_results, f)
    with open(f'{results_dir}/configs-source-target-models_{arg_string}_{timestamp}.json', 'w') as f:
        json.dump([(args.benchmark, num_source_models, all_source_models, all_target_models)], f)
    with open(f'{results_dir}/configs-seen-unseen-idxs_{arg_string}_{timestamp}.json', 'w') as f:
        json.dump(all_datasets_configs, f)
    with open(f'{results_dir}/times_{arg_string}_{timestamp}.json', 'w') as f:
        json.dump(times, f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()

    results_dir = f'./results'
    if args.same_points:
        results_dir += '-same-points'
    if args.combine_subtasks:
        if args.same_points:
            results_dir = './results-combine-subtasks'
        else:
            results_dir = './results-combine-subtasks-same-percent'
    if args.fixed_target_models:
        results_dir += '_fixed-target-models'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    available_techniques = {
        "Random": random_selection_naive,
        "Anchor_Points_Weighted": anchor_points_weighted,
        "Anchor_Points_Predictor": anchor_points_predictor,
        "DPP": dpp_selection,
        "tinyBenchmarks": tinybenchmarks_all_num_medoids,
        "tinyBenchmarks_pirt": tinybenchmarks_all_num_medoids_pirt,
        "tinyBenchmarks_gpirt": tinybenchmarks_all_num_medoids_gpirt,
        'Random_Subtask_Stratified_Equal': random_selection_subtask_stratified_equal,
        'Random_Subtask_Stratified_Proportional': random_selection_subtask_stratified_proportional,
        "Stratified_Random_Sampling": stratified_random_sampling
    }

    techniques = {
        k: v for k, v in available_techniques.items() if k in args.selection_techniques
    }
    
    benchmark_to_datasets = {
        'mmlu': [
                'mmlu_abstract_algebra',
                'mmlu_anatomy',
                'mmlu_astronomy',
                'mmlu_business_ethics',
                'mmlu_clinical_knowledge',
                'mmlu_college_biology',
                'mmlu_college_chemistry',
                'mmlu_college_computer_science',
                'mmlu_college_mathematics',
                'mmlu_college_medicine',
                'mmlu_college_physics',
                'mmlu_computer_security',
                'mmlu_conceptual_physics',
                'mmlu_econometrics',
                'mmlu_electrical_engineering',
                'mmlu_elementary_mathematics',
                'mmlu_formal_logic',
                'mmlu_global_facts',
                'mmlu_high_school_biology',
                'mmlu_high_school_chemistry',
                'mmlu_high_school_computer_science',
                'mmlu_high_school_european_history',
                'mmlu_high_school_geography',
                'mmlu_high_school_government_and_politics',
                'mmlu_high_school_macroeconomics',
                'mmlu_high_school_mathematics',
                'mmlu_high_school_microeconomics',
                'mmlu_high_school_physics',
                'mmlu_high_school_psychology',
                'mmlu_high_school_statistics',
                'mmlu_high_school_us_history',
                'mmlu_high_school_world_history',
                'mmlu_human_aging',
                'mmlu_human_sexuality',
                'mmlu_international_law',
                'mmlu_jurisprudence',
                'mmlu_logical_fallacies',
                'mmlu_machine_learning',
                'mmlu_management',
                'mmlu_marketing',
                'mmlu_medical_genetics',
                'mmlu_miscellaneous',
                'mmlu_moral_disputes',
                'mmlu_moral_scenarios',
                'mmlu_nutrition',
                'mmlu_philosophy',
                'mmlu_prehistory',
                'mmlu_professional_accounting',
        ],
        'bbh': [
            "bbh_boolean_expressions",
            "bbh_causal_judgement",
            "bbh_date_understanding",
            "bbh_disambiguation_qa",
            "bbh_formal_fallacies",
            "bbh_geometric_shapes",
            "bbh_hyperbaton",
            "bbh_logical_deduction_five_objects",
            "bbh_logical_deduction_seven_objects",
            "bbh_logical_deduction_three_objects",
            "bbh_movie_recommendation",
            "bbh_navigate",
            "bbh_object_counting",
            "bbh_penguins_in_a_table",
            "bbh_reasoning_about_colored_objects",
            "bbh_ruin_names",
            "bbh_salient_translation_error_detection",
            "bbh_snarks",
            "bbh_sports_understanding",
            "bbh_temporal_sequences",
            "bbh_tracking_shuffled_objects_five_objects",
            "bbh_tracking_shuffled_objects_seven_objects",
            "bbh_tracking_shuffled_objects_three_objects",
            "bbh_web_of_lies",
        ],
        'mmlu-pro': [
            'mmlu-pro_philosophy',
            'mmlu-pro_law',
            'mmlu-pro_engineering',
            'mmlu-pro_chemistry',
            'mmlu-pro_business',
            'mmlu-pro_psychology',
            'mmlu-pro_health',
            'mmlu-pro_biology',
            'mmlu-pro_history',
            'mmlu-pro_other',
            'mmlu-pro_economics',
            'mmlu-pro_physics',
            'mmlu-pro_computer science',
            'mmlu-pro_math'
        ],
        'gpqa': [
            'gpqa_main'
        ],
    }

    possible_fixed_target_models = {
        'mmlu-pro_7b_instruct': ['Intel__neural-chat-7b-v3', 'Deci__DeciLM-7B-instruct', 'togethercomputer__Llama-2-7B-32K-Instruct', 'internlm__internlm2_5-7b-chat', 'google__gemma-7b-it', 'mistralai__Mistral-7B-Instruct-v0.2', 'ibm-granite__granite-7b-instruct', 'deepseek-ai__deepseek-llm-7b-chat', 'tiiuae__Falcon3-Mamba-7B-Instruct', 'google__gemma-1.1-7b-it', 'Qwen__Qwen2-7B-Instruct', 'allenai__OLMo-2-1124-7B-Instruct', 'tiiuae__falcon-7b-instruct', 'tiiuae__Falcon3-7B-Instruct', 'Qwen__Qwen2.5-Coder-7B-Instruct', 'meta-llama__Llama-2-7b-chat-hf', 'Qwen__Qwen1.5-7B-Chat', 'allenai__OLMo-7B-Instruct-hf', 'Qwen__Qwen2.5-7B-Instruct', 'Qwen__Qwen2.5-7B-Instruct-1M', 'Qwen__Qwen2-VL-7B-Instruct', 'Qwen__Qwen2.5-Math-7B-Instruct', 'mistralai__Mistral-7B-Instruct-v0.3', 'togethercomputer__RedPajama-INCITE-7B-Instruct', 'nvidia__AceInstruct-7B', 'mistralai__Mistral-7B-Instruct-v0.1', 'nvidia__AceMath-7B-Instruct', 'argilla__notus-7b-v1', 'mlabonne__NeuralBeagle14-7B', 'HuggingFaceH4__zephyr-7b-gemma-v0.1', 'mlabonne__AlphaMonarch-7B', 'lmsys__vicuna-7b-v1.3', 'CohereForAI__c4ai-command-r7b-12-2024', 'NousResearch__Nous-Hermes-2-Mistral-7B-DPO', 'ibm__merlinite-7b', 'NousResearch__Nous-Hermes-llama-2-7b', 'togethercomputer__RedPajama-INCITE-7B-Chat', 'Intel__neural-chat-7b-v3-2', 'Intel__neural-chat-7b-v3-3', 'Intel__neural-chat-7b-v3-1', 'cognitivecomputations__dolphin-2.9.2-qwen2-7b', 'HuggingFaceH4__zephyr-7b-beta', 'teknium__OpenHermes-7B', 'Open-Orca__Mistral-7B-OpenOrca', 'cognitivecomputations__dolphin-2.9.3-mistral-7B-32k', 'berkeley-nest__Starling-LM-7B-alpha', 'HuggingFaceH4__zephyr-7b-alpha', 'databricks__dolly-v2-7b', 'lmsys__vicuna-7b-v1.5', 'teknium__CollectiveCognition-v1.1-Mistral-7B', 'nvidia__AceMath-7B-RM', 'togethercomputer__LLaMA-2-7B-32K', 'microsoft__Orca-2-7b', 'teknium__OpenHermes-2.5-Mistral-7B', 'deepseek-ai__DeepSeek-R1-Distill-Qwen-7B', 'NousResearch__Hermes-2-Pro-Mistral-7B'],
        'mmlu-pro_7b_base': ['togethercomputer__RedPajama-INCITE-7B-Base', 'google__gemma-7b', 'deepseek-ai__deepseek-llm-7b-base', 'tiiuae__Falcon3-Mamba-7B-Base', 'Qwen__Qwen1.5-7B', 'Qwen__Qwen2-7B', 'allenai__OLMo-1.7-7B-hf', 'ibm-granite__granite-7b-base', 'Qwen__Qwen2.5-7B', 'tiiuae__falcon-7b', 'tiiuae__Falcon3-7B-Base', 'mlabonne__Beyonder-4x7B-v3', 'mistralai__Mistral-7B-v0.3', 'mistralai__Mistral-7B-v0.1', 'Qwen__Qwen2.5-Coder-7B', 'mosaicml__mpt-7b', 'mistral-community__Mistral-7B-v0.2', 'Qwen__Qwen2-Math-7B', 'NousResearch__Yarn-Mistral-7b-128k', 'NousResearch__Yarn-Mistral-7b-64k', 'allenai__OLMo-7B-hf', 'meta-llama__Llama-2-7b-hf', 'Deci__DeciLM-7B', 'bigscience__bloom-7b1', 'tiiuae__falcon-mamba-7b', 'NousResearch__Yarn-Llama-2-7b-128k', 'bigcode__starcoder2-7b', 'Qwen__Qwen2.5-Math-7B'],
        'mmlu-pro_8b_instruct': ['meta-llama__Llama-3.1-8B', 'CohereForAI__aya-expanse-8b', 'openchat__openchat-3.6-8b-20240522', 'ibm-granite__granite-3.2-8b-instruct', 'nvidia__Mistral-NeMo-Minitron-8B-Instruct', 'mlabonne__OrpoLlama-3-8B', 'mistralai__Ministral-8B-Instruct-2410', 'gradientai__Llama-3-8B-Instruct-Gradient-1048k', 'ibm-granite__granite-3.1-8b-instruct', 'allenai__Llama-3.1-Tulu-3-8B', 'mlabonne__Meta-Llama-3.1-8B-Instruct-abliterated', 'meta-llama__Llama-3.1-8B-Instruct', 'NousResearch__Hermes-3-Llama-3.1-8B', 'NousResearch__Hermes-2-Pro-Llama-3-8B', 'argilla-warehouse__Llama-3.1-8B-MagPie-Ultra', 'Salesforce__LLaMA-3-8B-SFR-Iterative-DPO-R', 'allenai__Llama-3.1-Tulu-3-8B-DPO', 'TencentARC__LLaMA-Pro-8B-Instruct', 'nvidia__OpenMath2-Llama3.1-8B', 'cognitivecomputations__dolphin-2.9.4-llama3.1-8b', 'allenai__Llama-3.1-Tulu-3-8B-SFT', 'cognitivecomputations__Dolphin3.0-Llama3.1-8B', 'internlm__internlm2_5-1_8b-chat', 'ibm-granite__granite-3.0-8b-instruct', 'meta-llama__Meta-Llama-3-8B-Instruct', 'internlm__internlm2-chat-1_8b', 'CohereForAI__aya-23-8B', 'NousResearch__Hermes-2-Theta-Llama-3-8B', 'mlabonne__Daredevil-8B', 'mlabonne__NeuralDaredevil-8B-abliterated', 'mlabonne__Daredevil-8B-abliterated', 'mlabonne__ChimeraLlama-3-8B-v3', 'mlabonne__ChimeraLlama-3-8B-v2', 'abacusai__Llama-3-Smaug-8B'],
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

    args.datasets_to_run = benchmark_to_datasets[args.benchmark]

    ##args.point_counts = [0.02, 0.04, 0.08, 0.16, 0.24]#, 0.32, 0.4]
    args.point_counts = [0.02, 0.04, 0.08, 0.16, 0.24, 0.32, 0.4]
    if args.same_points:
        args.point_counts = [2, 5, 10, 20, 30, 40, 50]

    if args.combine_subtasks:
        if args.same_points:
            args.point_counts = [10, 25, 50, 100, 250, 500, 1000]
        else:
            args.point_counts = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.24]

    if args.benchmark == 'gpqa':
        args.point_counts = [10, 25, 50, 100, 200]

    # get the total number of models
    temp_all_data = np.load(f'./open-llm-leaderboard-results/{args.datasets_to_run[0]}_models-by-confidence.npy')
    num_models = temp_all_data.shape[0]
    print(f'Total number of models: {num_models}')

    args.num_source_models = [int(n) for n in args.num_source_models]

    for num_source_models in args.num_source_models:

        arg_string = f'{args.benchmark}_{args.point_counts}-points_{num_source_models}-source-models_{args.num_runs}-runs'

        if args.combine_subtasks:
            arg_string += '_combine-subtasks'
        
        if args.fixed_target_models:
            arg_string += f'_fixed-target-models-{args.fixed_target_models}'

        # decide which models are source and which are target for each run
        # these are the same across datasets
        all_source_models = []
        all_target_models = []
        if args.reuse_configs:
            with open(f'./configs-to-reuse/configs-source-target-models_{arg_string}.json', 'r') as f:
                d = json.load(f)
                all_source_models = d[0][2]
                all_target_models = d[0][3]
        else:
            # generate new source and target model splits
            for _ in range(args.num_runs):
                if args.fixed_target_models:
                    with open(f'./open-llm-leaderboard-results/{args.datasets_to_run[0]}_model-names-in-order.json', 'r') as f:
                        all_model_names_in_order = json.load(f)
                    target_model_names = possible_fixed_target_models[args.fixed_target_models]
                    target_models = [all_model_names_in_order.index(model_name) for model_name in target_model_names]
                else:
                    target_models = random.sample(range(num_models), k=50)
                source_models = random.sample(list(set(range(num_models)).difference(target_models)), num_source_models)
                all_target_models.append(target_models)
                all_source_models.append(source_models)
        
        # load dataset seen/unseen splits if necessary
        dataset_to_all_splits_reused = None
        if args.reuse_configs:
            with open(f'./configs-to-reuse/configs-seen-unseen-idxs_{arg_string}.json', 'r') as f:
                config_data = json.load(f)
            dataset_to_all_splits_reused = {}
            for d in config_data:
                dataset_to_all_splits_reused[d[1]] = (d[2], d[3])

        if args.combine_subtasks:
            combine_subtasks_loop(args, arg_string, all_source_models, all_target_models,
                                   dataset_to_all_splits_reused, techniques)
        else:
            separate_subtasks_loop(args, arg_string, all_source_models, all_target_models,
                                   dataset_to_all_splits_reused, techniques)
        
    print('Done')