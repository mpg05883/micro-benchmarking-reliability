import numpy as np
import math
import random
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from dpp_src.samplers import draw_OPE, draw_discrete_OPE, generate_Jacobi_parameters

import kmedoids
from tinybenchmarks_reimplemented import *
from sklearn.cluster import KMeans

def random_selection_naive(
    ds,
    all_data,
    gt_labels,
    models_by_correct,
    num_medoids,
    num_medoids_fraction,
    num_source_models,
    true_scores,
    source_models,
    target_models,
    seen_idxs,
    unseen_idxs,
    run_idx,
    seen_subtask_idxs = None,
):
    # partition our data
    val_slice1 = all_data[source_models, :]
    val_slice2 = all_data[target_models, :]
    val_slice1 = val_slice1[:, seen_idxs]
    val_slice2 = val_slice2[:, seen_idxs]

    models_by_correct_seen = models_by_correct[:, seen_idxs]
    models_by_correct_unseen = models_by_correct[:, unseen_idxs]
    target_models_by_correct_seen = models_by_correct_seen[target_models]
    target_models_by_correct_unseen = models_by_correct_unseen[target_models]
    model_accuracies_seen = np.sum(models_by_correct_seen, axis=1) / models_by_correct_seen.shape[1]
    model_accuracies_unseen = np.sum(models_by_correct_unseen, axis=1) / models_by_correct_unseen.shape[1]
    target_models_accuracies_seen = model_accuracies_seen[target_models]
    target_models_accuracies_unseen = model_accuracies_unseen[target_models]

    selected_idxs = np.random.choice(
        target_models_by_correct_seen.shape[1], num_medoids, replace=False
    )

    target_models_by_predicted_correct_selected = target_models_by_correct_seen[:, selected_idxs]
    estimated_scores = np.mean(target_models_by_correct_seen[:, selected_idxs], axis=1)
    true_target_model_scores = np.array(true_scores)[target_models]

    print(f"Random uniform, avg. error: {np.abs(estimated_scores-true_target_model_scores).mean():.3f}")


    return ('Random', ds, num_medoids, num_medoids_fraction, num_source_models,
            int(run_idx.split('_')[0]), target_models_accuracies_seen.tolist(),
            target_models_accuracies_unseen.tolist(), true_target_model_scores.tolist(),
            estimated_scores.tolist())

def random_selection_subtask_stratified_equal(
    ds,
    all_data,
    gt_labels,
    models_by_correct,
    num_medoids,
    num_medoids_fraction,
    num_source_models,
    true_scores,
    source_models,
    target_models,
    seen_idxs,
    unseen_idxs,
    run_idx,
    seen_subtask_idxs = None,
):
    # partition our data
    val_slice1 = all_data[source_models, :]
    val_slice2 = all_data[target_models, :]
    val_slice1 = val_slice1[:, seen_idxs]
    val_slice2 = val_slice2[:, seen_idxs]

    models_by_correct_seen = models_by_correct[:, seen_idxs]
    models_by_correct_unseen = models_by_correct[:, unseen_idxs]
    target_models_by_correct_seen = models_by_correct_seen[target_models]
    target_models_by_correct_unseen = models_by_correct_unseen[target_models]
    model_accuracies_seen = np.sum(models_by_correct_seen, axis=1) / models_by_correct_seen.shape[1]
    model_accuracies_unseen = np.sum(models_by_correct_unseen, axis=1) / models_by_correct_unseen.shape[1]
    target_models_accuracies_seen = model_accuracies_seen[target_models]
    target_models_accuracies_unseen = model_accuracies_unseen[target_models]

    num_subtasks = len(seen_subtask_idxs)
    num_points_per_subtask = np.zeros(num_subtasks).astype(int)
    num_points_per_subtask[:] = math.floor(num_medoids / num_subtasks)
    num_leftover = num_medoids % num_subtasks
    #if num_medoids < num_subtasks:
    chosen_subtasks = random.sample(range(num_subtasks), k=num_leftover)
    num_points_per_subtask[chosen_subtasks] += 1
    selected_idxs = []
    for k, idxs in zip(num_points_per_subtask, seen_subtask_idxs):
        assert(k < len(idxs))
        if k > 0:
            selected_idxs.extend(random.sample(idxs, k=k))
    selected_idxs = np.array(selected_idxs)

    target_models_by_predicted_correct_selected = target_models_by_correct_seen[:, selected_idxs]
    estimated_scores = np.mean(target_models_by_correct_seen[:, selected_idxs], axis=1)
    true_target_model_scores = np.array(true_scores)[target_models]

    print(f"Random equal, avg. error: {np.abs(estimated_scores-true_target_model_scores).mean():.3f}")

    return ('Random_Subtask_Stratified_Equal', ds, num_medoids, num_medoids_fraction, num_source_models,
            int(run_idx.split('_')[0]), target_models_accuracies_seen.tolist(),
            target_models_accuracies_unseen.tolist(), true_target_model_scores.tolist(),
            estimated_scores.tolist())

def random_selection_subtask_stratified_proportional(
    ds,
    all_data,
    gt_labels,
    models_by_correct,
    num_medoids,
    num_medoids_fraction,
    num_source_models,
    true_scores,
    source_models,
    target_models,
    seen_idxs,
    unseen_idxs,
    run_idx,
    seen_subtask_idxs = None,
):
    # partition our data
    val_slice1 = all_data[source_models, :]
    val_slice2 = all_data[target_models, :]
    val_slice1 = val_slice1[:, seen_idxs]
    val_slice2 = val_slice2[:, seen_idxs]

    models_by_correct_seen = models_by_correct[:, seen_idxs]
    models_by_correct_unseen = models_by_correct[:, unseen_idxs]
    target_models_by_correct_seen = models_by_correct_seen[target_models]
    target_models_by_correct_unseen = models_by_correct_unseen[target_models]
    model_accuracies_seen = np.sum(models_by_correct_seen, axis=1) / models_by_correct_seen.shape[1]
    model_accuracies_unseen = np.sum(models_by_correct_unseen, axis=1) / models_by_correct_unseen.shape[1]
    target_models_accuracies_seen = model_accuracies_seen[target_models]
    target_models_accuracies_unseen = model_accuracies_unseen[target_models]

    num_subtasks = len(seen_subtask_idxs)
    example_weights = np.zeros(len(seen_idxs))
    for subtask_idxs in seen_subtask_idxs:
        example_weights[subtask_idxs] = 1 / len(subtask_idxs)
    example_weights /= np.sum(example_weights)
    selected_idxs = np.random.choice(
        target_models_by_correct_seen.shape[1], num_medoids, replace=False, p=example_weights
    )
    
    target_models_by_predicted_correct_selected = target_models_by_correct_seen[:, selected_idxs]
    estimated_scores = np.mean(target_models_by_correct_seen[:, selected_idxs], axis=1)
    true_target_model_scores = np.array(true_scores)[target_models]

    print(f"Random proportional, avg. error: {np.abs(estimated_scores-true_target_model_scores).mean():.3f}")

    return ('Random_Subtask_Stratified_Proportional', ds, num_medoids, num_medoids_fraction, num_source_models,
            int(run_idx.split('_')[0]), target_models_accuracies_seen.tolist(),
            target_models_accuracies_unseen.tolist(), true_target_model_scores.tolist(),
            estimated_scores.tolist())

def stratified_random_sampling(
    ds,
    all_data,
    gt_labels,
    models_by_correct,
    num_medoids,
    num_medoids_fraction,
    num_source_models,
    true_scores,
    source_models,
    target_models,
    seen_idxs,
    unseen_idxs,
    run_idx,
    seen_subtask_idxs = None,
):
    
    '''
    The strata are defined by the mean confidence of the source models on the seen points on the correct class.
    '''

    val_slice = all_data[source_models, :][:, seen_idxs]
    gt_labels_seen = gt_labels[seen_idxs].astype(int)

    confidences = np.zeros(val_slice.shape[:2])
    for example in range(val_slice.shape[1]):
        for model in range(val_slice.shape[0]):
            confidences[model, example] = val_slice[model, example, gt_labels_seen[example]]    
    proxies = np.transpose(confidences)

    # mean across all source models for all seen points
    proxies = np.mean(proxies, axis=1)

    models_by_correct_unseen = models_by_correct[:, unseen_idxs]
    models_by_correct_seen = models_by_correct[:, seen_idxs]
    num_seen = models_by_correct_seen.shape[1]
    target_models_by_correct_seen = models_by_correct_seen[target_models]
    

    num_strata = 10 # number of proxy stratums
    kmeans = KMeans(n_clusters=num_strata, random_state=42, n_init='auto')
    strata_labels = kmeans.fit_predict(proxies.reshape(-1, 1))

    strata = [[] for _ in range(num_strata)]
    for idx, label in enumerate(strata_labels):
        strata[label].append(idx)

    selected_idxs = []
    for stratum in strata:
        stratum_size = len(stratum)
        if stratum_size == 0:
            continue

        sample_size = int(np.round(num_medoids * (stratum_size / num_seen))) # sample size for the strata
        sample_size = min(sample_size, stratum_size)

        if sample_size > 0:
            selected_idxs.extend(np.random.choice(stratum, sample_size, replace=False)) 

    remaining = num_medoids - len(selected_idxs)
    if remaining > 0:
        all_indices = np.arange(num_seen)
        unsampled = np.setdiff1d(all_indices, selected_idxs)
        if len(unsampled) >= remaining:
            selected_idxs.extend(np.random.choice(unsampled, remaining, replace=False))
        else:
            selected_idxs.extend(np.random.choice(all_indices, remaining, replace=True))
    selected_idxs = np.array(selected_idxs)

    sampling_probs = np.zeros(num_seen)
    for s, stratum in enumerate(strata):
        stratum_size = len(stratum)
        if stratum_size == 0:
            continue

        sample_size = int(np.round(num_medoids * (stratum_size / num_seen)))
        sample_size = min(sample_size, stratum_size)
        prob = sample_size / stratum_size if stratum_size > 0 else 0 # probability of sampling from the stratum
        for idx in stratum:
            sampling_probs[idx] = prob

    for idx in selected_idxs:
        if sampling_probs[idx] == 0:
            sampling_probs[idx] = num_medoids / num_seen

    estimated_scores = []
    for model_idx in range(len(target_models)):
        model_correctness = target_models_by_correct_seen[model_idx]
        ht_sum = 0.0
        for idx in selected_idxs:
            prob = sampling_probs[idx]
            if prob > 0:
                ht_sum += model_correctness[idx] / prob

        ht_estimate = ht_sum / num_seen
        estimated_scores.append(ht_estimate)

    return (
        'Stratified_Random_Sampling',
        ds,
        num_medoids,
        num_medoids_fraction,
        num_source_models,
        int(run_idx.split('_')[0]),
        np.mean(models_by_correct_seen[target_models], axis=1).tolist(),
        np.mean(models_by_correct_unseen[target_models], axis=1).tolist(),
        np.array(true_scores)[target_models].tolist(),
        estimated_scores
    )

def dpp_selection(
    ds,
    all_data,
    gt_labels,
    models_by_correct,
    num_medoids,
    num_medoids_fraction,
    num_source_models,
    true_scores,
    source_models,
    target_models,
    seen_idxs,
    unseen_idxs,
    run_idx,
    seen_subtask_idxs = None,
):
    # partition our data
    val_slice1 = all_data[source_models, :]
    val_slice2 = all_data[target_models, :]
    val_slice1 = val_slice1[:, seen_idxs]
    val_slice2 = val_slice2[:, seen_idxs]
    
    models_by_predictions = np.argmax(all_data, axis=2)
    models_by_correct_seen = models_by_correct[:, seen_idxs]
    models_by_correct_unseen = models_by_correct[:, unseen_idxs]
    target_models_by_correct_seen = models_by_correct_seen[target_models, :]
    model_accuracies_seen = np.sum(models_by_correct_seen, axis=1) / models_by_correct_seen.shape[1]
    model_accuracies_unseen = np.sum(models_by_correct_unseen, axis=1) / models_by_correct_unseen.shape[1]
    target_models_accuracies_seen = model_accuracies_seen[target_models]
    target_models_accuracies_unseen = model_accuracies_unseen[target_models]
    gt_labels_seen = gt_labels[seen_idxs].astype(int)

    estimated_scores = []
    true_target_model_scores = np.array(true_scores)[target_models]

    # DPP selection
    # get just the confidence in the correct class
    # for some reason the vectorized version doesn't yield the right shape...
    confidences = np.zeros(val_slice1.shape[:2])
    for example in range(val_slice1.shape[1]):
        for model in range(val_slice1.shape[0]):
            confidences[model, example] = val_slice1[model, example, gt_labels_seen[example]]    
    X = np.transpose(confidences)
    X = PCA(n_components=min(min(X.shape), 4), svd_solver='randomized').fit_transform(X)
    thrs = 0.01
    low, high = np.quantile(X, thrs, 0),  np.quantile(X, 1-thrs, 0)
    sli_tail = np.all(low <= X, 1) & np.all(X <= high, 1)
    final_idxs = np.arange(0, X.shape[0], 1)[sli_tail]
    X = X[sli_tail]
    X = 0.99 * (2 * (X - X.min(0)) / (X.max(0) - X.min(0)) - 1)
    n = len(X)
    # Init OPE sampler by pre-computing KDE on data
    # kde = KernelDensity(kernel="epanechnikov", bandwidth="scott").fit(X)
    # kde_distr = np.exp(kde.score_samples(X))
    # ab_coeff = generate_Jacobi_parameters(X)
    samples, weights = draw_discrete_OPE(X, num_medoids, 1)
    samples = np.squeeze(samples)
    weights = np.squeeze(weights)
    weights_normalized = weights / np.sum(weights)
    selected_idxs_seen = final_idxs[samples]
    estimated_scores = []
    true_target_model_scores = np.array(true_scores)[target_models]
    target_models_by_predicted_correct_selected = target_models_by_correct_seen[:, selected_idxs_seen]
    for model_idx in range(len(target_models)):
        # our estimated score is just the weighted mean gold label prob of correct class on selected points
        #gt_selected_points = gt_labels_seen[selected_idxs_seen]
        corrects = target_models_by_correct_seen[model_idx, selected_idxs_seen] 
        # weighted is really bad right now....
        est_score = np.sum(corrects)/corrects.shape[0]
        estimated_scores.append(est_score)
    
    return ('DPP', ds, num_medoids, num_medoids_fraction, num_source_models,
            int(run_idx.split('_')[0]), target_models_accuracies_seen.tolist(),
            target_models_accuracies_unseen.tolist(), true_target_model_scores.tolist(),
            estimated_scores)


def anchor_points_weighted(
    ds,
    all_data,
    gt_labels,
    models_by_correct,
    num_medoids,
    num_medoids_fraction,
    num_source_models,
    true_scores,
    source_models,
    target_models,
    seen_idxs,
    unseen_idxs,
    run_idx,
    seen_subtask_idxs = None,
):
    
    models_by_correct_confidences = np.zeros(all_data.shape[:2])
    for example in range(all_data.shape[1]):
        label = int(gt_labels[example])
        for model in range(all_data.shape[0]):
            models_by_correct_confidences[model, example] = all_data[model, example, label] 

    # partition our data
    val_slice1 = models_by_correct_confidences[source_models, :]
    val_slice2 = models_by_correct_confidences[target_models, :]
    val_slice1 = val_slice1[:, seen_idxs]
    val_slice2 = val_slice2[:, seen_idxs]
    # unseen_points = all_data[target_models, :]
    # unseen_points = all_data[:, unseen_idxs]

    models_by_predictions = np.argmax(all_data, axis=2)
    models_by_correct_seen = models_by_correct[:, seen_idxs]
    models_by_correct_unseen = models_by_correct[:, unseen_idxs]
    target_models_by_correct_seen = models_by_correct_seen[target_models, :]
    model_accuracies_seen = np.sum(models_by_correct_seen, axis=1) / models_by_correct_seen.shape[1]
    model_accuracies_unseen = np.sum(models_by_correct_unseen, axis=1) / models_by_correct_unseen.shape[1]
    target_models_accuracies_seen = model_accuracies_seen[target_models]
    target_models_accuracies_unseen = model_accuracies_unseen[target_models]
    gt_labels_seen = gt_labels[seen_idxs]

    estimated_scores = []
    true_target_model_scores = np.array(true_scores)[target_models]

    # perform k-medoids over embeddings of our known points

    corrs = np.corrcoef(val_slice1, rowvar=False)

    selected_idxs = kmedoids.fasterpam(
        1 - corrs, num_medoids, init="random"
    ).medoids

    # get cluster sizes
    cluster_members = np.argmax(corrs[selected_idxs, :], axis=0)
    unique, cluster_sizes = np.unique(cluster_members, return_counts=True)
    cluster_sizes = list(cluster_sizes)

    # if clusters are empty, assign size of 0
    for i in range(num_medoids):
        if i not in unique:
            cluster_sizes.insert(i, 0)

    weights = cluster_sizes / np.sum(cluster_sizes)

    for model_idx in range(val_slice2.shape[0]):
        # our estimated score is just the mean gold label prob of correct class on selected points
        votes = val_slice2[model_idx, selected_idxs]
        est_score = np.sum(weights * votes)
        estimated_scores.append(est_score)

    print(f"Anchor points weighted, avg. error: {np.abs(estimated_scores-true_target_model_scores).mean():.3f}")


    return ('Anchor_Points_Weighted', ds, num_medoids, num_medoids_fraction,
            num_source_models, int(run_idx.split('_')[0]),
            target_models_accuracies_seen.tolist(),
            target_models_accuracies_unseen.tolist(),
            true_target_model_scores.tolist(), estimated_scores)

def tinybenchmarks(
    ds,
    all_data,
    gt_labels,
    models_by_correct,
    num_medoids,
    num_medoids_fraction,
    num_source_models,
    true_scores,
    source_models,
    target_models,
    seen_idxs,
    unseen_idxs,
    run_idx,
    seen_subtask_idxs = None,
):  
    print(f'{ds}, tinyBenchmarks run {run_idx.split("_")[0]}, {num_source_models} source models, {num_medoids} num_medoids, {num_medoids_fraction} fraction of points')

    # partition our data
    val_slice1 = all_data[source_models, :]
    val_slice2 = all_data[target_models, :]
    val_slice1 = val_slice1[:, seen_idxs]
    val_slice2 = val_slice2[:, seen_idxs]
    
    models_by_predictions = np.argmax(all_data, axis=2)
    models_by_correct_seen = models_by_correct[:, seen_idxs]
    models_by_correct_unseen = models_by_correct[:, unseen_idxs]
    target_models_by_correct_seen = models_by_correct_seen[target_models, :]
    model_accuracies_seen = np.sum(models_by_correct_seen, axis=1) / models_by_correct_seen.shape[1]
    model_accuracies_unseen = np.sum(models_by_correct_unseen, axis=1) / models_by_correct_unseen.shape[1]
    target_models_accuracies_seen = model_accuracies_seen[target_models]
    target_models_accuracies_unseen = model_accuracies_unseen[target_models]
    gt_labels_seen = gt_labels[seen_idxs]

    true_target_model_scores = np.array(true_scores)[target_models]

    # get all target model predicted scores in order
    estimated_scores, target_models_by_predicted_correct_selected, selected_point_weights = \
        tinybenchmarks_wrapper(models_by_correct_seen, source_models, target_models, num_medoids, run_idx)
    
    return ('tinyBenchmarks', ds, num_medoids, num_medoids_fraction, num_source_models, int(run_idx.split('_')[0]), target_models_accuracies_seen.tolist(), target_models_accuracies_unseen.tolist(), true_target_model_scores.tolist(), estimated_scores.tolist())

def tinybenchmarks_all_num_medoids(
    ds,
    all_data,
    gt_labels,
    models_by_correct,
    all_num_medoids,
    all_num_medoids_fraction,
    num_source_models,
    true_scores,
    source_models,
    target_models,
    seen_idxs,
    unseen_idxs,
    run_idx,
    seen_subtask_idxs = None,
):  
    print(f'{ds}, tinyBenchmarks run {run_idx.split("_")[0]}, {num_source_models} source models, {all_num_medoids} num_medoids')

    # partition our data
    val_slice1 = all_data[source_models, :]
    val_slice2 = all_data[target_models, :]
    val_slice1 = val_slice1[:, seen_idxs]
    val_slice2 = val_slice2[:, seen_idxs]
    
    models_by_predictions = np.argmax(all_data, axis=2)
    models_by_correct_seen = models_by_correct[:, seen_idxs]
    models_by_correct_unseen = models_by_correct[:, unseen_idxs]
    target_models_by_correct_seen = models_by_correct_seen[target_models, :]
    model_accuracies_seen = np.sum(models_by_correct_seen, axis=1) / models_by_correct_seen.shape[1]
    model_accuracies_unseen = np.sum(models_by_correct_unseen, axis=1) / models_by_correct_unseen.shape[1]
    target_models_accuracies_seen = model_accuracies_seen[target_models]
    target_models_accuracies_unseen = model_accuracies_unseen[target_models]
    gt_labels_seen = gt_labels[seen_idxs]

    true_target_model_scores = np.array(true_scores)[target_models]

    # get all target model predicted scores in order
    all_estimated_scores, all_target_models_by_predicted_correct_selected, all_selected_point_weights = \
        tinybenchmarks_wrapper_all_num_medoids(models_by_correct_seen, source_models, target_models, all_num_medoids, run_idx)
    
    results = []
    for num_medoids, num_medoids_fraction, estimated_scores, target_models_by_predicted_correct_selected, selected_point_weights \
      in zip(all_num_medoids, all_num_medoids_fraction, all_estimated_scores, all_target_models_by_predicted_correct_selected, all_selected_point_weights):
        results.append(('tinyBenchmarks', ds, num_medoids, num_medoids_fraction,
                        num_source_models, int(run_idx.split('_')[0]),
                        target_models_accuracies_seen.tolist(),
                        target_models_accuracies_unseen.tolist(),
                        true_target_model_scores.tolist(),
                        estimated_scores.tolist()))
    return results

def tinybenchmarks_all_num_medoids_pirt(
    ds,
    all_data,
    gt_labels,
    models_by_correct,
    all_num_medoids,
    all_num_medoids_fraction,
    num_source_models,
    true_scores,
    source_models,
    target_models,
    seen_idxs,
    unseen_idxs,
    run_idx,
    seen_subtask_idxs = None,
):  
    print(f'{ds}, tinyBenchmarks p-irt run {run_idx.split("_")[0]}, {num_source_models} source models, {all_num_medoids} num_medoids')

    # partition our data
    val_slice1 = all_data[source_models, :]
    val_slice2 = all_data[target_models, :]
    val_slice1 = val_slice1[:, seen_idxs]
    val_slice2 = val_slice2[:, seen_idxs]
    
    models_by_predictions = np.argmax(all_data, axis=2)
    models_by_correct_seen = models_by_correct[:, seen_idxs]
    models_by_correct_unseen = models_by_correct[:, unseen_idxs]
    target_models_by_correct_seen = models_by_correct_seen[target_models, :]
    model_accuracies_seen = np.sum(models_by_correct_seen, axis=1) / models_by_correct_seen.shape[1]
    model_accuracies_unseen = np.sum(models_by_correct_unseen, axis=1) / models_by_correct_unseen.shape[1]
    target_models_accuracies_seen = model_accuracies_seen[target_models]
    target_models_accuracies_unseen = model_accuracies_unseen[target_models]
    gt_labels_seen = gt_labels[seen_idxs]

    true_target_model_scores = np.array(true_scores)[target_models]

    # get all target model predicted scores in order
    all_estimated_scores, all_target_models_by_predicted_correct_selected = \
        tinybenchmarks_wrapper_all_num_medoids_pirt(models_by_correct_seen, source_models, target_models, all_num_medoids, run_idx)
    
    results = []
    for num_medoids, num_medoids_fraction, estimated_scores, target_models_by_predicted_correct_selected \
      in zip(all_num_medoids, all_num_medoids_fraction, all_estimated_scores, all_target_models_by_predicted_correct_selected):
        results.append(('tinyBenchmarks (p-IRT)', ds, num_medoids, num_medoids_fraction,
                        num_source_models, int(run_idx.split('_')[0]),
                        target_models_accuracies_seen.tolist(),
                        target_models_accuracies_unseen.tolist(),
                        true_target_model_scores.tolist(),
                        estimated_scores.tolist()))
    return results

def tinybenchmarks_all_num_medoids_gpirt(
    ds,
    all_data,
    gt_labels,
    models_by_correct,
    all_num_medoids,
    all_num_medoids_fraction,
    num_source_models,
    true_scores,
    source_models,
    target_models,
    seen_idxs,
    unseen_idxs,
    run_idx,
    seen_subtask_idxs = None,
):  
    print(f'{ds}, tinyBenchmarks gp-irt run {run_idx.split("_")[0]}, {num_source_models} source models, {all_num_medoids} num_medoids')

    # partition our data
    val_slice1 = all_data[source_models, :]
    val_slice2 = all_data[target_models, :]
    val_slice1 = val_slice1[:, seen_idxs]
    val_slice2 = val_slice2[:, seen_idxs]
    
    models_by_predictions = np.argmax(all_data, axis=2)
    models_by_correct_seen = models_by_correct[:, seen_idxs]
    models_by_correct_unseen = models_by_correct[:, unseen_idxs]
    target_models_by_correct_seen = models_by_correct_seen[target_models, :]
    model_accuracies_seen = np.sum(models_by_correct_seen, axis=1) / models_by_correct_seen.shape[1]
    model_accuracies_unseen = np.sum(models_by_correct_unseen, axis=1) / models_by_correct_unseen.shape[1]
    target_models_accuracies_seen = model_accuracies_seen[target_models]
    target_models_accuracies_unseen = model_accuracies_unseen[target_models]
    gt_labels_seen = gt_labels[seen_idxs]

    true_target_model_scores = np.array(true_scores)[target_models]

    # get all target model predicted scores in order
    all_estimated_scores, all_target_models_by_predicted_correct_selected = \
        tinybenchmarks_wrapper_all_num_medoids_gpirt(models_by_correct_seen, source_models, target_models, all_num_medoids, run_idx)
    
    results = []
    for num_medoids, num_medoids_fraction, estimated_scores, target_models_by_predicted_correct_selected \
      in zip(all_num_medoids, all_num_medoids_fraction, all_estimated_scores, all_target_models_by_predicted_correct_selected):
        results.append(('tinyBenchmarks (gp-IRT)', ds, num_medoids, num_medoids_fraction,
                        num_source_models, int(run_idx.split('_')[0]),
                        target_models_accuracies_seen.tolist(),
                        target_models_accuracies_unseen.tolist(),
                        true_target_model_scores.tolist(),
                        estimated_scores.tolist()))
    return results