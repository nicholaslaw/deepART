import numpy as np
import itertools

def segregation(X, labels):
    '''
    X: numpy matrix
        contains all cluster samples
    labels: iterable, list preferred
        contains corresponding labels for each sample in X
    returns a dictionary containing samples segregated according to clusters, a list of distinct clusters
    and centroid vectors corresponding to each cluster
    '''
    segregated = {}
    for sample, label in list(zip(X, labels)):
        label = str(label)
        if label not in segregated.keys():
            segregated[label] = [sample]
        else:
            temp = segregated[label].copy()
            temp.append(sample)
            segregated[label] = temp
    for key, val in segregated.items():
        segregated[key] = np.array(val) # rows are samples
    all_clusters = list(set(labels))
    all_centroids = [val.mean(axis=0) for val in segregated.values()]
    return segregated, all_clusters, all_centroids

def dunn_val_idx(X, labels, inter_cluster=2, size=2):
    '''
    ref: https://en.wikipedia.org/wiki/Dunn_index
    X: numpy matrix
        contains all cluster samples
    labels: iterable, list preferred
        contains corresponding labels for each sample in X
    inter_cluster: integer in [0, 2]
        0: distance between the closest two data points, one in each cluster
        1: distance between the furthest two data points, one in each cluster
        2: distance between centroids of two clusters
    size: integer in [0, 2] specifying how to calculate diameter or size of a cluster
        0: maximum distance between any two points in a cluster
        1: mean distance between any two points in a cluster
        2: average distance of all points from the mean
    returns Dunn Validity Index for All Cluster
    NOTE: Depends on function segregation
    '''
    if len(X.shape) == 1:
        raise ValueError("Samples in X have inconsistent dimensions")
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array")
    if len(labels) != len(X):
        raise ValueError("Numbers of training samples and labels do not match")
    if not isinstance(inter_cluster, int) or not 0<=inter_cluster<=2:
        raise ValueError("inter_cluster must be an integer between 0 and 2 inclusive")
    if not isinstance(size, int) or not 0<=size<=2:
        raise ValueError("size must be an integer between 0 and 2 inclusive")

    # Concatenate and Sort Training Samples and Labels
    segregated, all_clusters, all_centroids = segregation(X, labels)
    # Compute Numerator
    numerator_choices = []
    if 0<=inter_cluster<=1:
        for pair in list(itertools.combinations(all_clusters, 2)):
            first = pair[0]
            second = pair[1]
            first_samples = segregated[str(first)]
            second_samples = segregated[str(second)]
            pair_result = []
            for sample in first_samples:
                for sam in second_samples:
                    pair_result.append(np.linalg.norm(sample-sam, ord=2))
            if inter_cluster == 0:
                numerator_choices.append(min(pair_result))
            else:
                numerator_choices.append(max(pair_result))
    else:
        for pair in list(itertools.combinations(all_centroids, 2)):
            first = pair[0]
            second = pair[1]
            numerator_choices.append(np.linalg.norm(first-second, ord=2))
    numerator = min(numerator_choices)

    # Compute Denominator
    denominator_choices = []
    if 0<=size<=1:
        for key, val in segregated.items():
            pair_result = []
            for pair in list(itertools.combinations(val, 2)):
                first = pair[0]
                second = pair[1]
                pair_result.append(np.linalg.norm(first-second, ord=2))
            if size == 0:
                denominator_choices.append(max(pair_result))
            else:
                denom = len(pair_result) * (len(pair_result) - 1)
                denominator_choices.append(sum(pair_result) / denom)
    else:
        for key, val in segregated.items():
            pair_result = []
            cluster_centroid = all_centroids[int(key)]
            for sample in val:
                pair_result.append(np.linalg.norm(sample-cluster_centroid, ord=2))
            denominator_choices.append(np.mean(pair_result))
    denominator = max(denominator_choices)

    return numerator / denominator

def connectivity_idx(X, labels, num_nn):
    '''
    X: numpy matrix
        contains all cluster samples
    labels: iterable, list preferred
        contains corresponding labels for each sample in X
    num_nn: int
        number of nearest neighbors to use
    returns connectivity index for all clusters
    '''
    if len(X.shape) == 1:
        raise ValueError("Samples in X have inconsistent dimensions")
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array")
    if len(labels) != len(X):
        raise ValueError("Numbers of training samples and labels do not match")
    if not isinstance(num_nn, int) or not num_nn > 0:
        raise ValueError("num_nn must be a positive integer")
    concat = list(zip(X, labels))
    samples_copy = concat.copy()
    result = []
    for idx, sam in enumerate(concat):
        sample = sam[0]
        lab = sam[1]
        temp = samples_copy.copy()
        del temp[idx]
        sample_results = []
        for other in temp:
            sample_results.append([np.linalg.norm(sample-other[0], ord=2), other[1]])
        sample_results.sort(key=lambda x: x[0])
        sample_lst = []
        for i in range(num_nn):
            if sample_results[i][1] == lab:
                sample_lst.append(0)
            else:
                sample_lst.append(1/(i+1))
        result.append(sum(sample_lst))
    return sum(result)

def jaccard_idx(labels):
    '''
    X: numpy matrix
        contains all cluster samples
    labels: list of lists
        contains corresponding labels (multilabels) for each sample in X
    return jaccard index for each cluster pair
    '''
    distinct_labs = list(set([j for i in labels for j in i]))
    all_clusters_pairs = list(itertools.combinations(distinct_labs, 2))
    results = {}
    for pair in all_clusters_pairs:
        print("pair", pair)
        first = pair[0]
        second = pair[1]
        p_01 = 0
        p_10 = 0
        p_11 = 0
        for sample in labels:
            print("sample", sample)
            if first in sample:
                if second in sample:
                    print("p11")
                    p_11 += 1
                else:
                    print("p10")
                    p_10 += 1
            elif second in sample:
                print("p01")
                p_01 += 1
        results[str(first) + "-" + str(second)] = p_11 / (p_01 + p_10 + p_11)
        print("\n")
    return results