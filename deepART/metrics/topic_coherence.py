import itertools
import numpy as np

def factorial(num):
    '''
    num: int
        an non negative integer
    factorial function, eg. 5! = 120
    '''
    if isinstance(num, float) or num < 0:
        raise ValueError("num must be a non negative integer")
    elif num == 1 or num == 0:
        return 1
    else:
        return num * factorial(num-1)

def coherence(topic):
    '''
    topics: list
        a list of lists where each list contains all pairwise metrics calculated
        for that particular topic
    returns topic coherence score based on the metric used previously
    '''
    n = len(topic)
    return sum(topic) / (factorial(n) / (2*factorial(n-2)))

def cos_sim(topics):
    '''
    topics: list 
        a list of lists where each list contains words and each
        word is represented by its corresponding vector
    returns pairwise cosine similarity for all possible pairs of words belonging to each topic
    NOTE: this function will break if a topic contains only 2 words or fewer
    '''
    result = []
    for topic in topics:
        combi_two = list(itertools.combinations(topic, 2))
        topic_result = []
        for pair in combi_two:
            numerator = np.dot(pair[0], pair[1])
            denominator = np.linalg.norm(pair[0], ord=2) * np.linalg.norm(pair[1], ord=2)
            cos = numerator / denominator
            topic_result.append(cos)
        result.append(coherence(topic_result))
    return result

def dice_coeff(topics):
    '''
    topics: list 
        a list of lists where each list contains words and each
        word is represented by its corresponding vector
    returns pairwise dice coefficient for all possible pairs of words belonging to each topic
    NOTE: this function will break if a topic contains only 2 words or fewer
    '''
    result = []
    for topic in topics:
        combi_two = list(itertools.combinations(topic, 2))
        topic_result = []
        for pair in combi_two:
            numerator = 2 * sum(np.minimum(pair[0], pair[1]))
            denominator = sum(pair[0] + pair[1])
            dice = numerator / denominator
            topic_result.append(dice)
        result.append(coherence(topic_result))
    return result

def jaccard(topics):
    '''
    topics: list 
        a list of lists where each list contains words and each
        word is represented by its corresponding vector
    returns pairwise jaccard similarity for all possible pairs of words belonging to each topic
    '''
    result = []
    for topic in topics:
        combi_two = list(itertools.combinations(topic, 2))
        topic_result = []
        for pair in combi_two:
            numerator = sum(np.minimum(pair[0], pair[1]))
            denominator = sum(np.maximum(pair[0], pair[1]))
            jac = numerator / denominator
            topic_result.append(jac)
        result.append(coherence(topic_result))
    return result