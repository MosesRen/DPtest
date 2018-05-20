

from collections import Counter
import random
import numpy as np
from math import pow

def shuffle_in_unison(a, b):
    """ Shuffles two lists of equal length and keeps corresponding elements in the same index. """
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def entropy(Y):
    """ In information theory, entropy is a measure of the uncertanty of a random sample from a group. """
    
    distribution = Counter(Y)
    s = 0.0
    total = len(Y)
    for y, num_y in list(distribution.items()):
        probability_y = (num_y/total)
        s += (probability_y)*np.log(probability_y)
    return -s


def information_gain(y, y_true, y_false):
    """ The reduction in entropy from splitting data into two groups. """
    return entropy(y) - (entropy(y_true)*len(y_true) + entropy(y_false)*len(y_false))/len(y)

def gini_index(y, y_true, y_false):
    '''返回划分后的左右子树的基尼指数之和'''
    return len(y_true)/len(y)*gini(y_true)+len(y_false)/len(y)*gini(y_false)

def laplacenoise(epsilon):
    return np.random.laplace(0., 2/epsilon, 1)[0]

def gini(Y):
    gini = 0.0
    num_sample = len(Y)
    distribution = Counter(Y)
    for y, num_y in list(distribution.items()):
        num_y += laplacenoise(0.01)
        # num_sample+=num_y
    for y, num_y in list(distribution.items()):
        probability_y = (num_y/num_sample)
        gini+=pow(probability_y,2)
    return 1-gini