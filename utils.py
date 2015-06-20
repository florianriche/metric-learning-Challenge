# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy import io
import bisect
import numpy as np

def readfile(file, test=True):
    mat_file = io.loadmat(file)
    X = mat_file['X']
    if test:
        Y = mat_file['pairs']
    else:
        Y = mat_file['label']
    return X,Y


def roc_report(pairs_label,dist,name,plot=True):
    fpr, tpr, thresholds = metrics.roc_curve(pairs_label, -dist)
    if plot:
        plt.plot(fpr, tpr, label= name)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
    score_facile = 1.0 - tpr[bisect.bisect(fpr, 0.001) - 1]
    idx = (np.abs(fpr + tpr - 1.)).argmin()
    score_difficile = (fpr[idx]+(1-tpr[idx]))/2
    return score_facile,score_difficile
    
    
    
def generate_pairs(label, n_pairs, positive_ratio, random_state=42):
    """Generate a set of pair indices
    
    Parameters
    ----------
    label : array, shape (n_samples, 1)
        Label vector
    n_pairs : int
        Number of pairs to generate
    positive_ratio : float
        Positive to negative ratio for pairs
    random_state : int
        Random seed for reproducibility
        
    Output
    ------
    pairs_idx : array, shape (n_pairs, 2)
        The indices for the set of pairs
    label_pairs : array, shape (n_pairs, 1)
        The pair labels (+1 or -1)
    """
    rng = np.random.RandomState(random_state)
    n_samples = label.shape[0]
    pairs_idx = np.zeros((n_pairs, 2), dtype=int)
    pairs_idx[:, 0] = rng.randint(0, n_samples, n_pairs)
    rand_vec = rng.rand(n_pairs)
    for i in range(n_pairs):
        if rand_vec[i] <= positive_ratio:
            idx_same = np.where(label == label[pairs_idx[i, 0]])[0]
            idx2 = rng.randint(idx_same.shape[0])
            pairs_idx[i, 1] = idx_same[idx2]
        else:
            idx_diff = np.where(label != label[pairs_idx[i, 0]])[0]
            idx2 = rng.randint(idx_diff.shape[0])
            pairs_idx[i, 1] = idx_diff[idx2]
    pairs_label = 2.0 * (label[pairs_idx[:, 0]] == label[pairs_idx[:, 1]]) - 1.0
    return pairs_idx, pairs_label