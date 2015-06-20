# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

from  sklearn.preprocessing import normalize
from utils import readfile, roc_report, generate_pairs
from metric_learn import distances_pairs
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

X_train_facile, Y_train_facile = readfile('data_train_facile',test=False)

X_train_facile = normalize(X_train_facile)
pairs_idx, pairs_label = generate_pairs(Y_train_facile, 1000, 0.1)
#X_train_facile = normalize(X_train_facile[:10000])
#pairs_idx, pairs_label = generate_pairs(Y_train_facile[:10000], 1000, 0.1)

scores = []
possible_distances =[("Cosine",distance.cosine),
                     ("BrayCurtis",distance.braycurtis),
("Euclidean",distance.euclidean),("Manhattan",distance.cityblock),("Chebyshev",distance.chebyshev),("Hamming",distance.hamming),
("Correlation",distance.correlation) ]

for (name,func) in possible_distances:
    print name
    dist = distances_pairs(X_train_facile, pairs_idx,func)
    #print dist
    score_facile , score_difficile = roc_report(pairs_label,dist,name)
    scores.append((name,score_facile,score_difficile))

scores = pd.DataFrame(scores,columns = ["id","facile","difficile"])
plt.legend(loc='best')
plt.show()


number_features = range(10,1500,100)
rng = np.random.RandomState(42)
scores_nbFeatures_easy = []
scores_nbFeatures_hard = []

for nb_feature in number_features:
    sample = np.random.sample(nb_feature)
    I = rng.randint(0,X_train_facile.shape[1],size=nb_feature)
    X_tmp = X_train_facile[:,I]
    print X_tmp.shape
    dist = distances_pairs(X_tmp, pairs_idx,distance.braycurtis)
    score_facile , score_difficile = roc_report(pairs_label,dist,"braycurtis",False)
    scores_nbFeatures_easy.append(score_facile)
    scores_nbFeatures_hard.append(score_difficile)

plt.plot(number_features,scores_nbFeatures_easy,label='score facile')
plt.plot(number_features,scores_nbFeatures_hard,label='score difficile')
plt.legend(loc='best')
plt.savefig('Importance du nombre de features.png')
plt.show()

number_features = range(10,1500,100)
rng = np.random.RandomState(42)
scores_nbFeatures_easy = []
scores_nbFeatures_hard = []
scores_PCA_easy = []
scores_PCA_hard = []

for nb_feature in number_features:
    PCA_model = PCA(number_features)
    X_tmp = PCA_model.fit_transform(X_train_facile)
    print X_tmp.shape
    score_easy = {}
    score_hard = {}
    for (name,func) in possible_distances:
        dist = distances_pairs(X_tmp, pairs_idx,func)
        score_facile , score_difficile = roc_report(pairs_label,dist,name,False)
        score_easy[name] = score_facile
        score_hard[name] = score_difficile
    scores_PCA_easy.append(score_easy)
    scores_PCA_hard.append(score_hard)
    print pd.dataframe(scores_PCA_easy)
    
plt.plot(number_features,scores_nbFeatures_easy,label='score facile')
plt.plot(number_features,scores_nbFeatures_hard,label='score difficile')
plt.legend(loc='best')
plt.savefig('Importance de la PCA.png')
plt.show()

'''
X_train_hard, Y_train_hard = readfile('data_train_difficile')
X_test_facile, Y_test_facile = readfile('data_test_facile')
X_test_hard, Y_test_hard = readfile('data_test_difficile')
'''