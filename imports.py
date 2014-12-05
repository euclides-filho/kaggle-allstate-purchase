"""
Project: http://www.kaggle.com/c/allstate-purchase-prediction-challenge
Ranking: 9th from 1571 teams
Work Period: 12-may-2014 to 19-may-2014
Author:  Euclides Fernandes Filho
email:   euclides5414@gmail.com
"""
#-------------------------------------------------------------
from os import path
ROOT = path.abspath('..') + '\\'


import numpy as np
import pandas as pd
from os import path
from time import time
from datetime import datetime
import random
random.seed(int(time()))
SEED = 17

import multiprocessing
from multiprocessing import Pool

CORES =  multiprocessing.cpu_count()

from sklearn.preprocessing import Imputer
from sklearn.metrics import f1_score,precision_score,recall_score, accuracy_score
from sklearn.metrics import make_scorer
#----------------------------------------------------------------------
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import  LogisticRegression
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier

def F1(y_true,y_pred):
    average= 'weighted'
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred,average=average)
    recall = recall_score(y_true, y_pred,average=average)
    f1 = f1_score(y_true, y_pred,average=average)    
    print 'Accuracy\t%0.4f\tPrecision\t%0.4f\tRecall\t%0.4f\tF1\t%0.4f' % (accuracy,precision,recall,f1)
    return f1

class AdaBoostClassifierDTC(AdaBoostClassifier):
    def __init__(self,n_estimators=50, learning_rate=1.0, algorithm='SAMME.R',\
        criterion='gini', splitter='best', max_depth=5, min_samples_split=2, min_samples_leaf=1,\
        max_features=None, random_state=None, min_density=None, compute_importances=None):

        base_estimator=DecisionTreeClassifier()
        self.base_estimator = base_estimator
        self.base_estimator_class = self.base_estimator.__class__
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.splitter = splitter
        self.max_depth = max_depth
        self.criterion = criterion
        self.max_features = max_features
        self.min_density = min_density
        self.random_state = random_state
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.compute_importances = compute_importances
        
        self.estimator = self.base_estimator_class(criterion=self.criterion, splitter=self.splitter, max_depth=self.max_depth,\
                min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,\
                random_state=self.random_state, min_density=self.min_density, compute_importances=self.compute_importances)
        
        AdaBoostClassifier.__init__(self, base_estimator=self.estimator, n_estimators=self.n_estimators, learning_rate=self.learning_rate, algorithm=self.algorithm)


