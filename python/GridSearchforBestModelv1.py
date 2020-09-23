# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 06:07:43 2019

@author: asoni
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
 #Perform hyper-parameter optimization
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier

models = {
        'ExtraTreesClassifier': ExtraTreesClassifier(),
        'SGDClassifier': SGDClassifier(),
        'RandomForestClassifier': RandomForestClassifier(),
        'AdaBoostClassifier': AdaBoostClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        'SVC': SVC(),
    }
par = {
  #      'LogisticRegression':{
  #         'penalty': ['l2'],
  #         'C': [0.001, 0.01, 0.1, 1, 10, 100]
  #      },
    
       'SGDClassifier': {
            'penalty': ['none','l2', 'elasticnet', 'l1'],
            'max_iter': [50, 80],
            'tol': [1e-4],
            'loss': ['log', 'modified_huber']
        },
    
        'ExtraTreesClassifier': { 'n_estimators': [16, 32] },
        'RandomForestClassifier': {'max_depth': [2, 3, 5, 10], 'criterion': ['gini','entropy']},
        'AdaBoostClassifier':  { 'n_estimators': [16, 32] },
        'GradientBoostingClassifier': { 'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0] },
        'SVC': [
            {'kernel': ['linear'], 'probability': [True],'C': [1, 10]},
            {'kernel': ['rbf'],'probability': [True], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
        ]
    }


class EstimatorSelectionHelper:
    
   
       
    def __init__(self,options):

        self.par = { # 'LogisticRegression':{
                     #          'penalty': ['l2', 'l1'],
                     #          'C': [0.001, 0.01, 0.1, 1, 10, 100]
                     #       },
                        
                        'SGDClassifier': {
                                'penalty': ['none','l2', 'elasticnet', 'l1'],
                                'max_iter': [50, 80],
                                'tol': [1e-4],
                                'loss': ['log', 'modified_huber']
                            },
                        
                        'ExtraTreesClassifier': { 'n_estimators': [16, 32] },
                        'RandomForestClassifier': {'max_depth': [2, 3, 5, 10], 'criterion': ['gini','entropy']},
                        'AdaBoostClassifier':  { 'n_estimators': [16, 32] },
                        'GradientBoostingClassifier': { 'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0] },
                        'SVC': [
                                {'kernel': ['linear'], 'probability': [True],'C': [1, 10]}, 
                                {'kernel': ['rbf'],'probability': [True], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
                              ]
                            }
        self.models = {
                            'ExtraTreesClassifier': ExtraTreesClassifier(),
                            'SGDClassifier': SGDClassifier(),
                            'RandomForestClassifier': RandomForestClassifier(),
                            'AdaBoostClassifier': AdaBoostClassifier(),
                            'GradientBoostingClassifier': GradientBoostingClassifier(),
                            'SVC': SVC(),
                        }
        self.keys = self.models.keys()
        self.grid_searches = {}
        self.options = options
    
    #**grid_kwargs
    def fit(self, X, y):
        kfold = self.options['cv']
        scoring = self.options['scoring']
        for key in self.keys:
            print('Running GridSearchCV for %s.' % key)
            model = self.models[key]
            par = self.par[key]
            grid_search = GridSearchCV(model, par, cv = kfold, scoring = scoring)
            grid_search.fit(X, y)
            self.grid_searches[key] = grid_search
        print('******* GridSearch Complete ******')    #**grid_kwargs
        
          
    
    def score_summary(self, sort_by='mean_test_score'):
        frames = []
        for name, grid_search in self.grid_searches.items():
            frame = pd.DataFrame(grid_search.cv_results_)
            frame = frame.filter(regex='^(?!.*par_).*$')
            frame['estimator'] = len(frame)*[name]
            frames.append(frame)
        df = pd.concat(frames)
        
        df = df.sort_values([sort_by], ascending=False)
        df = df.reset_index()
        df = df.drop(['rank_test_score', 'index'], 1)
        
        columns = df.columns.tolist()
        columns.remove('estimator')
        columns = ['estimator']+columns
        df = df[columns]
        return df
    
def fitModelParam(modelname,params):
    if modelname == 'RandomForestClassifier':
        model = RandomForestClassifier(**params)
    elif modelname ==  'SVC':
        model = SVC(**params)
    elif modelname == 'ExtraTreesClassifier':
        model = ExtraTreesClassifier(**params)
    elif modelname == 'SGDClassifier':
        model = SGDClassifier(**params)
    elif modelname == 'AdaBoostClassifier':
        model = AdaBoostClassifier(**params)
    elif modelname == 'GradientBoostingClassifier':
        model = GradientBoostingClassifier(**params)
    else:
        model = RandomForestClassifier(**params)
    return model

def SelectAndTrainModel(i,grid,x_train, y_train): 
    gs = grid.score_summary()

    modelname = gs['estimator'][i]
    par = gs['params'][i]
    
    text_clf = Pipeline([
                                #('wscaler', StandardScaler(with_std=True,with_mean=False)),
                                ('clf', fitModelParam(modelname,par))
                        ])
    model = text_clf.fit(x_train, y_train)
    return model

