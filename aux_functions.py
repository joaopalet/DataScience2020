### Aux Functions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import ds_functions as ds
import seaborn as sns
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek



# Feature Selection

def remove_corr_features(dataframe, bound):
    df = dataframe.copy()
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > bound)]
    df = df.drop(df[to_drop], axis=1)
    
    return df

def remove_percentile_features(X, y, percentile):
    test = SelectPercentile(score_func=chi2, percentile=percentile)
    fit = test.fit(X, y)
    features = fit.transform(X)
    
    return features, y

def remove_low_variance(X, y, t):
    selector = VarianceThreshold(threshold=t)
    X_new = selector.fit_transform(X)
    
    return X_new, y

def select_k_best(dataframe, k):
    df = dataframe.copy()
    y: np.ndarray = df.pop('DEATH_EVENT').values
    X: np.ndarray = df.values
        
    test = SelectKBest(score_func=f_classif, k=k)
    selector = test.fit(X, y)
    X_new = selector.transform(X)
    
    return X_new, y, selector.get_support(indices=True)

def best_feature_selection_data1(dataframe):
    df = dataframe.copy()
    y: np.ndarray = df.pop('DEATH_EVENT').values
    X: np.ndarray = df.values
        
    test = SelectKBest(score_func=f_classif, k=2)
    X_new = test.fit_transform(X, y)
    
    return X_new, y

def best_feature_selection_data2(dataframe):
    df = dataframe.copy()
    df = remove_corr_features(df, 0.75)
    
    y: np.ndarray = df.pop(df.columns[-1]).values
    X: np.ndarray = df.values
        
    X, y = remove_low_variance(X, y, .98 * (1 - .98))
        
    X, y = remove_percentile_features(X, y, 30)
        
    return X, y
    
    
    
# Data Balancing

def balance_SMOTE(X, y):
    smote = SMOTE(sampling_strategy='minority', random_state=30)
    smote_X, smote_y = smote.fit_sample(X, y)
    return smote_X, smote_y

def balance_mix(X, y):
    smote = SMOTE(sampling_strategy='minority', random_state=30)
    tl = TomekLinks(sampling_strategy='all')
    smotetl = SMOTETomek(smote=smote, tomek=tl, random_state=30)
    smotetl_X, smotetl_y = smotetl.fit_resample(X, y)
    return smotetl_X, smotetl_y

def balance_undersample(X, y):
    rus = RandomUnderSampler(random_state=None) 
    X_rus, y_rus = rus.fit_resample(X, y)
    return X_rus, y_rus

def balance_oversample(X, y):
    ros = RandomOverSampler()
    X_ros, y_ros = ros.fit_resample(X, y)
    return X_ros, y_ros
    
    
    
# Scaling

def normalize_zscore(dataframe):
    transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(dataframe)
    norm_data_zscore = pd.DataFrame(transf.transform(dataframe), columns= dataframe.columns)

    return norm_data_zscore

def min_max_scaler(dataframe):
    transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(dataframe)
    norm_data = pd.DataFrame(transf.transform(dataframe), columns= dataframe.columns)
    
    return norm_data