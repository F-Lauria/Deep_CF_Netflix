import pandas as pd
import numpy as np
import math

import fastai
from fastai import *
from fastai.collab import *
from fastai.callbacks import *



def MF_fastai_Gsearch(train_val,n_batch_sizes,lr,nr_epochs):
    column_names = n_batch_sizes #batch sizes 
    df = pd.DataFrame(columns = column_names)
    row = []
    
    for i in lr:
        for j in n_batch_sizes:
            data = CollabDataBunch.from_df(train_val, valid_pct=0.2,seed=2,bs=j)
            learn = collab_learner(data, n_factors=10, y_range=(1,5) , metrics=[rmse],
                           callback_fns=[partial(EarlyStoppingCallback, min_delta=0.01, patience=2, mode="min", monitor="root_mean_squared_error")])
            learn.fit_one_cycle(nr_epochs,i)
            result = learn.validate()
            row.append((math.sqrt(result[0])))
        a_series = pd.Series(row, index = df.columns)
        df = df.append(a_series, ignore_index=True)
        row = []
        
    df.insert(0, "Learning_rate", lr, True)
    df.set_index('Learning_rate',inplace = True)
    
    return df
    
def MLP_fastai_find_best_layers(train_val,n_batch_sizes,lr,nr_epochs,layers):
    
    rmse = []
    
    for i in layers:
        data = CollabDataBunch.from_df(train_val, valid_pct=0.2,seed=2,bs=n_batch_sizes)
        learn = collab_learner(data, use_nn=True, emb_szs={'userID': 10, 'movieID':10}, layers=i, y_range=[1, 5])
        learn.fit_one_cycle(nr_epochs,lr)
        result = learn.validate()
        rmse.append((math.sqrt(result[0])))
        
    min_rmse = min(rmse)
    index = np.argmin(rmse)
    best_layers = layers[index]
    
    return (min_rmse,best_layers)
    
def MLP_fastai_factors(train_val,n_batch_sizes,lr,nr_epochs,layers,latent_factors):
    
    rmse = []
    n_latent_factors = []
    
    for i in latent_factors:
        data = CollabDataBunch.from_df(train_val, valid_pct=0.2,seed=2,bs=n_batch_sizes)
        learn = collab_learner(data, use_nn=True, emb_szs={'userID': i, 'movieID':i}, layers=layers, y_range=[1, 5])
        learn.fit_one_cycle(nr_epochs,lr)
        result = learn.validate()
        rmse.append((math.sqrt(result[0])))
        n_latent_factors.append(i)
        
    
    return (rmse,n_latent_factors)
    

def MLP_fastai_layers(train_val,n_batch_sizes,lr,nr_epochs,layers,latent_factors):
    
    n_layers = []
    for layer in layers:
        num = len(layer)
        n_layers.append(num)
        
    rmse = []
    
    for i in layers:
        data = CollabDataBunch.from_df(train_val, valid_pct=0.2,seed=2,bs=n_batch_sizes)
        learn = collab_learner(data, use_nn=True, emb_szs={'userID': latent_factors, 'movieID':latent_factors},
        layers=i, y_range=[1, 5])
        learn.fit_one_cycle(nr_epochs,lr)
        result = learn.validate()
        rmse.append((math.sqrt(result[0])))
        
    return (rmse,n_layers)