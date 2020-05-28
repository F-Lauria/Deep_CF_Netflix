import pandas as pd
from sklearn.metrics import mean_squared_error
import math
import surprise
import numpy as np


# function from microsoft
def predict(
    algo,
    data,
    usercol,
    itemcol,
    predcol,
):
    """Computes predictions of an algorithm from Surprise on the data. Can be used for computing rating metrics like RMSE.
    
    Args:
        algo (surprise.prediction_algorithms.algo_base.AlgoBase): an algorithm from Surprise
        data (pd.DataFrame): the data on which to predict
        usercol (str): name of the user column
        itemcol (str): name of the item column
    
    Returns:
        pd.DataFrame: dataframe with usercol, itemcol, predcol
    """
    predictions = [
        algo.predict(getattr(row, usercol), getattr(row, itemcol))
        for row in data.itertuples()
    ]
    predictions = pd.DataFrame(predictions)
    predictions = predictions.rename(
        index=str, columns={"uid": usercol, "iid": itemcol, "est": predcol}
    )
    return predictions.drop(["details", "r_ui"], axis="columns")

def compute_rmse(true,pred):
    return math.sqrt(mean_squared_error(true, pred))

def fit_predict(train,test,algo,n_factors,n_epochs, use_PMF = False):
    
    if use_PMF == False:
        #print("ImprovedRSVD created")
        model = algo(random_state=0, n_factors=n_factors, n_epochs=n_epochs, verbose=False)
    else:
        #print("PMF created")
        model = algo(random_state=0, n_factors=n_factors, n_epochs=n_epochs, verbose=False, biased = False) 
        
    model.fit(train) # fit
    model_pred = predict(model, test , usercol='userID', itemcol='movieID',predcol="rating") # predict
    
    true_pred = test["rating"].values
    pred = model_pred["rating"].values
    rmse = compute_rmse(true_pred,pred)
    #print("Finished")

    return (rmse,model)
    
    
def epochs_tuning(train,val,n_epochs, PMF = False):
    rmse = []
    for i in n_epochs:
        result = fit_predict(train,val,surprise.SVD,n_factors=10,n_epochs=i,use_PMF = PMF)
        rmse.append((result[0],i))
        
    #min_rmse = min(rmse)
    #index = np.argmin(rmse)
    
    return rmse
    
        
    
    
def run_Gsearch(train,val,column_names,n_factors,n_epochs, PMF = False): #column_names = n_epochs
    row = []
    df = pd.DataFrame(columns = column_names)
    for i in n_factors :
        for j in n_epochs:
            result = fit_predict(train,val,surprise.SVD,n_factors=i,n_epochs=j,use_PMF = PMF)
            row.append(result[0])
        a_series = pd.Series(row, index = df.columns)
        df = df.append(a_series, ignore_index=True)
        row = []
    
    df = df.rename_axis('n_factors').rename_axis('n_epochs', axis='columns')
    
    return df