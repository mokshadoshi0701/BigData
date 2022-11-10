import array
# from random import random
# import pickle
# import random
# from typing import Optional
from fastapi import FastAPI
# import uvicorn
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import joblib
from fastapi.responses import FileResponse

# two dimensional example
from numpy import array


from pydantic import BaseModel


# class Sevir_data(BaseModel):
#     q000_ir: float
#     q001_ir: float
#     q010_ir: float
#     q025_ir: float
#     q050_ir: float
#     q075_ir: float
#     q090_ir: float
#     q099_ir: float
#     q100_ir: float
#     q000_wv: float
#     q001_wv: float
#     q010_wv: float
#     q025_wv: float
#     q050_wv: float
#     q075_wv: float
#     q090_wv: float
#     q099_wv: float
#     q100_wv: float

#     q000_vi: float
#     q001_vi: float
#     q010_vi: float
#     q025_vi: float
#     q050_vi: float
#     q075_vi: float
#     q090_vi: float
#     q099_vi: float
#     q100_vi: float
#     q000_vl: float
#     q001_vl: float
#     q010_vl: float
#     q025_vl: float
#     q050_vl: float
#     q075_vl: float
#     q090_vl: float
#     q099_vl: float
#     q100_vl: float
 

app = FastAPI()

import sys
print(sys.path)
# sys.path.insert(1, 'scripts/')
from aux_functions import load_n_combine_df
(X_train,y_train),(X_validate,y_validate),(X_test,y_test) = load_n_combine_df(path_to_data='dataset/',features_to_keep=np.arange(0,1,1),class_labels=False)

column_names = ['q000_ir','q001_ir',
 'q010_ir',
 'q025_ir',
 'q050_ir',
 'q075_ir',
 'q090_ir',
 'q099_ir',
 'q100_ir',
 'q000_wv',
 'q001_wv',
 'q010_wv',
 'q025_wv',
 'q050_wv',
 'q075_wv',
 'q090_wv',
 'q099_wv',
 'q100_wv',
 'q000_vi',
 'q001_vi',
 'q010_vi',
 'q025_vi',
 'q050_vi',
 'q075_vi',
 'q090_vi',
 'q099_vi',
 'q100_vi',
 'q000_vl',
 'q001_vl',
 'q010_vl',
 'q025_vl',
 'q050_vl',
 'q075_vl',
 'q090_vl',
 'q099_vl',
 'q100_vl']

# X_df = pd.DataFrame(X_test,columns=column_names)
# print(X_df.head())


# path= 'LinearRegression.pkl'
loaded_model = joblib.load(open('modelLinearRegression.pkl', 'rb'))


def pred(X_validate,y_validate):

    yhat = loaded_model.predict(X_validate)
    mae = np.mean(np.abs(y_validate-yhat))
    rmse = np.sqrt(np.mean((y_validate-yhat)**2))
    bias = np.mean(y_validate-yhat)
    r2 = 1 - (np.sum((y_validate-yhat)**2))/(np.sum((y_validate-np.mean(y_validate))**2))

    out= 'MAE:{} flashes, RMSE:{} flashes, Bias:{} flashes, Rsquared:{}'.format(np.round(mae,2),np.round(rmse,2),np.round(bias,2),np.round(r2,2))
    return out

@app.get('/scores')
def prediction():
    return {"Predction Scores:" : pred(X_validate,y_validate)}


# #make figure  
# fig = plt.figure(figsize=(5,5))
# #set background color to white so we can copy paste out of the notebook if we want 
# fig.set_facecolor('w')

# #get axis for drawing
# ax = plt.gca()

# #plot data 
# ax.scatter(yhat,y_validate,color=random,s=1,marker='+')
# ax.plot([0,3500],[0,3500],'-k')
# ax.set_xlabel('ML Prediction, [$number of flashes$]')
# ax.set_xlabel('GLM measurement, [$number of flashes$]')
# ax.savefig('/images/y_pred.png')


#get predictions 
@app.get("/predicting/")
def model_predict():
    yhat = loaded_model.predict(X_validate)

#make figure  
    fig = plt.figure(figsize=(5,5))
#set background color to white so we can co
# py paste out of the notebook if we want 
    fig.set_facecolor('w')

#get axis for drawing
    ax = plt.gca()

#plot data 
    ax.scatter(yhat,y_validate,s=1,marker='+')
    ax.plot([0,3500],[0,3500],'-k')
    ax.set_ylabel('ML Prediction, [$number of flashes$]')
    ax.set_xlabel('GLM measurement, [$number of flashes$]')

    fig.savefig('slr1.png')
    return FileResponse('slr1.png')

   
@app.get("/predict/{input}")
def model_predict(input:float):
    # dict= data.dict()
    y_values= array([input]).reshape(-1,1)
    # y_pred = loaded_model.predict(X_validate[:])
    
    # y_pred = loaded_model.predict(y_values)
    # y_pred
    # print(y_pred)
    y_pred = loaded_model.predict(y_values)
    # yo = (y_test[1])
    # result = loaded_model.score(X_test, y_test)
    return {"predictions":list(y_pred)}

