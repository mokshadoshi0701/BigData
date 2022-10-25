#needed packages 
import xarray as xr
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import joblib

import uvicorn
from fastapi import FastAPI


import boto3
from botocore.handlers import disable_signing
resource = boto3.resource('s3')
resource.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
bucket=resource.Bucket('sevir_bigdata/data/sevir')


#import some helper functions for our other directory.
import sys
print(sys.path)
sys.path.insert(1, 'E:/DAMG_7245_BigData/BigData/Assignment2/model/sevir_model/scripts/')
from aux_functions import load_n_combine_df
(X_train,y_train),(X_validate,y_validate),(X_test,y_test) = load_n_combine_df(path_to_data='E:/DAMG_7245_BigData/BigData/Assignment2/model/sevir_model/data/sevir/',features_to_keep=np.arange(0,1,1),class_labels=False)


app = FastAPI()
# @app.get('/')
# def index():
#     return {'message': 'Hello, World'}



# print("1 column",X_train.shape)
# print("Selects first row",X_train[0])
# print("Selects second row, first column, x_train",X_train[1][0])
# print("Selects second row, first column,y_train",y_train[1])
# print("xtrain dtype",X_train.dtype )
# print("type of X_train",type(X_train))

# print("Max value in X_train",X_train.max())

# @app.get('/')
# def index():
#     return {'X_train max': X_train.max()}



#load model from sklearn
from sklearn.linear_model import LinearRegression

#initialize
model = LinearRegression()

print(model)

#train the model
model1 = model.fit(X_train,y_train)

# Evaluate  ML model

# As a sanity check, we will first look at the *one-to-one* plot 
# where the x-axis is the predicted number of flashes, and the y-axis 
# is the true number of flashes. A perfect prediction will be directly 
# along the diagonal. 

#get predictions 
# yhat = model.predict(X_validate)

####uncomment to remove zeroes
# (X_train,y_train),(X_validate,y_validate),(X_test,y_test) = load_n_combine_df(path_to_data='E:/DAMG_7245_BigData/BigData/Assignment2/model/sevir_model/data/sevir/',features_to_keep=np.arange(0,1,1),class_labels=False,dropzeros=True)

# #initialize
# model2 = LinearRegression()

# model2 = model2.fit(X_train,y_train)

# #get predictions 
# yhat2 = model2.predict(X_validate)

# from gewitter_functions import get_mae,get_rmse,get_bias,get_r2

# yhat = model.predict(X_validate)
# mae = get_mae(y_validate,yhat)
# rmse = get_rmse(y_validate,yhat)
# bias = get_bias(y_validate,yhat)
# r2 = get_r2(y_validate,yhat)

# #print them out so we can see them 
# print('MAE:{} flashes, RMSE:{} flashes, Bias:{} flashes, Rsquared:{}'.format(np.round(mae,2),np.round(rmse,2),np.round(bias,2),np.round(r2,2)))


# import pickle
name = 'LinearRegression.pkl'
start_path = 'E:\DAMG_7245_BigData\BigData\Assignment2\model\sevir_model\models\model'
savefile = open(start_path + name,'wb')
joblib.dump(model,savefile)


# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)