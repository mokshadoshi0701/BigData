import pickle
from typing import Optional
from fastapi import FastAPI
import uvicorn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from fastapi.responses import FileResponse


from pydantic import BaseModel

 

# {
#   "q000": -61.31,
#   "q001": -59.45,
#   "q010": -55.30,
#   "q025": -52.59,
#   "q050": -47.00,
#   "q075": -36.92,
#   "q090": -26.560,
#   "q099": -4.4163,
#   "q100": 15.19
# }

# Thunderstorm Wind

app = FastAPI()

import sys
print(sys.path)
sys.path.insert(1, 'M:/BigData/BigData/Assignment2/WAF_ML_Tutorial_Part1/scripts/')
from aux_functions import load_n_combine_df
(X_train,y_train),(X_validate,y_validate),(X_test,y_test) = load_n_combine_df(path_to_data='M:/BigData/BigData/Assignment2/model/sevir_model/data/sevir/',features_to_keep=np.arange(0,1,1),class_labels=False,dropzeros=True)

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


path= 'M:/BigData/BigData/Assignment2/model/sevir_model/models/modelLinearRegression.pkl'
loaded_model = pickle.load(open(path, 'rb'))

model = loaded_model.fit(X_train,y_train)
   


#get predictions 
@app.get("/predicting/")
def model_predict():
    yhat = model.predict(X_validate)

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

    fig.savefig('slr.png')
    return FileResponse('slr.png')



#Printing accuracy scores
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


if __name__ == '__main__':
     uvicorn.run(app, host='127.0.0.1', port=8000)

