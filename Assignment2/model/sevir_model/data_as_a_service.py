#needed packages 
import mimetypes
import xarray as xr
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from fastapi.responses import FileResponse

import os
import h5py # needs conda/pip install h5py
import matplotlib.pyplot as plt

import uvicorn
from fastapi import FastAPI



app = FastAPI()

catalog = pd.read_csv('E:\DAMG_7245_BigData\BigData\Assignment2\model\sevir_model\data\sevir\CATALOG.csv',parse_dates=['time_utc'],low_memory=False)
print(catalog.head(5))

@app.get('/')
def index():
    return {"message": "Hello World"}

#Finding files by event id
@app.get("/event-id/{event_no}")
def get_event_id(event_no: int):
    e= list(catalog[catalog.event_id == event_no].file_name)
    return {str(len(e))+' related files found': e}


#Retrieving files by modality name
@app.get("/{modality_name}")
def get_modality(modality_name: str):
    e= list(catalog[catalog.img_type == modality_name].file_name)
    return {str(len(e))+' related files found'}


#func2
@app.get("/mod-eventtype/{mod}/{event_type}")
def get_eventtype_mod(mod: str, event_type: str):
    files_of_mod= catalog[catalog.img_type == mod]
    data = list(files_of_mod[files_of_mod.event_type == event_type].file_name)
    return {'No of Hail event type in year 2018': len(data)}

#func1
@app.get("/feature/{feature_name}")
def get_unique(feature_name: str ):
    return {'unique items': list(catalog[feature_name].fillna('').unique()) }

#lastfunc
@app.get("/getimage/{path}")
def get_img(path: str):
    
    file_index = 14
    DATA_PATH ='M:/BigData/BigData/Assignment2/model/sevir_model/data'

    with h5py.File('%s/ir069/2019/SEVIR_IR069_STORMEVENTS_2019_0101_0630.h5' % DATA_PATH,'r') as hf:
        event_id = hf['id'][file_index]
        vil      = hf['ir069'][file_index] 
        
    print('Event ID:',event_id)
    print('Image shape:',vil.shape)

    
    fig,axs=plt.subplots(1,4,figsize=(10,5))
    axs[0].imshow(vil[:,:,10])
    axs[1].imshow(vil[:,:,20])
    axs[2].imshow(vil[:,:,30])
    axs[3].imshow(vil[:,:,40])
        
    fig.savefig('graph.png')
    return FileResponse('graph.png')

# @app.get("/mod/")
# def get_mod(modality:str,year:int):
#     files_of_year= catalog[catalog['time_utc'].dt.strftime('%Y')==year]
#     data = list(files_of_year[files_of_year.img_type == modality].file_name)
#     return data







#835047





if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)





