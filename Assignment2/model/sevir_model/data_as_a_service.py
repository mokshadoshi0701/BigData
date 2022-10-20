#needed packages 
import xarray as xr
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

import uvicorn
from fastapi import FastAPI

catalog = pd.read_csv('M:\BigData\BigData\Assignment2\model\sevir_model\data\sevir\CATALOG.csv',parse_dates=['time_utc'],low_memory=False)
print(catalog.head(5))

app = FastAPI()

@app.get('/')
def index():
    return("Hello World")

#Finding files by event id
@app.get("/event-id/{event_no}")
def get_event_id(event_no: int):
    e= list(catalog[catalog.event_id == event_no].file_name)
    return {str(len(e))+' related files found': e}


#Retrieving files by modality name
@app.get("/{modality_name}")
def get_modality(modality_name: str):
    e= list(catalog[catalog.img_type == modality_name].file_name)
    return {str(len(e))+' related files found': e}

@app.get("/year-eventtype/{year}/{event_type}")
def get_event_modality(year: int, event_type: str):
    evis = list(catalog[catalog.event_type == event_type and catalog.img_type== vis].img_type)
    return {'Files Found':len(e)}


img_types = set(['vis','ir069','ir107','vil'])
# Group by event id, and filter to only events that have all desired img_types
events = catalog.groupby('id').filter(lambda x: img_types.issubset(set(x['img_type']))).groupby('id')
#e= list(catalog[catalog.event_id == event_no & catalog.img_type == modality].file_name)
print(events)
#835047





if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)





