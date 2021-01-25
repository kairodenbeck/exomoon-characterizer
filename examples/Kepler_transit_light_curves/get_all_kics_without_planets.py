from __future__ import print_function, division
import os

import numpy as np
import requests

import json
import sys

def get_available_kics(quarter):
    # api-endpoint 
    URL = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"

    # defining a params dict for the parameters to be sent to the API 
    PARAMS = {'table':"keplertimeseries","quarter":str(quarter), "format":"json","where":"object_status=0 and targettype='long cadence'","select":"star_id, targettype, radius, surface_gravity, kepmag, eff_temp"} 
    # sending get request and saving the response as response object 
    r = requests.get(url = URL, params = PARAMS)
    # extracting data in json format 
    data = r.json()

    for d in data:
        if d["radius"] and d["surface_gravity"]:
            d["mass"]=d["radius"]**2.*10.**(d["surface_gravity"]-4.43775)
        else:
            d["mass"]=None

    print("Nr of targets:", len(data))
    return data

kics={}
for used_quarter in range(0,17):
    kics_per_quarter=get_available_kics(used_quarter)
    
    kics_quarter_dict={kic["star_id"]:kic for kic in kics_per_quarter}
    #print(kics_quarter_dict)
    print("Start merging quarter", used_quarter)
    kics.update(kics_quarter_dict)
    print("End merging quarter", used_quarter)

print("Total number of kics:", len(kics))

with open('kics_without_planets.json', 'w') as f:
    json.dump(kics, f)
