
import os

os.environ["HOME"]=os.path.expanduser("~")

import numpy as np
import pylab as pl
import lightkurve as lk
import requests

import h5py
import sys

from exomoon_characterizer.fitting import model_one_moon

def get_available_kics(quarter):
    # api-endpoint 
    URL = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"

    # defining a params dict for the parameters to be sent to the API 
    PARAMS = {'table':"keplertimeseries","quarter":str(quarter), "format":"json","where":"object_status=0 and targettype='long cadence'","select":"star_id, targettype, radius, surface_gravity"} 
    # sending get request and saving the response as response object 
    r = requests.get(url = URL, params = PARAMS)
    # extracting data in json format 
    data = r.json()

    for d in data:
        if d["radius"] and d["surface_gravity"]:
            d["mass"]=d["radius"]**2.*10.**(d["surface_gravity"]-4.43775)
        else:
            d["mass"]=None

    print "Nr of targets:", len(data)
    return data

def get_lightcurve_from_kic(kic_nr,quarter):
    kic_name="KIC "+"%09i"%kic_nr
    download_dir=os.path.abspath('./.lightkurve_cache')
    if not os.path.isdir(download_dir):
        os.mkdir(download_dir)
    search_res=lk.search_lightcurvefile(kic_name,quarter=quarter,cadence="long")
    selector=0
    if len(search_res)>1:
        kplrname="kplr%09i"%kic_nr
        for sel, ta in enumerate(search_res.table["target_name"]):
            if ta==kplrname:
                selector=sel
    try:
        lcf=search_res[selector].download(download_dir=download_dir)
        
        lc=lcf.PDCSAP_FLUX.normalize()
    except Exception as E:
        print "Trying to correct Error:", E
        cache_name=[download_dir+"/mastDownload/Kepler/"+o_id+"/"+pFn for o_id, pFn in zip(search_res.table["obs_id"],search_res.table["productFilename"])]
        os.system("rm -f %s"%(cache_name[selector]))
        lcf=search_res[selector].download(download_dir=download_dir)
        
        lc=lcf.PDCSAP_FLUX.normalize()
    return lc.to_pandas(["time","flux","flux_err"])

def radius_mass_relation(planet_mass):
    if planet_mass<121.:
        return planet_mass**(0.5)
    else:
        return 11.

def generate_light_curve(time,flux,star_radius=1.,star_mass=1.,moonness=True,verbosity=0,seed=None,return_param_dict=False):
    
    if seed:
        np.random.seed(seed)

    if len(flux)!= len(time) or len(flux)==0:
        raise ValueError

    planet_mass = np.random.uniform(100.,1000.)
    planet_radius = radius_mass_relation(planet_mass)

    ratio_P = planet_radius/109.1/star_radius 

    bary_period = 10.**(np.random.uniform(1.,3))

    bary_t0 = np.random.uniform(np.min(lc.time),min(np.max(time),np.min(lc.time)+bary_period))

    bary_phase = (bary_t0/bary_period) % 1.

    bary_sma =  ((bary_period/365.25)**2./star_mass )**(1./3.) 

    bary_a_o_R = bary_sma *215. / star_radius

    bary_impact = np.random.uniform(-1.,1.)

    moon_mass = np.random.uniform(1.,20.)
    moon_radius = radius_mass_relation(moon_mass)
    
    if moonness:

        ratio_S = moon_radius/109.1/star_radius 

        moon_period = np.random.uniform(.3,5.) # hill period and Roche limit

        moon_phase = np.random.uniform(0,1.)

        moon_sma =  ((moon_period/365.25)**2./planet_mass )**(1./3.)

        a_o_R_S = moon_sma *215. / star_radius

        i_S=np.random.uniform(-np.pi,np.pi)
        Omega_S=np.random.uniform(-np.pi,np.pi)
    else:
        moon_mass=0.0
        moon_radius=0.0
        ratio_S=0.
        moon_period=1.
        moon_phase=0.
        moon_sma=.01
        a_o_R_S=1.
        
        i_S=np.pi*0.5
        Omega_S=0.0
        
    if return_param_dict:
        params_out=dict()
        params_out["R_star"]=star_radius
        params_out["M_star"]=star_mass
        params_out["R_p"]= planet_radius
        params_out["M_p"]= planet_mass
        params_out["ratio_p"] = ratio_P
        params_out["impact_b"] = bary_impact
        params_out["M_p"] = planet_mass
        params_out["a_o_R"] = bary_sma
        params_out["P_b"] = bary_period
        params_out["phi_b"] = bary_phase
        params_out["T0_b"] = bary_t0
        
        params_out["R_s"] = moon_radius
        params_out["M_s"] = moon_mass
        params_out["ratio_s"] = ratio_S
        params_out["M_s"] = moon_mass
        params_out["a_S"] = moon_sma
        params_out["a_o_R_s"] = a_o_R_S
        params_out["P_s"] = moon_period
        params_out["phi_s"] = moon_phase
        params_out["i_s"] = i_S
        params_out["Omega_S"] = Omega_S
    
    if verbosity>1:
        
        print "R_p:", planet_radius
        print "ratio_p", ratio_P
        print "M_p:", planet_mass
        print "a_o_R:", bary_sma
        print "P_b:", bary_period
        print "t_0:", bary_t0

        print "R_s:", moon_radius
        print "ratio_s", ratio_S
        print "M_s:", moon_mass
        print "a_o_R_s:", moon_sma
        print "P_s:", moon_period

    model = model_one_moon(time,ratio_P,bary_a_o_R,bary_impact,bary_phase, bary_period,0.4,0.4,
              ratio_S,a_o_R_S,moon_phase,moon_period,moon_mass/(planet_mass),i_S,Omega_S)

    flux_sim=model-1.+flux
    
    sys.stdout.flush()
    if return_param_dict:
        return flux_sim, params_out
    return flux_sim


if "true" == sys.argv[1].lower():
    moonness=True
elif "false" == sys.argv[1].lower():
    moonness=False
else:
    raise NotImplementedError

print "Moonness:", moonness
    
used_quarter=int(sys.argv[2])

kics=get_available_kics(used_quarter)

#Number of light curves in the sample per quarter
N_lc=5000

output_file_name="output_ml_"
if moonness:
    output_file_name+="one_moon_"
else:
    output_file_name+="no_moon_"
    
output_file_name+="quarter_"+str(used_quarter)+".h5"


if os.path.exists(output_file_name):
    os.remove(output_file_name)

outf=h5py.File(output_file_name,"w")

i=0
i_succ=0
while i_succ < N_lc and i<len(kics):
    try:
        kic_nr=kics[i]["star_id"]
        radius=kics[i]["radius"]
        mass=kics[i]["mass"]
        print "KIC %09i"%kic_nr
        lc=get_lightcurve_from_kic(kic_nr,used_quarter)
        
        flux_sim,po=generate_light_curve(lc.time,lc.flux,star_radius=radius,star_mass=mass,moonness=moonness,return_param_dict=True)
        
        lc.flux=flux_sim
        cg=outf.create_group("light curve "+str(i_succ))
        cg.create_dataset("time",data=lc.time,compression='gzip', compression_opts=4,dtype=np.float64)
        cg.create_dataset("flux",data=lc.flux,compression='gzip', compression_opts=4,dtype=np.float64)
        cg.attrs.create("KIC",kic_nr)
        for key, val in po.items():
            cg.attrs.create(key,val)
        i_succ+=1
    except Exception as E:
        print "No LC saved:", E
    i+=1
if i_succ<N_lc-1:
    print "Less samples than expected were created"
outf.close()
