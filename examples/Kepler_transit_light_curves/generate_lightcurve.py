from __future__ import print_function, division
import os

import numpy as np
import pylab as pl
import lightkurve as lk

import h5py
import json
import sys

from exomoon_characterizer.fitting import model_one_moon, model_no_moon

use_mpi=True  #still needs a good way to detect automatically when to use mpi
if use_mpi:
    from schwimmbad import MPIPool


#toggle to decide if light curves with or without moon are genereated
if "true" == sys.argv[1].lower():
    moonness=True
elif "false" == sys.argv[1].lower():
    moonness=False
else:
    raise NotImplementedError

#number of light curves generated
N_lc=int(sys.argv[2])

#in which order to go through the kepler light curves
#note: In general, data sets with a moon use uneven KICs
#and those without moons even KICs (see below)
if len(sys.argv)<4:
    sort_mode = "none"
else:
    sort_mode = sys.argv[3]

def get_lightcurve_from_kic(kic_nr,quarter):
    """Fetches the time, light curve and uncertainty of the star sepcified 
    by the KIC number for the given quarter.
    """
    kic_name="KIC "+"%09i"%kic_nr
    download_dir=os.path.abspath('./.lightkurve_cache')
    if not os.path.isdir(download_dir):
        os.mkdir(download_dir)
    search_res=lk.search_lightcurvefile(kic_name,quarter=quarter,cadence="long")
    selector=0
    if len(search_res)==0:
        return None
    try:
        lcf=search_res[selector].download(download_dir=download_dir)
        
        lc=lcf.PDCSAP_FLUX.normalize()
    except Exception as E:
        print("Trying to correct Error:", E)
        cache_name = [ download_dir + "/mastDownload/Kepler/" + o_id+"/"+pFn 
        for o_id, pFn in zip(search_res.table["obs_id"], search_res.table["productFilename"]):
            os.system("rm -f %s"%(cache_name[selector]))
        lcf=search_res[selector].download(download_dir=download_dir)
        
        lc=lcf.PDCSAP_FLUX.normalize()
    return lc.to_pandas(["time","flux","flux_err"])

def radius_mass_relation(mass, add_dispersion=False):
    """Calculates the right planet radius given a planet mass
    using the scaling relation defined in 
    https://arxiv.org/abs/1603.08614
    Arguments:
       - mass: mass of the planet
    Keyword arguments:
       - add_dispersion: if True, scatters radius according to
         the formula in the above reference
    Output:
       - radius of the star 
    """
    if mass<2.04:
        radius = 1.08*mass**0.279
        if add_dispersion:
            frac_dev = np.random.randn()*0.043
            radius*=1+frac_dev
    elif mass<131.6:
        radius = 1.08*2.04**0.279*(mass/2.04)**0.589
        if add_dispersion:
            frac_dev = np.random.randn()*0.146
            radius*=1+frac_dev
    else:
        radius = 1.08*2.04**0.279*(131.6/2.04)**0.589*(mass/131.6)**-0.044
        if add_dispersion:
            frac_dev = np.random.randn()*0.0737
            radius*=1+frac_dev
    return radius

def generate_light_curve(time, flux, star_radius=1., star_mass=1., moonness=True, verbosity=0, seed=None, return_param_dict=False):
    """Add a transit light curve with or without moon to an existing
    light curve (mostly with noise).
    """ 
    #planet and moon orbit their common barycenter, which orbits the star
    #If no moon is present, the planet is at the barycenter
    if seed:
        np.random.seed(seed)

    if len(flux)!= len(time) or len(flux)==0:
        raise ValueError

    planet_mass = np.random.uniform(100.,1000.)
    planet_radius = radius_mass_relation(planet_mass,add_dispersion=True)

    ratio_P = planet_radius/109.1/star_radius 

    bary_period = 10.**(np.random.uniform(np.log10(50),np.log10(500)))

    bary_t0 = np.random.uniform(np.min(time[np.isfinite(flux)]), 
                            min(np.max(time[np.isfinite(flux)]), 
                            np.min(time[np.isfinite(flux)])+bary_period))

    bary_phase = (bary_t0/bary_period) % 1.

    bary_sma =  ((bary_period/365.25)**2./star_mass )**(1./3.) 

    bary_a_o_R = bary_sma *215. / star_radius

    bary_impact = np.random.uniform(-1.,1.)

    moon_mass = np.random.uniform(1.,20.)
    moon_radius = radius_mass_relation(moon_mass,add_dispersion=True)

    #very approximate LD. Using https://arxiv.org/pdf/1503.07020.pdf for
    #solar like values and dispersion and https://arxiv.org/pdf/1308.0009.pdf
    #to constrain u1,u2 region (values outside might cause brightening).
    u1= np.random.normal(0.25,0.05)
    if u1<0:
        u1=0
    elif u1>2:
        u1=2
    u2= np.random.normal(0.3,0.05)
    if u2>1-u1:
        u2=1-u1
    elif u2<-u1/2.:
        u2=-u1/2.
    
    if moonness:

        ratio_S = moon_radius/109.1/star_radius 

        roche_limit = moon_radius / 23455. * (2.*planet_mass/moon_mass)**(1./3.)
        hill_radius = bary_sma * (planet_mass/333.e3/(3.*star_mass))**(1./3.)

        moon_sma = np.random.uniform(roche_limit,0.5*hill_radius)

        moon_period = np.sqrt(moon_sma**3./(planet_mass/333.e3))*365.25

        moon_phase = np.random.uniform(0,1.)

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
        params_out["u1"] = u1
        params_out["u2"] = u2
        params_out["R_p"]= planet_radius
        params_out["M_p"]= planet_mass
        params_out["ratio_p"] = ratio_P
        params_out["impact_b"] = bary_impact
        params_out["M_p"] = planet_mass
        params_out["a_o_R"] = bary_a_o_R
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
        
        print("R_star", star_radius)
        print("M_star", star_mass)
        print("R_p:", planet_radius)
        print("M_p:", planet_mass)
        print("ratio_p", ratio_P)
        print("impact_b", bary_impact)
        print("a_o_R:", bary_sma)
        print("P_b:", bary_period)
        print("t_0:", bary_t0)

        print("R_s:", moon_radius)
        print("ratio_s", ratio_S)
        print("M_s:", moon_mass)
        print("a_o_R_s:", moon_sma)
        print("P_s:", moon_period)

    model = model_one_moon(time, 
            ratio_P, bary_a_o_R, bary_impact, bary_phase, bary_period, 
            u1, u2, 
            ratio_S, a_o_R_S, moon_phase,moon_period, moon_mass/(planet_mass), 
            i_S, Omega_S, fix_blocking=True)

    flux_sim=model*flux
    
    sys.stdout.flush()
    if return_param_dict:
        return flux_sim, params_out
    return flux_sim
    

def fit_planet_parameters(time,flux,param_dict, plot_initial_guess=False):
    from scipy.optimize import curve_fit

    time = time[np.isfinite(flux)]
    flux = flux[np.isfinite(flux)]

    ratio = param_dict["ratio_p"]
    period = param_dict["P_b"]
    phase = param_dict["phi_b"]
    a_o_R = param_dict["a_o_R"]
    impact = param_dict["impact_b"]
    u1 = param_dict["u1"]
    u2=param_dict["u2"]
    q1 = (u1+u2)**2
    q2 = u1/(2.*u1+2.*u2)
    params = [np.abs(ratio),a_o_R,np.abs(impact),phase,period,q1,q2]

    def model(time,ratio,a_o_R,impact,phase,period,q1,q2):
        if impact>1.0+ratio:#outside valid bounds
            #make sure gradient points back to valid region
            return np.ones(len(time))*100.*(impact)
        return model_no_moon(time, ratio, a_o_R, impact, phase, period, 
                                q1, q2, kipping_LD=True)
    if plot_initial_guess:
        model = model(time,*params)
        pl.figure()
        pl.scatter(time,flux,s=5,linewidth=0,c="k")
        pl.plot(time,model)
        pl.show()

    try:
        res = curve_fit(model,time,flux,params, 
                    bounds=[[0,1.0,0.0,0.0,0.1,0.0,0.0],
                            [0.2,1000,1.2,1.0,1000,1.,1.]])
    except RuntimeError as E:
        print("Runtime error:", E)
        return [params]
    except Exception as E:
        raise E
    return res

    
def build_lightcurve(kic_nr, radius, mass, moonness, return_param_dict=True):
    fluxes=[]
    for quarter in range(0,17):
        data = get_lightcurve_from_kic(kic_nr,quarter)
        if data is not None:
            fluxes.extend(np.array(data["flux"]))
        else:
            fluxes.extend(np.ones(quarter_data_points[quarter])*np.nan)
    
    time = np.array(global_times)
    flux = np.array(fluxes)
    
    flux_before=np.copy(flux)
    flux, po = generate_light_curve(time, flux, star_radius = radius, star_mass = mass, moonness = moonness, return_param_dict = True, verbosity = 0)
    
    if return_param_dict:
        return time, flux, po
    return time, flux

def callback_build_lightcurve(data):
    kic_nr, radius, mass, kepmag = data
    try:
        return build_lightcurve(kic_nr, radius, mass, moonness, return_param_dict = True)
    except:
        return None, None, None

def get_number_of_data_points_per_quarter(kic_nr_ref = 2163351):
       
    #We know that KIC 2163351 has data in all quarters
    quarter_data_points=[]
    global_times=[]
    for quarter in range(0,17):
            data = get_lightcurve_from_kic(kic_nr_ref,quarter)
            global_times.extend(np.array(data["time"]))
            quarter_data_points.append(len(data["time"]))
    return global_times, quarter_data_points
    
def generate_output_file_name(moonness,sort_mode):
    output_file_name="output_ml_"
    if sort_mode != "none":
        output_file_name+="sortedby_"+sort_mode+"_"
    if moonness:
        output_file_name+="one_moon"
    else:
        output_file_name+="no_moon"
    output_file_name+=".h5"
    return output_file_name

### Open list of kics without planets.        
with open("kics_without_planets.json","r") as k: 
    kics = json.load(k)
available_kic_nrs = [kics[kic]["star_id"] for kic in kics if kics[kic]["radius"] and 0.84<kics[kic]["radius"]<1.15]
print("Number of G-type stars:", len(available_kic_nrs))


if "none" == sort_mode:
    print("no sorting of kics")
elif "kepmag" == sort_mode:
    print("sorting kics by Kepler magnitude")
    kepmags = [kics[str(k)]["kepmag"] for k in available_kic_nrs]
    sort_index = np.argsort(kepmags)
    available_kic_nrs = list(np.array(available_kic_nrs)[sort_index])
else:
    raise NotImplementedError

#We need to find the number of data points in each quarter
#bc lightkurve doesn't provide quarters with no data at all
#so we need to fill them with NaNs later manually
global_times, quarter_data_points = get_number_of_data_points_per_quarter()

#Set use_mpi to True when starting with mpiexec (e.g. on a compute cluster)       
if use_mpi:
    pool = MPIPool()
    
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

#Generate output file name and delete old files with the same name
#(hdf5 doesn't like trying to override old files)
output_file_name=generate_output_file_name(moonness,sort_mode)
if os.path.exists(output_file_name):
    os.remove(output_file_name)
outf=h5py.File(output_file_name,"w", libver='latest')

i_succ=0
i_tot=0

if use_mpi:
    i_batch=100
else:
    i_batch=1

while i_succ<N_lc and i_tot<len(available_kic_nrs):
    
    print("i_succ:", i_succ, "i_tot", i_tot)
    i_this_batch=min(N_lc-i_succ,min(i_batch,len(available_kic_nrs)-i_tot))
    kic_nrs_batch=available_kic_nrs[i_tot:i_tot+i_this_batch]

    #we process i_batch light curves in one go to make sure all cores are busy.
    print("i_batch:", i_batch, "i_this_batch", i_this_batch)

    #preparing input parameters for transit generation function
    radii = []
    masses = []
    kic_nrs=[]
    kepmags = []
    for kic_nr in kic_nrs_batch:
        if kic_nr%2==moonness*1:
            radius=kics[str(kic_nr)]["radius"]
            mass=kics[str(kic_nr)]["mass"]
            if mass==None or radius==None:
                radius=1.
                mass=1.
            kic_nrs.append(kic_nr)
            radii.append(radius)
            masses.append(mass)
            kepmags.append(kics[str(kic_nr)]["kepmag"])
    data = [[kic_nr, radius, mass, kepmag] for kic_nr, radius, mass,kepmag in zip(kic_nrs, radii, masses, kepmags)]
    

    if use_mpi:
        output = list(pool.map(callback_build_lightcurve, data))
    else:
        output = list(map(callback_build_lightcurve,data))    

    times, fluxes, params = [], [], []
    for o in output:
        times.append(o[0])
        fluxes.append(o[1])
        params.append(o[2])
    
    i_tot+=i_batch


    for kic_nr,kepmag, time, flux, param in zip(kic_nrs, kepmags, times,fluxes, params):
        try:
            if time is not None:
                cg=outf.create_group("light curve KIC "+str(kic_nr))
                cg.create_dataset("time",data=time,compression='gzip', compression_opts=4,dtype=np.float64)
                cg.create_dataset("flux",data=flux,compression='gzip', compression_opts=4,dtype=np.float64)
                cg.attrs.create("KIC",kic_nr)
                param["kepmag"]=kepmag
                
                #We fit a transit model to the light curve to get planetary tranist parameters
                #We do this for light curves containing a moon and no moon to ensure consistency
                res_fit = fit_planet_parameters(time,flux,param)
                q1 = res_fit[0][-2]
                q2 = res_fit[0][-1]
                u1 = 2.*np.sqrt(q1)*q2
                u2 = np.sqrt(q1)*(1.-2.*q2)
                mu = np.sqrt(max(0,1.0-res_fit[0][2]**2))
                depth = res_fit[0][0]**2.*(1.0-u1*(1-mu)-u2*(1.0-mu)**2) \
                                         /(1.0-u1/3.-u2/6.0)

                duration = res_fit[0][4]*2*np.sqrt((1.+res_fit[0][0])**2-res_fit[0][2]**2)/(2.*np.pi*res_fit[0][1])

                period = res_fit[0][4]

                phase = res_fit[0][3]

                T0 = res_fit[0][3]*res_fit[0][4]

                test_attr = {"depth":depth, "duration": duration, "phi_b":phase, "T0_b": T0, "P_b":period}
                
                test_attr["kepmag"]=kepmag

                #training parameters set (=parameters that whould be avaiable for Kepler data)
                #should be the one available to the ML algorithm for training
                hf_new_train = cg.create_group("training parameters")
                hf_new_test = cg.create_group("test parameters")

                for a in test_attr:
                    hf_new_test.attrs.create(a,test_attr[a])

                for a in param:
                    hf_new_train.attrs.create(a,param[a])

                i_succ+=1
        except Exception as E:
            print("No LC saved:", E)
            sys.stdout.flush()
    sys.stdout.flush()
outf.close()
pool.close()


