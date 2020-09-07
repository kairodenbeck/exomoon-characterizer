from __future__ import print_function, division
import os

os.environ["HOME"]=os.path.expanduser("~")

import numpy as np
import pylab as pl

import h5py
import sys

from exomoon_characterizer.fitting import model_one_moon,model_no_moon


#roughly after https://arxiv.org/pdf/1701.07654.pdf
def radius_mass_relation(planet_mass):
    if planet_mass<121.:
        return planet_mass**(0.5)
    else:
        return 11.

def generate_light_curve(time,flux,star_radius=1.,star_mass=1., period=300,moonness=True,moon_radius=None,verbosity=0,seed=None,return_param_dict=False):
    
    if seed:
        np.random.seed(seed)

    if len(flux)!= len(time) or len(flux)==0:
        raise ValueError

    planet_mass = 10.**np.random.uniform(np.log10(100.),np.log10(1000.))
    planet_radius = radius_mass_relation(planet_mass)

    ratio_P = planet_radius/109.1/star_radius 

    bary_period = period#10.**(np.random.uniform(1.,3))

    bary_t0 = 25. #np.random.uniform(np.min(lc.time),min(np.max(time),np.min(lc.time)+bary_period))

    bary_phase = (bary_t0/bary_period) % 1.

    bary_sma =  ((bary_period/365.25)**2.*star_mass )**(1./3.) 

    bary_a_o_R = bary_sma *215. / star_radius

    bary_impact = np.random.uniform(-1.,1.)

    if not moon_radius:
        moon_mass = np.random.uniform(.5625,22.5625)
        moon_radius = radius_mass_relation(moon_mass)
    else:
        moon_mass = 1.

    q1=np.random.uniform(0.3,0.8)
    q2=np.random.uniform(0.3,0.8)


    c1=2.0*np.sqrt(q1)*q2
    c2=np.sqrt(q1)*(1.0-2.0*q2)
   
    if moonness:

        ratio_S = moon_radius/109.1/star_radius 

	roche_limit = moon_radius / 23455. * (2.*planet_mass/moon_mass)**(1./3.)
	hill_radius = bary_sma * (planet_mass/333.e3/(3.*star_mass))**(1./3.)

	moon_sma = np.random.uniform(roche_limit,0.5*hill_radius)

	moon_period = np.sqrt(moon_sma**3./(planet_mass/333.e3))*365.25
        #moon_period = np.random.uniform(.5,10.) # hill period and Roche limit

        moon_phase = np.random.uniform(0,1.)

        #moon_sma =  ((moon_period/365.25)**2.*(planet_mass/333.e3) )**(1./3.)

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
	params_out["bary_sma"] = bary_sma
        params_out["a_o_R"] = bary_a_o_R
        params_out["P_b"] = bary_period
        params_out["phi_b"] = bary_phase
        params_out["T0_b"] = bary_t0
	params_out["q1"] = q1
	params_out["q2"] = q2
        
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
        
        print("R_p:", planet_radius)
        print("ratio_p", ratio_P)
        print("M_p:", planet_mass)
        print("a_o_R:", bary_sma)
        print("P_b:", bary_period)
        print("t_0:", bary_t0)

        print("R_s:", moon_radius)
        print("ratio_s", ratio_S)
        print("M_s:", moon_mass)
        print("a_o_R_s:", moon_sma)
        print("P_s:", moon_period)

    if moonness:
        model = model_one_moon(time,ratio_P,bary_a_o_R,bary_impact,bary_phase, bary_period,c1,c2,
                ratio_S,a_o_R_S,moon_phase,moon_period,moon_mass/(planet_mass),i_S,Omega_S,
		fix_blocking=True)
    else:
	model = model_no_moon(time,ratio_P,bary_a_o_R,bary_impact,bary_phase, bary_period,c1,c2)
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

print("Moonness:", moonness)

fix_noise=False
fix_radius=False
if "fix_noise"==sys.argv[3].lower():
    fix_noise=True
elif "fix_radius"==sys.argv[3].lower():
    fix_radius=True
else:
    print(sys.argv[3])
    raise NotImplementedError


print(sys.argv[3])
print("fix_noise?", fix_noise)
print("fix_radius?", fix_radius)


N_lc=int(sys.argv[2])

output_file_name="output_ml_DeltaBIC_"
if fix_noise:
    output_file_name+="fixed_noise_"
elif fix_radius:
    output_file_name+="fixed_radius_"
else:
    raise NotImplementedError

if moonness:
    output_file_name+="one_moon.h5"
else:
    output_file_name+="no_moon.h5"


if os.path.exists(output_file_name):
    os.remove(output_file_name)

outf=h5py.File(output_file_name,"w",swmr=True)
outf.swmr=True

from time import clock

start_time=clock()

i=0
i_succ=0
while i_succ < N_lc:
    try:
        
        radius=np.random.uniform(0.8,1.4)
        mass=radius**2.4
        cadence=29.4/60./24.
	period=10.**np.random.uniform(np.log10(100.),np.log10(300.))
	time_complete=np.arange(-10,4*period+60.,cadence)
	
	time=[]
	for t in range(4):
            midp=25.+period*t
            midp=(int(midp/cadence)+0.5)*cadence
            time.extend(time_complete[np.abs(time_complete-midp)<25.])
        time=np.array(time)

        noise_level=np.random.uniform(25e-6,525e-6)

        if fix_noise:
            noise_level=100.e-6

	lc=np.ones(len(time))+np.random.normal(size=len(time))*noise_level
        moon_radius=None
	if fix_radius:
	    moon_radius=1.
        flux_sim,po=generate_light_curve(time,lc,star_radius=radius,star_mass=mass,period=period,moonness=moonness,moon_radius=moon_radius,return_param_dict=True)
        
        po["noise level"]=noise_level
        flux_sim
        cg=outf.create_group("light curve "+str(i_succ))
        cg.create_dataset("time",data=time,compression='gzip', compression_opts=4,dtype=np.float64)
        cg.create_dataset("flux",data=flux_sim,compression='gzip', compression_opts=4,dtype=np.float64)
        #cg.attrs.create("KIC",kic_nr)
        
	for key, val in list(po.items()):
            cg.attrs.create(key,val)
        i_succ+=1
	if i_succ%100==0:
	    time_difference=clock()-start_time
	    print(i_succ, "/", N_lc, "generated. Time remaining (min):", time_difference*1./i_succ*(N_lc-i_succ)/60.)
    except Exception as E:
        print("No LC saved:", E)
    i+=1
if i_succ<N_lc-1:
    print("Less samples than expected were created")
outf.close()
