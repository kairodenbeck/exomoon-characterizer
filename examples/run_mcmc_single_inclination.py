#!/usr/bin/python

import sys
import os, os.path, errno
import numpy as np
import pylab as pl
import h5py

from exomoon_characterizer import run_ptmcmc_buildin_detrending
from exomoon_characterizer.fitting import model_one_moon

used_model = "one_moon"
run_one_moon_model=True


use_MPI=False

print "Use MPI?", use_MPI

rho_star = 1.#in solar density

period=283.3
t0=39.2

ratio_P = 0.1
a_o_R = (period/365.25)**(2./3.)*215.
impact_B=0.0
phase_B=t0/period

ratio_S = 0.02
a_s_o_R = 0.5
period_S= 2.3
phase_S = 0.154
mass_ratio_SP = 0.01

N_transits=3


noise_levels=[100e-6,100e-6,30e-6]
q1s = [0.5,0.5,0.1]
q2s = [0.3,0.3,0.2]

u1s = [2.0*np.sqrt(q1)*q2 for q1, q2 in zip(q1s,q2s)]
u2s = [np.sqrt(q1)*(1.0-2.0*q2) for q1, q2 in zip(q1s,q2s)]

LD_indices=[0,0,1]

time=[]
flux = []
fmod= []
sigma_flux = []

for i in range(N_transits):
    t = np.linspace(t0+i*period-10.,t0+i*period+10.,1000)

    model = model_one_moon(t,ratio_P,a_o_R,impact_B,t0/period,period,u1s[i],u2s[i],ratio_S,a_s_o_R, phase_S,period_S,mass_ratio_SP,fix_blocking=True)

    lc= model + np.random.randn(len(model))*noise_levels[i]

    time.append(t)
    flux.append(lc)
    fmod.append(model)
    sigma_flux.append(np.ones(len(lc))*noise_levels[i])



print len(time)
print len(flux)
print len(sigma_flux)


for t,f,fm,fe in zip(time,flux,fmod,sigma_flux):
    pl.scatter(t,f,c="k",linewidth=0,s=5)
    pl.plot(t,fm,c='C1',linewidth=1)
pl.show()

detrend_order=3

npd=(detrend_order+1)*len(flux)




#beware of units!!!
bound_no_moon_low = [0.0, 1.0, 0.00,  phase_B-0.1, 280.0, 0.0, 0.0,0.0,0.0]
bound_no_moon_high= [0.2, 215.0, 0.99,  phase_B+0.1, 290.0, 1.0, 1.0,1.0,1.0]
bounds_no_moon= (bound_no_moon_low+list([-0.1]*npd),bound_no_moon_high+list([0.1]*npd))

first_guess_no_moon_low = [ratio_P*0.95, a_o_R*0.95, 0.0, phase_B-0.01, period*0.99, q1s[0]-0.1, q2s[0]-0.1,q1s[-1]-0.1, q2s[-1]-0.1]
first_guess_no_moon_high= [ratio_P*1.05, a_o_R*1.05, 0.1, phase_B+0.01, period*1.01, q1s[0]+0.1, q2s[0]+0.1,q1s[-1]+0.1, q2s[-1]+0.1]
first_guess_no_moon =(first_guess_no_moon_low,first_guess_no_moon_high)


if False:
    for t,f,sf in zip(time,flux,sigma_flux):
        pl.errorbar(t,f,yerr=sf)
        model = model_no_moon_nested_2(t,*first_guess_no_moon_low)
        pl.plot(t,model,c="k",zorder=10)
        model = model_no_moon_nested_2(t,*first_guess_no_moon_high)
        pl.plot(t,model,c="k",ls=":",zorder=10)
    pl.show()

    exit()

'''
No moon parameters:
00 ratio_P [0,1]
01 a_o_R [1,inf]
02 impact_B [0,1]
03 phase_B [0,1]
04 period_B [0,inf] in days
05 q1 [0,1]
06 q2 [0,1]
/08 q1 "hubble" [0,1]
/09 q2 "hubble" [0,1]
Moon parameters:
07/09 ratio_M [0,ratio_P]
08/10 semi_msajor_axis_S_o_R [R_Roche/R_star,eta*R_Hill/R_star]
10/12 phase_ [0,1]
09/11 period_S [0,inf] in days
11/13 mass_ratio_SP [0,1]
12/14 i [-pi,pi]
13/15 Omega_i [-pi,pi]
'''
bound_one_moon_low = bound_no_moon_low + [0.0,0.0, 0.0,0.0,0.0,-np.pi,-np.pi]
bound_one_moon_high= bound_no_moon_high+ [0.1,6.0,1.0,100.0,1.0,np.pi,np.pi]
bounds_one_moon= (bound_one_moon_low,bound_one_moon_high)

first_guess_one_moon_low = first_guess_no_moon_low + [0.005,0.8, 0.0, 5., 0.001,-np.pi,-np.pi]
first_guess_one_moon_high= first_guess_no_moon_high+ [0.04, 6.0, 1., 35.0, 0.02,np.pi,np.pi]
first_guess_one_moon =(first_guess_one_moon_low,first_guess_one_moon_high)


pr_args={"verbosity":1,"star_mass_prior":None}#(1.079,0.138,0.100)}

output_file_name="test_" + used_model+".h5"


#one moon
if run_one_moon_model:
    ndim=len(bounds_one_moon[0])
    nwalkers=40
    n_run=5000
    n_burn=5000
    n_temp=1
    sampler_one_moon=run_ptmcmc_buildin_detrending(time,flux,sigma_flux,"one_moon",bounds_one_moon,first_guess_one_moon,ndim,nwalkers,n_run,n_burn,n_temp, save=False,verbosity=1,save_between=1000,density_prior=False, restart_from=output_file_name, save_between_path=output_file_name,LD_indices=LD_indices,use_inclination=True,fix_blocking=True, use_kipping_LD_param=True,interpolate_occultquad=None,use_mpi=use_MPI, detrend_order=detrend_order)
    del sampler_one_moon


