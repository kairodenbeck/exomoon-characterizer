from __future__ import print_function, division
import numpy as np
import numpy.random as rnd

import os.path
import h5py
import emcee
from emcee.autocorr import integrated_time
from emcee.autocorr import AutocorrError


from fitting import model_one_moon
from fitting import model_no_moon

R_sun_in_au = 0.00465
R_earth_in_R_sun = 0.009157694
R_earth_in_au = 23455.0

def test_h5_file(output_file_name,n_run=None):
    assert(".h5" in output_file_name)
    import os.path
    if not os.path.isfile(output_file_name):
        return None
    try:
        import os
        os.system("h5clear -s "+output_file_name)
        with h5py.File(output_file_name,"r") as hf:
            
            test_prob=hf["lnprob"]
            test_chain=hf["chain"]

            restart_name=output_file_name
            print("restarting from", restart_name)
            print("shape:", test_chain.shape)
            if n_run and len(test_chain[0,0])==n_run:
                print("Trying to restart from an finished run. Set n_run higher if you wish to continue. Exiting.")
                exit()
    except Exception as E:
        import sys
        sys.stderr.write("Can't open restart file. Starting from scratch.")

        sys.stderr.write(E.message + "\n")
        for a in E.args:
           sys.stderr.write(str(a) + "\n")

        restart_name=None
    return restart_name

def test_npz_file(output_file_name, n_run=None):
    import os.path
    if not os.path.isfile(output_file_name+".npz"):
        return None
    try:
        test_data=np.load(output_file_name+".npz")
        test_prob=test_data["lnprob"]
        test_chain=test_data["chain"]
        restart_name=output_file_name+".npz"
        print("restarting from", restart_name)
        print("shape:", test_chain.shape)
        if n_run and len(test_chain[0])==n_run:
            print("Trying to restart from an finished run. Set n_run higher if you wish to continue. Exiting.")
            exit()
        del test_prob
        del test_chain
        del test_data
    except Exception as E:
       sys.stderr.write("Can't open restart file. Starting from scratch.")

       sys.stderr.write(E.message + "\n")
       for a in E.args:
           sys.stderr.write(str(a) + "\n")

       restart_name=None
    return restart_name

def open_MCMC_results(file_name,n_burn,flat=True):
    """
    Returns chain and lnprob saved in file_name.

    arguments:
    file_name -- name of file to be opened

    Keyword arguments:
    n_burn -- number of discarded points at the beginning of the chains (default 0)
    flat -- Whether to flatten the chain for easier use. 

    Non-flattened output has the shape (n_walker,n_run-n_burn,n_dim). Flattend output has the shape (n_walker*(n_run-n_burn),b_dim)
    """

    temp=False
    if ".h5" in file_name:
        import h5py
        hf = h5py.File(file_name,"r",swmr=True)
        chain = hf["chain"][:]
        lnprob = hf["lnprob"][:]
        dims_chain=chain.shape
        n_dim = dims_chain[-1]
        n_run = dims_chain[-2]
        n_walker=dims_chain[-3]
        if len(dims_chain)==4:
            temp=True
            n_temp=dims_chain[0]

    assert(n_run>n_burn)

    if temp:
        chain = chain[:,:,n_burn:,]
        lnprob = lnprob[:,:,n_burn:]

        if flat:
            chain = chain.reshape(n_temp,(n_run-n_burn)*n_walker,n_dim)
            lnprob = lnprob.reshape(n_temp,(n_run-n_burn)*n_walker)

    else:
        chain = chain[:,n_burn:,]
        lnprob = lnprob[:,n_burn:]

        if flat:
            chain = chain.reshape(-1,n_dim)
            lnprob = lnprob.reshape(-1)

    return chain, lnprob

def split_param_vector_in_dict(param,n_obs=1,n_limb_dark=1,flat_system=False,detrending_type="none",detrening_build_in=False,moon_exists=False):

    n_detr_per_transit=0
    if detrending_type == "Offset":
        n_detr_per_transit=1
    if detrending_type == "Linear":
        n_detr_per_transit=2
    if detrending_type == "Poly2":
        n_detr_per_transit=3
    if detrending_type == "Poly3":
        n_detr_per_transit=4
    if detrending_type == "Poly4":
        n_detr_per_transit=5
    if detrending_type == "Poly5":
        n_detr_per_transit=6



def log_likelihood_with_detrend(params,time,obs,sigma,b_l=None,b_u=None,detrend_order=2, minus_inf=-np.inf, mode="both",moonness="no_moon",use_inclination=False,LD_indices=None,use_kipping_LD_param=True, interpolate_occultquad=None,verbosity=0,plots=False,split_moon_period=False, fix_blocking=False, density_prior=False,stellar_density=None,oversample_factor=1,return_model=False,exposure_time=None):
    single_loglike=log_likelihood_no_moon
    if moonness=="one_moon":
        single_loglike=log_likelihood_one_moon
    n_detr=len(obs)*(detrend_order+1)

    detrend_params=np.array(params[-n_detr:]).reshape(len(obs),detrend_order+1)

    if verbosity>2:
        print("Boundary lengths:", len(b_l),len(b_u))

    result=0
    if return_model:
        model_lc=[]
    for t,o,s,f,d,i in zip(time,obs,sigma,oversample_factor,detrend_params,range(len(time))):
        detrend_model=np.polyval(d,t-t[len(t)/2])+1.
        if np.any(detrend_model<=0.9) or np.any(detrend_model>=1.1):
            if verbosity>1:
                print("Out of bounds detrending:", d)
            if verbosity>3:
                pl.plot(t,detrend_model)
                pl.scatter(t,o,s=1,c="k")
                pl.show()
            return minus_inf
        if verbosity>3:
            pl.plot(t,detrend_model)
            pl.scatter(t,o,s=1,c="k")
            pl.show()
        detrended_obs=o/detrend_model
        detrended_sig=s/detrend_model

        result-=np.sum(d**2.0)

        params_only_transit=params[:-n_detr]

        if verbosity>2:
            print("LD index:", LD_indices[i])
        if LD_indices:
            #params_only_transit[9],params_only_transit[10]=params_only_transit[10],params_only_transit[9]
            if b_l:
                b_l_single=np.concatenate((b_l[:5],
                                            b_l[5+2*LD_indices[i]:5+2*LD_indices[i]+2],
                                            b_l[5+2*np.max(LD_indices)+2:]))
            else: b_l_single=None
            if b_u:
                b_u_single=np.concatenate((b_u[:5],
                                                b_u[5+2*LD_indices[i]:5+2*LD_indices[i]+2],
                                                b_u[5+2*np.max(LD_indices)+2:]))
            else:
                b_u_single=None
            if moonness=="one_moon":
                params_only_transit=np.concatenate((params[:5],
                                                params[5+2*LD_indices[i]:5+2*LD_indices[i]+2],
                                                params[5+2*np.max(LD_indices)+2:]))

            else:
                params_only_transit=np.concatenate((params[:5],
                                                params[5+2*LD_indices[i]:5+2*LD_indices[i]+2],
                                                params[5+2*np.max(LD_indices)+2:]))
        else:
            params_only_transit=np.copy(params)
        
        try:
            loglike=single_loglike(params_only_transit,t,detrended_obs,detrended_sig,b_l_single,b_u_single,
                minus_inf=minus_inf,mode=mode,use_kipping_LD_param=use_kipping_LD_param,
                interpolate_occultquad=interpolate_occultquad,verbosity=verbosity,
                fix_blocking=fix_blocking,
                density_prior=density_prior,stellar_density=stellar_density,
                oversample_factor=f,exposure_time=exposure_time,use_inclination=use_inclination)
            if return_model:
                m=single_loglike(params_only_transit,t,detrended_obs,detrended_sig,b_l_single,b_u_single,
                minus_inf=minus_inf,mode="model light curve",use_kipping_LD_param=use_kipping_LD_param,
                interpolate_occultquad=interpolate_occultquad,verbosity=verbosity,
                fix_blocking=fix_blocking,
                density_prior=density_prior,stellar_density=stellar_density,
                oversample_factor=f,exposure_time=exposure_time,use_inclination=use_inclination)
                model_lc.append(m)
            if verbosity>2:
                print("loglike:", loglike, "transit:", i)
            if minus_inf==loglike:
                return minus_inf
            result+=loglike
        except ValueError:
            if verbosity>2:
                print("return minusinf")
            return minus_inf
    if return_model:
        return model_lc
    return result

def model_one_moon_with_detrend(time,params,detrend_order=2, LD_indices=[0,0,0,1],use_inclination=False,fix_blocking=False):
    n_detr=len(time)*(detrend_order+1)

    detrend_params=np.array(params[-n_detr:]).reshape(len(time),detrend_order+1)
    transit_params=params[:-n_detr]
    lc=[]
    for t,d,i in zip(time,detrend_params,range(len(time))):
        transit_params_loc=np.concatenate((transit_params[:5],
                                                transit_params[5+2*LD_indices[i]:5+2*LD_indices[i]+2],
                                                transit_params[5+2*np.max(LD_indices)+2:]))
        transit_params_loc[9],transit_params_loc[10]=transit_params_loc[10],transit_params_loc[9]
        m=model_one_moon(t,*list(transit_params_loc))
        
        detrend_model=np.polyval(d,t-t[len(t)/2])+1.

        lc.append(m*detrend_model)
    return lc

def model_no_moon_with_detrend(time,params,detrend_order=2, use_inclination=False,LD_indices=[0,0,0,1],fix_blocking=False):
    n_detr=len(time)*(detrend_order+1)

    detrend_params=np.array(params[-n_detr:]).reshape(len(time),detrend_order+1)
    transit_params=params[:-n_detr]
    lc=[]
    for t,d,i in zip(time,detrend_params,range(len(time))):
        transit_params_loc=np.concatenate((transit_params[:5],
                                                transit_params[5+2*LD_indices[i]:5+2*LD_indices[i]+2]))
        
        m=model_no_moon(t,*list(transit_params_loc))

        detrend_model=np.polyval(d,t-t[len(t)/2])+1.

        lc.append(m*detrend_model)
    return lc


def polydetrend(time,flux,sigma,polyorder=2,return_parameters=False):
    p=np.polyfit(time,flux,polyorder,w=1./sigma**2.)
    m=np.polyval(p,time)
    if return_parameters:
        return p
    return flux/m, sigma/m, p, m

def log_likelihood_buildin_detrend(params,time,obs,sigma,b_l=None,b_u=None,evaluation_times=None,detrend_order=2, minus_inf=-np.inf, mode="both",moonness="no_moon",LD_indices=None,use_inclination=False,use_kipping_LD_param=True, interpolate_occultquad=None,split_period=False,verbosity=0,plots=False, fix_blocking=False, density_prior=False,stellar_density=None,oversample_factor=1,exposure_time=None):
    single_model=model_no_moon
    single_loglike=log_likelihood_no_moon
    if moonness=="one_moon":
        single_model=model_one_moon
        single_loglike=log_likelihood_one_moon

    if evaluation_times == None:
        evaluation_times=time
    result=0
    if verbosity>2:
        print("mode:", mode)
        print("params:", params)
        print("bound_l:", b_l)
        print("bound_h:", b_u)
    detrend_params=[]
    model_lc=[]
    if "projected distance"==mode:
        z_P=[]
        z_M=[]
    for t,et,o,s,f,i in zip(time,evaluation_times,obs,sigma,oversample_factor,range(len(time))):
        if LD_indices:
            #params_only_transit[9],params_only_transit[10]=params_only_transit[10],params_only_transit[9]
            if b_l:
                b_l_single=np.concatenate((b_l[:5],
                                            b_l[5+2*LD_indices[i]:5+2*LD_indices[i]+2],
                                            b_l[5+2*np.max(LD_indices)+2:]))
            else: b_l_single=None
            if b_u:
                b_u_single=np.concatenate((b_u[:5],
                                                b_u[5+2*LD_indices[i]:5+2*LD_indices[i]+2],
                                                b_u[5+2*np.max(LD_indices)+2:]))
            else:
                b_u_single=None
            if moonness=="one_moon":
                params_only_transit=np.concatenate((params[:5],
                                                params[5+2*LD_indices[i]:5+2*LD_indices[i]+2],
                                                params[5+2*np.max(LD_indices)+2:]))

            else:
                params_only_transit=np.concatenate((params[:5],
                                                params[5+2*LD_indices[i]:5+2*LD_indices[i]+2]))
        else:
            params_only_transit=np.copy(params)

        if "prior"==mode or "both"==mode:
            #try:
            result+=single_loglike(params_only_transit,t,o,s,b_l_single,b_u_single,
                    minus_inf=minus_inf,mode=mode,use_kipping_LD_param=use_kipping_LD_param,
                    interpolate_occultquad=interpolate_occultquad,verbosity=verbosity,split_period=split_period,fix_blocking=fix_blocking,
                    density_prior=density_prior,stellar_density=stellar_density,
                    oversample_factor=f,exposure_time=exposure_time)
            
            if not np.isfinite(result):
                if verbosity>1:
                    print("minus inf prior")
                return minus_inf
        if "probability"==mode or "detrend parameters" ==mode or "model light curve"==mode or "both"==mode or "projected distance"==mode:
            #calculate model light curve 
            if "one_moon" == moonness:
                params_only_transit[9],params_only_transit[10]=params_only_transit[10],params_only_transit[9]
                if split_period:
                    n_s=np.round(params_only_transit[12])
                    delta_phi=params_only_transit[9]
                    params_only_transit[9]=params_only_transit[4]/(n_s+delta_phi)
                    params_only_transit=params_only_transit[:-1]
            if use_kipping_LD_param:
                q1,q2=params_only_transit[5],params_only_transit[6]
                u1=2.0*np.sqrt(q1)*q2
                u2=np.sqrt(q1)*(1.0-2.0*q2)
                params_only_transit[5],params_only_transit[6]=u1,u2
            m=single_model(t,*list(params_only_transit),interpolate_occultquad=None)

            if "projected distance"==mode:
                print("Calculating projected distance")
                z_P_tr,z_M_tr=single_model(t,*list(params_only_transit),interpolate_occultquad=None,return_z=True)
                z_P.append(z_P_tr)
                z_M.append(z_M_tr)

            if detrend_order>=0:
                od,sod,poly, detr=polydetrend(t,o/m,s,polyorder=detrend_order)
            else:
                od=np.copy(o/m)
                poly=[0.0]
                sod=np.copy(s)
                detr=np.ones(len(t))


            detrend_params.append(poly)
            if "model light curve" == mode:
                em=single_model(et,*list(params_only_transit),interpolate_occultquad=None)
                if detrend_order>=0:
                    detrend_curve=np.polyval(poly,et)
                else:
                    detrend_curve=np.ones(len(et))
                model_lc.append(em*detrend_curve)

            if False:
                pl.scatter(et,o)
                pl.plot(t,m*detr)
                pl.plot(t,od)
                pl.show()

            if "probability"==mode or "both"==mode:
                result+=log_likelihood_no_signal(t,od-1.0,sod)
                if not np.isfinite(result):
                    if verbosity>1:
                        print("minus inf prob")
                    return minus_inf
    if "projected distance"==mode:
        return z_P, z_M
    if "detrend parameters"==mode:
        return detrend_params
    if "model light curve"==mode:
        return model_lc

    if verbosity>1:
        print("prob:", result)
    return result

def log_likelihood_one_moon(params,time,obs,sigma,b_l,b_u, minus_inf=-np.inf, mode="both",use_kipping_LD_param=True, interpolate_occultquad=None,verbosity=0,plots=False,split_period=False, fix_blocking=False, density_prior=False,stellar_density=None,oversample_factor=1,exposure_time=None,use_inclination=False):
    #verbosity=2

    if verbosity>2:
        print("with inclination?", use_inclination)
        print("parameter size:", len(params))
        print("mode:", mode)
        print("shapes:")
        print("  params:", np.array(params).shape)
        print("  time:", np.array(time).shape)
        print("  obs:", np.array(obs).shape)
        print("  b_l:", np.array(b_l).shape)
        print("  b_u:", np.array(b_u).shape)

    if oversample_factor==1:
        time=np.copy(time)
        obs_time=np.copy(time)
    else:
        if exposure_time is None:
            exposure_time=time[1]-time[0]
        obs_time=np.copy(time)
        time=(np.array(time)[:,None]+(np.arange(oversample_factor+2)[None,1:-1]*1.0/(oversample_factor+1.0)-0.5)*exposure_time).reshape(-1)
    if mode == "both":
        calc_prior=True
        calc_prob=True
    if mode=="prior":
        calc_prior=True
        calc_prob=False
    if mode=="probability":
        calc_prior=False
        calc_prob=True
    if mode=="model light curve":
        calc_prior=True
        calc_prob=True


    params[0]=np.abs(params[0])
    params[2]=np.abs(params[2])
    params[7]=np.abs(params[7])
    params[11]=np.abs(params[11])
    #params[3]=params[3]%1.0
    #params[10]=params[10]%1.0


    #first calculate the prior
    if verbosity>2:
        print("Parameters:", params)
    pr=1.0

    if verbosity>2:
        print("prior:", pr)

    ratio_P  = params[0]
    a_o_R = params[1]
    impact = params[2]
    phase_bary=params[3]
    per_B = params[4]
    ratio_M  = params[7]
    a_o_R_PM= params[8]
    phase_moon = params[9]
    per_PM = params[10]
    mass_ratio_MP = params[11]

    if split_period:
        n_s=round(params[12])
        delta_phi=per_PM
        per_PM=per_B/(n_s+delta_phi)

    if use_inclination:
        i_s=params[12]
        Omega_s=params[13]


    if np.any(np.array(params)<np.array(b_l)) or np.any(np.array(params)>np.array(b_u)):
        if verbosity > 1:
            print("Exiting because params is not in bounds! params:", params, "\np<b:", np.array(params)<np.array(b_l), "\np>b:", np.array(params)>np.array(b_u))
        return minus_inf


    if verbosity>2:
        print("(2)prior:", pr)
    if calc_prior:
        if verbosity>2:
            print("Prior verbosity:", verbosity)
        if ratio_P<ratio_M:
            if verbosity>1:
                print("wrong ratio!")
            return minus_inf
        roche_limit_o_R_star = ratio_M*(2.0/mass_ratio_MP)**(1.0/3.0)
        factor_hill=(per_PM/per_B)**2.0-(a_o_R_PM/a_o_R)**3.0
        if 0.5**3.0/(3.0*(1.0+mass_ratio_MP)) < factor_hill:
            if verbosity>1:
                print("Hill radius violation!")
            return minus_inf
        if a_o_R_PM < roche_limit_o_R_star:
            if verbosity>1:
                print("Roche radius violation: R_Roche/R_star =", roche_limit_o_R_star, ", a_ps/R_star =", a_o_R_PM)
            return minus_inf

        if density_prior and ratio_P>0. and ratio_M>0.0:
            fact_A = a_o_R_PM**3./((1.0+mass_ratio_MP)*4.*np.pi/3.)*(365.25/per_PM)**2.
            density_p = 1.0/9.286e6*fact_A/ratio_P**3. # in rho_Earth 
            density_s = 1.0/9.286e6*fact_A*mass_ratio_MP/ratio_M**3. # in rho_Earth
            if verbosity>2:
                print("Planet density:", density_p, "Moon density:", density_s)
                print("parameters:", params)
            if density_p<0.1 or density_p > 5.0:
                if verbosity>1:
                    print("planet density violation: rho[rho_Earth] =", density_p, "a_s/R_star =", a_o_R_PM, "P_s[d] =", per_PM)
                return minus_inf
            if density_s<0.1 or density_s > 5.0:
                if verbosity>1:
                    print("moon density violation: rho[rho_Earth] =", density_s, "a_s/R_star =", a_o_R_PM, "P_s[d] =", per_PM)
                return minus_inf
        if stellar_density is not None:
            st_dens=a_o_R**3.*9.934e6*(365.25/per_B)**2.
            pr*=np.exp(-0.5*(stellar_density["mean"]/stellar_density["sigma"])**2.)
            if verbosity>2:
                print("Stellar density [rho_sun]:", st_dens)
        if split_period:
            pr*=per_PM
    ln_pr=np.log(pr)

    if verbosity>1:
        print("prior:", ln_pr)
    if "prior" == mode:
        return ln_pr

    ln_P=0.0
    ln_P_const=np.sum(-0.5*np.log(2.0*np.pi*sigma**2.0))
    if calc_prob:


        if use_kipping_LD_param:
            q1=params[5]
            q2=params[6]
            u1=2.0*np.sqrt(q1)*q2
            u2=np.sqrt(q1)*(1.0-2.0*q2)

        if use_inclination:
            model_lc = model_one_moon(time,ratio_P,a_o_R,impact,phase_bary,per_B, u1,u2,
            ratio_M, a_o_R_PM, phase_moon, per_PM, mass_ratio_MP, i_s, Omega_s,
            plots=plots,verbosity=verbosity,interpolate_occultquad=interpolate_occultquad,return_z=False,fix_blocking=fix_blocking)
        else:
            model_lc = model_one_moon(time,ratio_P,a_o_R,impact,phase_bary,per_B, u1,u2,
            ratio_M, a_o_R_PM, phase_moon, per_PM, mass_ratio_MP,
            plots=plots,verbosity=verbosity,interpolate_occultquad=interpolate_occultquad,return_z=False,fix_blocking=fix_blocking)

        if np.any(model_lc>1.0+1.0e-5):
            if verbosity>2:
                print("model_LC>1 encountered")
            return minus_inf

        if plots or verbosity>2:
            pl.scatter(obs_time,obs,c="k", linewidth=0,s=3)
            pl.plot(time, model_lc,c="C1",linewidth=2)
            pl.show()
        
        if mode=="model light curve":
            return np.mean(model_lc.reshape(-1,oversample_factor),axis=1)

        if oversample_factor==1:
            ln_P=np.sum(-0.5*(model_lc-obs)**2.0/sigma**2.0)+ln_P_const
        else:
            ln_P=np.sum(-0.5*(np.mean(model_lc.reshape(-1,oversample_factor),axis=1)-obs)**2.0/sigma**2.0)+ln_P_const
    if verbosity>1:
        print("ln_P:", ln_P, "ln_pr:", ln_pr)
    return ln_P+ln_pr



def log_likelihood_no_moon(params,time,obs,sigma,b_l,b_u, minus_inf=-np.inf, mode="both",use_kipping_LD_param=True, interpolate_occultquad=None,verbosity=0,plots=False,split_moon_period=False, fix_blocking=False,use_inclination=False, density_prior=False,stellar_density=None,split_period=False,oversample_factor=1,exposure_time=None):
    if oversample_factor==1:
        time=np.copy(time)
        obs_time=np.copy(time)
    else:
        if exposure_time is None:
            exposure_time=time[1]-time[0]
        obs_time=np.copy(time)
        time=(np.array(time)[:,None]+(np.arange(oversample_factor+2)[None,1:-1]*1.0/(oversample_factor+1.0)-0.5)*exposure_time).reshape(-1)
    calc_prior=True
    calc_prob=True
    if mode=="prior":
        calc_prior=True
        calc_prob=False
    if mode=="probability":
        calc_prior=False
        calc_prob=True


    params[0]=np.abs(params[0])
    params[2]=np.abs(params[2])
    params[3]=params[3]%1.0

    ratio_P  = params[0]
    a_o_R = params[1]
    impact = params[2]
    phase_bary=params[3]
    per_B = params[4]


    #first calculate the prior
    if verbosity>2:
        print(params)
    pr=1.0
    if calc_prior:
        if np.any(np.array(params)<np.array(b_l)) or np.any(np.array(params)>np.array(b_u)):
            if verbosity > 1:
                print("Exiting because params is not in bounds! params:", params)
            return minus_inf
        if verbosity>2:
            print("Prior verbosity:", verbosity)
        if stellar_density is not None:
            st_dens=a_o_R**3.*9.934e6*(365.25/per_B)**2.
            pr*=np.exp(-0.5*(stellar_density["mean"]/stellar_density["sigma"])**2.)
            if verbosity>3:
                print("Stellar density [rho_sun]:", st_dens)

    ln_pr=np.log(pr)

    ln_P=0.0
    ln_P_const=np.sum(-0.5*np.log(2.0*np.pi*sigma**2.0))

    if calc_prob:
        if use_kipping_LD_param:
            q1=params[5]
            q2=params[6]
            u1=2.0*np.sqrt(q1)*q2
            u2=np.sqrt(q1)*(1.0-2.0*q2)

        model_lc = model_no_moon(time,ratio_P,a_o_R,impact,phase_bary,per_B, u1,u2,
            plots=plots,verbosity=verbosity,interpolate_occultquad=interpolate_occultquad,return_z=False)


        if plots:
            pl.scatter(obs_time,obs,c="k", linewidth=0,s=3)
            pl.plot(time, model_lc,c="C1",linewidth=2)
            pl.show()
        
        if oversample_factor==1:
            ln_P=np.sum(-0.5*(model_lc-obs)**2.0/sigma**2.0)+ln_P_const
        else:
            ln_P=np.sum(-0.5*(np.mean(model_lc.reshape(-1,oversample_factor),axis=1)-obs)**2.0/sigma**2.0)+ln_P_const
    if verbosity>1:
        print("ln_P:", ln_P, "ln_pr:", ln_pr)
    return ln_P+ln_pr



def log_likelihood_no_signal(time,obs,sigma, minus_inf=-np.inf,verbosity=0):
    ln_P_const=np.sum(-0.5*np.log(2.0*np.pi*sigma**2.0))
    ln_P=np.sum(-0.5*(obs)**2.0/sigma**2.0)+ln_P_const
    if verbosity>1:
        print("ln_P:", ln_P, "ln_pr:", ln_pr)
    return ln_P



def run_mcmc_2(time,flux,sigma_flux,model,bounds,first_guess,ndim,nwalkers,n_run,n_burn, save=False,verbosity=0,save_between=None,save_between_path=None, use_kipping_LD_param=True,interpolate_occultquad=None,split_moon_period=False, allow_oversampling=True,fix_blocking=False,use_own_method=True, density_prior=False,stellar_density=None, emcee_a=2., use_mpi=False,restart_from=None):
	
	
    oversampling_factor=1
    exposure_time=time[1]-time[0]
    if allow_oversampling:
        oversampling_factor=int(exposure_time/(5./60./24.))+1


    #select model
    lnprobkwargs={"use_kipping_LD_param":use_kipping_LD_param, "interpolate_occultquad":interpolate_occultquad, "verbosity":verbosity, "fix_blocking":fix_blocking, "stellar_density":stellar_density}
    if model=="one_moon" and density_prior:
        lnprobkwargs["density_prior"]=density_prior
    if model=="no_moon":
        label_ar = ["ratio_P", "a_o_R", "impact", "phase", "period", "LD q1", "LD q2"]
        lnprob=log_likelihood_no_moon
        lnprobkwargs["oversample_factor"]=oversampling_factor
    if model=="one_moon":
        lnprob=log_likelihood_one_moon
        lnprobkwargs["oversample_factor"]=oversampling_factor
        label_ar = ["ratio_P", "a_B_o_R", "impact_B", "phase_B", "period_B", "LD q1", "LD q2","ratio_M","a_pm_o_R","P_pm","phase_M", "mass ratio"]

    print("Use mpi?", use_mpi)
    if use_mpi:
        from emcee.utils import MPIPool
        pool = MPIPool()
        print("Is master?", pool.is_master())
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

    if restart_from is None:
        #randomize start param
        pos = [np.array(first_guess[0]) + rnd.rand(ndim) * (np.array(first_guess[1]) - np.array(first_guess[0])) for i in range(nwalkers)]

        for i in range(len(pos)):
            p=pos[i]
            while not np.isfinite(lnprob(p,time,flux,sigma_flux,first_guess[0],first_guess[1],plots=False,**lnprobkwargs)):
                if verbosity>1:
                    print(i)
                p=np.array(first_guess[0]) + rnd.rand(ndim) * (np.array(first_guess[1]) - np.array(first_guess[0]))
            pos[i]=p
            print("init param", i+1, "/", len(pos),"done!")


        
        print("Start burn-in.")

        #starting smaller chains that are kitted together later
        walkers_pos=[]
        n_walkers_tot=0
        n_walkers_burnin=2*len(pos[0])+2

        nwalkers_over=n_walkers_burnin*(int(nwalkers/n_walkers_burnin)+1)-nwalkers
        pos_ex = [np.array(first_guess[0]) + rnd.rand(ndim) * (np.array(first_guess[1]) - np.array(first_guess[0])) for i in range(nwalkers_over)]

        for i in range(len(pos_ex)):
            p=pos_ex[i]
            while not np.isfinite(lnprob(p,time,flux,sigma_flux,first_guess[0],first_guess[1],**lnprobkwargs)):
                p=np.array(first_guess[0]) + rnd.rand(ndim) * (np.array(first_guess[1]) - np.array(first_guess[0]))
            pos_ex[i]=p
            print("init param", i+1, "/", len(pos_ex),"done!")

        pos_2=pos+pos_ex

        print("Running", int(nwalkers/n_walkers_burnin)+1, "subchains.")
        i_subch=0
        while n_walkers_tot<nwalkers:
            i_subch+=1
            if use_mpi:
                sampler = emcee.EnsembleSampler(n_walkers_burnin, ndim, lnprob, args=(time,flux,sigma_flux,bounds[0],bounds[1]),kwargs=lnprobkwargs, pool=pool,a=emcee_a)
            else:
                sampler = emcee.EnsembleSampler(n_walkers_burnin, ndim, lnprob, args=(time,flux,sigma_flux,bounds[0],bounds[1]),kwargs=lnprobkwargs, threads=1,a=emcee_a)
            sampler.run_mcmc(pos_2[n_walkers_tot:n_walkers_tot+n_walkers_burnin],n_burn)
            if n_walkers_tot+n_walkers_burnin>nwalkers:
                walkers_pos.extend(sampler.chain[:nwalkers-n_walkers_tot,-1,:])
            else:
                walkers_pos.extend(sampler.chain[:,-1,:])
            del sampler
            n_walkers_tot+=n_walkers_burnin
            print("Burn-in of subchain", i_subch, "done.")
    if use_mpi:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(time,flux,sigma_flux,bounds[0],bounds[1]),kwargs=lnprobkwargs, pool=pool,a=emcee_a)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(time,flux,sigma_flux,bounds[0],bounds[1]),kwargs=lnprobkwargs, threads=1,a=emcee_a)
    
    if restart_from is None:
        sampler.run_mcmc(walkers_pos,n_burn)
        pos_b=sampler.chain[:,-1,:]
        sampler.reset()
    else:
        data=np.load(restart_from)
        sampler.chain=np.copy(data["chain"])
        sampler.lnprobability=np.copy(data["lnprob"])

    if verbosity>0:
        print("Burn-in complete. Starting for real.")
    if save_between_path is None:
        save_between_path="mcmc_run_between_"+model

    n_run_part=n_run
    if save_between is not None:
        n_run_part=min(n_run,save_between)
        print("Choosing sub-chain-length:", n_run_part)
        if restart_from is None:
            sampler.run_mcmc(pos_b,n_run_part)
        else:
            sampler.run_mcmc(None,n_run_part)
        while len(sampler.chain[0])<n_run:
            np.savez_compressed(save_between_path, chain=sampler.chain, lnprob=sampler.lnprobability,
                        labels=label_ar, mcmc_params={"n_dim":ndim,"n_walkers":nwalkers, "Acc_fr":sampler.acceptance_fraction})
            if verbosity>0:
                print("Saved progress. Chain length:", len(sampler.chain[0]))
            sampler.run_mcmc(None,n_run_part)
            if verbosity>0:
                try:
                    auto_time=sampler.get_autocorr_time(c=0.1)
                    print("Min/Max AutoCorrTime:",np.min(auto_time), np.max(auto_time))
                except AutocorrError as AcE:
                    print(AcE.message)
                except Exception as E:
                    print(E.message)
        np.savez_compressed(save_between_path, chain=sampler.chain, lnprob=sampler.lnprobability,
                        labels=label_ar, mcmc_params={"n_dim":ndim,"n_walkers":nwalkers, "Acc_fr":sampler.acceptance_fraction})
    else:
        if restart_from is None:
            sampler.run_mcmc(pos_b,n_run)
        else:
            sampler.run_mcmc(None,n_run)

    if use_mpi:
        pool.close()

    if verbosity>0:
        print("Run done.")
    if save:
        np.savez_compressed("mcmc_run_"+model, chain=sampler.chain, lnprob=sampler.lnprobability,
             labels=label_ar, mcmc_params={"n_dim":ndim,"n_walkers":nwalkers, "Acc_fr":sampler.acceptance_fraction})
    return sampler

    
def run_ptmcmc(time,flux,sigma_flux,model,bounds,first_guess,ndim,nwalkers,n_run,n_burn,n_temp,save=False,verbosity=0,save_between=None,save_between_path=None,use_kipping_LD_param=True,interpolate_occultquad=None,allow_oversampling=True,fix_blocking=False,use_own_method=True, density_prior=False, emcee_a=2.):

    oversampling_factor=1
    exposure_time=time[1]-time[0]
    if allow_oversampling:
        oversampling_factor=int(exposure_time/(5./60./24.))+1


    system=star_system(time,flux,sigma_flux,oversample_factor=oversampling_factor,exposure_time=exposure_time)

    system.use_own_method=use_own_method
    #select model
    lnprobkwargs={"use_kipping_LD_param":use_kipping_LD_param, "interpolate_occultquad":interpolate_occultquad, "verbosity":verbosity, "fix_blocking":fix_blocking}
    if model=="one_moon" and density_prior:
        lnprobkwargs["density_prior"]=density_prior
    if model=="no_moon":
        label_ar = ["ratio_P", "a_o_R", "impact", "phase", "period", "LD q1", "LD q2"]
        lnprob=system.log_likelihood_no_moon
    if model=="one_moon":
        lnprob=system.log_likelihood_one_moon
        label_ar = ["ratio_P", "a_B_o_R", "impact_B", "phase_B", "period_B", "LD q1", "LD q2","ratio_M","a_pm_o_R","P_pm","phase_M", "mass ratio"]

    lnpriorkwargs=dict(lnprobkwargs)
    lnpriorkwargs["mode"]="prior"
    lnprobkwargs["mode"]="probability"

    lnprobargs=(bounds[0],bounds[1])
    lnpriorargs=(bounds[0],bounds[1])

    def lnprobability(param):
        return lnprob(param,*lnprobargs,**lnprobkwargs)
    def lnprior(param):
        return lnprob(param,*lnpriorargs,**lnpriorkwargs)
    #randomize start param
    pos = [np.array(first_guess[0]) + rnd.rand(nwalkers,ndim) * (np.array(first_guess[1]) - np.array(first_guess[0])) for i in range(n_temp)]

    for j in range(n_temp):
        for i in range(nwalkers):
            p=pos[j][i]
            while (not np.isfinite(lnprior(p))) and (not np.isfinite(lnprobabilty(p))):
                if verbosity>1:
                    print(i)
                p=np.array(first_guess[0]) + rnd.rand(ndim) * (np.array(first_guess[1]) - np.array(first_guess[0]))
            pos[j][i]=p
        print("init params for temp", j+1, "/", len(pos),"done!")


    
    print("Start burn-in.")


    sampler=emcee.PTSampler(n_temp,nwalkers,ndim,lnprobabilty,lnprior)
    
    sampler.run_mcmc(pos, n_burn)
    pos_b=sampler.chain[:,:,-1,:]
    sampler.reset()

    if verbosity>0:
        print("Burn-in complete. Starting for real.")
    if save_between_path is None:
        save_between_path="ptmcmc_run_between_"+model

    n_run_part=n_run
    if save_between is not None:
        n_run_part=min(n_run,save_between)
        print("Choosing sub-chain-length:", n_run_part)
        sampler.run_mcmc(pos_b,n_run_part)
        while len(sampler.chain[0])<n_run:
            np.savez_compressed(save_between_path, chain=sampler.chain, lnprob=sampler.lnprobability,
                        labels=label_ar, mcmc_params={"n_dim":ndim,"n_walkers":nwalkers,"n_temp":n_temp, "Acc_fr":sampler.acceptance_fraction})
            if verbosity>0:
                print("Saved progress. Chain shape:", sampler.chain.shape)
            sampler.run_mcmc(None,n_run_part)
            if verbosity>0:
                try:
                    auto_time=sampler.get_autocorr_time(c=0.1)
                    print("Min/Max AutoCorrTime:",np.min(auto_time), np.max(auto_time))
                except AutocorrError as AcE:
                    print(AcE.message)
                except Exception as E:
                    print(E.message)
        np.savez_compressed(save_between_path, chain=sampler.chain, lnprob=sampler.lnprobability,
                        labels=label_ar, mcmc_params={"n_dim":ndim,"n_walkers":nwalkers,"n_temp":n_temp, "Acc_fr":sampler.acceptance_fraction})
    else:
        sampler.run_mcmc(pos_b,n_run)

    if verbosity>0:
        print("Run done.")
    if save:
        np.savez_compressed("ptmcmc_run_"+model, chain=sampler.chain, lnprob=sampler.lnprobability,
             labels=label_ar, mcmc_params={"n_dim":ndim,"n_walkers":nwalkers, "Acc_fr":sampler.acceptance_fraction})
    return sampler

def run_ptmcmc_with_detrending(time,flux,sigma_flux,model,bounds,first_guess, ndim,nwalkers,n_run,n_burn,n_temp,use_inclination=False, LD_indices=None,save=False,verbosity=0,save_between=None, restart_from=None,save_between_path=None, use_kipping_LD_param=True,interpolate_occultquad=None, allow_oversampling=True,fix_blocking=False,use_own_method=True, density_prior=False, emcee_a=2.,detrend_order=2,use_mpi=False):

    oversampling_factor=[]
    for t in time:
        oversampling_factor_tr=1
        exposure_time=t[1]-t[0]
        if allow_oversampling:
            oversampling_factor_tr=int(exposure_time/(5./60./24.))+1
        oversampling_factor.append(oversampling_factor_tr)

    #select model
    lnprob=log_likelihood_with_detrend
    lnprobkwargs={  "use_kipping_LD_param":use_kipping_LD_param,
                    "interpolate_occultquad":interpolate_occultquad,
                    "verbosity":verbosity,
                    "detrend_order":detrend_order,
                    "fix_blocking":fix_blocking,
                    "LD_indices":LD_indices}
    #                "stellar_density":stellar_density}
    if model=="one_moon" and density_prior:
        lnprobkwargs["density_prior"]=density_prior
    if model=="no_moon":
        label_ar = ["ratio_P", "a_o_R", "impact", "phase", "period", "LD q1", "LD q2"]
        lnprobkwargs["moonness"]="no_moon"
        lnprobkwargs["oversample_factor"]=oversampling_factor
    if model=="one_moon":
        lnprobkwargs["moonness"]="one_moon"
        lnprobkwargs["use_inclination"]=use_inclination
        lnprobkwargs["oversample_factor"]=oversampling_factor
        label_ar = ["ratio_P", "a_B_o_R", "impact_B", "phase_B", "period_B", "LD q1", "LD q2","ratio_M","a_pm_o_R","P_pm","phase_M", "mass ratio"]

    lnpriorkwargs=dict(lnprobkwargs)
    lnpriorkwargs["mode"]="prior"
    lnprobkwargs["mode"]="probability"

    lnprobargs=(time,flux,sigma_flux,bounds[0],bounds[1])
    lnpriorargs=(time,flux,sigma_flux,bounds[0],bounds[1])

    #randomize start param
    #pos = [np.array(first_guess[0]) + rnd.rand(nwalkers,ndim) * (np.array(first_guess[1]) - np.array(first_guess[0])) for i in range(n_temp)]

    print("Use mpi?", use_mpi)
    import sys
    if use_mpi:
        from emcee.utils import MPIPool
        pool = MPIPool(loadbalance=True)
        print("Is master?", pool.is_master())
        if not pool.is_master():
            try:
                pool.wait()
                sys.exit(0)
            except Exception as E:
                print("Exception:", E, "encountered.")
                sys.exit(0)


    chain=[]
    lnprobability=[]



    if restart_from:

        if ".h5" in restart_from:
            with h5py.File(restart_from,"r") as hf_old:
                pos=hf_old["chain"][:,:,-1,:]
        else:
            old_data=np.load(restart_from)
            chain=list(np.swapaxes(old_data["chain"],0,2))
            lnprobability=list(np.swapaxes(old_data["lnprob"],0,2))

            pos=np.swapaxes(chain[-1],0,1)
        print(pos.shape)
        assert(np.array(pos).shape==(n_temp,nwalkers,ndim))

    else:
        pos = [np.array(first_guess[0]) + rnd.rand(nwalkers,ndim) * (np.array(first_guess[1]) - np.array(first_guess[0])) for i in range(n_temp)]

        for j in range(n_temp):
            print("start temp:", j)
            for i in range(nwalkers):
                p=pos[j][i]
                while (not np.isfinite(lnprob(p,*lnpriorargs,**lnpriorkwargs))) or (not np.isfinite(lnprob(p,*lnprobargs,**lnprobkwargs))):
                    if verbosity>1:
                        print(i, p)
                    p=np.array(first_guess[0]) + rnd.rand(ndim) * (np.array(first_guess[1]) - np.array(first_guess[0]))
                pos[j][i]=p
                print("walker", i, "found")
            if False:
                for p in pos[j]:
                    print(p, LD_indices)
                    if "one_moon" == model:
                        model_lc=model_one_moon_with_detrend(time,p,detrend_order=detrend_order, LD_indices=LD_indices,fix_blocking=fix_blocking,use_inclination=use_inclination)
                    if "no_moon" == model:
                        model_lc=model_no_moon_with_detrend(time,p,detrend_order=detrend_order, LD_indices=LD_indices,fix_blocking=fix_blocking)
                    for t,lc,f in zip(time,model_lc,flux):
                        pl.scatter(t,f,color="k",s=2,linewidth=0)
                        pl.plot(t,lc,c="C0",alpha=0.01,lw=2)
                pl.show()
            print("init params for temp", j+1, "/", len(pos),"done!")

    if ".h5" in save_between_path:
        if not restart_from:
            if os.path.isfile(save_between_path):
                os.remove(save_between_path)
        old_exists=False
        if os.path.isfile(save_between_path):
            old_exists=True
        print("old file", save_between_path, "exists?", old_exists)
        #hf_out=h5py.File(save_between_path,"a")
        if old_exists:
            hf_out=h5py.File(save_between_path,"a")
            hf_out.swmr_mode = True
        else:
            hf_out=h5py.File(save_between_path,"w",libver="latest")
            hf_out.swmr_mode = True
        print("HDF5 file", save_between_path, "opened. SWMR mode?", hf_out.swmr_mode)
        if old_exists:
            hf_lnprob=hf_out["lnprob"]
            hf_chain=hf_out["chain"]
            assert(hf_chain.attrs["use inclination"]==use_inclination)
            assert(hf_chain.attrs["detrend order"]==detrend_order)
            print("opened old file successfully.")
        else:
            chunk_size_lnprob=1000000/(nwalkers*n_temp)
            hf_lnprob=hf_out.create_dataset("lnprob",(n_temp,nwalkers,0),maxshape=(n_temp,nwalkers,None),dtype=np.float64,chunks=(n_temp,nwalkers,chunk_size_lnprob),compression="gzip")
            chunk_size_chain=1000000/(nwalkers*n_temp*ndim)
            hf_chain=hf_out.create_dataset("chain",(n_temp,nwalkers,0,ndim),maxshape=(n_temp,nwalkers,None,ndim),dtype=np.float64,chunks=(n_temp,nwalkers,chunk_size_chain,ndim),compression="gzip")
            
            hf_chain.attrs.create("use inclination",use_inclination)
            hf_chain.attrs.create("detrend order", detrend_order)
            hf_chain.attrs.create("labels", label_ar)
        import signal
        def graceful_close_file(sig,frame):
            print("SIGNAL", sig, "sent. Attempting to close hdf5 file and exit after.")
            hf_out.flush()
            hf_out.close()
            sys.exit(-1)
        signal.signal(signal.SIGINT, graceful_close_file)


    betas=[0.1]**np.arange(n_temp)
    #betas[-1]=0.0
    
    print("Start burn-in.")

    if use_mpi:
        sampler = emcee.PTSampler(n_temp,nwalkers,ndim,
                                    lnprob,lnprob,
                                    loglargs=lnprobargs,loglkwargs=lnprobkwargs,
                                    logpargs=lnpriorargs,logpkwargs=lnpriorkwargs,
                                    pool=pool)
        #sampler = emcee.PTSampler(n_temp,nwalkers,ndim,lnprobabiltiy,lnprior, pool=pool)
    else:
        sampler = emcee.PTSampler(n_temp,nwalkers,ndim,
                                    lnprob,lnprob,
                                    loglargs=lnprobargs,loglkwargs=lnprobkwargs,
                                    logpargs=lnpriorargs,logpkwargs=lnpriorkwargs,
                                    threads=1)
        #sampler = emcee.PTSampler(n_temp,nwalkers,ndim,lnprobabiltiy,lnprior,betas=betas,threads=1)
    
    sampler.run_mcmc(pos, n_burn)
    pos_b=sampler.chain[:,:,-1,:]
    print(pos_b)
    print(sampler.lnprobability)
    sampler.reset()


    if verbosity>0:
        print("Burn-in complete. Starting for real.")
    if save_between_path is None:
        save_between_path="ptmcmc_run_between_"+model

    import thread
    write_finished=True

    def write_output(new_chain,new_lnprob):
        write_finished=False
        chain.extend(np.swapaxes(new_chain,0,2))
        lnprobability.extend(np.swapaxes(new_lnprob,0,2))

        print("lnprob:",np.array(lnprobability).shape)
        print("chain:", np.array(chain).shape)
        print("pos_b:", np.array(pos_b).shape)
        print("start writing file.")
        np.savez_compressed(save_between_path, chain=np.swapaxes(np.array(chain),0,2), lnprob=np.swapaxes(np.array(lnprobability),0,2),
                    labels=label_ar, mcmc_params={"n_dim":ndim,"n_walkers":nwalkers,"n_temp":n_temp, "detrend order":detrend_order, "inclination":use_inclination})
        print("ended writing file.")
        write_finished=True
        return

    n_run_part=n_run
    ln_prob_len=0
    if save_between is not None:
        n_run_part=min(n_run,save_between)
        print("Choosing sub-chain-length:", n_run_part)
        while ln_prob_len<n_run:
            sampler.run_mcmc(pos_b,n_run_part)
            pos_b=sampler.chain[:,:,-1,:]
            if verbosity>0:
                print("Saving progress. Chain shape:", sampler.chain.shape)

            while not write_finished:
                print("Waiting for writing to finish. Recheck in 20 seconds")
                sleep(20)

            if ".h5" in save_between_path:
                hf_lnprob.resize(hf_lnprob.shape[2]+n_run_part,axis=2)
                hf_lnprob[:,:,-n_run_part:]=sampler.lnprobability
                hf_chain.resize(hf_chain.shape[2]+n_run_part,axis=2)
                hf_chain[:,:,-n_run_part:,:]=sampler.chain
                hf_out.flush()
            else:
                write_output(sampler.chain, sampler.lnprobability)
            ln_prob_len+=n_run_part
            if verbosity>0:
                try:
                    auto_time=sampler.get_autocorr_time(c=0.1)
                    print("Min/Max AutoCorrTime:",np.min(auto_time), np.max(auto_time))
                except AutocorrError as AcE:
                    print(AcE.message)
                except Exception as E:
                    print(E.message)
                try:
                    if n_temp>1:
                        thermodynamic_ev=sampler.thermodynamic_integration_log_evidence(fburnin=0)
                        print("Thermodynamic integration log evidence:",thermodynamic_ev)
                except Exception as E:
                    print(E.message)
            sampler.reset()
        #np.savez_compressed(save_between_path, chain=sampler.chain, lnprob=sampler.lnprobability,
        #                labels=label_ar, mcmc_params={"n_dim":ndim,"n_walkers":nwalkers,"n_temp":n_temp, "Acc_fr":sampler.acceptance_fraction})
    else:
        sampler.run_mcmc(pos_b,n_run)
        if save:
            np.savez_compressed("ptmcmc_run_"+model, chain=sampler.chain, lnprob=sampler.lnprobability,
                 labels=label_ar, mcmc_params={"n_dim":ndim,"n_walkers":nwalkers, "detrend order":detrend_order})

    if verbosity>0:
        print("Run done.")
    return sampler

def run_ptmcmc_buildin_detrending(time,flux,sigma_flux,model,bounds,first_guess,ndim,nwalkers,n_run,n_burn,n_temp,save=False,verbosity=0,LD_indices=None,save_between=None,save_between_path=None,save_skip=100,use_kipping_LD_param=True,split_period=False,use_inclination=False, restart_from=None,interpolate_occultquad=None,beta=1.0/np.sqrt(2.0),allow_oversampling=True,fix_blocking=False, use_own_method=True, density_prior=False, emcee_a=2.,detrend_order=2,use_mpi=False):

    oversampling_factor=[]
    for t in time:
        oversampling_factor_tr=1
        exposure_time=t[1]-t[0]
        if allow_oversampling:
            oversampling_factor_tr=int(exposure_time/(5./60./24.))+1
        oversampling_factor.append(oversampling_factor_tr)

    #select model
    lnprob=log_likelihood_buildin_detrend
    lnprobkwargs={  "use_kipping_LD_param":use_kipping_LD_param,
                    "interpolate_occultquad":interpolate_occultquad,
                    "verbosity":verbosity,
                    "detrend_order":detrend_order,
                    "fix_blocking":fix_blocking,
                    "LD_indices":LD_indices}
    #                "stellar_density":stellar_density}
    if model=="one_moon" and density_prior:
        lnprobkwargs["density_prior"]=density_prior
    if model=="no_moon":
        label_ar = ["ratio_P", "a_o_R", "impact", "phase", "period", "LD q1", "LD q2"]
        lnprobkwargs["moonness"]="no_moon"
        lnprobkwargs["oversample_factor"]=oversampling_factor
    if model=="one_moon":
        lnprobkwargs["split_period"]=split_period
        lnprobkwargs["moonness"]="one_moon"
        lnprobkwargs["oversample_factor"]=oversampling_factor
        lnprobkwargs["use_inclination"]=use_inclination
        label_ar = ["ratio_P", "a_B_o_R", "impact_B", "phase_B", "period_B", "LD q1", "LD q2","ratio_M","a_pm_o_R","P_pm","phase_M", "mass ratio"]

    
    lnpriorkwargs=dict(lnprobkwargs)
    detrendparamskwargs=dict(lnprobkwargs)
    lnpriorkwargs["mode"]="prior"
    lnprobkwargs["mode"]="probability"
    detrendparamskwargs["mode"]="detrend parameters"

    lnprobargs=(time,flux,sigma_flux,bounds[0],bounds[1])
    lnpriorargs=(time,flux,sigma_flux,bounds[0],bounds[1])
    detrendparamsargs=(time,flux,sigma_flux,bounds[0],bounds[1])

    print("Use mpi?", use_mpi)
    import sys
    if use_mpi:
        from emcee.utils import MPIPool
        pool = MPIPool()#(loadbalance=True)#
        print("Is master?", pool.is_master())
        if not pool.is_master():
            try:
                pool.wait()
                sys.exit(0)
            except Exception as E:
                print("Exception:", E, "encountered.")
                sys.exit(0)


    chain=[]
    lnprobability=[]

    if restart_from:
        if ".h5" in restart_from:
            with h5py.File(restart_from,"r") as hf_old:
                pos=hf_old["chain"][:,:,-1,:]
        else:
            old_data=np.load(restart_from)
            chain=list(np.swapaxes(old_data["chain"],0,2))
            lnprobability=list(np.swapaxes(old_data["lnprob"],0,2))

            pos=np.swapaxes(chain[-1],0,1)
        print(pos.shape)
        assert(np.array(pos).shape==(n_temp,nwalkers,ndim))

    else:
        pos = [np.array(first_guess[0]) + rnd.rand(nwalkers,ndim) * (np.array(first_guess[1]) - np.array(first_guess[0])) for i in range(n_temp)]

        for j in range(n_temp):
            print("starting with temp", j)
            for i in range(nwalkers):
                p=pos[j][i]
                while (not np.isfinite(lnprob(p,*lnpriorargs,**lnpriorkwargs))) or (not np.isfinite(lnprob(p,*lnprobargs,**lnprobkwargs))):
                    if verbosity>1:
                        print(i, p)
                    p=np.array(first_guess[0]) + rnd.rand(ndim) * (np.array(first_guess[1]) - np.array(first_guess[0]))
                pos[j][i]=p
                print("walker", i, "found")
            if False:
                for p in pos[j]:
                    print(p, LD_indices)
                    if "one_moon" == model:
                        model_lc=model_one_moon_with_detrend(time,p,detrend_order=detrend_order, LD_indices=LD_indices,fix_blocking=fix_blocking,use_inclination=use_inclination)
                    if "no_moon" == model:
                        model_lc=model_no_moon_with_detrend(time,p,detrend_order=detrend_order, LD_indices=LD_indices,fix_blocking=fix_blocking)
                    for t,lc,f in zip(time,model_lc,flux):
                        pl.scatter(t,f,color="k",s=2,linewidth=0)
                        pl.plot(t,lc,c="C0",alpha=0.01,lw=2)
                pl.show()
            print("init params for temp", j+1, "/", len(pos),"done!")

    if ".h5" in save_between_path:
        if not restart_from:
            if os.path.isfile(save_between_path):
                os.remove(save_between_path)
        old_exists=False
        if os.path.isfile(save_between_path):
            old_exists=True
        print("old file", save_between_path, "exists?", old_exists)
        #hf_out=h5py.File(save_between_path,"a")
        if old_exists:
            hf_out=h5py.File(save_between_path,"a")
            hf_out.swmr_mode = True
        else:
            hf_out=h5py.File(save_between_path,"w",libver="latest")
            hf_out.swmr_mode = True
        print("HDF5 file", save_between_path, "opened. SWMR mode?", hf_out.swmr_mode)
        if old_exists:
            hf_lnprob=hf_out["lnprob"]
            hf_chain=hf_out["chain"]
            assert(hf_chain.attrs["use inclination"]==use_inclination)
            assert(hf_chain.attrs["detrend order"]==detrend_order)
            print("opened old file successfully.")
        else:
            chunk_size_lnprob=1000000/(nwalkers*n_temp)
            hf_lnprob=hf_out.create_dataset("lnprob",(n_temp,nwalkers,0),maxshape=(n_temp,nwalkers,None),dtype=np.float64,chunks=(n_temp,nwalkers,chunk_size_lnprob),compression="gzip")
            chunk_size_chain=1000000/(nwalkers*n_temp*ndim)
            hf_chain=hf_out.create_dataset("chain",(n_temp,nwalkers,0,ndim),maxshape=(n_temp,nwalkers,None,ndim),dtype=np.float64,chunks=(n_temp,nwalkers,chunk_size_chain,ndim),compression="gzip")

            for i in range(len(time) ):
                observ_name="Observation %i" % i
                grp =  hf_out.create_group(observ_name)
                ob=grp.create_dataset("observations",(3,len(time[i])))
                ob[:]=np.vstack([time[i],flux[i],sigma_flux[i]])
                ob.attrs.create("type", "light curve")
            
            hf_chain.attrs.create("use inclination",use_inclination)
            hf_chain.attrs.create("detrend order", detrend_order)
            hf_chain.attrs.create("labels", label_ar)




    betas=beta**np.arange(n_temp)
    #betas[-1]=0.0
    
    print("Start burn-in.")

    if use_mpi:
        sampler = emcee.PTSampler(n_temp,nwalkers,ndim,
                                    lnprob,lnprob,
                                    loglargs=lnprobargs,loglkwargs=lnprobkwargs,
                                    logpargs=lnpriorargs,logpkwargs=lnpriorkwargs,betas=betas,
                                    pool=pool)
        #sampler = emcee.PTSampler(n_temp,nwalkers,ndim,lnprobabiltiy,lnprior, pool=pool)
    else:
        sampler = emcee.PTSampler(n_temp,nwalkers,ndim,
                                    lnprob,lnprob,
                                    loglargs=lnprobargs,loglkwargs=lnprobkwargs,
                                    logpargs=lnpriorargs,logpkwargs=lnpriorkwargs,betas=betas,
                                    threads=1)
        #sampler = emcee.PTSampler(n_temp,nwalkers,ndim,lnprobabiltiy,lnprior,betas=betas,threads=1)
    
    sampler.run_mcmc(pos, n_burn)
    pos_b=sampler.chain[:,:,-1,:]
    print(pos_b)
    print(sampler.lnprobability)
    sampler.reset()


    if verbosity>0:
        print("Burn-in complete. Starting for real.")
    if save_between_path is None:
        save_between_path="ptmcmc_run_between_"+model

    n_run_part=n_run
    ln_prob_len=0
    if save_between is not None:
        n_run_part=min(n_run,save_between)
        print("Choosing sub-chain-length:", n_run_part)
        while ln_prob_len<n_run:
            sampler.run_mcmc(pos_b,n_run_part)
            pos_b=sampler.chain[:,:,-1,:]
            if verbosity>0:
                print("Saving progress. Chain shape:", sampler.chain.shape)

            if ".h5" in save_between_path:
                hf_lnprob=hf_out["lnprob"]
                hf_chain=hf_out["chain"]
                hf_lnprob.resize(hf_lnprob.shape[2]+n_run_part/save_skip,axis=2)
                hf_lnprob[:,:,-n_run_part/save_skip:]=sampler.lnprobability[:,:,::save_skip]
                hf_chain.resize(hf_chain.shape[2]+n_run_part/save_skip,axis=2)
                hf_chain[:,:,-n_run_part/save_skip:,:]=sampler.chain[:,:,::save_skip,:]
                hf_out.flush()
                print("lnprob:",hf_lnprob.shape)
                print("chain:", hf_chain.shape)
                print("pos_b:", np.array(pos_b).shape)
            else:
                write_output(sampler.chain, sampler.lnprobability)
            ln_prob_len+=n_run_part
            if verbosity>0:
                try:
                    auto_time=sampler.get_autocorr_time(c=0.1)
                    print("Min/Max AutoCorrTime:",np.min(auto_time), np.max(auto_time))
                except AutocorrError as AcE:
                    print(AcE.message)
                except Exception as E:
                    print(E.message)
                try:
                    if n_temp>1:
                        thermodynamic_ev=sampler.thermodynamic_integration_log_evidence(fburnin=0)
                        print("Thermodynamic integration log evidence:",thermodynamic_ev)
                except Exception as E:
                    print(E.message)
            sampler.reset()
        #np.savez_compressed(save_between_path, chain=sampler.chain, lnprob=sampler.lnprobability,
        #                labels=label_ar, mcmc_params={"n_dim":ndim,"n_walkers":nwalkers,"n_temp":n_temp, "Acc_fr":sampler.acceptance_fraction})
    else:
        sampler.run_mcmc(pos_b,n_run)
        if save:
            np.savez_compressed("ptmcmc_run_"+model, chain=sampler.chain, lnprob=sampler.lnprobability,
                 labels=label_ar, mcmc_params={"n_dim":ndim,"n_walkers":nwalkers, "detrend order":detrend_order})

    if verbosity>0:
        print("Run done.")
    return sampler
    
def calculate_M(sma,per):#sma in au, per in days. output: Mass in earth mass
    return sma**3.0/(per/365.25)**2.0/3.0e-6

def calculate_P(sma,mass):#Mass in earth mass, sma in au. output in days
    return 365.25*np.sqrt((sma)**3.0*4*np.pi**2.0/(4*np.pi**2.0*(mass)*3e-6))
