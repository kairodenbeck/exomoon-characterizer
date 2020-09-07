from __future__ import print_function, division
import numpy as np

import occultquad as oq

def my_occultquad(z,u1,u2,p0):
    """
    A wrapper for the original occultquad to fit my code better. 
    """
    return oq.occultquad(z,p0,[u1,u2])

#constants
R_sun_in_au = 0.00465
R_earth_in_R_sun = 0.009157694

def model_no_moon(time,ratio_P,a_o_R,impact_B,phase_B,period_B, c1,c2,verbosity=0,return_z=False, plots=False):
    const=a_o_R
    if verbosity>1:
        print(const)
    ratio_P=np.abs(ratio_P)
    z_B_x=const*np.sin(2.0*np.pi*(time/period_B-phase_B))
    z_B_y=np.abs(impact_B)*np.cos(2.0*np.pi*(time/period_B-phase_B))
    z_coord=np.cos(2.0*np.pi*(time/period_B-phase_B))
    z_B_y[z_coord<0.0]=10.0*a_o_R#exclude if behind star

    z_B=np.sqrt(z_B_x**2.0+z_B_y**2.0)

    if return_z:
        return ([z_B_x,z_B_y])

    u1 = c1#0.4089
    u2 = c2#0.2556

    transit_signal_P=np.ones(len(z_B))

    transit_signal_P[z_B<1.0+ratio_P] = my_occultquad(z_B[z_B<1.0+ratio_P], u1, u2, ratio_P)

    return transit_signal_P


def model_one_moon(time, ratio_P,a_o_R,impact_B,phase_B,period_B, c1,c2, ratio_M, semi_major_axis_PM_o_R, phase_M, period_PM, mass_ratio_MP, i_s=0,Omega_s=0, full_out=False,plots=False,verbosity=0, return_z=False,fix_blocking=False,blocking_res=300.):

    const=a_o_R
    if verbosity>1:
        print(const)
    z_B_x=const*np.sin(2.0*np.pi*(time/period_B-phase_B))
    z_B_y=impact_B*np.cos(2.0*np.pi*(time/period_B-phase_B))
    z_B_y[np.cos(2.0*np.pi*(time)/period_B-phase_B)<0.0]+=10.0*a_o_R#exclude if behind star

    pot_tr_mask = z_B_x**2.+z_B_y**2. < (1.0+semi_major_axis_PM_o_R+ratio_P)**2.

    if return_z:
        pot_tr_mask=time>-1e9
    
    total_transit_all=np.ones(len(time))

    if not np.any(pot_tr_mask):
        return total_transit_all

    time_p=time[pot_tr_mask]

    E=2.0*np.pi*(time_p/period_PM+phase_M) #need to solve for E from M iff ecc!=0

    ecc=0.0

    p=np.cos(E) - ecc
    q=np.sin(E) * np.sqrt(1.0-ecc**2.0)

    omega_s=0.0#argument of periapsis

    z=np.cos(omega_s)*p-np.sin(omega_s)*q
    x=np.sin(omega_s)*p+np.cos(omega_s)*q

    #i_s: inclination

    y=np.sin(i_s)*z
    z=np.cos(i_s)*z

    #Omega_s: longitude of ascending node

    z,x=np.cos(Omega_s)*z-np.sin(Omega_s)*x,np.sin(Omega_s)*z+np.cos(Omega_s)*x

    const_P=mass_ratio_MP/(1.0+mass_ratio_MP)*semi_major_axis_PM_o_R

    z_P_x= const_P*x
    z_P_y= const_P*y

    const_M=-1.0/(1.0+mass_ratio_MP)*semi_major_axis_PM_o_R

    if verbosity>1:
        print(const_M)
    
    z_M_x=const_M*x
    z_M_y=const_M*y

    if plots:
        import pylab as pl
        pl.plot(time[0:1000],z_M_x[0:1000],c="C0", linestyle="-")
        pl.plot(time[0:1000],z_M_y[0:1000],c="C0", linestyle=":")
        pl.plot(time[0:1000],z_P_x[0:1000],c="C1", linestyle="-")
        pl.plot(time[0:1000],z_P_y[0:1000],c="C1", linestyle=":")
        pl.show()
    
    z_B_P_x=z_B_x[pot_tr_mask]+z_P_x
    z_B_P_y=z_B_y[pot_tr_mask]+z_P_y
    
    z_B_M_x=z_B_x[pot_tr_mask]+z_M_x
    z_B_M_y=z_B_y[pot_tr_mask]+z_M_y
    
    z_B_P=np.sqrt(z_B_P_x**2.0+z_B_P_y**2.0)
    z_B_M=np.sqrt(z_B_M_x**2.0+z_B_M_y**2.0)

    if return_z:
        return ([z_B_P_x,z_B_P_y],[z_B_M_x,z_B_M_y])

    if plots:
        pl.plot(time_p,z_B_P,label="Planet")
        pl.plot(time_p,z_B_M,label="Moon")
        pl.legend()
        pl.show()


    u1 = c1#0.4089
    u2 = c2#0.2556

    transit_signal_P = my_occultquad(z_B_P, u1, u2, ratio_P)
    transit_signal_M = my_occultquad(z_B_M, u1, u2, ratio_M)

    transit_signal_P=np.array(transit_signal_P)
    transit_signal_M=np.array(transit_signal_M)

    total_transit=1.0-(1.0-transit_signal_P)-(1.0-transit_signal_M)
    if fix_blocking:

        overlap=(z_B_P<(1.0+ratio_P))*(z_B_M<(1.0+ratio_M))
        total_overlap=np.copy(overlap)
        total_overlap[total_overlap]=(z_B_P_x[overlap]-z_B_M_x[overlap])**2.0+(z_B_P_y[overlap]-z_B_M_y[overlap])**2.0<(ratio_P-ratio_M)**2.0
        overlap[overlap]=(z_B_P_x[overlap]-z_B_M_x[overlap])**2.0+(z_B_P_y[overlap]-z_B_M_y[overlap])**2.0<(ratio_P+ratio_M)**2.0
        total_transit[total_overlap]=transit_signal_P[total_overlap]
        overlap[total_overlap]=False


        if np.any(overlap):
            if verbosity>1:
                print("Fixing", np.sum(overlap), "overlaps.")
            fl_cor=[]
            n_o=1
            for zPx,zPy,zMx,zMy in zip(z_B_P_x[overlap],z_B_P_y[overlap],z_B_M_x[overlap],z_B_M_y[overlap]):
                n_o+=1
                bound_low_x=min(zPx-ratio_P,zMx-ratio_M)
                bound_high_x=max(zPx+ratio_P,zMx+ratio_M)
                bound_low_y=min(zPy-ratio_P,zMy-ratio_M)
                bound_high_y=max(zPy+ratio_P,zMy+ratio_M)
                N_pix_x=int(blocking_res*(bound_high_x-bound_low_x))+10#1225.
                N_pix_y=int(blocking_res*(bound_high_y-bound_low_y))+10#1225.

                if verbosity>2:
                    print("Fixing overlap", n_o)
                    print("Size of patch:", bound_high_x-bound_low_x, "x", bound_high_y-bound_low_y)
                    print("Pixels:", N_pix_x, "x", N_pix_y)

                x_pos=np.linspace(bound_low_x,bound_high_x,N_pix_x)
                y_pos=np.linspace(bound_low_y,bound_high_y,N_pix_y)
                dist=np.sqrt(x_pos[None,:]**2.0+y_pos[:,None]**2.0)
                
                mu=np.sqrt(1.0-np.minimum(dist,1.0)**2.0)
                brightness=(1.0-u1*(1.0-mu)-u2*(1.0-mu)**2.0)/(1.0-u1/3.0-u2/6.0)
                brightness[dist>1.0]=0.0
                flux_no_block=np.sum(brightness)*(x_pos[1]-x_pos[0])*(y_pos[1]-y_pos[0])/np.pi
                
                mask_P=(x_pos[None,:]-zPx)**2.0+(y_pos[:,None]-zPy)**2.0<ratio_P**2.0
                mask_S=(x_pos[None,:]-zMx)**2.0+(y_pos[:,None]-zMy)**2.0<ratio_M**2.0
                brightness[mask_P]=0.0
                brightness[mask_S]=0.0
                flux_with_tr=np.sum(brightness)*(x_pos[1]-x_pos[0])*(y_pos[1]-y_pos[0])/np.pi
                fl_cor.append(1.0-(flux_no_block-flux_with_tr))
            total_transit[overlap]=fl_cor


    if verbosity>1:
        print(transit_signal_P.shape)

    total_transit_all[pot_tr_mask]=total_transit

    if full_out:
        return total_transit_all,transit_signal_P, transit_signal_M
    return total_transit_all


def get_coords(time, ratio_P, a_o_R, impact_B, phase_B, period_B, c1, c2, ratio_M, semi_major_axis_PM_o_R, phase_M, period_PM, mass_ratio_MP, i_s=0, Omega_s=0, verbosity=0):
    const=a_o_R
    if verbosity>1:
        print(const)
    z_B_x=const*np.sin(2.0*np.pi*(time/period_B-phase_B))
    z_B_y=impact_B*np.cos(2.0*np.pi*(time/period_B-phase_B))
    z_B_z=-const*np.cos(2.0*np.pi*(time/period_B-phase_B))

    pot_tr_mask = z_B_x**2.+z_B_y**2. < (1.0+semi_major_axis_PM_o_R+ratio_P)**2.

    E=2.0*np.pi*(time_p/period_PM+phase_M) #need to solve for E from M iff ecc!=0

    ecc=0.0

    p=np.cos(E) - ecc
    q=np.sin(E) * np.sqrt(1.0-ecc**2.0)

    omega_s=0.0#argument of periapsis

    z=np.cos(omega_s)*p-np.sin(omega_s)*q
    x=np.sin(omega_s)*p+np.cos(omega_s)*q

    #i_s: inclination

    y=np.sin(i_s)*z
    z=np.cos(i_s)*z

    #Omega_s: longitude of ascending node

    z,x=np.cos(Omega_s)*z-np.sin(Omega_s)*x,np.sin(Omega_s)*z+np.cos(Omega_s)*x
    
    const_P=mass_ratio_MP/(1.0+mass_ratio_MP)*semi_major_axis_PM_o_R

    z_P_x= const_P*x
    z_P_y= const_P*y
    z_P_z=-const_P*z


    const_M=-1.0/(1.0+mass_ratio_MP)*semi_major_axis_PM_o_R

    if verbosity>1:
        print(const_M)
    
    z_M_x= const_M*x
    z_M_y= const_M*y
    z_M_z=-const_M*z

    return (z_B_x+z_P_x,z_B_y+z_P_y,z_B_z+z_P_z),(z_B_x+z_M_x,z_B_y+z_M_y,z_B_z+z_M_z)

