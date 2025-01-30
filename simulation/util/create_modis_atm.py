import h5py
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from er3t.util.modis import modis_07

def create_modis_atm(sat=None, o2mix=0.20935, output='zpt.h5', new_h_edge=None):
    """
    Use MODIS 07 product to create a vertical profile of temperature, dew temperature, pressure, O2 and H2O number density, and H2O volume mixing ratio.
    """
    # --------- Constants ------------
    Rd = 287.052874
    EPSILON = 0.622
    kb = 1.380649e-23
    g = 9.81
    # ---------------------------------
    if sat == None:
        sys.exit("[Error] sat information must be provided!")
    elif sat != None:
        # Get reanalysis from met and CO2 prior sounding data
        mod07 = modis_07(fnames=sat.fnames['mod_07'], extent=sat.extent)
        cld_mask = mod07.data['cld_mask']['data']
        ### cloud mask: 0=cloudy, 1=uncetain, 2=probably clear, 3=confident clear
        pprf_l_single = mod07.data['p_level']['data']                          # pressure in hPa
        hprf_l = mod07.data['h_level_retrieved']['data']
        tprf_l = mod07.data['T_level_retrieved']['data']                # temperature in K
        dewTprf_l = mod07.data['dewT_level_retrieved']['data']
        mwvmxprf_l = mod07.data['wvmx_level_retrieved']['data']
        sfc_p = mod07.data['p_sfc']['data']                             # surface pressure in hPa
        sfc_h = mod07.data['h_sfc']['data']                             # surface height in m
        sfc_gph = sfc_h                                                 # assume surface geopotential height is the same as surface height (in m)
        sza = np.nanmean(mod07.data['sza']['data'])
        vza = np.nanmean(mod07.data['vza']['data'])
        
        pprf_l = np.repeat(pprf_l_single, mwvmxprf_l.shape[1]).reshape(mwvmxprf_l.shape)
        
   
        
        
        r = mwvmxprf_l# mass mixing ratio
        eprf_l = pprf_l*r/(EPSILON+r)
        #Tv = tprf_l/(1-eprf_l/pprf_l*(1-EPSILON))
        Tv = tprf_l/(1-(r/(r+EPSILON))*(1-EPSILON))

        
        air_layer = pprf_l/(kb*tprf_l)/1e6  # air number density in molec/cm3
        dry_air_layer = (pprf_l-eprf_l)/(kb*tprf_l)/1e6  # air number density in molec/cm3
        o2_layer = dry_air_layer*o2mix          # O2 number density in molec/cm3
        h2o_layer = eprf_l/(kb*tprf_l)/1e6  # H2O number density in molec/cm3
        air_ml = 28.0134*(1-o2mix) + 31.999*o2mix
        h2o_vmr = h2o_layer/dry_air_layer       # H2O volume mixing ratio
        

        sfc_h_mean = np.nanmean(sfc_h)
        pprf_lev_mean   = np.nanmean(pprf_l, axis=1)    # pressure mid grid in hPa
        tprf_lev_mean   = np.nanmean(tprf_l, axis=1)         # temperature mid grid in K
        dewTprf_lev_mean= np.nanmean(dewTprf_l, axis=1)      # dew temperature mid grid in K
        d_o2_lev_mean   = np.nanmean(o2_layer, axis=1)
        d_h2o_lev_mean  = np.nanmean(h2o_layer, axis=1)
        hprf_lev_mean   = np.nanmean(hprf_l, axis=1)/1000     # height mid grid in km
        h2o_vmr_mean    = np.nanmean(h2o_vmr, axis=1)

    if new_h_edge is not None:
        levels = new_h_edge
    else:
        levels = np.concatenate((np.linspace(sfc_h_mean, 5, 11), 
                                np.array([5.5,  6. ,  6.5,  7. ,  7.5,  8. ,  8.5,  9. ,  9.5, 10. , 11. , 12. , 13. , 14. , 20. , 25. , 30. , 35. , 40. ])))
    print(levels)
    if os.path.isfile(output): 
        print(f'[Warning] Output file {output} exists - overwriting!')
    print('Saving to file '+output)
    h5_output = h5py.File(output, 'w')
    h5_output.create_dataset('sfc_h',       data=sfc_h_mean)
    h5_output.create_dataset('level_sim',      data=levels)
    h5_output.create_dataset('h_lev',       data=hprf_lev_mean)
    h5_output.create_dataset('p_lev',       data=pprf_lev_mean)
    h5_output.create_dataset('t_lev',       data=tprf_lev_mean)
    h5_output.create_dataset('dewT_lev',    data=dewTprf_lev_mean)
    h5_output.create_dataset('d_o2_lev',    data=d_o2_lev_mean)
    h5_output.create_dataset('d_h2o_lev',   data=d_h2o_lev_mean)
    h5_output.create_dataset('h2o_vmr',     data=h2o_vmr_mean)
    h5_output.create_dataset('o2_mix',      data=o2mix)
    h5_output.create_dataset('sza',         data=sza)
    h5_output.create_dataset('vza',         data=vza)

    zpt_plot(pprf_lev_mean, tprf_lev_mean, dewTprf_lev_mean, h2o_vmr_mean, output=output.replace('.h5', '.png'))
    return None


def zpt_plot(p_lay, t_lay, dewT_lay, h2o_vmr, output):
    import metpy.calc as mpcalc
    from metpy.plots import SkewT
    from metpy.units import units

    p_prf = p_lay * units.hPa
    T_prf = (t_lay * units.kelvin).to(units.degC)
    Td_prf = (dewT_lay * units.kelvin).to(units.degC)

    fig, (ax1, ax2) =plt.subplots(1, 2, figsize=(12, 6.75))
    # fig = plt.figure(figsize=(4, 6.75))
    ax1.set_visible(False)
    # ax2.set_visible(False)
    skew = SkewT(fig=fig, subplot=(1, 2, 1), aspect=120.5)
    skew.plot(p_prf, T_prf, 'r', label='Temperature', linewidth=3)
    skew.plot(p_prf, Td_prf, 'g', label='Dew Point', linewidth=3)

    # Set some better labels than the default
    skew.ax.set_xlabel('Temperature ($\N{DEGREE CELSIUS}$)', fontsize=14)
    skew.ax.set_ylabel('Pressure (hPa)', fontsize=14)
    # Add the relevant special lines
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()
    skew.ax.set_ylim(1000, 100)
    skew.ax.legend()
    skew.ax.text(0.02, 1.07, '(a)', transform=skew.ax.transAxes, fontsize=16, fontweight='bold', va='center', ha='left')


    lns2 = ax2.plot(h2o_vmr, p_prf, 'b:', label='H$_2$O', linewidth=3)

    # Set some better labels than the default
    
    # Add the relevant special lines
    ax2.set_ylim(1000, 100)
    # ax2.set_xlim(406, 409)
    #// set yaxis to log scale
    ax2.set_yscale('log')
    #// set y ticks format in integer
    ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))
    ax2.yaxis.set_minor_formatter(ticker.StrMethodFormatter('{x:.0f}'))
    ax2.set_ylabel('Pressure (hPa)', fontsize=14)
    ax2.set_xlabel('H$_2$O mixing ratio', fontsize=14, color='b')
    # added these three lines
    lns = lns2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs)
    ax2.text(1.28, 1.07, '(b)', transform=skew.ax.transAxes, fontsize=16, fontweight='bold', va='center', ha='left')
    #// set y grid line
    ax2.grid(which='minor', axis='y', linestyle='-', linewidth=1, color='lightgrey')
    #// combi

    fig.tight_layout()
    fig.savefig(output, dpi=300)

