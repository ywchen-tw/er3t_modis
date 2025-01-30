import os, sys
import h5py
import numpy as np
from scipy import stats as st
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib import ticker
from matplotlib.colors import LogNorm
import matplotlib.image as mpl_img
import platform
from er3t.pre.abs import abs_16g
from er3t.pre.sfc import sfc_sat
from er3t.pre.pha import pha_mie_wc 
from er3t.rtm.mca import mca_atm_1d, mca_atm_3d, mca_sfc_2d
from er3t.rtm.mca import mcarats_ng
from er3t.rtm.mca import mca_out_ng
from er3t.rtm.mca import mca_sca

from util.modis_atmmod import atm_atmmod
# from er3t.pre.atm import atm_atmmod
from simulation.util.util import sat_tmp
from simulation.util.modis_cld_sat import cld_sat


def ax_lonlat_setting(ax, label_size=16):
    ax.set_xlabel('Longititude ($^\circ$)', fontsize=label_size)
    ax.set_ylabel('Latitude ($^\circ$)', fontsize=label_size)
    ax.xaxis.set_major_locator(FixedLocator(np.arange(-180.0, 181.0, 0.1)))
    ax.yaxis.set_major_locator(FixedLocator(np.arange(-90.0, 91.0, 0.1)))

def cal_modis_rad(sat, zpt_file, wavelength, cfg_info, fdir='tmp-data', solver='3D', case_name_tag='default', overwrite=False):
    """
    Simulate MODIS radiance
    """
    if wavelength not in [650, 1640, 2130]:
        raise ValueError('Wavelength must be either 650, 1640, or 2130 nm for MODIS radiance simulation!')

    # atm object
    # =================================================================================
    fname_atm = '%s/atm.pk' % fdir
    atm0      = atm_atmmod(zpt_file=zpt_file, fname=fname_atm, overwrite=True)
    # =================================================================================

    # abs object
    # =================================================================================
    fname_abs = '%s/abs_%d.pk' % (fdir, wavelength)
    abs0      = abs_16g(wavelength=wavelength, fname=fname_abs, atm_obj=atm0, overwrite=overwrite)
    # =================================================================================

    # solar zenith/azimuth angles and sensor zenith/azimuth angles
    # =================================================================================
    with h5py.File(f'{sat.fdir_pre_data}/pre-data.h5', 'r') as f:
        sza = f['mod/geo/sza'][...].mean()
        saa = f['mod/geo/saa'][...].mean()
        vza = f['mod/geo/vza'][...].mean()
        vaa = f['mod/geo/vaa'][...].mean()
    # =================================================================================
    print('sza: %.2f, saa: %.2f, vza: %.2f, vaa: %.2f' % (sza, saa, vza, vaa))
    print(np.cos(sza/180*np.pi))
    # sys.exit()

    # sfc object
    # =================================================================================
    data = {}
    with h5py.File(f'{sat.fdir_pre_data}/pre-data.h5', 'r') as f:
        data['alb_2d'] = dict(data=f[f'mod/sfc/alb_43_{wavelength:d}'][...], name='Surface albedo (lambertian)', units='N/A')
        data['lon_2d'] = dict(data=f['mod/sfc/lon'][...], name='Longitude', units='degrees')
        data['lat_2d'] = dict(data=f['mod/sfc/lat'][...], name='Latitude' , units='degrees')

    fname_sfc = '%s/sfc_%d.pk' % (fdir, wavelength)
    mod09 = sat_tmp(data)
    sfc0      = sfc_sat(sat_obj=mod09, fname=fname_sfc, extent=sat.extent, verbose=True, overwrite=overwrite)
    sfc_2d    = mca_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/mca_sfc_2d_%d.bin' % (fdir, wavelength), overwrite=overwrite)
    # =================================================================================

    # cld object
    # =================================================================================
    data = {}
    with h5py.File(f'{sat.fdir_pre_data}/pre-data.h5', 'r') as f_pre_data:
        data['lon_2d'] = dict(name='Gridded longitude'               , units='degrees'    , data=f_pre_data['lon'][...])
        data['lat_2d'] = dict(name='Gridded latitude'                , units='degrees'    , data=f_pre_data['lat'][...])
        data['rad_2d'] = dict(name='Gridded radiance'                , units='km'         , data=f_pre_data[f'mod/rad/rad_650'][...])
        if solver.lower() == 'ipa':
            suffix = 'ipa'     # with wind correction only
        elif solver.lower() == '3d':
            suffix = '3d'      # with parallex and wind correction
        data['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'        , data=f_pre_data[f'mod/cld/cot_{suffix}_650'][...])
        data['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micro'      , data=f_pre_data[f'mod/cld/cer_{suffix}_650'][...])
        data['cth_2d'] = dict(name='Gridded cloud top height'        , units='km'         , data=f_pre_data[f'mod/cld/cth_{suffix}_650'][...])

    modl1b    =  sat_tmp(data)
    fname_cld = '%s/cld.pk' % fdir

    cth0 = modl1b.data['cth_2d']['data']
    cth0[cth0>10.0] = 10.0
    cgt0 = np.zeros_like(cth0)
    cgt0[cth0>0.0] = 1.0                  # all clouds have geometrical thickness of 1 km
    cgt0[cth0>4.0] = cth0[cth0>4.0]-3.0   # high clouds (cth>4km) has cloud base at 3 km
    cld0      = cld_sat(zpt_file=zpt_file, fname_atm=fname_atm, sat_info=sat,
                        sat_obj=modl1b, fname=fname_cld, cth=cth0, cgt=cgt0, dz=0.5,#np.unique(atm0.lay['thickness']['data'])[0],
                        overwrite=overwrite)
    # =================================================================================


    # mca_sca object
    # =================================================================================
    pha0 = pha_mie_wc(wavelength=wavelength, overwrite=overwrite)
    sca  = mca_sca(pha_obj=pha0, fname='%s/mca_sca_%d.bin' % (fdir, wavelength), overwrite=overwrite)
    # =================================================================================

    # mca_cld object
    # =================================================================================
    atm3d0  = mca_atm_3d(cld_obj=cld0, atm_obj=atm0, pha_obj=pha0, fname='%s/mca_atm_3d.bin' % fdir)
   
    # homogeneous 1d mcarats "atmosphere"
    # =================================================================================
    atm1d0  = mca_atm_1d(atm_obj=atm0, abs_obj=abs0)

    if cfg_info['_aerosol'] == 'TRUE':
        # add homogeneous 1d mcarats "atmosphere", aerosol layer
        with h5py.File(f'{sat.fdir_pre_data}/pre-data.h5', 'r') as f:
            AOD_550_land_mean = f['mod/aod/AOD_550_land_mean'][...]
            Angstrom_Exponent_land_mean = f['mod/aod/Angstrom_Exponent_land_mean'][...]
            SSA_land_mean = f['mod/aod/SSA_660_land_mean'][...]

            aod = AOD_550_land_mean*((wavelength/550)**(Angstrom_Exponent_land_mean*-1))
            ssa = SSA_land_mean
            cth_mode = st.mode(cth0[np.logical_and(cth0>0, cth0<4)])
        
        asy    = float(cfg_info['asy']) # aerosol asymmetry parameter
        z_bot  = np.min(levels)         # altitude of layer bottom in km
        z_top  = cth_mode.mode[0]       # altitude of layer top in km
        aer_ext = aod / (z_top-z_bot) / 1000

        atm1d0.add_mca_1d_atm(ext1d=aer_ext, omg1d=ssa, apf1d=asy, z_bottom=z_bot, z_top=z_top)
    # =================================================================================

    atm_1ds = [atm1d0]
    atm_3ds = [atm3d0]
    # =================================================================================

    
    # solar zenith/azimuth angles and sensor zenith/azimuth angles
    # =================================================================================
    with h5py.File(f'{sat.fdir_pre_data}/pre-data.h5', 'r') as f:
        sza = f['mod/geo/sza'][...].mean()
        saa = f['mod/geo/saa'][...].mean()
        vza = f['mod/geo/vza'][...].mean()
        vaa = f['mod/geo/vaa'][...].mean()
    # =================================================================================

    # run mcarats
    # =================================================================================
    # cpu number used
    if platform.system() in ['Windows', 'Darwin']:
        Ncpu=os.cpu_count()-1
    else:
        Ncpu=int(cfg_info['Ncpu'])
    Nphotons = float(cfg_info['modis_650_N_photons'])
    temp_dir = '%s/%.4fnm/rad_%s' % (fdir, wavelength, solver.lower())
    run = False if os.path.isdir(temp_dir) and overwrite==False else True
    mca0 = mcarats_ng(
            date=sat.date,
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            surface_albedo=sfc_2d,
            sca=sca,
            Ng=abs0.Ng,
            target='radiance',
            solar_zenith_angle   = sza,
            solar_azimuth_angle  = saa,
            sensor_zenith_angle  = vza,
            sensor_azimuth_angle = vaa,
            fdir=temp_dir,
            Nrun=3,
            weights=abs0.coef['weight']['data'],
            photons=Nphotons,
            solver=solver,
            Ncpu=Ncpu,
            mp_mode='py',
            overwrite=run
            )

    # mcarats output
    out0 = mca_out_ng(fname='%s/mca-out-rad-modis-%s_%.4fnm.h5' % (fdir, solver.lower(), wavelength), mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)
    # =================================================================================

def modis_simulation_plot(sat, cfg_info, case_name_tag='default', fdir='tmp',
                               solver='3D', wvl=650, vmax=0.5, plot=False):

    # create data directory (for storing data) if the directory does not exist
    # ==================================================================================================
    extent_list = sat.extent
    extent_analysis = sat.extent_analysis
    mod_img = mpl_img.imread(sat.fnames['mod_rgb'][0])
    mod_img_wesn = sat.extent
    ref_threshold = float(cfg_info['ref_threshold'])
    # ==================================================================================================

    # read in MODIS measured radiance
    # ==================================================================================================
    with h5py.File('data/%s/pre-data.h5' % case_name_tag, 'r') as f:
        extent = f['extent'][...]
        lon_mod = f['lon'][...]
        lat_mod = f['lat'][...]
        rad_mod = f[f'mod/rad/rad_{wvl}'][...]
        ref_mod = f[f'mod/rad/ref_{wvl}'][...]
        cth_mod = f['mod/cld/cth_l2'][...]
        cot_3d_650 = f['mod/cld/cot_3d_650'][...]
        cer_3d_650 = f['mod/cld/cer_3d_650'][...]
        cth_3d_650 = f['mod/cld/cth_3d_650'][...]
        cot_3d_ipa = f['mod/cld/cot_ipa'][...]
        cer_3d_ipa = f['mod/cld/cer_ipa'][...]
        cth_3d_ipa = f['mod/cld/cth_ipa'][...]
        sza = f['mod/geo/sza'][...].mean()
        if solver.lower() == 'ipa':
            cth_mod = f['mod/cld/cth_ipa_650'][...]
        elif solver.lower() == '3d':
            cth_mod = f['mod/cld/cth_3d_650'][...]
    # ==================================================================================================

    # read in EaR3T simulations
    # ==================================================================================================
    fname = '%s/mca-out-rad-modis-%s_%.4fnm.h5' % (fdir, solver.lower(), wvl)
    with h5py.File(fname, 'r') as f:
        rad_rtm_sim     = f['mean/rad'][...]
        rad_rtm_sim_std = f['mean/rad_std'][...]
        toa = f['mean/toa'][...]
    # ==================================================================================================

    # save data
    # ==================================================================================================
    with h5py.File('data/%s/post-data_1640.h5' % case_name_tag, 'r+') as f:
        for key in ['wvl', 'lon', 'lat', 'extent', 'rad_obs', 'ref_obs', 'ref_threshold', 
                    f'rad_sim_{solver.lower()}', f'rad_sim_{solver.lower()}_std', f'ref_sim_{solver.lower()}']:
            if key in f.keys():
                del f[key]
        f['wvl'] = wvl
        f['lon'] = lon_mod
        f['lat'] = lat_mod
        f['extent']         = extent
        f['rad_obs']        = rad_mod
        f['ref_obs']        = ref_mod
        f['ref_threshold']  = ref_threshold
        f[f'rad_sim_{solver.lower()}']     = rad_rtm_sim
        f[f'rad_sim_{solver.lower()}_std'] = rad_rtm_sim_std
        f[f'ref_sim_{solver.lower()}']     = rad_rtm_sim*np.pi/(toa*np.cos(sza/180*np.pi))
    # ==================================================================================================
    
    if plot:
        label_size = 16
        tick_size = 12
        # ==================================================================================================
        
        fig = plt.figure(figsize=(15, 10))
        ax1 = fig.add_subplot(234)
        ax1.imshow(mod_img, extent=mod_img_wesn)
        ax1.pcolormesh(lon_mod, lat_mod, rad_mod, cmap='Greys_r', vmin=0.0, vmax=vmax)
        ax1.set_title('MODIS Measured Radiance')

        logic = (lon_mod>=extent_analysis[0]) & (lon_mod<=extent_analysis[1]) &\
                (lat_mod>=extent_analysis[2]) & (lat_mod<=extent_analysis[3])

        xedges = np.arange(-0.01, 0.61, 0.005)
        yedges = np.arange(-0.01, 0.61, 0.005)
        heatmap, xedges, yedges = np.histogram2d(rad_mod[logic], rad_rtm_sim[logic], bins=(xedges, yedges))
        YY, XX = np.meshgrid((yedges[:-1]+yedges[1:])/2.0, (xedges[:-1]+xedges[1:])/2.0)

        levels = np.concatenate((np.arange(1.0, 10.0, 1.0),
                                 np.arange(10.0, 200.0, 10.0),
                                 np.arange(200.0, 1000.0, 100.0),
                                 np.arange(1000.0, 10001.0, 5000.0)))
        
        ax3 = fig.add_subplot(232)
        ax3.imshow(mod_img, extent=mod_img_wesn)
        ax3.pcolormesh(lon_mod, lat_mod, rad_rtm_sim, cmap='Greys_r', vmin=0.0, vmax=vmax)
        ax3.set_title(f'EaR$^3$T Simulated {solver} Radiance')
        
        ax13 = fig.add_subplot(233)
        diff = ax13.imshow((rad_rtm_sim-rad_mod).T, cmap='bwr', extent=extent, origin='lower', vmin=-0.15, vmax=0.15)
        cbar_13 = fig.colorbar(diff, ax=ax13)
        cbar_13.set_label('Simulated - MODIS Radiance')
        ax13.set_title(f'{solver} Simulated - MODIS Radiance')
        
        ax2 = fig.add_subplot(231)
        cs = ax2.contourf(XX, YY, heatmap, levels, extend='both', locator=ticker.LogLocator(), cmap='jet')
        ax2.plot([0.0, 1.0], [0.0, 1.0], lw=1.0, ls='--', color='gray', zorder=3)
        ax2.set_xlim(0.0, vmax)
        ax2.set_ylim(0.0, vmax)
        ax2.set_xlabel('MODIS Measured Radiance')
        ax2.set_ylabel(f'Simulated {solver} Radiance')

        ax22 = fig.add_subplot(235)
        ax22.imshow(mod_img, extent=mod_img_wesn)
        cth_mask = ~np.isnan(cth_mod)
        ax22.scatter(lon_mod, lat_mod, cth_mod, c='r')
        ax22.set_title(f'EaR$^3$T Cloud mask\n(ref threshold: {ref_threshold})')

        ax4 = fig.add_subplot(236) 
        cth_img = ax4.imshow(cth_mod.T, cmap='jet', extent=extent, origin='lower', vmin=0.0, vmax=10)
        fig.colorbar(cth_img, ax=ax4)
        ax4.set_title('EaR$^3$T CTH')

        for ax in [ax1, ax3, ax13, ax22, ax4]:
            ax_lonlat_setting(ax, label_size=16)
            ax.set_xlim(extent_list[0]+0.15, extent_list[1]-0.15)
            ax.set_ylim(extent_list[2]+0.15, extent_list[3]-0.15)

        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.savefig(f'{sat.fdir_pre_data}/modis_{wvl}_{case_name_tag}_{solver}.png', bbox_inches='tight')
        
        # ==================================================================================================
        
        plt.clf()
        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_axes([0.05, 0.05, 0.21, 0.9])
        ax1.imshow(mod_img, extent=mod_img_wesn)
        c1 = ax1.pcolormesh(lon_mod, lat_mod, rad_mod, cmap='Greys_r', vmin=0.0, vmax=vmax)
        cbar = fig.colorbar(c1, ax=ax1, orientation='vertical', pad=0.05, fraction=0.0665)
        cbar.set_label('MODIS Measured Radiance ($\mathrm{W\;m^{-2}\;\mu\;m^{-1}\;sr^{-1}}$)', fontsize=label_size-2  )

        ax2 = fig.add_axes([0.37, 0.05, 0.21, 0.9])
        ax2.imshow(mod_img, extent=mod_img_wesn)
        c2 = ax2.pcolormesh(lon_mod, lat_mod, rad_rtm_sim, cmap='Greys_r', vmin=0.0, vmax=vmax)
        cbar2 = fig.colorbar(c2, ax=ax2, orientation='vertical', pad=0.05,fraction=0.0665)
        cbar2.set_label('Simulated %s Radiance ($\mathrm{W\;m^{-2}\;\mu\;m^{-1}\;sr^{-1}}$)' %solver, fontsize=label_size-2)
                       
        ax3 = fig.add_axes([0.70, 0.05, 0.275, 0.9])
        ax3.set_aspect('equal', 'box')
        cs = ax3.contourf(XX, YY, heatmap, levels, extend='both', locator=ticker.LogLocator(), cmap='jet')
        ax3.plot([0.0, 1.0], [0.0, 1.0], lw=1.0, ls='--', color='gray', zorder=3)
        ax3.set_xlim(0.0, vmax)
        ax3.set_ylim(0.0, vmax)
        ax3.set_xlabel(r'MODIS Measured Radiance ($\mathrm{W\;m^{-2}\;nm^{-1}\;sr^{-1}}$)', fontsize=label_size)
        ax3.set_ylabel(r'Simulated %s Radiance ($\mathrm{W\;m^{-2}\;nm^{-1}\;sr^{-1}}$)' %solver, fontsize=label_size)

        for ax in [ax1, ax2]:
            ax.set_xlim(extent_analysis[0], extent_analysis[1])
            ax.set_ylim(extent_analysis[2], extent_analysis[3])
            ax.xaxis.set_major_locator(FixedLocator(np.arange(-180.0, 181.0, 0.1)))
            ax.yaxis.set_major_locator(FixedLocator(np.arange(-90.0, 91.0, 0.1)))
            ax.set_xlabel('Longititude ($^\circ$E)', fontsize=label_size)
            ax.set_ylabel('Latitude ($^\circ$N)', fontsize=label_size)

        for ax, label_ord in zip([ax1, ax2, ax3], ['(a)', '(b)', '(c)']):
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            ax.tick_params(axis='both', labelsize=tick_size)
            ax.text(xmin+0.0*(xmax-xmin), ymin+1.025*(ymax-ymin), 
                    label_ord, fontsize=label_size, color='k')

        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.savefig(f'{sat.fdir_pre_data}/modis_{wvl}_{case_name_tag}_{solver}_comparison.png', bbox_inches='tight', dpi=300)
        
        plt.clf()
        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_axes([0.05, 0.05, 0.21, 0.9])
        ax1.imshow(mod_img, extent=mod_img_wesn)
        c1 = ax1.pcolormesh(lon_mod, lat_mod, rad_mod, cmap='Greys_r', vmin=0.0, vmax=vmax)
        cbar = fig.colorbar(c1, ax=ax1, orientation='vertical', pad=0.05, fraction=0.0665)
        cbar.set_label('MODIS Measured Radiance ($\mathrm{W\;m^{-2}\;nm^{-1}\;sr^{-1}}$)', fontsize=label_size-2 )

        ax2 = fig.add_axes([0.37, 0.05, 0.21, 0.9])
        ax2.imshow(mod_img, extent=mod_img_wesn)
        c2 = ax2.pcolormesh(lon_mod, lat_mod, rad_rtm_sim, cmap='Greys_r', vmin=0.0, vmax=vmax)
        cbar2 = fig.colorbar(c2, ax=ax2, orientation='vertical', pad=0.05,fraction=0.0665)
        cbar2.set_label('Simulated %s Radiance ($\mathrm{W\;m^{-2}\;nm^{-1}\;sr^{-1}}$)' %solver, fontsize=label_size-2)
                       
        ax3 = fig.add_axes([0.70, 0.05, 0.275, 0.9])
        ax3.set_aspect('equal', 'box')
        # Calculate the point density
        xy = np.vstack([rad_mod[logic], rad_rtm_sim[logic]])
        Z = gaussian_kde(xy)(xy)
        # Sort the points by density, so that the densest points are plotted last
        idx = Z.argsort()
        x_mod, y_rtm_3d, Z_den = rad_mod[logic][idx], rad_rtm_sim[logic][idx], Z[idx]
        cs = ax3.scatter(x_mod, y_rtm_3d, 
                         c=Z_den, s=5, 
                         norm=LogNorm(vmin=1, vmax=300), cmap='jet',
                         )
        ax3.plot([0.0, 1.0], [0.0, 1.0], lw=1.0, ls='--', color='gray', zorder=3)
        cbar3 = fig.colorbar(cs, ax=ax3, orientation='vertical', pad=0.05, fraction=0.0465, extend='both')  
        # set colobar scale to log
        cbar3.set_ticks([1, 10, 100, 300], [1, 10, 100, 300])
        cbar3.set_label('Density', fontsize=label_size)
        ax3.set_xlim(0.0, vmax)
        ax3.set_ylim(0.0, vmax)
        ax3.set_xlabel('MODIS Measured Radiance\n($\mathrm{W\;m^{-2}\;nm^{-1}\;sr^{-1}}$)', fontsize=label_size)
        ax3.set_ylabel('Simulated %s Radiance\n($\mathrm{W\;m^{-2}\;nm^{-1}\;sr^{-1}}$)' %solver, fontsize=label_size)

        # calculate the correlation coefficient and slope
        slope, intercept, r_value, p_value, std_err = st.linregress(rad_mod[logic], rad_rtm_sim[logic])
        print(f'{wvl} nm R2: {r_value**2:.2f}, Slope: {slope:.2f}')
        
        for ax in [ax1, ax2]:
            ax.set_xlim(extent_analysis[0], extent_analysis[1])
            ax.set_ylim(extent_analysis[2], extent_analysis[3])
            ax_lonlat_setting(ax, label_size)

        for ax, label_ord in zip([ax1, ax2, ax3], ['(a)', '(b)', '(c)']):
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            ax.tick_params(axis='both', labelsize=tick_size)
            ax.text(xmin+0.0*(xmax-xmin), ymin+1.025*(ymax-ymin), 
                    label_ord, fontsize=label_size, color='k')

        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.savefig(f'{sat.fdir_pre_data}/modis_{wvl}_{case_name_tag}_{solver}_comparison_scatter.png', bbox_inches='tight', dpi=300)
  
        plt.clf()
        fig = plt.figure(figsize=(8, 6))
        ax3 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
        ax3.set_aspect('equal', 'box')
        mask = cot_3d_650[logic] > 0
        # Calculate the point density
        xy = np.vstack([rad_mod[logic][mask], rad_rtm_sim[logic][mask]])
        Z = gaussian_kde(xy)(xy)
        # Sort the points by density, so that the densest points are plotted last
        idx = Z.argsort()
        x_mod, y_rtm_3d, Z_den = rad_mod[logic][mask][idx], rad_rtm_sim[logic][mask][idx], Z[idx]
        cs = ax3.scatter(x_mod, y_rtm_3d, c=Z_den, s=5, 
                         norm=LogNorm(vmin=1, vmax=300), cmap='jet',
                         )
        ax3.plot([0.0, 1.0], [0.0, 1.0], lw=1.0, ls='--', color='gray', zorder=3)
        cbar3 = fig.colorbar(cs, ax=ax3, orientation='vertical', pad=0.05, fraction=0.0465, extend='both')  
        # set colobar scale to log
        cbar3.set_ticks([1, 10, 100, 300], [1, 10, 100, 300])
        cbar3.set_label('Density', fontsize=label_size)
        ax3.set_xlim(0.0, vmax)
        ax3.set_ylim(0.0, vmax)
        ax3.set_xlabel('MODIS Measured Radiance\n($\mathrm{W\;m^{-2}\;nm^{-1}\;sr^{-1}}$)', fontsize=label_size)
        ax3.set_ylabel('Simulated %s Radiance\n($\mathrm{W\;m^{-2}\;nm^{-1}\;sr^{-1}}$)' %solver, fontsize=label_size)

        # calculate the correlation coefficient and slope
        slope, intercept, r_value, p_value, std_err = st.linregress(rad_mod[logic], rad_rtm_sim[logic])
        print(f'{wvl} nm R2: {r_value**2:.2f}, Slope: {slope:.2f}')
        

        for ax in [ax3]:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            ax.tick_params(axis='both', labelsize=tick_size)

        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.savefig(f'{sat.fdir_pre_data}/modis_{wvl}_{case_name_tag}_{solver}_cloud_comparison_scatter.png', bbox_inches='tight', dpi=300)

        # ==================================================================================================
        
        plt.clf()
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_axes([0.05, 0.05, 0.25, 0.9])
        ax2 = fig.add_axes([0.40, 0.05, 0.25, 0.9])
        ax3 = fig.add_axes([0.75, 0.05, 0.25, 0.9])
        
        cs1 = ax1.scatter(lon_mod, lat_mod, c=cot_3d_650, cmap='jet', vmin=0.0, vmax=50.0)
        cb1 = fig.colorbar(cs1, ax=ax1, orientation='vertical', pad=0.05)
        cb1.set_label('COT', fontsize=label_size)

        cs2 = ax2.scatter(lon_mod, lat_mod, c=cer_3d_650, cmap='jet', vmin=0.0, vmax=30.0)
        cb2 = fig.colorbar(cs2, ax=ax2, orientation='vertical', pad=0.05)
        cb2.set_label('CER ($\mathrm{\mu m}$)', fontsize=label_size)

        cs3 = ax3.scatter(lon_mod, lat_mod, c=cth_3d_650, cmap='jet', vmin=0.0, vmax=10.0)
        cb3 = fig.colorbar(cs3, ax=ax3, orientation='vertical', pad=0.05)
        cb3.set_label('CTH (km)', fontsize=label_size)

        for ax, label_ord in zip([ax1, ax2, ax3], ['(a)', '(b)', '(c)']):
            ax.imshow(mod_img, extent=mod_img_wesn)
            ax.set_xlim(extent_analysis[0], extent_analysis[1])
            ax.set_ylim(extent_analysis[2], extent_analysis[3])
            ax_lonlat_setting(ax, label_size)
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            ax.tick_params(axis='both', labelsize=tick_size)
            ax.text(xmin+0.0*(xmax-xmin), ymin+1.025*(ymax-ymin), 
                    label_ord, fontsize=label_size, color='k')

        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.savefig(f'{sat.fdir_pre_data}/modis_3d_cloud_information.png', bbox_inches='tight')
        
        # ==================================================================================================
        
        plt.clf()
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_axes([0.05, 0.05, 0.25, 0.9])
        ax2 = fig.add_axes([0.40, 0.05, 0.25, 0.9])
        ax3 = fig.add_axes([0.75, 0.05, 0.25, 0.9])
        
        cs1 = ax1.scatter(lon_mod, lat_mod, c=cot_3d_ipa, cmap='jet',
                       vmin=0.0, vmax=50.0)
        cb1 = fig.colorbar(cs1, ax=ax1, orientation='vertical', pad=0.05)
        cb1.set_label('COT', fontsize=label_size)

        cs2 = ax2.scatter(lon_mod, lat_mod, c=cer_3d_ipa, cmap='jet', vmin=0.0, vmax=30.0)
        cb2 = fig.colorbar(cs2, ax=ax2, orientation='vertical', pad=0.05)
        cb2.set_label('CER ($\mathrm{\mu m}$)', fontsize=label_size)

        cs3 = ax3.scatter(lon_mod, lat_mod, c=cth_3d_ipa, cmap='jet',  vmin=0.0, vmax=10.0)
        cb3 = fig.colorbar(cs3, ax=ax3, orientation='vertical', pad=0.05)
        cb3.set_label('CTH (km)', fontsize=label_size)

        for ax, label_ord in zip([ax1, ax2, ax3], ['(a)', '(b)', '(c)']):
            ax.imshow(mod_img, extent=mod_img_wesn)
            ax.set_xlim(extent_analysis[0], extent_analysis[1])
            ax.set_ylim(extent_analysis[2], extent_analysis[3])
            ax_lonlat_setting(ax, label_size)
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            ax.tick_params(axis='both', labelsize=tick_size)
            ax.text(xmin+0.0*(xmax-xmin), ymin+1.025*(ymax-ymin), 
                    label_ord, fontsize=label_size, color='k')

        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.savefig(f'{sat.fdir_pre_data}/modis_ipa_cloud_information.png', bbox_inches='tight')
        # ==================================================================================================
