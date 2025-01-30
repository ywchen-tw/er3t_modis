import os
import sys
import h5py
import numpy as np
import datetime
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import rcParams
from er3t.util.modis import modis_l1b, modis_l2, modis_03, modis_04, modis_09a1, modis_43a3
from er3t.util import grid_by_dxdy
import matplotlib.image as mpl_img
from mpl_toolkits.axes_grid1 import make_axes_locatable


def sfc_alb_mask_inter(lon_alb, lat_alb, sfc_alb, lon_2d, lat_2d):
    """
    Interpolate surface albedo values at (lon_2d, lat_2d) using nearest neighbor method
    """
    # Ensure sfc_alb values are within [0, 1] range
    sfc_alb = np.clip(sfc_alb, 0.0, 1.0)

    # Create a mask for valid sfc_alb values
    mask = sfc_alb >= 0

    # Create an array of valid (lon, lat) points
    points = np.column_stack((lon_alb[mask].flatten(), lat_alb[mask].flatten()))
    #points = np.transpose(np.vstack((lon_alb[mask].flatten(), lat_alb[mask].flatten())))

    # Interpolate sfc_alb values at lon_2d and lat_2d using nearest neighbor method
    sfc_alb_inter = interpolate.griddata(points, sfc_alb[mask].flatten(), 
                                        (lon_2d, lat_2d), method='nearest')
    
    return sfc_alb_inter
    


def cdata_sat_raw(sat0, dx, dy, overwrite=False, plot=True):
    """
    Purpose: Collect satellite data for OCO-2 retrieval
    oco_band: 'o2a', 'wco2', 'sco2'
    """

    # Check if preprocessed data exists and return if overwrite is False
    if os.path.isfile(f'{sat0.fdir_pre_data}/pre-data.h5') and not overwrite:
        print(f'Message [pre_data]: {sat0.fdir_pre_data}/pre-data.h5 exsit.')
        return None
    else:
        # Open the HDF file and create MODIS data groups
        f0 = h5py.File(f'{sat0.fdir_pre_data}/pre-data.h5', 'w')
        f0['extent'] = sat0.extent

        # MODIS data groups in the HDF file
        #/--------------------------------------------------------------\#
        g  = f0.create_group('mod')
        g0 = g.create_group('geo')
        g1 = g.create_group('rad')
        g2 = g.create_group('cld')
        g3 = g.create_group('sfc')
        g4 = g.create_group('aod')

        #\--------------------------------------------------------------/#

        # Process MODIS RGB imagery
        mod_rgb = mpl_img.imread(sat0.fnames['mod_rgb'][0])
        g['rgb'] = mod_rgb
        print('Message [cdata_sat_raw]: the processing of MODIS RGB imagery is complete.')


        # Process MODIS L1B radiance/reflectance data
        modl1b = modis_l1b(fnames=sat0.fnames['mod_02'], extent=sat0.extent)
        lon0, lat0 = modl1b.data['lon']['data'], modl1b.data['lat']['data']
        ref_650_raw, rad_650_raw = modl1b.data['ref']['data'][0, ...], modl1b.data['rad']['data'][0, ...]
        
        grid_by_dxdy_nearest_args = {'extent': sat0.extent, 'dx': dx, 'dy': dy, 'method': 'nearest'}
        grid_by_dxdy_linear_args = {'extent': sat0.extent, 'dx': dx, 'dy': dy, 'method': 'linear'}
        lon_2d, lat_2d, ref_650_2d = grid_by_dxdy(lon0, lat0, ref_650_raw, **grid_by_dxdy_nearest_args)
        _, _, rad_650_2d = grid_by_dxdy(lon0, lat0, rad_650_raw, **grid_by_dxdy_nearest_args)

        g1.update({'ref_650': ref_650_2d, 'rad_650': rad_650_2d})

        # Process MODIS L1B data at 500m resolution
        modl1b_500m = modis_l1b(fnames=sat0.fnames['mod_02_hkm'], extent=sat0.extent)
        lon0_500m, lat0_500m = modl1b_500m.data['lon']['data'], modl1b_500m.data['lat']['data']

        ref_2d_470_raw, ref_2d_555_raw = modl1b_500m.data['ref']['data'][2, ...], modl1b_500m.data['ref']['data'][3, ...]
        ref_2d_1640_raw, rad_2d_1640_raw = modl1b_500m.data['ref']['data'][5, ...], modl1b_500m.data['rad']['data'][5, ...]
        ref_2d_2130_raw, rad_2d_2130_raw = modl1b_500m.data['ref']['data'][6, ...], modl1b_500m.data['rad']['data'][6, ...]

        # Create 2D grids of MODIS L1B data at 500m resolution
        lon_2d_500m, lat_2d_500m, ref_2d_470 = grid_by_dxdy(lon0_500m, lat0_500m, ref_2d_470_raw, **grid_by_dxdy_nearest_args)
        _, _, ref_2d_555  = grid_by_dxdy(lon0_500m, lat0_500m, ref_2d_555_raw, **grid_by_dxdy_nearest_args)
        _, _, ref_2d_1640 = grid_by_dxdy(lon0_500m, lat0_500m, ref_2d_1640_raw, **grid_by_dxdy_nearest_args)
        _, _, rad_2d_1640 = grid_by_dxdy(lon0_500m, lat0_500m, rad_2d_1640_raw, **grid_by_dxdy_nearest_args)
        _, _, ref_2d_2130 = grid_by_dxdy(lon0_500m, lat0_500m, ref_2d_2130_raw, **grid_by_dxdy_nearest_args)
        _, _, rad_2d_2130 = grid_by_dxdy(lon0_500m, lat0_500m, rad_2d_2130_raw, **grid_by_dxdy_nearest_args)

        # Interpolate MODIS L1B data to lon_2d and lat_2d using linear method
        for var_name in ['ref_2d_470', 'ref_2d_555', 'ref_2d_1640', 'rad_2d_1640', 'ref_2d_2130', 'rad_2d_2130']:
            var = vars()[var_name]
            mask = var>=0
            points_mask = np.column_stack((lon_2d_500m[mask].flatten(), lat_2d_500m[mask].flatten()))
            #points_mask = np.transpose(np.vstack((lon_2d_500m[mask].flatten(), lat_2d_500m[mask].flatten())))
            #vars()[f'{var_name}_inter'] = interpolate.griddata(points_mask, var[mask].flatten(), (lon_2d, lat_2d), method='linear')

            vars()[f'{var_name}_inter_linear'] = interpolate.griddata(points_mask, var[mask].flatten(), (lon_2d, lat_2d), method='linear')
            mask = vars()[f'{var_name}_inter_linear']>=0
            points_mask = np.column_stack((lon_2d[mask].flatten(), lat_2d[mask].flatten()))

            vars()[f'{var_name}_inter'] = interpolate.griddata(points_mask, vars()[f'{var_name}_inter_linear'][mask].flatten(), (lon_2d, lat_2d), method='linear')

        # Add MODIS L1B data to HDF groups
        g1.update({'ref_470': vars()[f'ref_2d_470_inter'], 'ref_555': vars()[f'ref_2d_555_inter'], 
                   'ref_1640': vars()[f'ref_2d_1640_inter'], 'rad_1640': vars()[f'rad_2d_1640_inter'], 
                   'ref_2130': vars()[f'ref_2d_2130_inter'], 'rad_2130': vars()[f'rad_2d_2130_inter']})

        print('Message [cdata_sat_raw]: the processing of MODIS L1B radiance/reflectance at 470, 555, 1640, 2130 nm is complete.')

        plt.clf()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        c1 = ax1.imshow(vars()[f'ref_2d_1640_inter'], origin='lower')
        ax1.set_title('MODIS L1B Reflectance at 1640 nm')
        fig.colorbar(c1, ax=ax1)
        c2 = ax2.imshow(vars()[f'rad_2d_1640_inter'], origin='lower')
        ax2.set_title('MODIS L1B Radiance at 1640 nm')
        fig.colorbar(c2, ax=ax2)
        plt.show()
        
        # Save longitude and latitude to HDF group
        f0.update({'lon': lon_2d, 'lat': lat_2d})

        # Create 1D grids of longitude and latitude
        lon_1d = lon_2d[:, 0]
        lat_1d = lat_2d[0, :]


        # MODIS geo information - sza, saa, vza, vaa
        #/--------------------------------------------------------------\#
        mod03 = modis_03(fnames=sat0.fnames['mod_03'], extent=sat0.extent, vnames=['Height'])
        lon0, lat0, sza0, saa0, vza0, vaa0, sfh0 = [mod03.data[var]['data'] for var in ['lon', 'lat', 'sza', 'saa', 'vza', 'vaa', 'height']]
        # Convert height from m to km
        sfh0 = sfh0/1000.0 # units: km
        sfh0[sfh0<0.0] = np.nan

        _, _, sza_2d = grid_by_dxdy(lon0, lat0, sza0, **grid_by_dxdy_nearest_args)
        _, _, saa_2d = grid_by_dxdy(lon0, lat0, saa0, **grid_by_dxdy_nearest_args)
        _, _, vza_2d = grid_by_dxdy(lon0, lat0, vza0, **grid_by_dxdy_nearest_args)
        _, _, vaa_2d = grid_by_dxdy(lon0, lat0, vaa0, **grid_by_dxdy_nearest_args)
        _, _, sfh_2d = grid_by_dxdy(lon0, lat0, sfh0, **grid_by_dxdy_nearest_args)

        g0.update({'sza': sza_2d, 'saa': saa_2d, 'vza': vza_2d, 'vaa': vaa_2d, 'sfh': sfh_2d})

        print('Message [cdata_sat_raw]: the processing of MODIS geo-info is complete.')
        #\--------------------------------------------------------------/#


        # cloud properties
        #/--------------------------------------------------------------\#
        modl2 = modis_l2(fnames=sat0.fnames['mod_l2'], extent=sat0.extent, vnames=['cloud_top_height_1km'])
        lon0, lat0, cer0, cot0 = [modl2.data[var]['data'] for var in ['lon', 'lat', 'cer', 'cot']]

        cth0  = modl2.data['cloud_top_height_1km']['data']/1000.0 # units: km
        cth0[cth0<=0.0] = np.nan

        _, _, cer_2d_l2 = grid_by_dxdy(lon0, lat0, cer0, **grid_by_dxdy_nearest_args)
        cer_2d_l2[cer_2d_l2<=1.0] = np.nan

        _, _, cot_2d_l2 = grid_by_dxdy(lon0, lat0, cot0, **grid_by_dxdy_nearest_args)
        cot_2d_l2[cot_2d_l2<=0.0] = np.nan

        _, _, cth_2d_l2 = grid_by_dxdy(lon0, lat0, cth0, **grid_by_dxdy_linear_args)
        cth_2d_l2[cth_2d_l2<=0.0] = np.nan

        g2.update({'cot_l2': cot_2d_l2, 'cer_l2': cer_2d_l2, 'cth_l2': cth_2d_l2})

        print('Message [cdata_sat_raw]: the processing of MODIS cloud properties is complete.')
        #\--------------------------------------------------------------/#


        # surface
        #/--------------------------------------------------------------\#
        # Extract and grid MODIS surface reflectance
        #   band 1: 620  - 670  nm, index 0
        #   band 2: 841  - 876  nm, index 1
        #   band 3: 459  - 479  nm, index 2
        #   band 4: 545  - 565  nm, index 3
        #   band 5: 1230 - 1250 nm, index 4
        #   band 6: 1628 - 1652 nm, index 5
        #   band 7: 2105 - 2155 nm, index 6

        wavelength_list = [650, 860, 470, 555, 1240, 1640, 2130]
        mod09 = modis_09a1(fnames=sat0.fnames['mod_09'], extent=sat0.extent)
        mod43 = modis_43a3(fnames=sat0.fnames['mcd_43'], extent=sat0.extent)
        for wv_index in range(7):
            
            lon_2d_sfc, lat_2d_sfc, vars()[f'sfc_09_{wavelength_list[wv_index]:d}'] = grid_by_dxdy(mod09.data['lon']['data'], mod09.data['lat']['data'], mod09.data['ref']['data'][wv_index, :], **grid_by_dxdy_nearest_args)
            vars()[f'sfc_09_{wavelength_list[wv_index]:d}'] = sfc_alb_mask_inter(lon_2d_sfc, lat_2d_sfc, vars()[f'sfc_09_{wavelength_list[wv_index]:d}'], lon_2d, lat_2d)
            lon_2d_sfc, lat_2d_sfc, vars()[f'sfc_43_{wavelength_list[wv_index]:d}'] = grid_by_dxdy(mod43.data['lon']['data'], mod43.data['lat']['data'], mod43.data['wsa']['data'][wv_index, :], **grid_by_dxdy_nearest_args)
            vars()[f'sfc_43_{wavelength_list[wv_index]:d}'] = sfc_alb_mask_inter(lon_2d_sfc, lat_2d_sfc, vars()[f'sfc_43_{wavelength_list[wv_index]:d}'], lon_2d, lat_2d)

        sfc_43_o2a = vars()[f'sfc_43_860']
        sfc_43_wco2 = vars()[f'sfc_43_1640']
        sfc_43_sco2 = vars()[f'sfc_43_2130']


        g3.update({'lon': lon_2d_sfc, 'lat': lat_2d_sfc})
        for wavelength in wavelength_list:
            g3['alb_09_%d' % wavelength] = vars()[f'sfc_09_{wavelength:d}']
            g3['alb_43_%d' % wavelength] = vars()[f'sfc_43_{wavelength:d}']

        g3.update({'alb_43_o2a': sfc_43_o2a, 'alb_43_wco2': sfc_43_wco2, 'alb_43_sco2': sfc_43_sco2})

        print('Message [cdata_sat_raw]: the processing of MODIS surface properties is complete.')
        #\--------------------------------------------------------------/#

        # aerosol
        #/--------------------------------------------------------------\#
        mcd04 = modis_04(fnames=sat0.fnames['mod_04'], extent=sat0.extent, 
                        vnames=['Deep_Blue_Spectral_Single_Scattering_Albedo_Land', ])
        AOD_lon, AOD_lat, AOD_550_land = grid_by_dxdy(mcd04.data['lon']['data'], mcd04.data['lat']['data'], mcd04.data['AOD_550_land']['data'], **grid_by_dxdy_nearest_args)
        _, _, Angstrom_Exponent_land = grid_by_dxdy(mcd04.data['lon']['data'], mcd04.data['lat']['data'], mcd04.data['Angstrom_Exponent_land']['data'], **grid_by_dxdy_nearest_args)
        _, _, SSA_land_660 = grid_by_dxdy(mcd04.data['lon']['data'], mcd04.data['lat']['data'], mcd04.data['SSA_land']['data'][2, :], **grid_by_dxdy_nearest_args)

        _, _, AOD_550_land_grid = grid_by_dxdy(mcd04.data['lon']['data'], mcd04.data['lat']['data'], mcd04.data['AOD_550_land']['data'], **grid_by_dxdy_linear_args)
        #_, _, aerosol_type_land = grid_by_dxdy(mcd04.data['lon']['data'], mcd04.data['lat']['data'], mcd04.data['aerosol_type_land']['data'], **grid_by_dxdy_nearest_args)
        #_, _, aerosol_cloud_frac_land = grid_by_dxdy(mcd04.data['lon']['data'], mcd04.data['lat']['data'], mcd04.data['aerosol_cloud_frac_land']['data'], **grid_by_dxdy_nearest_args)

        AOD_550_land_nan = AOD_550_land.copy()
        AOD_550_land_nan[np.isnan(AOD_550_land_nan)] = np.nan
        AOD_550_land_nan[AOD_550_land_nan<0] = np.nan
        SSA_land_660_nan = SSA_land_660.copy()
        SSA_land_660_nan[np.isnan(SSA_land_660_nan)] = np.nan
        SSA_land_660_nan[SSA_land_660_nan<0] = np.nan

        AOD_550_land_mean = np.nanmean(AOD_550_land[(AOD_550_land>=0) & (~np.isnan(AOD_550_land))])
        Angstrom_Exponent_land_mean = np.nanmean(Angstrom_Exponent_land[AOD_550_land>=0])
        SSA_land_mean = np.nanmean(SSA_land_660[(SSA_land_660>=0) & (~np.isnan(SSA_land_660))])

        g4['AOD_550_land'] = AOD_550_land_grid
        g4['AOD_550_land_mean'] = AOD_550_land_mean
        g4['Angstrom_Exponent_land_mean'] = Angstrom_Exponent_land_mean
        g4['SSA_660_land_mean'] = SSA_land_mean

        #/--------------------------------------------------------------\#


    #/----------------------------------------------------------------------------\#

    if plot:
        with h5py.File(f'{sat0.fdir_pre_data}/pre-data.h5', 'r') as f0:
            extent = f0['extent'][...]
            rgb = f0['mod/rgb'][...]
            rad = f0['mod/rad/rad_650'][...]
            ref = f0['mod/rad/ref_650'][...]
            sza = f0['mod/geo/sza'][...]
            saa = f0['mod/geo/saa'][...]
            vza = f0['mod/geo/vza'][...]
            vaa = f0['mod/geo/vaa'][...]
            cot = f0['mod/cld/cot_l2'][...]
            cer = f0['mod/cld/cer_l2'][...]
            cth = f0['mod/cld/cth_l2'][...]
            sfh = f0['mod/geo/sfh'][...]
            alb09 = f0['mod/sfc/alb_09_860'][...]
            alb43 = f0['mod/sfc/alb_43_860'][...]
        # figure
        #/----------------------------------------------------------------------------\#
        plt.close('all')
        rcParams['font.size'] = 12
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        fig.suptitle('MODIS Products Preview')

        titles = ['RGB Imagery', f'L1B Radiance 650 nm)', f'L1B Reflectance 650 nm)', None, 
                  'Solar Zenith [°]', 'Solar Azimuth [°]', 'Viewing Zenith [°]', 'Viewing Azimuth [°]',
                  'L2 COT', 'L2 CER [µm]', 'L2 CTH [km]', 'Surface Height [km]', f'09A1 Reflectance at 860 nm',
                  f'43A3 WSA at 860 nm']

        data = [rgb, rad.T, ref.T, None, sza.T, saa.T, vza.T, vaa.T, cot.T, cer.T, cth.T, sfh.T, alb09.T, alb43.T]
        vmins = [None, 0.0, 0.0, None, None, None, None, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        vmaxs = [None, 0.5, 1.0, None, None, None, None, None, 50.0, 30.0, 15.0, 5.0, 0.4, 0.4]

        for idx, (ax, title, img, vmin, vmax) in enumerate(zip(np.ravel(axes), titles, data, vmins, vmaxs)):
            if idx == 3:
                ax.axis('off')
                continue
            if idx == 0:
                cs = ax.imshow(img, zorder=0, extent=extent)
            else:
                cs = ax.imshow(img, origin='lower', cmap='jet', zorder=0, extent=extent, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.set_xlim(extent[:2])
            ax.set_ylim(extent[2:])
            ax.set_xlabel('Longitude [°]')
            ax.set_ylabel('Latitude [°]')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', '5%', pad='3%')
            cbar = fig.colorbar(cs, cax=cax) if idx != 0 else cax.axis('off')
        
        for ax in np.ravel(axes)[-2:]:
            ax.axis('off')

        # save figure
        #/--------------------------------------------------------------\#
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        plt.savefig('%s/<%s>.png' % (sat0.fdir_pre_data, _metadata['Function']), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        #\----------------------------------------------------------------------------/#
