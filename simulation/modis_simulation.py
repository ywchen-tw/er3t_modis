#!/bin/env python
#SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=OCO2_test

import os
from pathlib import Path
import sys
import platform
import numpy as np
from datetime import datetime
from util.create_modis_atm import create_modis_atm
from util.modis_download import modis_download
from util.modis_raw_collect import cdata_sat_raw
from util.modis_cloud import cdata_cld_modis_only
from util.radiance_sim import cal_modis_rad, modis_simulation_plot
from util.util import path_dir, timing, grab_cfg, save_h5_info

@timing
def preprocess(cfg_info):
    # define date and region to study
    # ===============================================================
    date = datetime(int(cfg_info['date'][:4]),    # year
                    int(cfg_info['date'][4:6]),   # month
                    int(cfg_info['date'][6:]))    # day
    extent = [float(loc) + offset for loc, offset in zip(cfg_info['subdomain'], [-0.1, 0.1, -0.1, 0.1])]
    extent_analysis = [float(loc) for loc in cfg_info['subdomain']]
    print(f'simulation extent: {extent}')
    ref_threshold = float(cfg_info['ref_threshold'])
    name_tag = f"{cfg_info['cfg_name']}_modis"
    # ===============================================================

    # create data/name_tag directory if it does not exist
    # ===============================================================
    fdir_data = path_dir('/'.join(['data', name_tag]))
    # ===============================================================

    # download satellite data based on given date and region
    # ===============================================================
    fname_sat = '/'.join([fdir_data, 'sat.pk'])
    sat0 = modis_download(date=date, 
                              fdir_out=cfg_info['path_sat_data'], 
                              fdir_pre_data=fdir_data,
                              extent=extent,
                              extent_analysis=extent_analysis,
                              fname=fname_sat, overwrite=False)
    # sys.exit()
    # ===============================================================
    if not ('l2' in cfg_info.keys()):
        oco_data_dict = {'l2': 'oco_std',
                         'met': 'oco_met',
                         'l1b': 'oco_l1b',
                         'lt': 'oco_lite',
                         'dia': 'oco_dia',
                         'imap': 'oco_imap',
                         'co2prior': 'oco_co2prior'}
        for key, value in oco_data_dict.items():
            save_h5_info(cfg_info['cfg_path'], key, sat0.fnames[value][0].split('/')[-1])
    save_h5_info(cfg_info['cfg_path'], 'png', sat0.fnames['mod_rgb'][0].split('/')[-1])
    # create tmp-data/{name_tag} directory if it does not exist
    # ===============================================================
    fdir_cot_tmp = path_dir('tmp-data/%s/cot' % (name_tag))
    # ===============================================================

    # # create atmosphere based on OCO-Met and CO2_prior
    # # ===============================================================
    zpt_file = '/'.join([fdir_data, 'zpt.h5'])
    # os.path.abspath('/'.join([path_dir('/'.join(['data', '20181018_central_asia_2_test4_20181018'])), 'zpt.h5']))
    if 1:#not os.path.isfile(zpt_file):
        create_modis_atm(sat=sat0, o2mix=float(cfg_info['o2mix']), output=zpt_file)
        # the atmospheric layer used for the simulation is set to 
        # np.linspace(sfc_h_mean, 5, 11) + [5.5,  6. ,  6.5,  7. ,  7.5,  8. ,  8.5,  9. ,  9.5, 10. , 11. , 12. , 13. , 14. , 20. , 25. , 30. , 35. , 40. ]
    # # ===============================================================

    # # read out wavelength information from absorption file
    # # ===============================================================
    # nx = int(cfg_info['nx'])
    # for iband, band_tag in enumerate(['o2a', 'wco2', 'sco2']):
    #     fname_abs = f'{fdir_data}/atm_abs_{band_tag}_{(nx+1):d}.h5'
    #     if not os.path.isfile(fname_abs):
    #         oco_abs(cfg, sat0, 
    #                 zpt_file=zpt_file, iband=iband, 
    #                 nx=nx, 
    #                 Trn_min=float(cfg_info['Trn_min']), 
    #                 pathout=fdir_data,
    #                 reextract=False, plot=True)
    
    if not os.path.isfile(f'{fdir_data}/pre-data.h5') :
        cdata_sat_raw(sat0=sat０, dx=250, dy=250, overwrite=True, plot=True)
        cdata_cld_modis_only(sat０, fdir_cot_tmp, zpt_file, cfg_info, plot=True)
    # ===============================================================


    return date, extent, name_tag, fdir_data, sat0, zpt_file



def ax_lon_lat_label(ax, label_size=14, tick_size=12):
    ax.set_xlabel('Longitude ($^\circ$E)', fontsize=label_size)
    ax.set_ylabel('Latitude ($^\circ$N)', fontsize=label_size)
    ax.tick_params(axis='both', labelsize=tick_size)


@timing
def run_case_modis(cfg_info, preprocess_info):
    # Get information from cfg_info
    # ======================================================================
    name_tag = preprocess_info[2]
    sat0     = preprocess_info[4]
    zpt_file = preprocess_info[5]
    print('zpt_file:', zpt_file)
    # ======================================================================

    # run calculations for 650 nm
    # ======================================================================
    fdir_tmp_650 = path_dir(f'tmp-data/{name_tag}/modis_650')
    for solver in ['IPA',]:# '3D']:
        cal_modis_rad(sat0, zpt_file, 650, cfg_info, fdir=fdir_tmp_650, solver=solver,
                        overwrite=True, case_name_tag=name_tag)
        modis_simulation_plot(sat0, cfg_info, case_name_tag=name_tag, fdir=fdir_tmp_650,
                                   solver=solver, wvl=650, vmax=0.5, plot=True)
    # ======================================================================

    # run calculations for 1640 nm
    # ======================================================================
    # fdir_tmp_1640 = path_dir(f'tmp-data/{name_tag}/modis_1640')
    # for solver in ['IPA', '3D']:
    #     cal_modis_rad(sat0, zpt_file, 1640, cfg_info, fdir=fdir_tmp_1640, solver=solver,
    #                     overwrite=True, case_name_tag=name_tag)
    #     modis_simulation_plot(sat0, cfg_info, case_name_tag=name_tag, fdir=fdir_tmp_1640,
    #                                solver=solver, wvl=1640, vmax=0.12, plot=True)
    # ======================================================================


    # run calculations for 2130 nm
    # ======================================================================
    # fdir_tmp_2130 = path_dir(f'tmp-data/{name_tag}/modis_2130')
    # for solver in ['IPA', '3D']:
    #     cal_modis_rad(sat0, zpt_file, 2130, cfg_info, fdir=fdir_tmp_2130, solver=solver,
    #                     overwrite=True, case_name_tag=name_tag)
    #     modis_simulation_plot(sat0, cfg_info, case_name_tag=name_tag, fdir=fdir_tmp_2130,
    #                                solver=solver, wvl=2130, vmax=0.03, plot=True)
    # ======================================================================

    

def run_simulation(cfg, sfc_alb=None, sza=None):
    cfg_info = grab_cfg(cfg)
    preprocess_info = preprocess(cfg_info)
    run_case_modis(cfg_info, preprocess_info)


if __name__ == '__main__':
    

    cfg = 'cfg/20190815_pacific_modis.csv'

    print(cfg)
    run_simulation(cfg) #done
    
    







    



