import os
import sys
import pickle
import numpy as np
from er3t.util.modis import get_sinusoidal_grid_tag, get_filename_tag, download_modis_rgb
from er3t.util.daac import download_laads_https

class modis_download:
    """
    This class is used to download satellite data from MODIS and OCO-2
    """
    def __init__(self,
                 date=None,
                 satID='aqua',
                 extent=None,
                 extent_analysis=None,
                 fname=None,
                 fdir_out='data',
                 fdir_pre_data='data',
                 overwrite=False,
                 quiet=False,
                 verbose=False):
        """
        Initialize the SatelliteDownload class.
        """
        self.date     = date
        self.satID    = satID
        self.extent   = extent
        self.extent_analysis = extent_analysis
        self.fdir_out = fdir_out
        self.fdir_pre_data = fdir_pre_data
        self.quiet    = quiet
        self.verbose  = verbose

        if (fname is not None) and os.path.exists(fname) and (not overwrite):
            self.load(fname)
        elif ((date is not None) and (extent is not None) and (fname is not None) and os.path.exists(fname) and overwrite) or \
             ((date is not None) and (extent is not None) and (fname is not None) and not os.path.exists(fname)):
            self.run()
            self.dump(fname)

        elif date is not None and extent is not None and fname is None:
            self.run()
        else:
            raise FileNotFoundError('Error   [satellite_download]: Please check if \'%s\' exists or provide \'date\' and \'extent\' to proceed.' % fname)

    def load(self, fname):
        """
        Load the satellite data from the pickle file.
        """
        with open(fname, 'rb') as f:
            obj = pickle.load(f)
            if hasattr(obj, 'fnames') and hasattr(obj, 'extent') and hasattr(obj, 'fdir_out') and hasattr(obj, 'date'):
                if self.verbose:
                    print('Message [satellite_download]: Loading %s ...' % fname)
                self.date     = obj.date
                self.extent   = obj.extent
                self.fnames   = obj.fnames
                self.fdir_out = obj.fdir_out
            else:
                sys.exit('Error   [satellite_download]: File \'%s\' is not the correct pickle file to load.' % fname)

    def run(self, run=True):
        """
        Run the satellite download process.
        """
        lon = np.array(self.extent[:2])
        lat = np.array(self.extent[2:])
        if self.satID == 'aqua':
            sat_tag = '61/MYD'
        elif self.satID == 'terra':
            sat_tag = '61/MOD'

        self.fnames = {}

        self.fnames['mod_rgb'] = [download_modis_rgb(self.date, self.extent, fdir=self.fdir_pre_data, which='aqua', coastline=True)]

        # MODIS Level 2 Cloud Product and MODIS 03 geo file
        self.fnames['mod_l2'] = []
        self.fnames['mod_02_1km'] = []
        self.fnames['mod_02_hkm'] = []
        self.fnames['mod_02'] = []
        self.fnames['mod_03'] = []
        self.fnames['mod_04'] = []
        self.fnames['mod_07'] = []

        modis_parent_fdir = '/'.join([self.fdir_out, 'modis'])
        if not os.path.exists(modis_parent_fdir):
            os.makedirs(modis_parent_fdir)
        modis_fdir = f"{modis_parent_fdir}/{self.date.strftime('%Y%m%d')}"
        if not os.path.exists(modis_fdir):
            os.makedirs(modis_fdir)

        filename_tags_03 = get_filename_tag(self.date, lon, lat, satID=self.satID)
        for filename_tag in filename_tags_03:
            # fnames_l2 = download_modis_https(self.date, '61/MYD06_L2', filename_tag, day_interval=1, fdir_out=modis_fdir, run=run)
            # fnames_02_1km = download_modis_https(self.date, '61/MYD021KM', filename_tag, day_interval=1, fdir_out=modis_fdir, run=run)
            # fnames_02_hkm = download_modis_https(self.date, '61/MYD02HKM', filename_tag, day_interval=1, fdir_out=modis_fdir, run=run)
            # fnames_02 = download_modis_https(self.date, '61/MYD02QKM', filename_tag, day_interval=1, fdir_out=modis_fdir, run=run)
            # fnames_03 = download_modis_https(self.date, '61/MYD03'   , filename_tag, day_interval=1, fdir_out=modis_fdir, run=run)
            # fnames_04 = download_modis_https(self.date, '61/MYD04_L2'   , filename_tag, day_interval=1, fdir_out=modis_fdir, run=run)
            
            fnames_l2 = download_laads_https(self.date, f'{sat_tag}06_L2', filename_tag, day_interval=1, 
                                             server='https://ladsweb.modaps.eosdis.nasa.gov',
                                             fdir_prefix='/archive/allData',fdir_out=modis_fdir, run=run)
            fnames_02_1km = download_laads_https(self.date, f'{sat_tag}021KM', filename_tag, day_interval=1, 
                                             server='https://ladsweb.modaps.eosdis.nasa.gov',
                                             fdir_prefix='/archive/allData',fdir_out=modis_fdir, run=run)
            fnames_02_hkm = download_laads_https(self.date, f'{sat_tag}02HKM', filename_tag, day_interval=1, 
                                             server='https://ladsweb.modaps.eosdis.nasa.gov',
                                             fdir_prefix='/archive/allData',fdir_out=modis_fdir, run=run)
            fnames_02 = download_laads_https(self.date, f'{sat_tag}02QKM', filename_tag, day_interval=1, 
                                             server='https://ladsweb.modaps.eosdis.nasa.gov',
                                             fdir_prefix='/archive/allData',fdir_out=modis_fdir, run=run)
            fnames_03 = download_laads_https(self.date, f'{sat_tag}03', filename_tag, day_interval=1, 
                                             server='https://ladsweb.modaps.eosdis.nasa.gov',
                                             fdir_prefix='/archive/allData',fdir_out=modis_fdir, run=run)
            fnames_04 = download_laads_https(self.date, f'{sat_tag}04_L2', filename_tag, day_interval=1, 
                                             server='https://ladsweb.modaps.eosdis.nasa.gov',
                                             fdir_prefix='/archive/allData',fdir_out=modis_fdir, run=run)
            fnames_07 = download_laads_https(self.date, f'{sat_tag}07_L2', filename_tag, day_interval=1, 
                                             server='https://ladsweb.modaps.eosdis.nasa.gov',
                                             fdir_prefix='/archive/allData',fdir_out=modis_fdir, run=run)
            
            self.fnames['mod_l2'] += fnames_l2
            self.fnames['mod_02_1km'] += fnames_02_1km
            self.fnames['mod_02_hkm'] += fnames_02_hkm
            self.fnames['mod_02'] += fnames_02
            self.fnames['mod_03'] += fnames_03
            self.fnames['mod_04'] += fnames_04
            self.fnames['mod_07'] += fnames_07

        # MOD09A1 surface reflectance product
        self.fnames['mod_09'] = []
        filename_tags_09 = get_sinusoidal_grid_tag(lon, lat)
        for filename_tag in filename_tags_09:
            # fnames_09 = download_modis_https(self.date, '61/MOD09A1', filename_tag, day_interval=8, fdir_out=modis_fdir, run=run)
            fnames_09 = download_laads_https(self.date, f'{sat_tag}09A1', filename_tag, day_interval=8, 
                                             server='https://ladsweb.modaps.eosdis.nasa.gov',
                                             fdir_prefix='/archive/allData',fdir_out=modis_fdir, run=run)
            self.fnames['mod_09'] += fnames_09
        # MOD43A3 surface reflectance product
        self.fnames['mcd_43'] = []
        filename_tags_43 = get_sinusoidal_grid_tag(lon, lat)
        for filename_tag in filename_tags_43:
            # fnames_43 = download_modis_https(self.date, '61/MCD43A3', filename_tag, day_interval=16, fdir_out=modis_fdir, run=run)
            fnames_43 = download_laads_https(self.date, '61/MCD43A3', filename_tag, day_interval=8,
                                             server='https://ladsweb.modaps.eosdis.nasa.gov',
                                             fdir_prefix='/archive/allData',fdir_out=modis_fdir, run=run)
            self.fnames['mcd_43'] += fnames_43

    def dump(self, fname):
        """
        Save the SatelliteDownload object into a pickle file
        """
        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [satellite_download]: Saving object into %s ...' % fname)
            pickle.dump(self, f)

if __name__ == '__main__':
    None