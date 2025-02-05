import os
import sys
import copy
import pickle
import h5py
import numpy as np
import er3t.common


__all__ = ['atm_atmmod']


class atm_atmmod:

    """
    This class is modified from the original version in the er3t/pre/atm/atm_atmmod.py
    It reads the zpt file to get the gas number densities for the given levels and layers.

    Input:

        levels=      : keyword argument, numpy array, height in km
        fname=       : keyword argument, string, default=None, the atmoshpere file user wants to name
        fname_atmmod=: keyword argument, string, defult='mca-data/atmmod/afglus.dat', the base atmosphere file to interpolate to levels and layers
        overwrite=   : keyword argument, boolen, default=False, whether or not user wants to overwrite the atmosphere file
        verbose=     : keyword argument, boolen, default=False, whether or not print detailed messages

    Note:
    If levels is provided but fname does not exisit:
        calculate atmospheric gases profile and save data into fname

    if levels is not provided but fname is provided (also exists):
        read out the data from fname

    if levels and fname are neither provided:
        exit with error message

    Output:
        self.lev['pressure']
        self.lev['temperature']
        self.lev['altitude']
        self.lev['h2o']
        self.lev['o2']
        self.lev['o3']
        self.lev['co2']
        self.lev['no2']
        self.lev['ch4']
        self.lev['factor']

        self.lay['pressure']
        self.lay['temperature']
        self.lay['altitude']
        self.lay['thickness']
        self.lay['h2o']
        self.lay['o2']
        self.lay['o3']
        self.lay['co2']
        self.lay['no2']
        self.lay['ch4']
        self.lay['factor']
    """


    ID     = 'Atmosphere 1D'

    gases  = ['o3', 'o2', 'h2o', 'co2', 'no2', 'ch4']


    def __init__(self,                \
                 zpt_file     = None, \
                 fname        = None, \
                 fname_atmmod = '%s/afglus.dat' % er3t.common.fdir_data_atmmod, \
                 overwrite    = False, \
                 verbose      = False):

        self.verbose      = verbose
        self.fname_atmmod = fname_atmmod

        if ((fname is not None) and (os.path.exists(fname)) and (not overwrite)):

            self.load(fname)

        elif ((zpt_file is not None) and (fname is not None) and (os.path.exists(fname)) and (overwrite)) or \
             ((zpt_file is not None) and (fname is not None) and (not os.path.exists(fname))):

            self.run(zpt_file)
            self.dump(fname)

        elif ((zpt_file is not None) and (fname is None)):

            self.run(zpt_file)

        else:

            sys.exit('Error   [atm_atmmod]: Please check if \'%s\' exists or provide \'zpt_file \' to proceed.' % fname)


    def load(self, fname):

        with open(fname, 'rb') as f:
            obj = pickle.load(f)
            if hasattr(obj, 'lev') and hasattr(obj, 'lay'):
                if self.verbose:
                    print('Message [atm_atmmod]: Loading %s ...' % fname)
                self.fname = obj.fname
                self.lev   = obj.lev
                self.lay   = obj.lay
            else:
                sys.exit('Error   [atm_atmmod]: File \'%s\' is not the correct pickle file to load.' % fname)


    def run(self, zpt_file):
        
        with h5py.File(zpt_file, 'r') as modis_zpt:
            levels = modis_zpt['level_sim'][...]
            self.levels = levels
            self.layers = 0.5 * (levels[1:]+levels[:-1])
            self.o2mix = modis_zpt['o2_mix'][...]
            self.lev_h = modis_zpt['h_lev'][...]
            self.lev_t = modis_zpt['t_lev'][...]
            self.lev_p = modis_zpt['p_lev'][...]
            self.lev_d_o2 = modis_zpt['d_o2_lev'][...]
            self.lev_d_h2o = modis_zpt['d_h2o_lev'][...]

        # self.atm0: Python dictionary
        #   self.atm0['altitude']
        #   self.atm0['pressure']
        #   self.atm0['temperature']
        #   self.atm0['co2']
        #   self.atm0['no2']
        #   self.atm0['h2o']
        #   self.atm0['o3']
        #   self.atm0['o2']
        self.atmmod()

        # self.lev, self.lay: Python dictionary
        #   self.lev['altitude']    | self.lay['altitude']
        #   self.lev['pressure']    | self.lay['pressure']
        #   self.lev['temperature'] | self.lay['temperature']
        #   self.lev['co2']         | self.lay['co2']
        #   self.lev['no2']         | self.lay['no2']
        #   self.lev['h2o']         | self.lay['h2o']
        #   self.lev['o3']          | self.lay['o3']
        #   self.lev['o2']          | self.lay['o2']
        self.interp(self.o2mix)

        # add self.lev['ch4'] and self.lay['ch4']
        self.add_ch4()

        # covert mixing ratio [unitless] to number density [cm-3]
        self.cal_num_den()


    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [atm_atmmod]: Saving object into %s ...' % fname)
            pickle.dump(self, f)


    def atmmod(self):

        vnames = ['altitude', 'pressure', 'temperature', 'air', 'o3', 'o2', 'h2o', 'co2', 'no2']
        units  = ['km', 'mb', 'K', 'cm-3', 'cm-3', 'cm-3', 'cm-3', 'cm-3', 'cm-3']
        data   = np.genfromtxt(self.fname_atmmod)

        # read original data from *.dat file into Python dictionary that contains 'data', 'name', and 'units'
        self.atm0 = {}
        for i, vname in enumerate(vnames):
            self.atm0[vname] = {'data':data[:, i], 'name':vname, 'units':units[i]}

        # 1. change the values in array from descending order to ascending order
        indices = np.argsort(self.atm0['altitude']['data'])
        for key in self.atm0.keys():
            self.atm0[key]['data'] = self.atm0[key]['data'][indices]

        # 2. calculate the mixing ratio from volume number density for each gas
        for key in self.atm0.keys():
            if key in self.gases:
                self.atm0[key]['data']  = self.atm0[key]['data']/self.atm0['air']['data']
                self.atm0[key]['units'] = 'N/A'


    def interp(self, o2mix):

        # check whether the input height is within the atmosphere height range
        if self.levels.min() < self.atm0['altitude']['data'].min():
            sys.exit('Error   [atm_atmmod]: Input levels too low.')
        if self.levels.max() > self.atm0['altitude']['data'].max():
            sys.exit('Error   [atm_atmmod]: Input levels too high.')

        self.lev = {}
        self.lev = copy.deepcopy(self.atm0)
        self.lev['altitude']['data']  = self.levels

        self.lay = {}
        self.lay = copy.deepcopy(self.atm0)
        self.lay['altitude']['data']  = self.layers
        self.lay['thickness'] = { \
                 'name' : 'Thickness', \
                 'units':'km', \
                 'data':self.levels[1:]-self.levels[:-1]}

        # Linear interpolate to input levels and layers
        for key in self.atm0.keys():
            if key in ['co2', 'o3', 'no2']:
                self.lev[key]['data'] = np.interp(self.lev['altitude']['data'], self.atm0['altitude']['data'], self.atm0[key]['data'])
                self.lay[key]['data'] = np.interp(self.lay['altitude']['data'], self.atm0['altitude']['data'], self.atm0[key]['data'])
        for key, key_value in zip(['temperature', 'o2', 'h2o', 'air'], [self.lev_t, self.lev_d_o2, self.lev_d_h2o, self.lev_d_o2/self.o2mix]):
            self.lev[key]['data'] = np.interp(self.lev['altitude']['data'], self.lev_h, key_value)
            self.lay[key]['data'] = np.interp(self.lay['altitude']['data'], self.lev_h, key_value)
        
        # Use Barometric formula to interpolate pressure
        self.lev['pressure']['data'] = atm_interp_pressure(self.lev_p, self.lev_h, self.lev_t, \
                self.lev['altitude']['data'], self.lev['temperature']['data'])
        self.lay['pressure']['data'] = atm_interp_pressure(self.lev_p, self.lev_h, self.lev_t, \
                self.lay['altitude']['data'], self.lay['temperature']['data'])


    def add_ch4(self):

        ch4 = {'name':'ch4', 'units':'cm-3', 'data':atm_interp_ch4(self.levels)}
        self.lev['ch4'] = ch4

        ch4 = {'name':'ch4', 'units':'cm-3', 'data':atm_interp_ch4(self.layers)}
        self.lay['ch4'] = ch4


    def cal_num_den(self):

        self.lev['factor']  = { \
          'name':'number density factor', \
          'units':'cm-3', \
          'data':6.02214179e23/8.314472*self.lev['pressure']['data']/self.lev['temperature']['data']*1.0e-4}

        self.lay['factor']  = { \
          'name':'number density factor', \
          'units':'cm-3', \
          'data':6.02214179e23/8.314472*self.lay['pressure']['data']/self.lay['temperature']['data']*1.0e-4}

        for key in self.lev.keys():
            if key in ['ch4', 'no2']:#self.gases:
                self.lev[key]['data']  = self.lev[key]['data'] * self.lev['factor']['data']
                self.lev[key]['units'] = 'cm-3'
                self.lay[key]['data']  = self.lay[key]['data'] * self.lay['factor']['data']
                self.lay[key]['units'] = 'cm-3'



def atm_interp_pressure(pressure, altitude, temperature, altitude_to_interp, temperature_to_interp):

    """
    Use Barometric formula (https://en.wikipedia.org/wiki/Barometric_formula)
    to interpolate pressure from height and temperature

    Input:
        pressure: numpy array, original pressure in hPa
        altitude: numpy array, original altitude in km
        temperature: numpy array, original temperature in K
        altitude_to_interp: numpy array, altitude to be interpolate
        temperature_interp: numpy array, temperature to be interpolate

    Output:
        pn: interpolated pressure based on the input
    """

    indices = np.argsort(altitude)
    h       = np.float_(altitude[indices])
    p       = np.float_(pressure[indices])
    t       = np.float_(temperature[indices])

    indices = np.argsort(altitude_to_interp)
    hn      = np.float_(altitude_to_interp[indices])
    tn      = np.float_(temperature_to_interp[indices])

    n = p.size - 1
    a = 0.5*(t[1:]+t[:-1]) / (h[:-1]-h[1:]) * np.log(p[1:]/p[:-1])
    z = 0.5*(h[1:]+h[:-1])

    z0  = np.min(z) ; z1  = np.max(z)
    hn0 = np.min(hn); hn1 = np.max(hn)

    if hn0 < z0:
        a = np.hstack((a[0], a))
        z = np.hstack((hn0, z))
        if z0 - hn0 > 2.0:
            print('Warning [atm_interp_pressure]: Standard atmosphere not sufficient (lower boundary).')

    if hn1 > z1:
        a = np.hstack((a, z[n-1]))
        z = np.hstack((z, hn1))
        if hn1-z1 > 10.0:
            print('Warning [atm_interp_pressure]: Standard atmosphere not sufficient (upper boundary).')

    an = np.interp(hn, z, a)
    pn = np.zeros_like(hn)

    if hn.size == 1:
        hi = np.argmin(np.abs(hn-h))
        pn = p[hi]*np.exp(-an*(hn-h[hi])/tn)
        return pn

    for i in range(pn.size):
        hi = np.argmin(np.abs(hn[i]-h))
        pn[i] = p[hi]*np.exp(-an[i]*(hn[i]-h[hi])/tn[i])

    dp = pn[:-1] - pn[1:]
    pl = 0.5 * (pn[1:]+pn[:-1])
    zl = 0.5 * (hn[1:]+hn[:-1])

    for i in range(n-2):
        indices = (zl >= h[i]) & (zl < h[i+1])
        ind = np.where(indices==True)[0]
        ni  = indices.sum()
        if ni >= 2:
            dpm = dp[ind].sum()

            i0 = np.min(ind)
            i1 = np.max(ind)

            x1 = pl[i0]
            x2 = pl[i1]
            y1 = dp[i0]
            y2 = dp[i1]

            bb = (y2-y1) / (x2-x1)
            aa = y1 - bb*x1
            rescale = dpm / (aa+bb*pl[indices]).sum()

            if np.abs(rescale-1.0) > 0.1:
                print('------------------------------------------------------------------------------')
                print('Warning [atm_interp_pressure]:')
                print('Warning: pressure smoothing failed at ', h[i], '...', h[i+1])
                print('rescale=', rescale)
                print('------------------------------------------------------------------------------')
            else:
                dp[indices] = rescale*(aa+bb*pl[indices])

    for i in range(dp.size):
        pn[i+1] = pn[i] - dp[i]

    return pn



def atm_interp_ch4(altitude_inp):

    """
    input:
        levels: numpy array, height in km
    output:
        ch4mix: mixing ratio of CH4
    """

    # height
    ch4h   = np.array([ 0.000000,      0.100000,      0.200000,      0.300000, \
                        0.400000,      0.500000,      1.000000,      2.000000, \
                        3.000000,      4.000000,      5.000000,      6.000000, \
                        7.000000,      8.000000,      9.000000,     10.000000, \
                       11.000000,     12.000000,     13.000000,     14.000000, \
                       15.000000,     16.000000,     17.000000,     18.000000, \
                       19.000000,     20.000000,     21.000000,     22.000000, \
                       23.000000,     24.000000,     25.000000,     27.000000, \
                       29.000000,     31.000000,     33.000000,     35.000000, \
                       37.000000,     40.000000])

    # CH4 number concentration
    ch4m   = np.array([  1.70000e-06,   1.70000e-06,   1.70000e-06,   1.70000e-06, \
                         1.70000e-06,   1.70000e-06,   1.70000e-06,   1.70000e-06, \
                         1.70000e-06,   1.70000e-06,   1.70000e-06,   1.70000e-06, \
                         1.69900e-06,   1.69700e-06,   1.69300e-06,   1.68500e-06, \
                         1.67485e-06,   1.66200e-06,   1.64753e-06,   1.62915e-06, \
                         1.60500e-06,   1.58531e-06,   1.55875e-06,   1.52100e-06, \
                         1.48145e-06,   1.42400e-06,   1.38858e-06,   1.34258e-06, \
                         1.28041e-06,   1.19173e-06,   1.05500e-06,   1.02223e-06, \
                         9.63919e-07,   9.04935e-07,   8.82387e-07,   8.48513e-07, \
                         7.91919e-07,   0.000000000])

    ch4mix = np.interp(altitude_inp, ch4h, ch4m)

    return ch4mix



if __name__ == '__main__':

    pass
