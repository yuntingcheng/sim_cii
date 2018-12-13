import numpy as np
import matplotlib.pyplot as plt
import copy
from astropy import units as u
from astropy import constants as const
from astropy import cosmology
import pickle
from hmf import MassFunction


# cosmo params
cosmo = cosmology.Planck15
cosmo.hmf = MassFunction(Mmin = np.log10(1e6), Mmax = np.log10(1e15), cosmo_model=cosmo, hmf_model = 'SMT')

class HMFz:
    '''
    HMF, P(k), and T(k) at given redshift.
    '''
    def __init__(self, z, cosmo = cosmo):
        hmf = copy.deepcopy(cosmo.hmf)
        hmf.update(z=z)
        self.z = z
        self.hmf = hmf

    def sigma(self, Mmin = 1e8, Mmax = 1e15, dlog10m = 0.01, m_arr = []):
        '''
        mass variance
        output:
        =======
        dndm_arr []
        m_arr [Msun h^-1]
        '''
        
        if len(m_arr)==0:
            m_arr = 10**np.arange(np.log10(Mmin), np.log10(Mmax), dlog10m)
        
        logm_arr = np.log(m_arr)
        logm_dat_arr = np.log(self.hmf.m)
        sigma_dat_arr = self.hmf.sigma
        
        sigma_arr = np.interp(logm_arr, logm_dat_arr, sigma_dat_arr)
        return sigma_arr, m_arr

    def dndm(self, Mmin = 1e8, Mmax = 1e15, dlog10m = 0.01, m_arr = []):
        '''
        halo mass funciton
        output:
        =======
        dndm_arr [h^4 Mpc^-3 Msun^-1]
        m_arr [Msun h^-1]
        '''
        
        if len(m_arr)==0:
            m_arr = 10**np.arange(np.log10(Mmin), np.log10(Mmax), dlog10m)
        
        logm_arr = np.log(m_arr)
        logm_dat_arr = np.log(self.hmf.m)
        logdndm_dat_arr = np.log(self.hmf.dndm)
        
        logdndm_arr = np.interp(logm_arr, logm_dat_arr, logdndm_dat_arr)
        dndm_arr = np.exp(logdndm_arr)
        return dndm_arr, m_arr
    
    def dndlnm(self, Mmin = 1e8, Mmax = 1e15, dlog10m = 0.01, m_arr = []):        
        '''
        halo mass funciton
        output:
        =======
        dndlnm_arr [h^3 Mpc^-3]
        m_arr [Msun h^-1]
        '''
        dndm_arr, m_arr = self.dndm(Mmin = Mmin, Mmax = Mmax, dlog10m = dlog10m, m_arr = m_arr)
        
        return dndm_arr*m_arr, m_arr
    
    def dndlog10m(self, Mmin = 1e8, Mmax = 1e15, dlog10m = 0.01, m_arr = []):
        '''
        halo mass funciton
        output:
        =======
        dndlog10m_arr [h^3 Mpc^-3]
        m_arr [Msun h^-1]
        '''
        
        dndm_arr, m_arr = self.dndm(Mmin = Mmin, Mmax = Mmax, dlog10m = dlog10m, m_arr = m_arr)
        
        return dndm_arr*m_arr*np.log(10), m_arr
    
    def P(self, kmin = 1e-3, kmax = 1e1, dlog10k = 0.01, k_arr = []):
        '''
        Power spectrum
        output:
        =======
        P_arr [Mpc^3 h^-3]
        k_arr [h/Mpc]
        '''
        
        if len(k_arr)==0:
            k_arr = 10**np.arange(np.log10(kmin), np.log10(kmax), dlog10k)
        
        logk_arr = np.log(k_arr)

        logk_dat_arr = np.log(self.hmf.k)
        logP_dat_arr = np.log(self.hmf.power)
        
        logP_arr = np.interp(logk_arr, logk_dat_arr, logP_dat_arr)
        P_arr = np.exp(logP_arr)
        
        return P_arr, k_arr
    
    def Del2(self, kmin = 1e-3, kmax = 1e1, dlog10k = 0.01, k_arr = []):
        '''
        Dimensionless power spectrum
        output:
        =======
        Del2_arr []
        k_arr [h/Mpc]
        '''
        
        P_arr, k_arr = self.P(kmin = kmin, kmax = kmax, dlog10k = dlog10k, k_arr = k_arr)
        Del2_arr = P_arr*k_arr**3/2/np.pi**2
        
        return Del2_arr, k_arr
    
    def bias(self, Mmin = 1e8, Mmax = 1e15, dlog10m = 0.01, m_arr = []):
        '''
        Halo bias
        Sheth, Mo, Tormen 2001 eq 8
        '''
        if len(m_arr)==0:
            m_arr = 10**np.arange(np.log10(Mmin), np.log10(Mmax), dlog10m)
        
        del_sc = self.delta_sc(self.z)
        v = del_sc / self.sigma(m_arr = m_arr)[0]
        
        a = 0.707
        b = 0.5
        c = 0.6
        b_Lag = 1/np.sqrt(a)/del_sc *(np.sqrt(a)*a*v**2 + np.sqrt(a)*b*(a*v**2)**(1-c)\
                - (a*v**2)**c/((a*v**2)**c + b*(1-c)*(1-c/2)))
        b = 1 + b_Lag
        return b, m_arr
    
    def delta_sc(self, z):
        # Kitayama & Suto 1996 eq A6
        z = np.array(z)
        Omf = cosmo.Om0 * (1+z)**3 / (cosmo.Om0 * (1+z)**3 + cosmo.Ode0)
        deltasc = 3*(12*np.pi)**(2./3)/20 * (1 + 0.0123 * np.log10(Omf))
        return deltasc

class cosmo_dist:
    '''
    cosmo distance at z, in unit Mpc/h (kpc/h)
    '''
    def __init__(self, z, cosmo = cosmo):
        self.z = z
        self.h = cosmo.h
        self.hubble_distance = self._hdist(cosmo.hubble_distance)
        self.H = self._hdist_inv(cosmo.H(z))
        self.comoving_distance = self._hdist(cosmo.comoving_distance(z))
        self.angular_diameter_distance = self._hdist(cosmo.angular_diameter_distance(z))
        self.luminosity_distance = self._hdist(cosmo.luminosity_distance(z))
        self.comoving_transverse_distance = self._hdist(cosmo.comoving_transverse_distance(z))
        self.kpc_comoving_per_arcmin = self._hdist(cosmo.kpc_comoving_per_arcmin(z))
        self.kpc_proper_per_arcmin = self._hdist(cosmo.kpc_proper_per_arcmin(z))
        if 0 not in np.array(z):
            self.arcsec_per_kpc_comoving = self._hdist_inv(cosmo.arcsec_per_kpc_comoving(z))
            self.arcsec_per_kpc_proper = self._hdist_inv(cosmo.arcsec_per_kpc_proper(z))
    
    def _hdist(self, d):
        return d * self.h / u.h
    
    def _hdist_inv(self,dinv):
        return dinv / self.h * u.h
    
# spectral lines
class spec_lines():
    def __init__(self):
        self.CII = 157.7409 * u.um
        self.CO = self._CO
        self.Lya = 0.12157 * u.um
        self.Lyb = 0.10257 * u.um
        self.Ha = 0.6563 * u.um
        self.Hb = 0.4861 * u.um
        self.HI = 211061.140542 * u.um
        self.OII = 0.3727 * u.um
        self. OIII = 0.5007 * u.um
        
    def _CO(self, J = 1):
        return 2610./J * u.um
    
    def z_prj(self, j_prj, z_targ, j_targ):
        '''
        CII, CO prjected redshift. Return the projected z z_prj of line j_prj to 
        rest frame z_targ of line j_targ.
        '''
        wl_prj = self.CO(j_prj) if j_prj !=0 else self.CII
        wl_targ = self.CO(j_targ) if j_targ !=0 else self.CII
        zprj = (wl_targ / wl_prj)*(1 + z_targ) - 1
        return zprj.value
        
spec_lines = spec_lines()

