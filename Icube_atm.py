from cosmo_tools import *

class atm_sim_tools:
    def __init__(self, param):
        '''
        Inputs:
        =======
        Nx : # of grids in x direction
        Ny : # of grids in y direction
        dth : spatial resolution in x,y direction [arcmin]
        '''
        self.Nx = param.Nx
        self.Ny = param.Ny
        self.dth = param.dth
        self.Nnu = param.Nnu
        self.nu_min = param.nu_min
        self.nu_max = param.nu_max
        self.nu_cent = np.median(param.nu_bins)
        self.nu_bins = param.nu_bins
        self.nu_binedges = param.nu_binedges
    
    def get_atm_spec(self, nu_binedges = [], unit = 'K'):
        '''
        get the atmosphere spectrum
        
        Inputs:
        =======
        nu_binedges: desired nu binedge array, if not given, output the nu's in data file.
        unit: 'K' or 'Jy/sr'

        Outputs:
        ========
        nu_vec: freq [GHz]
        y_vec : spectruma [input unit]
        '''

        atm_dat = np.loadtxt('data/atm-results/kp_50deg_0.47', skiprows=1)
        nu_dat = atm_dat[:,0] # GHz
        
        if unit == 'K': y_dat = atm_dat[:,2] # K
        elif unit == 'Jy/sr': y_dat = atm_dat[:,3] * (u.W/u.m**2/u.Hz/u.sr).to(u.Jy/u.sr)
        else: raise ValueError('unit must be "K" or "Jy/sr".')
        
        if len(nu_binedges)==0:
            return nu_dat, y_dat
        else:
            y_vec = np.histogram(nu_dat, bins=nu_binedges, weights=y_dat)[0] \
                 / np.histogram(nu_dat, bins=nu_binedges)[0]
            nu_vec = (nu_binedges[:-1] + nu_binedges[1:]) / 2
            return nu_vec, y_vec
        
    def make_atm_cube(self):
        _,I_vec = self.get_atm_spec(nu_binedges = np.flipud(self.nu_binedges), unit = 'Jy/sr')
        I_vec = np.flipud(I_vec)
        self.I_vec = I_vec
        
        Icube_arr = np.zeros([self.Nx, self.Ny, self.Nnu])
        for idx,iI in enumerate(I_vec):
            Icube_arr[:,:,idx] = iI
        self.Icube_arr = Icube_arr
        return
    