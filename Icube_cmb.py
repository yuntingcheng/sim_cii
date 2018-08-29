from cosmo_tools import *

class CMB_sim_tools():
    '''
    Input the the survey geometry, return the simulated CMB intensity data cube.
    '''
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
        self.N = max(self.Nx, self.Ny)
        self.lmin = np.pi / (self.N * self.dth * u.arcmin.to(u.rad))
        self.lmax = np.pi / (self.dth * u.arcmin.to(u.rad))
        self.Nnu = param.Nnu
        self.nu_min = param.nu_min
        self.nu_max = param.nu_max
        self.nu_bins = param.nu_bins
        self.nu_binedges = param.nu_binedges
        self.nu_cent = np.median(self.nu_bins)
        
    def get_CMB_Cl(self, ell_arr = [], unit = 'muK'):
        '''
        get the cmb power spectrum in ell_arr, interpolate from the CAMB out put file
        data retrived from CAMB web interface https://lambda.gsfc.nasa.gov/toolbox/tb_camb_form.cfm
        query result page: https://lambda.gsfc.nasa.gov/tmp/camb/camb_82852245.cfm

        Inputs:
        =======
        ell_arr: desired ell scale array, if none, output the ells in data file.
        unit: 'K' or 'muK'

        Outputs:
        ========
        ell_vec,Cl_vec : CMB power spectrum ell & Cl
        '''

        if unit not in ['K','muK']:
            print('unit must be "K" or "muK"!')
            return

        camb_dat = np.loadtxt('data/camb_out.dat')
        ell_dat = camb_dat[:,0]
        Cl_dat = camb_dat[:,1] # l(l+1)Cl/2/pi in muK
        Cl_dat = Cl_dat * 2 * np.pi / (ell_dat*(ell_dat+1)) # Cl in muK
        if unit == 'K':
            Cl_dat *= 1e-12 # Cl in K

        if len(ell_arr) == 0:
            return [ell_dat, Cl_dat]
        else:
            Cl_interp = np.interp(ell_arr,ell_dat,Cl_dat)
            return [ell_arr, Cl_interp]

    def gen_CMB_map(self, N = None, seed=None, unit = 'dimless'):
        '''
        Generate a simulated CMB temperature (square size) map from Gaussian random field.
        ref: /Users/ytcheng/anaconda3/lib/python3.5/site-packages/lenstools/image/noise.py

        Inputs:
        =======
        N : # of pixels on each side [int]
            default: 2^n, smallest n s.t. 2^n > 2 * self.N
        seed: seed of random number generator [int]
        unit: 'muK', 'K', 'dimless' (Delta T/T0)

        Ouput:
        ======
        self.cmb_map_large: N x N simulated CMB map
        self.cmb_map: Nx x Ny simulated CMB map, sub-region of self.cmb_map_large
        ''' 

        if unit not in ['dimless','K','muK']:
            print('unit must be "K", "muK", "dimless"!')
            return

        if seed != None:
            assert isinstance(seed, int), "seed is not an integer: %r" % seed
            np.random.seed(seed)

        if N == None:
            N = int(2**(np.ceil(np.log2(2 * self.N))))
        
        dell = 2 * np.pi / (N * self.dth * u.arcmin.to(u.rad))
        lx = np.arange(0, N//2 + 1) * dell
        ly = np.fft.fftfreq(N) * N * dell
        l = np.sqrt(lx[np.newaxis,:]**2 + ly[:,np.newaxis]**2)
        [_,Cl_interp] = self.get_CMB_Cl(ell_arr = l, unit = 'muK')

        real_part = np.sqrt(abs(Cl_interp) / 2) * np.random.normal(size=l.shape) * dell / (2 * np.pi)
        im_part = np.sqrt(abs(Cl_interp) / 2) * np.random.normal(size=l.shape) * dell / (2 * np.pi)

        ft_map = (real_part + im_part*1.0j) * l.shape[0]**2
        cmb_map_large = np.fft.irfft2(ft_map)
        cmb_map_large -= np.mean(cmb_map_large) # muK
        
        # get the right units
        if unit == 'K':
            cmb_map_large *= 1e-6
        if unit == 'dimless':
            cmb_map_large /= cosmo.Tcmb0.to(u.uK).value
        
        self.cmb_map_unit = unit
        self.cmb_map_large = cmb_map_large
        
        #  Get the CMB data in pixels that will be used in generating I cube.
        Nlarge = self.cmb_map_large.shape[0]
        xstart = round(Nlarge/2. - self.Nx/2.)
        ystart = round(Nlarge/2. - self.Ny/2.)
        cmb_map = self.cmb_map_large[xstart : xstart + self.Nx, ystart : ystart + self.Ny]
        self.cmb_map = cmb_map
        return
    
    def T0cmb_sepctrum(self):
        '''
        I_nu(Tcmb) in nubin_vec [Jy/sr]
        '''
        nu_vec = self.nu_bins * u.GHz
        nu_vec = nu_vec.to(u.Hz)        
        x = (const.h * nu_vec  / const.k_B / cosmo.Tcmb0).decompose().value
        I_vec = 2 * const.h * nu_vec**3 / const.c**2 / (np.exp(x) - 1) / u.sr
        I_vec = I_vec.to(u.Jy / u.sr)
        return I_vec
    
    def make_CMB_cube(self):
        '''
        make the CMB intensity data cube from self.cmb_map
        
        Output:
        =======
        Icube_arr : CMB intensity cube [Jy/sr]
        '''
        I_vec = self.T0cmb_sepctrum()
        nu_vec = self.nu_bins * u.GHz
        
        # temperautre map Delta T/ T0
        if self.cmb_map_unit == 'dimless':
            T_map = self.cmb_map
        elif self.cmb_map_unit == 'muK':
            T_map = self.cmb_map / cosmo.Tcmb0.to(u.uK).value
        elif self.cmb_map_unit == 'K':
            T_map = self.cmb_map / cosmo.Tcmb0.to(u.K).value
        
        x_vec = (const.h * nu_vec / (const.k_B * cosmo.Tcmb0)).decompose().value
        
        Icube_arr = np.zeros([self.Nx, self.Ny, self.Nnu])
        for iz in range(self.Nnu):
            Inu = I_vec[iz].value
            x = x_vec[iz]
            for ix in range(self.Nx):
                for iy in range(self.Ny):
                    Icube_arr[ix, iy, iz] = Inu * (1 + T_map[ix,iy] * x)
        
        self.Icmb_arr = Icube_arr
        return