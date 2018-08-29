from survey_params import *
import scipy

class time_instr_noise:
    def __init__(self):
        self._time_par = TIME_param()
        self.nu_bins = self._time_par.nu_bins
        self.sig_noise = self._sigma_vec()
        
    def _read_nei_data(self):
        nei_data = np.loadtxt('./data/20171101_baseline_kp_0.6fudge_nei.txt', \
                              skiprows=1, delimiter = ',')
        nu_dat = nei_data[:,0] # GHz
        nei_dat = nei_data[:,1] * 1e6 # Jy/sr sqrt(sec)
        return nu_dat, nei_dat

    def _nei_interp(self):
        nu_dat,nei_dat = self._read_nei_data()
        f = scipy.interpolate.interp1d(nu_dat, nei_dat, fill_value='extrapolate')
        return f(self.nu_bins)
    
    def _sigma_vec(self):
        nei_vec = self._nei_interp()
        sig_vec = nei_vec / np.sqrt(self._time_par.t_pix)
        return sig_vec
    
    def sim_noise_2D(self, Nx):
        sig_map,_ = np.meshgrid(self.sig_noise, np.ones(Nx))
        sim_map = np.random.normal(scale = sig_map, size = sig_map.shape)
        return sim_map
    
    def sim_noise_3D(self, Nx, Ny):
        sig_map,_ = np.meshgrid(self.sig_noise, np.ones(Nx))
        sig_cube = sig_map.tolist() * Ny
        sig_cube = np.asarray(sig_cube).reshape(Nx,Ny,-1)
        sim_cube = np.random.normal(scale = sig_cube, size = sig_cube.shape)
        return sim_cube