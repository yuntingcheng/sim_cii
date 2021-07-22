import pandas as pd
from cosmo_tools import *
from sides.PYSIDES.pysides.gen_fluxes import *

class make_Icube_cib():
    '''
    Input the catalog and the survey geometry, return the intensity data cube.
    '''
    def __init__(self, df, Nx, Ny, dx, dy, nu_binedges, nu_bins = [], x_cent=0.7, y_cent=0.7):
        '''
        Inputs:
        =======
        df : data frame contain the catalog
        Nx : # of grids in x direction
        Ny : # of grids in y direction
        dx : spatial resolution in x direction [arcmin]
        dy : spatial resolution in y direction [arcmin]
        nu_binedges : frequency bin edges [GHz]
        x_cent: cube center x position in data frame ra [degree, default = 0.7 (center of the catalog)]
        y_cent: cube center x position in data frame dec [degree, default = 0.7(center of the catalog)]
        '''
        self.Nx = Nx
        self.Ny = Ny
        self.Nnu = len(nu_binedges) - 1
        self.dx = dx
        self.dy = dy
        self.x_cent = x_cent
        self.y_cent = y_cent
        self.xbinedge_vec, self.ybinedge_vec = _binedge_vectors(self)
        self.nu_binedges = nu_binedges
        if len(nu_bins)==0:
            self.nu_bins = (self.nu_binedges[:-1] + self.nu_binedges[1:]) / 2
        else:
            self.nu_bins = nu_bins
        
        self.lambda_bins = (const.c / (self.nu_bins * u.GHz)).to(u.um).value #[um]
        _select_df(self,df)
                        
    def make_cib_cube(self, SED_file, verbose = 1):
        '''
        make the I data cube of the line.
        Inputs:
        =======
        SED_file: SED pickle file name
        
        Oputputs:
        =========
        Icube_arr: Intensity data cube [(Nx, Ny, Nnu), Jy/sr]
        '''
        
        SED_dict = pickle.load(open(SED_file, "rb"))
        
        dOmega = (self.dx * u.arcmin).to(u.rad).value * (self.dy * u.arcmin).to(u.rad).value

        # make the data cube
        xbin_vec = _bin_label(np.array(self.df['ra']),self.xbinedge_vec)
        ybin_vec = _bin_label( np.array(self.df['dec']),self.ybinedge_vec)

        Scube_arr = np.zeros([self.Nx, self.Ny, self.Nnu])
        for ix in range(self.Nx):
            for iy in range(self.Ny):
                
                if verbose:
                    print('making CIB cube at pixel (%d,%d)'%(ix + 1,iy + 1))
                    
                sp = np.where((xbin_vec==ix) & (ybin_vec==iy))[0]
                redshift = np.asarray(self.df['redshift'])[sp]
                LIR = np.asarray(self.df['LIR'])[sp]
                Umean = np.asarray(self.df['Umean'])[sp]
                DL = np.asarray(self.df['Dlum'])[sp]
                issb  = np.asarray(self.df['issb'])[sp]
                
                Snu_arr = gen_Snu_arr(self.lambda_bins, SED_dict, redshift, LIR, Umean, DL, issb, sum = True)
                Scube_arr[ix,iy,:] = Snu_arr
        
       
        return Scube_arr / dOmega

def _binedge_vectors(line_class):
    '''
    define the binedges of the data cube grid

    Outputs:
    =======
    xbinedge_vec: binedge in x direction [(Nx+1,), deg]
    ybinedge_vec: binedge in y direction [(Ny+1,), deg]
    nu_binedges: binedge in nu direction [(Nnu+1,), GHz]

    line_class.xmin [deg]
    line_class.xmax [deg]
    line_class.ymin [deg]
    line_class.ymax [deg]
    '''
    line_class.xmin = line_class.x_cent - line_class.Nx * line_class.dx / 60. / 2. 
    line_class.xmax = line_class.x_cent + line_class.Nx * line_class.dx / 60. / 2.
    xbinedge_vec = np.linspace(line_class.xmin, line_class.xmax, line_class.Nx + 1)
    if line_class.xmin < 0:
        print('x_min exceed catalog range!!')
    if line_class.xmax > 1.4:
        print('x_max exceed catalog range!!')

    line_class.ymin = line_class.y_cent - line_class.Ny * line_class.dy / 60. / 2. 
    line_class.ymax = line_class.y_cent + line_class.Ny * line_class.dy / 60. / 2.
    ybinedge_vec = np.linspace(line_class.ymin, line_class.ymax, line_class.Ny + 1)
    if line_class.ymin < 0:
        print('y_min exceed catalog range!!')
    if line_class.ymax > 1.4:
        print('y_max exceed catalog range!!')

    return xbinedge_vec, ybinedge_vec

def _select_df(line_class,df):
    '''
    retrive the catalog with sources only within the data cube x,y range
    Inputs:
    =======
    df: input pandas data frame

    Outputs:
    ========
    line_class.df: new data frame with sources inside the cube and columns:
    ['redshift', 'ra', 'dec', 'issb', 'Umean', 'mu',\
           'ICO10', 'ICO21', 'ICO32', 'ICO43', 'ICO54', 'ICO65', 'ICO76', 'ICO87', 'ICII']
    '''
    df_new = df[(df.ra >= line_class.xmin) & (df.ra <= line_class.xmax) & \
                (df.dec >= line_class.ymin) & (df.dec <= line_class.ymax)].copy()
    df_new['LIR'] = df_new['SFR'] * 1e10
    df_new = df_new[['redshift', 'ra', 'dec', 'issb', 'Umean','mu','LIR',\
           'Dlum','ICO10', 'ICO21', 'ICO32', 'ICO43', 'ICO54', 'ICO65', 'ICO76', 'ICO87', 'ICII']].copy()
    line_class.df = df_new
    return

def _bin_label(data_vec, binedges_vec):
    # determine data points are belong to which bin
    # bin label from 0 to len(binedges_vec) - 2, -1 if data outside the bin
    binlabel = np.digitize(data_vec,binedges_vec)-1
    binlabel[data_vec == binedges_vec[0]] = 0
    binlabel[data_vec == binedges_vec[-1]] = len(binedges_vec) - 2
    binlabel[(data_vec < binedges_vec[0]) | (data_vec > binedges_vec[-1])] = -1
    return binlabel