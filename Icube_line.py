import pandas as pd
from cosmo_tools import *

class make_Icube_line():
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
        self.nu_min = min(nu_binedges)
        self.nu_max = max(nu_binedges)
        self.nu_cent = np.median(nu_binedges)
        
        # enfoce nu_binedges from high nu -> low nu (low-z -> high-z)
        if nu_binedges[1] - nu_binedges[0]>0:
            self.nu_binedges = np.flip(nu_binedges, axis=0)
        else:
            self.nu_binedges = nu_binedges
        
        if len(nu_bins)==0:
            self.nu_bins = (self.nu_binedges[:-1] + self.nu_binedges[1:]) / 2
        else:
            self.nu_bins = nu_bins
        _select_df(self,df)
    
    def info(self, linename, jco=1):
        '''
        print out the basic information of the data cube.
        '''
        if linename=='cii':
            freq_rest = spec_lines.CII.to(u.GHz, equivalencies = u.spectral()).value
            name = 'CII'
        elif linename=='co':
            if jco not in np.arange(1,9,1,dtype=int):
                print('jco data not exist! (jco best be in [1,2,...,8])')
                return
            freq_rest = spec_lines.CO(jco).to(u.GHz, equivalencies = u.spectral()).value
            name = 'CO(' + str(jco) + '-' + str(jco-1) + ')'
        
        dnu_cent = np.median(self.nu_binedges[:-1] - self.nu_binedges[1:])
        
        z_cent = (freq_rest / self.nu_cent) - 1
        zmax = (freq_rest / self.nu_min) - 1
        zmin = (freq_rest / self.nu_max) - 1
        zbinedge_vec = (freq_rest / self.nu_binedges) - 1
        Dcmbinedge_vec = cosmo_dist(zbinedge_vec).comoving_distance.value
        dDcmbinedge_vec = Dcmbinedge_vec[1:] - Dcmbinedge_vec[:-1]
        
        z_vec = np.array(self.df['redshift'])
        idx = np.where((z_vec >= zmin) & (z_vec <= zmax))[0]
        
        cd = cosmo_dist(z_cent)
        dx = self.dx * u.arcmin
        dy = self.dy * u.arcmin
        xMpc = (self.Nx * dx * cd.kpc_comoving_per_arcmin).to(u.Mpc / u.h).value
        yMpc = (self.Ny * dy * cd.kpc_comoving_per_arcmin).to(u.Mpc / u.h).value
        zMpc = (cosmo_dist(zmax).comoving_distance - cosmo_dist(zmin).comoving_distance).value
        dxMpc = (dx * cd.kpc_comoving_per_arcmin).to(u.Mpc / u.h).value
        dyMpc = (dy * cd.kpc_comoving_per_arcmin).to(u.Mpc / u.h).value
        dzMpc = np.median(dDcmbinedge_vec)
        
        print('data cube info:')
        print('='*30)
        print('{} x {} x {} voxels [ra, dec, LoS]'.format(self.Nx, self.Ny, self.Nnu))
        print('z (center) = {:.2f}, z range = [{:.2f}, {:.2f}] for {}'.format(z_cent, zmin, zmax, name))

        print('')
        print('survey size:')
        print('{:.2e} x {:.2e} deg ({:.2e} x {:.2e} arcmin) x {:.2e} GHz'\
              .format(self.Nx * self.dx / 60., self.Ny * self.dy / 60., \
                      self.Nx * self.dx, self.Ny * self.dy, self.nu_max - self.nu_min))
        print('{:.2e} x {:.2e} x {:.2e} [Mpc/h]'.format(xMpc, yMpc, zMpc))

        print('')
        print('voxel size (survey center voxel at z = {:2f}):'.format(np.median(zbinedge_vec)))
        print('{:.2e} x {:.2e} deg ({:.2e} x {:.2e} arcmin) x {:.2e} GHz'\
              .format(self.dx / 60., self.dy / 60., self.dx, self.dy, dnu_cent))
        print('{:.2e} x {:.2e} x {:.2e} [Mpc/h]'.format(dxMpc, dyMpc, dzMpc))

        return
    
    def make_line_cube(self, linename, jco = 1):
        '''
        make the I data cube of the line.
        Inputs:
        =======
        linename: 'cii' or 'co' [str]
        jco: CO jco->jco-1 [int, 1~8]
        
        Oputputs:
        =========
        self.z_cent: redshift of self.nu_cent
        self.zmin: redshift of self.nu_min
        self.zmax: redshift of self.nu_max
        self.zbinedges_vec: redshifts of self.nubinedges_vec
        self.Dcmbinedges_vec: comoving distances of self.zbinedges_vec [Mpc/h]
        
        Icube_arr: Intensity data cube [(Nx, Ny, Nnu), Jy/sr]
        Ncube_arr: source count data cube [(Nx, Ny, Nnu)]
        '''
        
        z_vec = np.array(self.df['redshift'])
        ra_vec = np.array(self.df['ra'])
        dec_vec = np.array(self.df['dec'])
  
        # retrieve I_vec [Jy GHz] of the line
        if linename=='cii':
            I_vec = np.array(self.df['ICII'])
            freq_rest = spec_lines.CII.to(u.GHz, equivalencies = u.spectral()).value
            name = 'CII'
        elif linename=='co':
            if jco not in np.arange(1,9,1,dtype=int):
                print('jco data not exist! (jco best be in [1,2,...,8])')
                return
            I_vec = np.array(self.df['ICO' + str(jco) + str(jco-1)])
            freq_rest = spec_lines.CO(jco).to(u.GHz, equivalencies = u.spectral()).value
            name = 'CO(' + str(jco) + '-' + str(jco-1) + ')'
            
        # add the z info to the data 
        self.z_cent = (freq_rest / self.nu_cent) - 1
        self.zmax = (freq_rest / self.nu_min) - 1
        self.zmin = (freq_rest / self.nu_max) - 1
        if self.zmin < 0:
            print('zmin < 0 !!')
        if self.zmax > 10:
            print('zmax exceed catalog range (z=10)!!')
        
        self.zbinedge_vec = (freq_rest / self.nu_binedges) - 1
        self.zbins = (self.zbinedge_vec[1:] + self.zbinedge_vec[:-1]) / 2
        
        
        Dcmbinedge_vec = cosmo_dist(self.zbinedge_vec).comoving_distance.value

        Dcmbinedge_vec = np.ones_like(self.zbinedge_vec) * -1.
        sp = np.where(self.zbinedge_vec > 0)[0]
        Dcmbinedge_vec[sp] = cosmo_dist(self.zbinedge_vec[sp]).comoving_distance.value
        self.Dcmbinedge_vec = Dcmbinedge_vec
        
        
        # select the sources in LoS direction
        idx = np.where((z_vec >= self.zmin) & (z_vec <= self.zmax))[0]
        I_vec = I_vec[idx]
        ra_vec = ra_vec[idx]
        dec_vec = dec_vec[idx]
        z_vec = z_vec[idx]
        
        # make the data cube
        xbin_vec = _bin_label(ra_vec,self.xbinedge_vec)
        ybin_vec = _bin_label(dec_vec,self.ybinedge_vec)
        zbin_vec = _bin_label(z_vec,self.zbinedge_vec)

        dfbin = pd.DataFrame({'xbin':xbin_vec, 'ybin':ybin_vec, 'zbin':zbin_vec, 'I':I_vec})
        grouped = dfbin.groupby(['xbin','ybin','zbin'],as_index=False)['I'].agg(['sum','count'])
        grouped = grouped.reset_index()
        
        Icube_arr = np.zeros([self.Nx, self.Ny, self.Nnu])
        Ncube_arr = np.zeros([self.Nx, self.Ny, self.Nnu])

        dOmega = (self.dx * u.arcmin).to(u.rad).value * (self.dy * u.arcmin).to(u.rad).value
        dnu_vec = self.nu_binedges[:-1]-self.nu_binedges[1:]
        _,_,dnu_arr= np.meshgrid(np.ones(self.Nx),np.ones(self.Ny),dnu_vec)
        dnudomg_arr = np.swapaxes(dnu_arr,0,1) * dOmega

        xbin = np.asarray(grouped['xbin'])
        ybin = np.asarray(grouped['ybin'])
        zbin = np.asarray(grouped['zbin'])
        Icube_arr[xbin,ybin,zbin] = np.asarray(grouped['sum'])
        Icube_arr /= dnudomg_arr
        Ncube_arr[xbin,ybin,zbin] = np.asarray(grouped['count'])

        return Icube_arr, Ncube_arr
    
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
    ['redshift', 'ra', 'dec', ...]
    '''
    df_new = df[(df.ra >= line_class.xmin) & (df.ra <= line_class.xmax) & \
                (df.dec >= line_class.ymin) & (df.dec <= line_class.ymax)].copy()
    df_new = df_new[['redshift', 'ra', 'dec', 'issb', 'Umean','mu',\
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