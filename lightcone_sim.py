import pandas as pd
from survey_params import *
from scipy import interpolate

def sim_zsrc_Neff(Nsamp, dth, zmin = 0, zmax = 10, dz = 0.001, Neff_scale = 1.):
    '''
    Simulate the source redshift in a light cone with Neff(z)
    
    Inputs:
    =======
    Nsamp: number of lightcone to generate
    dth: lightcone size dth^2[arcmin]
    zmin,zmax: range of z to be sampled
    dz: z resolution in the sim
    
    Output:
    =======
    zsrc_all = Nsamp element list, each element is a list of z of source in a lightcone
    '''
    dfNeff = pd.read_csv('data_internal/Neff.txt')
    z_dat = dfNeff['z'].values
    Neff_dat = dfNeff['Mstar'].values
    Neff_dat *= dz * dth**2
    z_vec = np.arange(zmin,zmax,dz)
    z_vec = z_vec[1:]
    Neff_vec = np.interp(z_vec,z_dat,Neff_dat)
    
    Neff_vec *= Neff_scale
    
    zsrc_all = []
    for i in range(Nsamp):
        samps = np.random.poisson([Neff_vec])[0]
        maxN = np.max(samps)
        zsamps = []
        for j in range(1,maxN+1,1):
            zsamps.extend(z_vec[samps>=j])
        zsamps = np.round(np.asarray(zsamps)/dz)*dz
        zsrc_all.append(zsamps)
    return zsrc_all

def add_line_bin(df,nu_binedges):
    nu_arr =  np.asarray(spec_lines.CII.to(u.GHz, equivalencies=u.spectral()).value / (1 + df['redshift']))
    binlabel = np.digitize(nu_arr,nu_binedges) - 1 # binlabels: 0 ~ len(nu_binedges)-2
    binlabel[nu_arr == nu_binedges[0]] = 0
    binlabel[nu_arr == nu_binedges[-1]] = len(nu_binedges) - 2
    binlabel[(nu_arr < nu_binedges[-1]) | (nu_arr > nu_binedges[0])] = -1
    df['binCII'] = binlabel
    for jco in range(1,9):
        nu_arr =  np.asarray(spec_lines.CO(jco).to(u.GHz, equivalencies=u.spectral()).value \
                             / (1 + df['redshift']))
        binlabel = np.digitize(nu_arr,nu_binedges) - 1
        binlabel[nu_arr == nu_binedges[0]] = 0
        binlabel[nu_arr == nu_binedges[-1]] = len(nu_binedges) - 2
        binlabel[(nu_arr < nu_binedges[-1]) | (nu_arr > nu_binedges[0])] = -1
        df['binCO' + str(jco) + str(jco-1)] = binlabel
    return df

def Ivox_from_zsrc(zsrc_all, dth, nu_binedges, juse, jtarg, Lsrc, sigL = [], Lratio = [], verbose = 0):
    '''
    Given list of source redshift and nu_binedges, calculate the inensity of the light cone.
    
    Inputs:
    =======
    zsrc_all[list]: Nsamp element list, each element is a list of z of source in a lightcone
    dth: pixel size dth^2 [arcmin]
    nu_binedges: [GHz]
    juse[list]: list of lines in the light cone
    jtarg[int]: targeting line 
    Lsrc[list]: the intrinsic luminosity of line for all sources [L_sun] 
    Lratio[list]: the intrinsic line luminosity of the individual source is Lratio time Lsrc.
                  Lratio has to be same dimension of zsrc_all. Default: all 1's.
    '''
    
    if jtarg not in juse:
        raise ValueError("jtarg must be in juse.")
        
    if len(Lratio) == 0:
        Lratio = np.ones(len(zsrc_all))
    elif len(Lratio) != len(zsrc_all):
        raise ValueError('zsrc_all and Lratio does not have the same dimension!!!')
    
    if len(sigL) == 0:
        sigL = np.zeros_like(Lsrc)
    elif len(sigL) != len(Lsrc):
        raise ValueError('sigL and Lsrc does not have the same dimension!!!')
    
    Nnu = len(nu_binedges) - 1
    Nset = len(zsrc_all)
    dsr = ((dth * u.arcmin).to(u.rad))**2
    dnus = abs(nu_binedges[1:] - nu_binedges[:-1])
    L_vec = Lsrc * u.Lsun
    
    idxtarg = juse.index(jtarg)
    
    usename = []
    for jco in juse:
        if jco==0:
            usename.extend(['binCII'])
        else:
            usename.extend(['binCO' + str(jco) + str(jco-1)])
    
    I_vec_all = np.zeros([Nset,Nnu])
    I_vec_targ = np.zeros([Nset,Nnu])
    for i in range(Nset):
        if len(zsrc_all[i])==0:
            I_vec_all[i,:] = 0.
            I_vec_targ[i,:] = 0.
        else:
            Lri = np.asarray(Lratio[i])
            dfi = pd.DataFrame(zsrc_all[i],columns=['redshift'])
            dfi = add_line_bin(dfi,nu_binedges)
            z_vec = dfi['redshift'].values
            DL_vec = cosmo.luminosity_distance(z_vec)
            A_vec = 1. / 4 / np.pi / DL_vec**2
            A_vec *= Lri
            L_vec_i = L_vec + np.random.normal(np.zeros_like(L_vec),sigL)*u.Lsun
            F_arr = A_vec.reshape(-1,1)*L_vec_i.reshape(1,-1)
            F_arr = F_arr.to(u.Jy * u.GHz).value

            bin_arr = dfi[usename].values
            dnu_arr = np.zeros_like(bin_arr,dtype=float)
            dnu_arr[bin_arr!=-1] = dnus[bin_arr[bin_arr!=-1]]

            I_arr = np.zeros_like(bin_arr,dtype=float)
            I_arr[dnu_arr!=0] = F_arr[dnu_arr!=0]/dnu_arr[dnu_arr!=0] / dsr


            I_vec = np.histogram(bin_arr,bins = np.arange(-0.5,Nnu,1), weights=I_arr)[0]
            I_vec_all[i,:] = I_vec

            I_vec = np.histogram(bin_arr[:,idxtarg],bins = np.arange(-0.5,Nnu,1), \
                                 weights=I_arr[:,idxtarg])[0]
            I_vec_targ[i,:] = I_vec
        
        if verbose:
            if (i+1)%100==0:
                print('produce light cone %d/%d (%d %%)'%(i+1,Nset,(i+1)*100./Nset))
    return I_vec_all,I_vec_targ

def gen_Ipred(z_coords, N_arr, dth, nu_binedges, juse, jtarg, Lsrc, verbose = 0):
    '''
    Generate I_arr with the N_arr from sparse approx.
    
    Inputs:
    =======
    z_coords[arr]: Nsamp element list, each element is a list of z of source in a lightcone
    N_arr[arr, Nset x len(z_coords)]: 
    dth: pixel size dth^2 [arcmin]
    nu_binedges: [GHz]
    juse[list]: list of lines in the light cone
    jtarg[int]: targeting line 
    Lsrc[list]: the intrinsic luminosity of line for all sources [L_sun] 
    Lratio[list]: the intrinsic line luminosity of the individual source is Lratio time Lsrc.
                  Lratio has to be same dimension of zsrc_all. Default: all 1's.
    '''

    Nsamp, Nz = N_arr.shape
     
    if Nz != len(z_coords):
        raise ValueError('N_arr 2nd dimension does not match len(z_coords).')
        
    zsrc_all = []
    Lratio = []
    for i in range(Nsamp):
        zsrc_all.append(z_coords)
        Lratio.append(N_arr[i,:])
        
    Ipred_all, Ipred_targ = Ivox_from_zsrc\
                    (zsrc_all, dth, nu_binedges, juse, jtarg, Lsrc, Lratio = Lratio,verbose = 0)
    
    return Ipred_all, Ipred_targ

def zround(z_true, z_coords):
    zround = []
    zidx = []
    for iset in range(len(z_true)):
        idx = [(np.abs(z - z_coords)).argmin() for z in z_true[iset]]
        zround.append(z_coords[idx])
        zidx.append(idx)
    return zround, zidx
    
def zlist_to_N(zsrc, z_coords_all, z_idx):
    Nall = np.zeros([len(zsrc),len(z_idx)])
    for iset in range(len(zsrc)):
        N = np.zeros_like(z_idx)
        idx = [(np.abs(zcii - z_coords_all)).argmin() for zcii in zsrc[iset]]
        for i in idx:
            N[np.where(i == z_idx)] += 1
            
        Nall[iset,:] = N
    return Nall

def Neff_tot( dth, zmin = 0, zmax = 10):
    '''
    Get the total Neff from zmin to zmax

    Inputs:
    =======
    dth: lightcone size dth^2[arcmin]
    zmin,zmax: range of z to be sampled
    
    Output:
    =======
    Neff: total Neff
    '''

    dfNeff = pd.read_csv('data_internal/Neff.txt')
    z_dat = dfNeff['z'].values
    Neff_dat = dfNeff['Mstar'].values
    Neff_dat *= dth**2
    tck = interpolate.splrep(z_dat, Neff_dat, s=0)
    zbinedges_arr = np.linspace(0,10,50)
    zbins_arr = (zbinedges_arr[:-1] + zbinedges_arr[1:]) / 2
    Nbins_arr = interpolate.splev(zbins_arr, tck, der=0)
    Nbins_arr[Nbins_arr < 0] = 0
    Ntot = np.sum(Nbins_arr) * (zbinedges_arr[1] - zbinedges_arr[0])

    return Ntot

def Neff_tot_idx(dth, z_coords_all, z_idx):
    '''
    The integrated Neff in z range specified by z_idx in z_coords_all.
    z_coords_all must be a linspace array
    '''
    dfNeff = pd.read_csv('data_internal/Neff.txt')
    z_dat = dfNeff['z'].values
    Neff_dat = dfNeff['Mstar'].values
    Neff_dat *= dth**2
    tck = interpolate.splrep(z_dat, Neff_dat, s=0)

    Nbins_arr = interpolate.splev(z_coords_all, tck, der=0)
    Nbins_arr[Nbins_arr < 0] = 0
    dz = z_coords_all[1] - z_coords_all[0]
    
    Ntot = np.sum(Nbins_arr[z_idx]) * dz

    return Ntot
