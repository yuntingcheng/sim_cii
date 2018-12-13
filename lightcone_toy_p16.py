import pandas as pd
from survey_params import *
from scipy import interpolate

def sim_zsrc_Neff(Nsamp, dth, zmin = 0, zmax = 10, dz = 0.001, Neff_scale = 1):
    '''
    Simulate the source redshift in a light cone with Neff(z)
    Neff is derived from Popping 16 Schechter LF model. 
    For z <=5, use CO(1-0) fitting function, for z > 5, use CII fitting funciton.
    
    Inputs:
    =======
    Nsamp: number of lightcone to generate
    dth: lightcone size dth^2[arcmin]
    zmin,zmax: range of z to be sampled
    dz: z resolution in the sim
    Neff_scale: scale the overall Neff
    
    Output:
    =======
    zsrc_all = Nsamp element list, each element is a list of z of source in a lightcone
    '''
    df = pd.read_csv('data_internal/P16NeffLs.txt')
    z_dat = df['z'].values
    Neff_dat = df['CO10_Neff'].values
    Neff_dat[z_dat > 5] = df['CII_Neff'].values[z_dat > 5]
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

def add_line_bin(df, nu_binedges, juse = [0,1,2,3,4,5,6]):
    '''
    Given a df with a column of 'redshift', and the survey nu_binedges, add the columns
    'binCO##' or 'binCII' specified in which nu bin we can observed the line.
    
    Inputs:
    =======
    df: df with each source a row. Contain column 'redshift'
    nu_binedges: frequency bin edges [GHz]
    juse: j lines to compute
    
    Output:
    =======
    df: df with columns 'binCO##', 'binCII'
    '''
    binlabel_arr = np.zeros([len(df), len(juse)], dtype=int)
    for jidx,jco in enumerate(juse):
        if jco ==0:
            name = 'CII'
            nu_arr = np.asarray(spec_lines.CII.to(u.GHz, equivalencies=u.spectral()).value \
                                 / (1 + df['redshift']))
        else:
            name = 'CO' + str(jco) + str(jco-1)
            nu_arr =  np.asarray(spec_lines.CO(jco).to(u.GHz, equivalencies=u.spectral()).value \
                                 / (1 + df['redshift']))
        
        # binlabels: 0 ~ len(nu_binedges)-2
        binlabel = np.digitize(nu_arr,nu_binedges) - 1
        binlabel[nu_arr == nu_binedges[0]] = 0
        binlabel[nu_arr == nu_binedges[-1]] = len(nu_binedges) - 2
        binlabel[(nu_arr < nu_binedges[-1]) | (nu_arr > nu_binedges[0])] = -1
        df['bin' + name] = binlabel
        binlabel_arr[:,jidx] = binlabel

    return df, binlabel_arr

def add_line_flux(df, juse = [0,1,2,3,4,5,6], muL = [], sigL = []):
    '''
    Given a df with a column of 'redshift', calculate line flux and add the columns
    'FCO##' or 'FCII'. The line instrinsic luminosity is the L* in Popping 16.
    muL, sigL are the bias and scatter in intrinsic luminosity. 
    
    Inputs:
    =======
    df: df with each source a row. Contain column 'redshift'
    juse: j lines to compute
    muL: intrinsic luminosity bias for all sources. Same dimemsion as juse. unit: L*
    sigL: intrinsic luminosity Gaussian scatter for all sources. Same dimemsion as juse. unit: L*
    
    Output:
    =======
    df: df with columns 'FCO##', 'FCII' [Jy GHz]
    '''    
    if len(sigL) == 0:
        sigL = np.zeros_like(juse)
    elif len(sigL) != len(juse):
        raise ValueError('sigL and juse does not have the same dimension!!!')
    if len(muL) == 0:
        muL = np.zeros_like(juse)
    elif len(muL) != len(juse):
        raise ValueError('sigL and juse does not have the same dimension!!!')

    dfdat = pd.read_csv('data_internal/P16NeffLs.txt')
    z_dat = dfdat['z'].values
    z_vec = df['redshift']
    DL_vec = cosmo.luminosity_distance(z_vec)
    
    F_arr = np.zeros([len(df), len(juse)])
    for jidx,jco in enumerate(juse):
        
        if jco == 0:
            name = 'CII'
        else:
            name = 'CO' + str(jco) + str(jco-1)
        Ls_dat = dfdat[name + '_Ls'].values
        L_vec = 10**np.interp(z_vec, z_dat, np.log10(Ls_dat))
        L_vec = L_vec * (1 + np.random.normal(muL[jidx], sigL[jidx], len(L_vec)))
        F_vec = L_vec * u.Lsun / 4 / np.pi / DL_vec**2
        F_vec = F_vec.to(u.Jy * u.GHz).value
        df['F' + name] = F_vec
        F_arr[:,jidx] = F_vec
    return df, F_arr

def Ivox_from_zsrc(zsrc_all, dth, nu_binedges, juse, jtarg, muL = [], sigL = [], Lratio = [], verbose = 0):
    '''
    Given list of source redshift and nu_binedges, calculate the inensity of the light cone.
    
    Inputs:
    =======
    zsrc_all[list]: Nsamp element list, each element is a list of z of source in a lightcone
    dth: pixel size dth^2 [arcmin]
    nu_binedges: [GHz]
    juse[list]: list of lines in the light cone
    jtarg[int or list]: targeting line
    muL: pass to add_line_flux
    sigL: pass to add_line_flux
    Lratio[list]: the intrinsic line luminosity of the individual source is Lratio time Lsrc.
                  Lratio has to be same dimension of zsrc_all. Default: all 1's.

    Outputs:
    ========
    I_vec_all: intensity from all the lines, Nsamp x Nnu array [Jy/sr]
    I_vec_targ: intensity from target lines, Nsamp x Nnu array [Jy/sr]
    '''
    
    if type(jtarg) is int:
        if jtarg not in juse:
            raise ValueError("jtarg must be in juse.")
    else:
        if not set(jtarg).issubset(juse):
            raise ValueError("jtarg must be in juse.")        

    if len(Lratio) != len(zsrc_all) and len(Lratio) !=0:
        raise ValueError('zsrc_all and Lratio does not have the same dimension!!!')

    Nnu = len(nu_binedges) - 1
    Nset = len(zsrc_all)
    dsr = ((dth * u.arcmin).to(u.rad))**2
    dnus = abs(nu_binedges[1:] - nu_binedges[:-1])
    
    usename = []
    for jco in juse:
        if jco==0:
            usename.extend(['binCII'])
        else:
            usename.extend(['binCO' + str(jco) + str(jco-1)])
    
    I_vec_all = np.zeros([Nset,Nnu])
    
    if type(jtarg) is int:
        idxtarg = juse.index(jtarg)
        I_vec_targ = np.zeros([Nset,Nnu])
    else:
        idxtarg_vec = [juse.index(jj) for jj in jtarg]
        I_vec_targ = np.zeros([len(jtarg),Nset,Nnu])
        
    for i in range(Nset):
        
        if len(zsrc_all[i])==0:
            I_vec_all[i,:] = 0.
            if type(jtarg) is int:
                I_vec_targ[i,:] = 0.
            else:
                I_vec_targ[:,i,:] = 0.
        
        else:
            if len(Lratio) == 0:
                Lri = np.ones(len(zsrc_all[i]))
            elif len(Lratio[i]) != len(zsrc_all[i]):
                raise ValueError('light cone %d zsrc_all and Lratio size not match!!!'%i)
            else:
                Lri = Lratio[i]
            Lri_arr = np.tile(np.asarray(Lri),(len(juse),1)).T
            df = pd.DataFrame(zsrc_all[i],columns=['redshift'])
            _, F_arr = add_line_flux(df, juse = juse, muL = muL, sigL = sigL)
            _, bin_arr = add_line_bin(df, nu_binedges, juse = juse)
            dnu_arr = np.zeros_like(bin_arr,dtype=float)
            dnu_arr[bin_arr!=-1] = dnus[bin_arr[bin_arr!=-1]]
            I_arr = np.zeros_like(bin_arr,dtype=float)
            I_arr[bin_arr!=-1] = F_arr[bin_arr!=-1] * Lri_arr[bin_arr!=-1]/dnu_arr[bin_arr!=-1] / dsr

            I_vec = np.histogram(bin_arr,bins = np.arange(-0.5,Nnu,1), weights=I_arr)[0]
            I_vec_all[i,:] = I_vec
            
            if type(jtarg) is int:
                I_vec = np.histogram(bin_arr[:,idxtarg],bins = np.arange(-0.5,Nnu,1), \
                                     weights=I_arr[:,idxtarg])[0]
                I_vec_targ[i,:] = I_vec
            else:
                for jj,idxtarg in enumerate(idxtarg_vec):
                    I_vec = np.histogram(bin_arr[:,idxtarg],bins = np.arange(-0.5,Nnu,1), \
                                         weights=I_arr[:,idxtarg])[0]
                    I_vec_targ[jj,i,:] = I_vec
        
        if verbose:
            if (i+1)%100==0:
                print('produce light cone %d/%d (%d %%)'%(i+1,Nset,(i+1)*100./Nset))
    
    return I_vec_all,I_vec_targ

def gen_Ipred(z_coords, N_arr, dth, nu_binedges, juse, jtarg, verbose = 0):
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
                    (zsrc_all, dth, nu_binedges, juse, jtarg, Lratio = Lratio, verbose = 0)
    
    return Ipred_all, Ipred_targ

def zround(z_true, z_coords):
    zround = []
    zidx = []
    for iset in range(len(z_true)):
        idx = [(np.abs(z - z_coords)).argmin() for z in z_true[iset]]
        zround.append(z_coords[idx])
        zidx.append(idx)
    return zround, zidx
    
def zlist_to_N(zsrc, z_coords_all, I_coords_all, z_idx, sp2):
    I_bl = np.copy(I_coords_all)
    I_bl[I_bl > 0] = 1
    Nall = np.zeros([len(zsrc),len(z_idx)])
    for iset in range(len(zsrc)):
        N = np.zeros_like(z_idx)
        idx_vec = np.array([(np.abs(zcii - z_coords_all)).argmin() for zcii in zsrc[iset]])
        idx_vec = idx_vec[np.sum(I_bl[:,idx_vec], axis = 0) >= 2]
        for idx in idx_vec:
            if idx <= min(z_idx[sp2]):
                N[0] += 1
            elif idx >= max(z_idx[sp2]):
                N[-1] += 1
            else:
                # pick the two neaest dictionary
                idx1 = idx - z_idx[sp2]
                idx1 = idx1[idx1 > 0]
                idx1 = idx - min(idx1)

                idx2 = idx - z_idx[sp2]
                idx2 = idx2[idx2 < 0]
                idx2 = idx - max(idx2)
                
                # get binary dict to see which one match
                if np.array_equal(I_bl[:,idx1],I_bl[:,idx]):
                    N[np.where(idx1 == z_idx)] += 1
                else:
                    N[np.where(idx2 == z_idx)] += 1

        Nall[iset,:] = N
    return Nall

def gen_lightcone_toy(Nlc, dth, nu_binedges, sp2, z_coords_all, I_coords_all, z_idx, juse, jtarg, Neff_scale = 1):
    
    zsrc = sim_zsrc_Neff(Nlc, dth, Neff_scale = Neff_scale)
    Ntrue = zlist_to_N(zsrc, z_coords_all, I_coords_all, z_idx, sp2)
    Ntrue = Ntrue[:,sp2]
    Itrue_all, Itrue_targ = Ivox_from_zsrc(zsrc, dth, nu_binedges, juse, jtarg, verbose=0)
    
    return Ntrue, Itrue_all, Itrue_targ

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

    dfNeff = pd.read_csv('data_internal/P16NeffLs.txt')
    z_dat = dfNeff['z'].values
    Neff_dat = dfNeff['CO10_Neff'].values
    Neff_dat[z_dat > 5] = dfNeff['CII_Neff'].values[z_dat > 5]
    Neff_dat *= dth**2
    tck = interpolate.splrep(z_dat, Neff_dat, s=0)
    zbinedges_arr = np.linspace(zmin,zmax,50)
    zbins_arr = (zbinedges_arr[:-1] + zbinedges_arr[1:]) / 2
    Nbins_arr = interpolate.splev(zbins_arr, tck, der=0)
    Nbins_arr[Nbins_arr < 0] = 0
    Ntot = np.sum(Nbins_arr) * (zbinedges_arr[1] - zbinedges_arr[0])

    return Ntot

def Neff_tot_idx(dth, z_coords):
    '''
    The integrated Neff in z range specified by z_coords.
    z_coords must be a linspace array, but not need to be continuous
    '''    
    
    dfNeff = pd.read_csv('data_internal/P16NeffLs.txt')
    z_dat = dfNeff['z'].values
    Neff_dat = dfNeff['CO10_Neff'].values
    Neff_dat[z_dat > 5] = dfNeff['CII_Neff'].values[z_dat > 5]
    Neff_dat *= dth**2
    tck = interpolate.splrep(z_dat, Neff_dat, s=0)

    Nbins_arr = interpolate.splev(z_coords, tck, der=0)
    Nbins_arr[Nbins_arr < 0] = 0
    
    dz_vec = z_coords[1:] - z_coords[:-1]
    values,counts = np.unique(dz_vec, return_counts=True)
    dz = values[np.argmax(counts)]   
    
    Ntot = np.sum(Nbins_arr) * dz
    
    return Ntot