import pandas as pd
from sklearn import preprocessing
from model_F17 import *

def sim_zsrc_Neff(Nsamp, dth, zmin = 0, zmax = 10, dz = 0.001, Neff_scale = 1, model = 'Be13'):
    '''
    Simulate the source redshift in a light cone with Neff(z)
    Neff is derived from Fonseca 17 Schechter LF model. 
    
    Inputs:
    =======
    Nsamp: number of lightcone to generate
    dth: lightcone size dth^2[arcmin]
    zmin,zmax: range of z to be sampled
    dz: z resolution in the sim
    Neff_scale: scale the overall Neff
    Model: SFR model -- 'F17' or 'Be13'
    Output:
    =======
    zsrc_all = Nsamp element list, each element is a list of z of source in a lightcone
    '''
    
    if model == 'F17':
        df = pd.read_csv('data_internal/F17NeffSFRs.txt')
    elif model == 'Be13':
        df = pd.read_csv('data_internal/Be13NeffSFRs.txt')
    else:
        raise ValueError("Model name incorrect.")
    
    z_dat = df['z'].values
    Neff_dat = df['Neff'].values
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

def sim_SFRlc_F17(Nsamp, dth, zbinedges, SFRbinedges, SFRf_arr, Neff_scale = 1, model = 'Be13'):
    '''
    Simulate the light cone with F17 Sim fit luminosity function. 
    
    Inputs:
    =======
    Nsamp: number of lightcone to generate
    dth: lightcone size dth^2[arcmin]
    zbinedges: the redshift bin edges
    SFRbinedges: the luminosity bin edges to sample the LF with Poisson distribution 
            [unit: dimensionless, SFR* of Schechter func]
    SFRf_arr: pre-computed SFR func at the given zbinedges, Lbinedges [h^3 / Mpc^3]
    
    Output:
    =======
    L_arr [Lsun]: Nsamp x Nz, L_arr(i,j) is the intrinsic luminoisty [Lsun] in lightcone j redshift j
    zbins: redshift bins for L_arr
    '''
    zbins = (zbinedges[1:] + zbinedges[:-1]) / 2
    Nz = len(zbins)
    
    CDedges = cosmo_dist(zbinedges).comoving_distance
    dCD = CDedges[1:] - CDedges[:-1]
    dth = dth * u.arcmin
    dtrans = cosmo_dist(zbins).kpc_comoving_per_arcmin * dth
    dV_zvec = dCD.to(u.Mpc / u.h) * dtrans.to(u.Mpc / u.h)**2
    
    SFRbins = np.sqrt(SFRbinedges[1:] * SFRbinedges[:-1])
    NSFR = len(SFRbins)
    dSFR_lvec = (SFRbinedges[1:] - SFRbinedges[:-1])

    if model == 'F17':
        df = pd.read_csv('data_internal/F17NeffSFRs.txt')
    elif model == 'Be13':
        df = pd.read_csv('data_internal/Be13NeffSFRs.txt')
    else:
        raise ValueError("Model name incorrect.")

    z_dat = df['z'].values
    sfrs_dat = df['SFRs'].values
    SFRs_vec = np.interp(zbins, z_dat, sfrs_dat)
    
    SFR_arr = np.zeros([Nsamp, Nz])
    N_arr = np.zeros([Nsamp, Nz])
    for iz, z in enumerate(zbins):
        Phi_lvec = SFRf_arr[iz,:] * Neff_scale
        Navg_lvec = Phi_lvec * dV_zvec[iz].value * dSFR_lvec
        N_vec = np.random.poisson(Navg_lvec, size = (Nsamp, NSFR))
        SFRtot_svec = np.matmul(N_vec, SFRbins.reshape([NSFR,-1])).flatten()
        N_arr[:,iz] = SFRtot_svec
        SFR_arr[:,iz] = SFRtot_svec * SFRs_vec[iz]
        
    return N_arr, zbins, SFR_arr

def add_line_bin(df, nu_binedges, line_use = ['Lya', 'Ha', 'Hb', 'OII', 'OIII']):
    '''
    Given a df with a column of 'redshift', and the survey nu_binedges, add the columns
    'binLINENAME' specified in which nu bin we can observed the line.
    
    Inputs:
    =======
    df: df with each source a row. Contain column 'redshift'
    nu_binedges: frequency bin edges [GHz]
    line_use: lines to compute
    
    Output:
    =======
    df: df with columns 'binLINENAME'
    '''
    binlabel_arr = np.zeros([len(df), len(line_use)], dtype=int)
    for jidx, name in enumerate(line_use):
        if name == 'Lya':
            nu_arr = np.asarray(spec_lines.Lya.to(u.GHz, equivalencies=u.spectral()).value \
                                 / (1 + df['redshift']))
        elif name == 'Ha':
            nu_arr = np.asarray(spec_lines.Ha.to(u.GHz, equivalencies=u.spectral()).value \
                                 / (1 + df['redshift']))
        elif name == 'Hb':
            nu_arr = np.asarray(spec_lines.Hb.to(u.GHz, equivalencies=u.spectral()).value \
                                 / (1 + df['redshift']))
        elif name == 'OII':
            nu_arr = np.asarray(spec_lines.OII.to(u.GHz, equivalencies=u.spectral()).value \
                                 / (1 + df['redshift']))
        elif name == 'OIII':
            nu_arr = np.asarray(spec_lines.OIII.to(u.GHz, equivalencies=u.spectral()).value \
                                 / (1 + df['redshift']))
        else:
            raise ValueError('Line name %s is invalid.'%name)
            
        # binlabels: 0 ~ len(nu_binedges)-2
        binlabel = np.digitize(nu_arr,nu_binedges) - 1
        binlabel[nu_arr == nu_binedges[0]] = 0
        binlabel[nu_arr == nu_binedges[-1]] = len(nu_binedges) - 2
        binlabel[(nu_arr < nu_binedges[-1]) | (nu_arr > nu_binedges[0])] = -1
        df['bin' + name] = binlabel
        binlabel_arr[:,jidx] = binlabel

    return df, binlabel_arr

def add_line_flux(df, line_use = ['Lya', 'Ha', 'Hb', 'OII', 'OIII'], muL = [], sigL = [], model = 'Be13'):
    '''
    Given a df with a column of 'redshift', calculate line flux and add the columns
    'FCO##' or 'FCII'. The line instrinsic luminosity is the L* in Fonseca 17.
    muL, sigL are the bias and scatter in intrinsic luminosity. 
    
    Inputs:
    =======
    df: df with each source a row. Contain column 'redshift'
    line_use: lines to compute
    muL: intrinsic luminosity bias for all sources. Same dimemsion as line_use. unit: L*
    sigL: intrinsic luminosity Gaussian scatter for all sources. Same dimemsion as line_use. unit: L*
    
    Output:
    =======
    df: df with columns 'FCO##', 'FCII' [Jy GHz]
    '''    
    if len(sigL) == 0:
        sigL = np.zeros(len(line_use))
    elif len(sigL) != len(line_use):
        raise ValueError('sigL and line_use does not have the same dimension!!!')
    if len(muL) == 0:
        muL = np.zeros(len(line_use))
    elif len(muL) != len(line_use):
        raise ValueError('sigL and line_use does not have the same dimension!!!')

    if model == 'F17':
        dfdat = pd.read_csv('data_internal/F17NeffSFRs.txt')
    elif model == 'Be13':
        dfdat = pd.read_csv('data_internal/Be13NeffSFRs.txt')
    else:
        raise ValueError("Model name incorrect.")

    z_dat = dfdat['z'].values
    z_vec = df['redshift']
    DL_vec = cosmo.luminosity_distance(z_vec)
    
    F_arr = np.zeros([len(df), len(line_use)])
    for jidx,line_name in enumerate(line_use):
        Ls_dat = dfdat[line_name + '_Ls'].values
        L_vec = 10**np.interp(z_vec, z_dat, np.log10(Ls_dat))
        L_vec = L_vec * (1 + np.random.normal(muL[jidx], sigL[jidx], len(L_vec)))
        F_vec = L_vec * u.Lsun / 4 / np.pi / DL_vec**2
        F_vec = F_vec.to(u.Jy * u.GHz).value
        df['F' + line_name] = F_vec
        F_arr[:,jidx] = F_vec
        
    return df, F_arr

def Ivox_from_zsrc(zsrc_all, dth, nu_binedges, line_use, line_targ, \
                   muL = [], sigL = [], Lratio = [], model = 'Be13', verbose = 0):
    '''
    Given list of source redshift and nu_binedges, calculate the inensity of the light cone.
    
    Inputs:
    =======
    zsrc_all[list]: Nsamp element list, each element is a list of z of source in a lightcone
    dth: pixel size dth^2 [arcmin]
    nu_binedges: [GHz]
    line_use[str list]: list of lines in the light cone
    line_targ[str or str list]: targeting line
    muL: pass to add_line_flux
    sigL: pass to add_line_flux
    Lratio[list]: the intrinsic line luminosity of the individual source is Lratio time Lsrc.
                  Lratio has to be same dimension of zsrc_all. Default: all 1's.

    Outputs:
    ========
    I_vec_all: intensity from all the lines, Nsamp x Nnu array [Jy/sr]
    I_vec_targ: intensity from target lines, Nsamp x Nnu array [Jy/sr]
    '''
    
    if type(line_targ) is str:
        if line_targ not in line_use:
            raise ValueError("line_targ must be in line_use.")
    else:
        if not set(line_targ).issubset(line_use):
            raise ValueError("line_targ must be in line_use.")        

    if len(Lratio) != len(zsrc_all) and len(Lratio) !=0:
        raise ValueError('zsrc_all and Lratio does not have the same dimension!!!')

    Nnu = len(nu_binedges) - 1
    Nset = len(zsrc_all)
    dsr = ((dth * u.arcmin).to(u.rad))**2
    dnus = abs(nu_binedges[1:] - nu_binedges[:-1])
    
    usename = []
    for name in line_use:
        usename.extend(['bin' + name])

    I_vec_all = np.zeros([Nset,Nnu])
    
    if type(line_targ) is str:
        idxtarg = line_use.index(line_targ)
        I_vec_targ = np.zeros([Nset,Nnu])
    else:
        idxtarg_vec = [line_use.index(jj) for jj in line_targ]
        I_vec_targ = np.zeros([len(line_targ),Nset,Nnu])
       
    for i in range(Nset):
        
        if len(zsrc_all[i])==0:
            I_vec_all[i,:] = 0.
            if type(line_targ) is str:
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
            Lri_arr = np.tile(np.asarray(Lri),(len(line_use),1)).T
            df = pd.DataFrame(zsrc_all[i],columns=['redshift'])
            _, F_arr = add_line_flux(df, line_use = line_use, muL = muL, sigL = sigL, model = model)
            _, bin_arr = add_line_bin(df, nu_binedges, line_use = line_use)
            dnu_arr = np.zeros_like(bin_arr,dtype=float)
            dnu_arr[bin_arr!=-1] = dnus[bin_arr[bin_arr!=-1]]
            I_arr = np.zeros_like(bin_arr,dtype=float)
            I_arr[bin_arr!=-1] = F_arr[bin_arr!=-1] * Lri_arr[bin_arr!=-1]/dnu_arr[bin_arr!=-1] / dsr

            I_vec = np.histogram(bin_arr,bins = np.arange(-0.5,Nnu,1), weights=I_arr)[0]
            I_vec_all[i,:] = I_vec
            
            if type(line_targ) is str:
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

def gen_Ipred(z_coords, N_arr, dth, nu_binedges, line_use, line_targ, model = 'Be13', verbose = 0):
    '''
    Generate I_arr with the N_arr from sparse approx.
    
    Inputs:
    =======
    z_coords[arr]: Nsamp element list, each element is a list of z of source in a lightcone
    N_arr[arr, Nset x len(z_coords)]: 
    dth: pixel size dth^2 [arcmin]
    nu_binedges: [GHz]
    line_use[str list]: list of lines in the light cone
    line_targ[str or str list]: targeting line
    '''

    Nsamp, Nz = N_arr.shape
     
    if Nz != len(z_coords):
        raise ValueError('N_arr 2nd dimension does not match len(z_coords).')
        
    zsrc_all = []
    Lratio = []
    for i in range(Nsamp):
        zsrc_all.append(z_coords)
        Lratio.append(N_arr[i,:])
        
    Ipred_all, Ipred_targ = Ivox_from_zsrc(zsrc_all, dth, nu_binedges, \
                    line_use, line_targ, Lratio = Lratio, model = model, verbose = 0)
    
    return Ipred_all, Ipred_targ

def zlist_to_N(zsrc, z_coords_all, I_coords_all, z_idx, Nsrc = []):
    I_bl = np.copy(I_coords_all)
    I_bl[I_bl > 0] = 1
    Nall = np.zeros([len(zsrc),len(z_idx)])
    for iset in range(len(zsrc)):
        N = np.zeros_like(z_idx, dtype = 'float')
        idx_vec = np.array([(np.abs(zcii - z_coords_all)).argmin() for zcii in zsrc[iset]])
        for isrc,idx in enumerate(idx_vec):
            if idx <= min(z_idx):
                if len(Nsrc) == 0:
                    N[0] += 1
                else:
                    N[0] += Nsrc[iset][isrc]
            elif idx >= max(z_idx):
                if len(Nsrc) == 0:
                    N[-1] += 1
                else:
                    N[-1] += Nsrc[iset][isrc]
            else:
                # pick the two neaest dictionary
                idx1 = idx - z_idx
                idx1 = idx1[idx1 > 0]
                idx1 = idx - min(idx1)

                idx2 = idx - z_idx
                idx2 = idx2[idx2 < 0]
                idx2 = idx - max(idx2)
                
                # get binary dict to see which one match
                if np.array_equal(I_bl[:,idx1],I_bl[:,idx]):
                    if len(Nsrc) == 0:
                        N[np.where(idx1 == z_idx)] += 1
                    else:
                        N[np.where(idx1 == z_idx)] += Nsrc[iset][isrc]
                else:
                    if len(Nsrc) == 0:
                        N[np.where(idx2 == z_idx)] += 1
                    else:
                        N[np.where(idx2 == z_idx)] += Nsrc[iset][isrc]

        Nall[iset,:] = N
    return Nall
def gen_lightcone_toy(Nlc, dth, nu_binedges, z_coords_all, \
                      I_coords_all, z_idx, line_use, line_targ, Neff_scale = 1, model = 'Be13'):
    
    zsrc = sim_zsrc_Neff(Nlc, dth, Neff_scale = Neff_scale, model = model)
    Ntrue = zlist_to_N(zsrc, z_coords_all, I_coords_all, z_idx)
    Itrue_all, Itrue_targ = Ivox_from_zsrc\
    (zsrc, dth, nu_binedges, line_use, line_targ, model = model, verbose=0)
    
    return Ntrue, Itrue_all, Itrue_targ

def gen_lightcone(Nlc, dth, nu_binedges, z_coords_all, \
                      I_coords_all, z_idx, line_use, line_targ, Neff_scale = 1, model = 'Be13'):
    
    zbinedges = np.arange(0,10 + 0.01,0.01)
    SFRbinedges = np.logspace(-3,1,100)
    
    if model == 'F17':
        SFRf_arr = np.load('data_internal/F17SFRfunc.npy')
    elif model == 'Be13':
        SFRf_arr = np.load('data_internal/Be13SFRfunc.npy')
    else:
        raise ValueError("Model name incorrect.")

    N_arr, zbins, _ = sim_SFRlc_F17(Nlc, dth, zbinedges, SFRbinedges, SFRf_arr, Neff_scale, model = model)
    zsrc = [zbins.tolist()]*Nlc
    Nsrc = N_arr.tolist()
    Ntrue = zlist_to_N(zsrc, z_coords_all, I_coords_all, z_idx, Nsrc = Nsrc)
    Itrue_all, Itrue_targ = gen_Ipred(z_coords_all[z_idx], Ntrue, \
                            dth, nu_binedges, line_use, line_targ, model = model, verbose = 0)
    return Ntrue, Itrue_all, Itrue_targ

def sparse_dict(dth, nu_binedges, line_use, dz = 0.0005, model = 'Be13'):

    z_coords_all = np.arange(dz,10,dz)
    z_coords_all_list = z_coords_all.reshape(len(z_coords_all),-1).tolist()
    I_coords_all,_ = Ivox_from_zsrc(z_coords_all_list, dth, nu_binedges, \
                                    line_use, "Lya", model = model, verbose=0)
    I_coords_all = I_coords_all.T
    z_coords_type_all = np.count_nonzero(I_coords_all, axis=0)

    I_bl = np.copy(I_coords_all)
    I_bl[I_bl > 0] = 1
    
    # dict for multiple bins
    z_coords = []
    z_idx = []
    ztemp = []
    for i,z in enumerate(z_coords_all):
        if len(ztemp) != 0 and np.array_equal(I_bl[:,i],I_bl[:,i-1]):
            ztemp.append(z)
        elif len(ztemp) != 0 and not np.array_equal(I_bl[:,i],I_bl[:,i-1]) and np.sum(I_bl[:,i]) >= 1:
            z_median = np.percentile(ztemp,50, interpolation= 'nearest')
            z_coords.append(z_median)
            z_idx.append(np.where(z_coords_all == z_median)[0][0])
            ztemp = [z]
        elif len(ztemp) != 0 and not np.array_equal(I_bl[:,i],I_bl[:,i-1]) and not np.sum(I_bl[:,i]) >= 1:
            z_median = np.percentile(ztemp,50, interpolation= 'nearest')
            z_coords.append(z_median)
            z_idx.append(np.where(z_coords_all == z_median)[0][0])
            ztemp = []
        elif len(ztemp) == 0 and np.sum(I_bl[:,i]) >= 1:
            ztemp = [z]

    z_idx = np.array(z_idx)
    z_coords = np.array(z_coords)

    A_raw = I_coords_all[:, z_idx]
    A, I_norm = preprocessing.normalize(A_raw,axis=0, copy=True, return_norm=True)
    # A_raw = A @ np.diag(I_norm)
    N_nu, N_z = A.shape

    return A, I_norm, z_coords, N_nu, N_z, z_coords_all, z_idx, I_coords_all
