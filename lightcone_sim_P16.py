from survey_params import *

def LF_P16(z, jco, L_arr = []):
    '''
    CO / CII Schechter luminosity function from Popping 2016 [1602.02761] 
    table 1 (CO) & table 2 (CII).
    
    Inputs:
    =======
    z: redshift, for z between the listed data, the Schechter params are linearly interpolate.
    jco: CO J line, 0 for CII
    L_arr: in unit L/L*
    
    Output:
    =======
    Phi_arr: dN/dV/d(L/L*) [(h/Mpc)^3]
    L_arr: in unit L/L*
    Ls: Schechter L* [L_sun]
    '''
    
    if len(L_arr)==0:
        L_arr = np.logspace(-3,1,100)
    else:
        L_arr = np.array(L_arr)
    if jco !=0:
        z_arr = [0,1,2,4,6]
        freq_rest = spec_lines.CO(jco).to(u.GHz, equivalencies = u.spectral()).value
        freq_obs = freq_rest / (1+z)

        if jco==1:
            alpha_arr = [-1.36, -1.49, -1.52, -1.71, -1.94]
            Ls_arr = [6.97, 7.25, 7.30, 7.26, 6.99]
            Phis_arr = [-2.85, -2.73, -2.63, -2.94, -3.46]
        elif jco ==2:
            alpha_arr = [-1.35, -1.47, -1.52, -1.75, -2.00]
            Ls_arr = [7.54, 7.84, 7.92, 7.89, 7.62]
            Phis_arr = [-2.85, -2.72, -2.66, -3.00, -3.56]
        elif jco ==3:
            alpha_arr = [-1.29, -1.47, -1.53, -1.76, -2.00]
            Ls_arr = [7.83, 8.23, 8.36, 8.26, 7.95]
            Phis_arr = [-2.81, -2.79, -2.78, -3.11, -3.60]
        elif jco ==4:
            alpha_arr = [-1.29, -1.45, -1.51, -1.80, -2.03]
            Ls_arr = [8.16, 8.50, 8.64, 8.70, 8.23]
            Phis_arr = [-2.93, -2.84, -2.85, -3.45, -3.78]
        elif jco ==5:
            alpha_arr = [-1.20, -1.47, -1.45, -1.76, -1.95]
            Ls_arr = [8.37, 8.80, 8.74, 8.73, 8.30]
            Phis_arr = [-2.94, -3.03, -2.80, -3.34, -3.67]
        elif jco ==6:
            alpha_arr = [-1.15, -1.41, -1.43, -1.73, -1.93]
            Ls_arr = [8.38, 8.74, 8.77, 8.84, 8.38]
            Phis_arr = [-2.92, -2.92, -2.80, -3.40, -3.72]
        else:
            raise ValueError('jco out of range (1-6)')
    elif jco ==0:
        z_arr = [0,1,2,3,4,6]
        freq_rest = spec_lines.CII.to(u.GHz, equivalencies = u.spectral()).value 
        freq_obs = freq_rest / (1+z)
        alpha_arr = [-1.25, -1.43, -1.52, -1.41, -1.53, -1.77]
        Ls_arr = [7.47, 7.66, 7.81, 7.80, 7.85, 7.80]
        Phis_arr = [-2.33, -2.15, -2.20, -2.12, -2.37, -2.95]
    else:
        raise ValueError('jco has to be 0, or 1-6!!')
    
    if z <= 6:
        alpha = np.interp(z, z_arr, alpha_arr)
        Ls = 10**np.interp(z, z_arr, Ls_arr)
        Phis = 10**np.interp(z, z_arr, Phis_arr)
    else:
        alpha = np.poly1d(np.polyfit(z_arr[-2:], alpha_arr[-2:],1))(z)
        if alpha < -2 : alpha = -2.
        Ls = 10**np.poly1d(np.polyfit(z_arr[-2:], Ls_arr[-2:],1))(z)
        Phis = 10**np.poly1d(np.polyfit(z_arr[-2:], Phis_arr[-2:],1))(z)    
    
    if jco !=0:
        # L = LV * lambda_rest Obreschkow et al. 2009 Eq. A12
        Ls  = (Ls * u.Jy * u.km * u.Mpc**2 / u.s)* (freq_rest * u.GHz) / const.c
        Ls = Ls.to(u.Lsun).value
    
    Phis = Phis / cosmo.h**3 * np.log(10)
    Phi_arr=Phis*(L_arr**alpha)*np.exp(-L_arr)
    
    return Phi_arr, L_arr, Ls

def Neff_P16(z, jco, unit = 'obs'):
    '''
    Calculate Neff from P16 model.
    
    Neff = <L>^2 / <L^2>

    Inputs:
    =======
    z: redshift, for z between the listed data, the Schechter params are linearly interpolate.
    jco: CO J line, 0 for CII
    unit: 'obs'-- Neff per comoving volume[(h/Mpc)^3] 
          'cmv' -- Neff per arcmin^2 per z [1/(arcmin)^2/dz]
    
    Output:
    =======
    Neff
    
    '''
    Phi_arr, L_arr, Ls = LF_P16(z,jco)
    dL_arr = L_arr[1:] - L_arr[:-1]
    Lbins_arr = (L_arr[1:] + L_arr[:-1]) / 2
    Phibins_arr = (Phi_arr[1:] + Phi_arr[:-1]) / 2
    Ltot = np.sum(Lbins_arr * Phibins_arr * dL_arr)
    L2tot = np.sum(Lbins_arr**2 * Phibins_arr * dL_arr)
    Neff = Ltot**2 / L2tot
    
    if unit == 'cmv':
        return Neff
    else:
        Neff = Neff * cosmo.h**3 * (u.Mpc)**-3
        dA = (cosmo.kpc_comoving_per_arcmin(z).to(u.Mpc / u.arcmin))**2
        dchi_dz = (const.c / cosmo.H(z)).to(u.Mpc)
        dV = dA * dchi_dz
        Neff *= dV
        Neff = Neff.value
        return Neff

def sim_Llc_P16(Nsamp, dth, zbinedges = [], Lbinedges = [], jco = 1, Neff_scale = 1):
    '''
    Simulate the light cone with Popping et al. 2016 luminosity function. 
    
    Inputs:
    =======
    Nsamp: number of lightcone to generate
    dth: lightcone size dth^2[arcmin]
    zbinedges: the redshift bin edges
    Lbinedges: the luminosity bin edges to sample the LF with Poisson distribution 
            [unit: L_star of Schechter func]
    jco: CO J line, 0 for CII
    
    Output:
    =======
    L_arr [Lsun]: Nsamp x Nz, L_arr(i,j) is the intrinsic luminoisty [Lsun] in lightcone j redshift j
    zbins: redshift bins for L_arr
    '''
    
    if len(zbinedges) == 0:
        zbinedges = np.arange(0,10 + 0.01,0.01)
    zbins = (zbinedges[1:] + zbinedges[:-1]) / 2
    Nz = len(zbins)
    
    CDedges = cosmo_dist(zbinedges).comoving_distance
    dCD = CDedges[1:] - CDedges[:-1]
    dth = dth * u.arcmin
    dtrans = cosmo_dist(zbins).kpc_comoving_per_arcmin * dth
    dV_zvec = dCD.to(u.Mpc / u.h) * dtrans.to(u.Mpc / u.h)**2
    
    if len(Lbinedges) == 0:
        Lbinedges = np.logspace(-3,1,100)
    Lbins = np.sqrt(Lbinedges[1:] * Lbinedges[:-1])
    NL = len(Lbins)
    dL_lvec = (Lbinedges[1:] - Lbinedges[:-1])
    
    L_arr = np.zeros([Nsamp,Nz])
    N_arr = np.zeros([Nsamp,Nz])
    for iz,z in enumerate(zbins):
        Phi_lvec, _ , Ls = LF_P16(z, jco, Lbins)
        Navg_lvec = Phi_lvec * dV_zvec[iz].value * dL_lvec * Neff_scale
        Nsim_arr = np.random.poisson(Navg_lvec, size = (Nsamp, NL))
        Ltot_svec = np.matmul(Nsim_arr,Lbins.reshape([NL,-1])).flatten()
        Ltot_svec *= Ls
        L_arr[:,iz] = Ltot_svec
        N_arr[:,iz] = Ltot_svec / Ls
    
    return L_arr, N_arr, zbins