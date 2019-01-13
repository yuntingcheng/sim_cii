from cosmo_tools import *
import pandas as pd

def SFR_Be13(z, M_arr = []):
    '''
    Behroozi et al. 2013 SFR(M, z) function. 
    https://arxiv.org/pdf/1207.6105.pdf
    data downloaded from:
    https://www.peterbehroozi.com/data.html
    
    Input:
    ======
    M_arr: halo mass [Msun / h]
    z: redshift
    
    Output:
    =======
    M_arr: halo mass [Msun / h]
    sfr_arr: [Msun / yr]
    '''
    # get data
    df = pd.read_csv('data/Behroozi_SFR/sfr/sfr_release.dat', sep = ' ',\
                     skiprows = 2, names=['z','M','SFR','MSM'])
    df['z'].update(df['z'] - 1)
    df.drop(['MSM'], axis = 1, inplace = True)
    dfg = df.groupby('z')
    z_arr = np.sort(list(dfg.groups))
    
    # interpolate to the given redshift
    if z == 0:
        z1 = z_arr[0]
        z2 = z_arr[1]
    elif z >= z_arr[-1]:
        z1 = z_arr[-2]
        z2 = z_arr[-1]
    else:
        z1 = z_arr[np.searchsorted(z_arr, z) - 1]
        z2 = z_arr[np.searchsorted(z_arr, z)]
        
    group1 = dfg.get_group(z1)
    group2 = dfg.get_group(z2)
    logM_dat = group1.M.values + np.log10(cosmo.h) # log10(M)[Msun / h]
    logSFR_dat1 = group1.SFR.values # log10(SFR)[Msun/yr]
    logSFR_dat2 = group2.SFR.values # log10(SFR)[Msun/yr]
    logSFR_dat = logSFR_dat1 + (logSFR_dat2 - logSFR_dat1) * (z - z1) / (z2 - z1)
    
    if z > z_arr[-1]:
        logSFR_dat = logSFR_dat2
    
    # interpolate to the assigned M_arr
    if len(M_arr) == 0:
        M_arr = np.logspace(8, 15, 1000)
    else:
        M_arr = np.array(M_arr)
    logM_arr = np.log10(M_arr)
    logSFR_arr = np.interp(logM_arr, logM_dat, logSFR_dat)
    
    # For M < min M in data, use linear extrapolation
    
    logSFR_arr_ex = np.poly1d(np.polyfit(logM_dat[:2], logSFR_dat[:2],1))(logM_arr)
    logSFR_arr[logM_arr < logM_dat[0]] = logSFR_arr_ex[logM_arr < logM_dat[0]]
    
    logSFR_arr[logSFR_arr < -20] = -20.
    SFR_arr = 10**logSFR_arr
    SFR_arr[SFR_arr < 1e-19] = 0
    M_arr = 10**logM_arr
    
    return SFR_arr, M_arr


def SFR_func_Be13(z, sfr_arr = []):
    '''
    SFR func (dn/dSFR) in Be13 model.
        
    Input:
    ======
    z: redshift
    sfr_arr: SFR/SFR* [dimensionless]
    
    Output:
    =======
    dnd(sfr_arr/sfrs): SFR func dn/dSFR [h^3 / Mpc^3]
    sfr_arr: SFR/SFR* [dimensionless]
    sfrs: [Msun / yr]
    Neff: [h^3 / Mpc^3]
    sfrd: SFRD [h^3 / Mpc^3 / (Msun / yr)]
    '''

    if len(sfr_arr) == 0:
        sfr_arr = np.logspace(-3, 1, 1000)
    else:
        sfr_arr = np.array(sfr_arr)

    SFRbinedges_arr, Mbinedges_arr = SFR_Be13(z)
    sp0 = np.where(SFRbinedges_arr != 0)[0]
    SFRbinedges_arr = SFRbinedges_arr[sp0]
    Mbinedges_arr = Mbinedges_arr[sp0]
    SFRbins_arr = np.sqrt(SFRbinedges_arr[1:] * SFRbinedges_arr[:-1])
    Mbins_arr = np.sqrt(Mbinedges_arr[1:] * Mbinedges_arr[:-1])
    dMbins_arr = Mbinedges_arr[1:] - Mbinedges_arr[:-1]
    hmfz = HMFz(z)
    dndm_arr, _ = hmfz.dndm(m_arr = Mbins_arr)
    dn_arr = dndm_arr * dMbins_arr
    
    SFRbinedges_dat = np.logspace(np.log10(min(SFRbinedges_arr)),\
                                  np.log10(max(SFRbinedges_arr)), 51)
    SFRbins_dat = np.sqrt(SFRbinedges_dat[1:] * SFRbinedges_dat[:-1])
    dndSFR_dat = np.histogram(SFRbins_arr, bins = SFRbinedges_dat, weights = dn_arr)[0]
    dndSFR_dat /= (SFRbinedges_dat[1:] - SFRbinedges_dat[:-1])
    
    # calculate Neff, sfrs, SFRD
    dlnSFR = np.log(SFRbins_dat[1]) - np.log(SFRbins_dat[0])
    sfrd = np.sum(dndSFR_dat * SFRbins_dat**2) * dlnSFR
    SFRsum2 = np.sum(dndSFR_dat * SFRbins_dat**3) * dlnSFR
    Neff = sfrd**2 / SFRsum2
    sfrs = SFRsum2 / sfrd
    
    # interpolate to the input sfr_arr
    SFRbins_dat /= sfrs
    dndSFR_dat *= sfrs
    
    dndsfr_arr = np.zeros_like(sfr_arr)
    sp = np.where((sfr_arr >= SFRbins_dat[0]) & (sfr_arr <= SFRbins_dat[-1]))[0]
    spn = np.where((sfr_arr < SFRbins_dat[0]))[0]
    spp = np.where((sfr_arr > SFRbins_dat[-1]))[0]
    
    x = np.log(SFRbins_dat[dndSFR_dat > 0])
    y = np.log(dndSFR_dat[dndSFR_dat > 0])
    dndsfr_arr[sp] = np.exp(np.interp(np.log(sfr_arr[sp]), x, y))
    dndsfr_arr[spp] = 0.
    dndsfr_arr[spn] = np.exp(y[0]+(np.log(sfr_arr[spn])-x[0])*(y[1]-y[0])/(x[1]-x[0]))

    return dndsfr_arr, sfr_arr, sfrs, Neff, sfrd

def SFRs_Be13(z):
    '''
    Return SFRD of Be13 model.
    Inputs:
    =======
    z: redshift, for z between the listed data, the Schechter params are linearly interpolate.

    Output:
    =======
    SFRs: [Msun / yr]

    '''
    _, _, sfrs, _, _ = SFR_func_Be13(z)
    return sfrs


def SFRD_Be13(z):
    '''
    Return SFRs of Be13 model.
    Inputs:
    =======
    z: redshift, for z between the listed data, the Schechter params are linearly interpolate.

    Output:
    =======
    sfrd: SFRD [h^3 / Mpc^3 / (Msun / yr)]

    '''
    M_arr = np.logspace(8, 15, 1000)
    dlnm = np.log(M_arr)[1] - np.log(M_arr)[0]
    
    hmfz = HMFz(z)
    dndlnm_arr, _ = hmfz.dndlnm(m_arr = M_arr)
    
    SFR_arr,_ = SFR_Be13(z, M_arr)
    SFRD = np.sum(dlnm * dndlnm_arr * SFR_arr)
    
    return SFRD

def Neff_Be13(z, unit = 'obs'):
    '''
    Calculate Neff from Be13 model.
    
    Neff = <L>^2 / <L^2>

    Inputs:
    =======
    z: redshift, for z between the listed data, the Schechter params are linearly interpolate.
    unit: 'obs'-- Neff per comoving volume[(h/Mpc)^3] 
          'cmv' -- Neff per arcmin^2 per z [1/(arcmin)^2/dz]
    
    Output:
    =======
    Neff
    
    '''

    _, _, _, Neff, _ = SFR_func_Be13(z)
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

def SFR_F17(z, M_arr = []):
    '''
    Fonseca et al. 2017 Eq 11, SFR(M, z) function. Paramters value in table 1.
    All the params in other redshift are linearly interpolate, except for M0 
    linear interp in log space.
    
    Input:
    ======
    M_arr: halo mass [Msun / h]
    z: redshift
    
    Output:
    =======
    M_arr: halo mass [Msun / h]
    sfr_arr: [Msun / yr]
    '''
    h = cosmo.h
    
    if len(M_arr) == 0:
        M_arr = np.logspace(8, 15, 1000)
    else:
        M_arr = np.array(M_arr)
    
    M_arr /= h
    
    z_dat = [0., 1., 2., 3., 4., 5.]
    M0_dat = [3.0e-10, 1.7e-9, 4.0e-9, 1.1e-8, 6.6e-8, 7.0e-7]
    Mb_dat = [6.0e10, 9.0e10, 7.0e10, 5.0e10, 5.0e10, 6.0e10]
    Mc_dat = [1.0e12, 2.0e12, 2.0e12, 3.0e12, 2.0e12, 2.0e12]
    a_dat = [3.15, 2.9, 3.1, 3.1, 2.9, 2.5]
    b_dat = [-1.7, -1.4, -2.0, -2.1, -2.0, -1.6]
    c_dat = [-1.7, -2.1, -1.5, -1.5, -1.0, -1.0]
    
    if z <= 5.:
        M0 = 10**np.interp(z, z_dat, np.log10(M0_dat))
        Mb = np.interp(z, z_dat, Mb_dat)
        Mc = np.interp(z, z_dat, Mc_dat)
        a = np.interp(z, z_dat, a_dat)
        b = np.interp(z, z_dat, b_dat)
        c = np.interp(z, z_dat, c_dat)
#     elif z <= 5.5:
#         M0 = 10**np.poly1d(np.polyfit(z_dat[-2:], np.log10(M0_dat[-2:]),1))(z)
#         Mb = np.poly1d(np.polyfit(z_dat[-2:], Mb_dat[-2:],1))(z)
#         Mc = np.poly1d(np.polyfit(z_dat[-2:], Mc_dat[-2:],1))(z)
#         a = np.poly1d(np.polyfit(z_dat[-2:], a_dat[-2:],1))(z)
#         b = np.poly1d(np.polyfit(z_dat[-2:], b_dat[-2:],1))(z)
#         c = np.poly1d(np.polyfit(z_dat[-2:], c_dat[-2:],1))(z)
    else:
        # the SFRD blows up if extrapolate, so use z = 5 for all high-z 
        M0 = 10**np.poly1d(np.polyfit(z_dat[-2:], np.log10(M0_dat[-2:]),1))(5)
        Mb = np.poly1d(np.polyfit(z_dat[-2:], Mb_dat[-2:],1))(5)
        Mc = np.poly1d(np.polyfit(z_dat[-2:], Mc_dat[-2:],1))(5)
        a = np.poly1d(np.polyfit(z_dat[-2:], a_dat[-2:],1))(5)
        b = np.poly1d(np.polyfit(z_dat[-2:], b_dat[-2:],1))(5)
        c = np.poly1d(np.polyfit(z_dat[-2:], c_dat[-2:],1))(5)
         
    Ma = 1e8
    
    sfr_arr = M0 * (M_arr / Ma)**a * (1 + (M_arr / Mb))**b * (1 + (M_arr / Mc))**c
    
    return sfr_arr, M_arr*h

def SFR_func_F17(z, sfr_arr = []):
    '''
    SFR func (dn/dSFR) in F17 model.
    
    dn/dSFR = dn/dm * dm/dSFR.
    
    Since the m(SFR) is not monotonic, dm/dSFR < 0 at very high mass halos,
    this function has to be compute separately at +/- part of dm/dSFR.
    
    Input:
    ======
    z: redshift
    sfr_arr: sfr / sfrs
    
    Output:
    =======
    dnd(sfr_arr/sfrs): SFR func dn/dSFR [h^3 / Mpc^3]
    sfr_arr: SFR/SFR*[]
    sfrs: [Msun / yr]
    Neff: [h^3 / Mpc^3]
    sfrd: SFRD [h^3 / Mpc^3 / (Msun / yr)]
    '''
    
    if len(sfr_arr) == 0:
        sfr_arr = np.logspace(-3, 1, 1000)
    else:
        sfr_arr = np.array(sfr_arr)
    
    M_arr = np.logspace(8, 15, 1000)
    dlnm = np.log(M_arr)[1] - np.log(M_arr)[0]
    
    hmfz = HMFz(z)
    dndm_dat, _ = hmfz.dndm(m_arr = M_arr)
    
    sfr_dat,_ = SFR_F17(z, M_arr)
    
    # sfr_arr to be interpolated to coadd +/- part of dm/dsfr
    sfr_arr1 = np.logspace(np.log10(min(sfr_dat)), np.log10(max(sfr_dat)), 1000)

    # calculate dm/dSFR
    M_arr1 = M_arr * 1.1
    M_arr2 = M_arr * 0.9
    sfr_dat1, _ = SFR_F17(z, M_arr1)
    sfr_dat2, _ = SFR_F17(z, M_arr2)
    dmdsfr_dat = (M_arr1 - M_arr2) / (sfr_dat1 - sfr_dat2)
    

    # seperate the positive and negative parts
    sp_p = np.where(dmdsfr_dat > 0)[0]
    sp_n = np.where(dmdsfr_dat < 0)[0]
    
    # interpolate SFR_func at sfr_arr
    if len(sp_n) > 0:
        sfr_datp = sfr_dat[sp_p]
        dndm_datp = dndm_dat[sp_p]
        dmdsfr_datp = dmdsfr_dat[sp_p]
        sfr_datn = sfr_dat[sp_n]
        dndm_datn = dndm_dat[sp_n]
        dmdsfr_datn = -dmdsfr_dat[sp_n]
        dndsfr_arrp = np.interp(sfr_arr1, sfr_datp, dndm_datp * dmdsfr_datp)
        dndsfr_arrn = np.interp(sfr_arr1, sfr_datn[::-1], dndm_datn[::-1] * dmdsfr_datn[::-1],\
                                left = 0, right = 0)
        dndsfr_arr1 = dndsfr_arrp + dndsfr_arrn
    else:
        dndsfr_arr1 = np.interp(sfr_arr1, sfr_dat, dndm_dat * dmdsfr_dat, left = 0, right = 0)
    
    # cut off the weird peak at high SFR
    sfr_arr1 = sfr_arr1[:-30]
    dndsfr_arr1 = dndsfr_arr1[:-30]
    
    # calculate Neff, sfrs, SFRD
    dlnsfr = np.log(sfr_arr1[1]) - np.log(sfr_arr1[0])
    Ncum_arr = np.cumsum(dndsfr_arr1[::-1] * sfr_arr1[::-1])[::-1] * dlnsfr
    sfrcum_arr = np.cumsum(dndsfr_arr1[::-1] * sfr_arr1[::-1]**2)[::-1] * dlnsfr
    sigsncum_arr = np.cumsum(dndsfr_arr1[::-1] * sfr_arr1[::-1]**3)[::-1] * dlnsfr
    Neff = sfrcum_arr[0]**2 / sigsncum_arr[0]
    sfrs = sigsncum_arr[0] / sfrcum_arr[0]
    sfrd = sfrcum_arr[0]
    
#     plt.plot(sfr_arr1, dndsfr_arr1)
#     plt.plot(sfr_arr1, Ncum_arr)
#     plt.plot(sfr_arr1, sfrcum_arr)
#     plt.plot(sfr_arr1, sigsncum_arr)
#     plt.yscale('log')
#     plt.xscale('log')
#     plt.grid()

    # interpolate to the input sfr_arr
    sfr_arr1 /= sfrs
    dndsfr_arr1 *= sfrs
    
    dndsfr_arr = np.zeros_like(sfr_arr)
    sp = np.where((sfr_arr >= sfr_arr1[0]) & (sfr_arr <= sfr_arr1[-1]))[0]
    spn = np.where((sfr_arr < sfr_arr1[0]))[0]
    spp = np.where((sfr_arr > sfr_arr1[-1]))[0]
    
    x = np.log(sfr_arr1)
    y = np.log(dndsfr_arr1)
    dndsfr_arr[sp] = np.exp(np.interp(np.log(sfr_arr[sp]), x, y))
    dndsfr_arr[spp] = 0.#np.exp(y[-1]+(np.log(sfr_arr[spp])-x[-1])*(y[-1]-y[-2])/(x[-1]-x[-2]))
    dndsfr_arr[spn] = np.exp(y[0]+(np.log(sfr_arr[spn])-x[0])*(y[1]-y[0])/(x[1]-x[0]))
    
    return dndsfr_arr, sfr_arr, sfrs, Neff, sfrd

def SFRs_F17(z):
    '''
    Return SFRD of F17 model.
    Inputs:
    =======
    z: redshift, for z between the listed data, the Schechter params are linearly interpolate.

    Output:
    =======
    SFRs: [Msun / yr]

    '''
    _, _, sfrs, _, _ = SFR_func_F17(z)
    return sfrs


def SFRD_F17(z):
    '''
    Return SFRs of F17 model.
    Inputs:
    =======
    z: redshift, for z between the listed data, the Schechter params are linearly interpolate.

    Output:
    =======
    sfrd: SFRD [h^3 / Mpc^3 / (Msun / yr)]

    '''
    M_arr = np.logspace(8, 15, 1000)
    dlnm = np.log(M_arr)[1] - np.log(M_arr)[0]
    
    hmfz = HMFz(z)
    dndlnm_arr, _ = hmfz.dndlnm(m_arr = M_arr)
    
    SFR_arr,_ = SFR_F17(z, M_arr)
    SFRD = np.sum(dlnm * dndlnm_arr * SFR_arr)
    
    return SFRD

def Neff_F17(z, unit = 'obs'):
    '''
    Calculate Neff from F17 model.
    
    Neff = <L>^2 / <L^2>

    Inputs:
    =======
    z: redshift, for z between the listed data, the Schechter params are linearly interpolate.
    unit: 'obs'-- Neff per comoving volume[(h/Mpc)^3] 
          'cmv' -- Neff per arcmin^2 per z [1/(arcmin)^2/dz]
    
    Output:
    =======
    Neff
    
    '''

    _, _, _, Neff, _ = SFR_func_F17(z)
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

def Lline_F17(line_name, z, sfr_arr):
    '''
    Fonseca et al. 2017 line luminosity:
    
    L = K(z)(SFR/[Msun/yr])**gamma
    K(z) = (f^UV_dust - f^UV_esc)f^line_esc(z) R_line
    f^UV_dust = 10^(-E_UV / 2.5)
    f^line_esc = 10^(-E_line / 2.5)
    Input:
    ======
    line_name: line name - ['Lya', 'Lyb', 'Hb', 'OII', 'OII']
    z: redshift
    sfr_arr: [Msun / yr]
    
    Output:
    =======
    L_arr: halo mass [Lsun]
    sfr_arr: [Msun / yr]

    '''
    sfr_arr = np.array(sfr_arr)
    line_name_arr = ['Lya', 'Ha', 'Hb', 'OII', 'OIII']
    
    if line_name not in line_name_arr:
        raise ValueError('Incorrect input line name.')
    
    f_UV_esc = 0.2
    f_UV_dust = 10 **(-1. / 2.5)
    
    if line_name == 'Lya':
        gamma = 1.
        R = 1.1e42 * (u.erg / u.s).to(u.Lsun)
        f_esc = 0.2
    elif line_name == 'Ha':
        gamma = 1.
        R = 1.3e41 * (u.erg / u.s).to(u.Lsun)
        f_esc = 10**(-1. / 2.5)
    elif line_name == 'Hb':
        gamma = 1.
        R = 4.45e40 * (u.erg / u.s).to(u.Lsun)
        f_esc = 10**(-1.38 / 2.5)
    elif line_name == 'OII':
        gamma = 1.
        R = 7.1e40 * (u.erg / u.s).to(u.Lsun)
        f_esc = 10**(-0.62 / 2.5)
    elif line_name == 'OIII':
        gamma = 1.
        R = 1.3e41 * (u.erg / u.s).to(u.Lsun)
        f_esc = 10**(-1.35 / 2.5)
    
    K = (f_UV_dust - f_UV_esc) * f_esc * R
    
    L_arr = K * (sfr_arr)**gamma
    
    return L_arr, sfr_arr

def Lline_G17(line_name, z, sfr_arr):
    '''
    Gong et al. 2017 line luminosity:
    (Eq. 3,4,5)
    Input:
    ======
    line_name: line name - ['Lya', 'Lyb', 'Hb', 'OII', 'OII']
    z: redshift
    sfr_arr: [Msun / yr]
    
    Output:
    =======
    L_arr: halo mass [Lsun]
    sfr_arr: [Msun / yr]

    '''
    sfr_arr = np.array(sfr_arr)
    line_name_arr = ['Ha', 'Hb', 'OII', 'OIII']
    
    if line_name not in line_name_arr:
        raise ValueError('Incorrect input line name.')
    
    Lsun2ergps = u.Lsun.to(u.erg / u.s)
    if line_name == 'Ha':
        L_arr = (sfr_arr / 7.9e-42) / Lsun2ergps
    elif line_name == 'Hb':
        L_arr = (sfr_arr / 7.9e-42) / Lsun2ergps
        L_arr *= 0.35
    elif line_name == 'OII':
        L_arr = (sfr_arr / 1.4e-41) / Lsun2ergps
    elif line_name == 'OIII':
        L_arr = (sfr_arr / 7.6e-42) / Lsun2ergps

    return L_arr, sfr_arr