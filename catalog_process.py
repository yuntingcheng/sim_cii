import pandas as pd
from cosmo_tools import *

def get_cat_df():
    '''
    retrive the catalog to a pandas df. 
    Throw out the unnecessary broad band flux data, convert the line flux to unit [Jy GHz]
    
    Oputputs:
    =========
    df: preprocessed data frame.
    '''
    fname = 'data_catalog/SIDES_with_lines_CO_CII.csv'
    df_all = pd.read_csv(fname)

    df = df_all[['redshift', 'ra', 'dec', 'Mhalo', 'Mstar', 'qflag', 'SFR', 'mu', 'issb', 'Umean',\
    'DL', 'ICO10', 'ICO21', 'ICO32', 'ICO43', 'ICO54', 'ICO65', 'ICO76', 'ICO87', 'ICII']].copy()

    # convert I columns in [Jy km s^-1] to flux in [Jy GHz]
    df['ICII'] = 3.445e-6 * spec_lines.CII.to(u.GHz, equivalencies=u.spectral()).value * df['ICII']
    for jco in range(1,9,1):
        name = 'ICO' + str(jco) + str(jco-1)
        df[name] = 3.445e-6 * spec_lines.CO(jco).to(u.GHz, equivalencies=u.spectral()).value * df[name] / (1 + df['redshift'])
    
    return df

def cat_SFRD(df, zedges_arr, z_arr = []):
    '''
    Calculate the catalog SFRD [Msun / yr / Mpc^3] with z-bins defined by zedges_arr.
    '''
    
    if len(z_arr)==0:
        z_arr = (zedges_arr[1:] + zedges_arr[:-1])/2
        
    dfsfr = df[['redshift', 'SFR']].copy()
    binlabel = np.digitize(dfsfr['redshift'], zedges_arr) - 1
    binlabel[(dfsfr['redshift'] < zedges_arr[0]) | (dfsfr['redshift'] > zedges_arr[-1])] = -1
    dfsfr['zbin'] = binlabel
    SFR_arr = np.asarray(dfsfr.groupby(['zbin'])['SFR'].agg(['sum'])).flatten()[1:] * u.Msun / u.year
    dx_arr = ((df['ra'].max() - df['ra'].min())* u.deg * cosmo.kpc_comoving_per_arcmin(z_arr)).to(u.Mpc)
    dy_arr = ((df['dec'].max() - df['dec'].min())* u.deg * cosmo.kpc_comoving_per_arcmin(z_arr)).to(u.Mpc)
    dz_arr = cosmo.comoving_distance(zedges_arr)[1:] - cosmo.comoving_distance(zedges_arr)[:-1]
    dV_arr = dx_arr * dy_arr * dz_arr
    SFRD_arr = SFR_arr / dV_arr
    
    return SFRD_arr, z_arr
