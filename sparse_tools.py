from sklearn import preprocessing
import cvxpy as cvx
from scipy import special
from lightcone_toy_p16 import *
from lightcone_sim_P16 import *
#from lightcone_sim import *
#from lightcone_sim_Sch import *

# def sparse_dict(dth, nu_binedges, nu_bins, juse, Lsrc):

#     z_coords_all = np.arange(0.01,10,0.01)
#     z_coords_all_list = z_coords_all.reshape(len(z_coords_all),-1).tolist()
#     I_coords_all,_ = Ivox_from_zsrc(z_coords_all_list, dth, nu_binedges, juse, 0, Lsrc, verbose=0)
#     I_coords_all = I_coords_all.T
#     z_coords_type_all = np.count_nonzero(I_coords_all, axis=0)
    
#     z_cii = spec_lines.CII.to(u.GHz, equivalencies=u.spectral()).value / nu_bins - 1
#     z_cii_idx = [(np.abs(zcii - z_coords_all)).argmin() for zcii in z_cii]

#     z_idx = np.append(np.where(z_coords_type_all >= 2)[0],z_cii_idx)
#     z_coords = z_coords_all[z_idx]
#     z_coords_type = np.append(z_coords_type_all[z_coords_type_all >= 2],np.ones_like(z_cii))
#     sp2 = np.where(z_coords_type >= 2)[0]

#     A_raw = I_coords_all[:,z_idx]
#     A, I_norm = preprocessing.normalize(A_raw,axis=0, copy=True, return_norm=True)
#     # A_raw = A @ np.diag(I_norm)
#     N_nu, N_z = A.shape
    
#     return A, I_norm, z_coords, N_nu, N_z, sp2, z_coords_all, z_idx

# def gen_lightcone_toy(Nlc, dth, nu_binedges, sp2, z_coords_all, z_idx, juse, jtarg, Lsrc):
    
#     zsrc = sim_zsrc_Neff(Nlc, dth, Neff_scale = 1)
#     Ntrue = zlist_to_N(zsrc, z_coords_all, z_idx)
#     Ntrue = Ntrue[:,sp2]
#     Itrue_all, Itrue_targ = Ivox_from_zsrc(zsrc, dth, nu_binedges, juse, jtarg, Lsrc, verbose=0)
    
#     return Ntrue, Itrue_all, Itrue_targ

# def gen_lightcone_Sch(N_lc, dth, nu_binedges, juse, jtarg, Lsrc, Lsrc_j, jbase = 1):
    
#     idx_jbase = Lsrc_j.index(jbase)
#     Lratio = np.array(Lsrc) / Lsrc[idx_jbase]
#     L_arr, zbins = sim_Llc_P16(N_lc, dth, jco = jbase)
    
#     N_nu = len(nu_binedges) - 1
#     Itrue_all = np.zeros([N_lc,N_nu])
    
#     if type(jtarg) is not int:
#         Itrue_targ = np.zeros([len(jtarg),N_lc,N_nu])
        
#     for jco in juse:
#         Itrue_jco = Ivox_from_L_arr(L_arr * Lratio[jco], zbins, dth, nu_binedges, jco).value
#         Itrue_all = Itrue_all + Itrue_jco
#         if type(jtarg) is int:
#             if jco == jtarg:
#                 Itrue_targ = Itrue_jco
#         else:
#             if jco in jtarg:
#                 Itrue_targ[jtarg.index(jco),:,:] = Itrue_jco
                
#     return Itrue_all, Itrue_targ 

def sparse_dict(dth, nu_binedges, juse, dz = 0.0005):

    z_coords_all = np.arange(dz,10,dz)
    z_coords_all_list = z_coords_all.reshape(len(z_coords_all),-1).tolist()
    I_coords_all,_ = Ivox_from_zsrc(z_coords_all_list, dth, nu_binedges, juse, 0, verbose=0)
    I_coords_all = I_coords_all.T
    z_coords_type_all = np.count_nonzero(I_coords_all, axis=0)

    df = pd.read_csv('data_internal/P16NeffLs.txt')
    z_dat = df['z'].values
    Neff_dat = df['CO10_Neff'].values
    Neff_dat[z_dat > 5] = df['CII_Neff'].values[z_dat > 5]
    Neff_dat *= dz * dth**2
    Neff_vec = np.interp(z_coords_all,z_dat,Neff_dat)

    I_bl = np.copy(I_coords_all)
    I_bl[I_bl > 0] = 1

    # dict for multiple bins
    z_coords2 = []
    z_idx2 = []
    ztemp = []
    for i,z in enumerate(z_coords_all):
        if len(ztemp) != 0 and np.array_equal(I_bl[:,i],I_bl[:,i-1]):
            ztemp.append(z)
        elif len(ztemp) != 0 and not np.array_equal(I_bl[:,i],I_bl[:,i-1]) and np.sum(I_bl[:,i]) >= 2:
            z_median = np.percentile(ztemp,50, interpolation= 'nearest')
            z_coords2.append(z_median)
            z_idx2.append(np.where(z_coords_all == z_median)[0][0])
            ztemp = [z]
        elif len(ztemp) != 0 and not np.array_equal(I_bl[:,i],I_bl[:,i-1]) and not np.sum(I_bl[:,i]) >= 2:
            z_median = np.percentile(ztemp,50, interpolation= 'nearest')
            z_coords2.append(z_median)
            z_idx2.append(np.where(z_coords_all == z_median)[0][0])
            ztemp = []
        elif len(ztemp) == 0 and np.sum(I_bl[:,i]) >= 2:
            ztemp = [z]

    # dict for single bin
    z_coords11 = []
    z_idx11 = []
    I1 = []
    Neff1 = []
    bin_idx1 = []
    ztemp = []
    Neffsum = 0.
    for i,z in enumerate(z_coords_all):
        if len(ztemp) != 0 and np.array_equal(I_bl[:,i],I_bl[:,i-1]):
            ztemp.append(z)
            Neffsum += Neff_vec[i]
        elif len(ztemp) != 0 and not np.array_equal(I_bl[:,i],I_bl[:,i-1]) and np.sum(I_bl[:,i]) == 1:
            z_median = np.percentile(ztemp,50, interpolation= 'nearest')
            zidx = np.where(z_coords_all == z_median)[0][0]
            nuidx = np.where(I_bl[:,zidx] == 1)[0][0]
            z_coords11.append(z_median)
            z_idx11.append(zidx)
            I1.append(I_coords_all[nuidx,zidx])
            Neff1.append(Neffsum)
            bin_idx1.append(nuidx)
            ztemp = [z]
            Neffsum += Neff_vec[i]
        elif len(ztemp) != 0 and not np.array_equal(I_bl[:,i],I_bl[:,i-1]) and not np.sum(I_bl[:,i]) == 1:
            z_median = np.percentile(ztemp,50, interpolation= 'nearest')
            zidx = np.where(z_coords_all == z_median)[0][0]
            nuidx = np.where(I_bl[:,zidx] == 1)[0][0]
            z_coords11.append(z_median)
            z_idx11.append(zidx)
            I1.append(I_coords_all[nuidx,zidx])
            Neff1.append(Neffsum)
            bin_idx1.append(nuidx)
            ztemp = []
            Neffsum =0.
        elif len(ztemp) == 0 and np.sum(I_bl[:,i]) == 1:
            ztemp = [z]
            Neffsum += Neff_vec[i]

    z_coords1 = []
    z_idx1 = []
    for nuidx in range(I_coords_all.shape[0]):
        idx = np.where((np.array(bin_idx1) == nuidx))[0]
        if len(idx) == 1:
            z_coords1.append(np.array(z_coords11)[idx][0])
            z_idx1.append(np.array(z_idx11)[idx][0])
        elif len(idx) == 2:
            z_coords1.append(np.array(z_coords11)[idx][0])
            z_idx1.append(np.array(z_idx11)[idx][0])
        else:
            idx = idx[:-1]
            Neff = np.array(np.log10(Neff1))[idx]
            idxuse = idx[np.argmax(Neff)]
            z_coords1.append(z_coords11[idxuse])
            z_idx1.append(z_idx11[idxuse])

    sp2 = np.arange(len(z_idx2))
    z_idx = z_idx2 + z_idx1
    z_coords = z_coords2 + z_coords1
    z_idx = np.array(z_idx)
    z_coords = np.array(z_coords)

    A_raw = I_coords_all[:, z_idx]
    A, I_norm = preprocessing.normalize(A_raw,axis=0, copy=True, return_norm=True)
    # A_raw = A @ np.diag(I_norm)
    N_nu, N_z = A.shape

    return A, I_norm, z_coords, N_nu, N_z, sp2, z_coords_all, z_idx, I_coords_all


def run_MP_sig(A, I_norm, Iobs_all, sigI, sig_th, iter_max = 500, return_iter = False):
    
    N_nu, N_z = A.shape
    N_lc = Iobs_all.shape[0]
    N_pred = np.zeros([N_lc, N_z])

    for ilc in range(N_lc):
        R_arr = Iobs_all[ilc].copy()
        R = np.sqrt(np.mean(R_arr**2))
        f_arr = np.zeros(N_nu)
        NI_arr = np.zeros(N_z)
        iter_count = 0
        while True:
            if iter_count == iter_max:
                break
                
            gamma = np.argmax(np.dot(R_arr.reshape(1,-1), A)[0])
            amp = np.sum(A[:,gamma] * R_arr)

            if amp < sig_th * sigI:
                break
            iter_count += 1
            u = amp * A[:,gamma]
            NI_arr[gamma] += amp
            R_arr -= u
            f_arr += u
            R = np.sqrt(np.mean(R_arr**2))

        N_pred[ilc,:] = NI_arr / I_norm
        
        if return_iter:
            return N_pred[:,:N_z - N_nu], iter_count
        #print('Light cone %d MP end in %d iterations.'%(ilc, iter_count))
    return N_pred[:,:N_z - N_nu]


def run_lasso(A, I_norm, Iobs_all, alpha, sp2, fit_bg = False):
    
    N_nu, N_z = A.shape
    N_lc = Iobs_all.shape[0]
    N_pred = np.zeros([N_lc, N_z - N_nu])

    if not fit_bg:
        for ilc in range(N_lc):
            NI_pred = cvx.Variable(N_z)
            objective = cvx.Minimize(cvx.sum_squares(A * NI_pred - Iobs_all[ilc]) / N_nu +\
                                     alpha * cvx.norm(NI_pred,1))
            prob = cvx.Problem(objective, [NI_pred >= 0])
            results = prob.solve(solver = cvx.SCS)
            N_pred[ilc,:] = NI_pred.value[sp2] / I_norm[sp2]
        
        return N_pred
    if fit_bg:
        for ilc in range(N_lc):
            NI_pred = cvx.Variable(N_z)
            C = cvx.Variable()
            objective = cvx.Minimize(cvx.sum_squares(A * NI_pred + C * np.ones(N_nu) - Iobs_all[ilc]) / N_nu +\
                                     alpha * cvx.norm(NI_pred,1))
            prob = cvx.Problem(objective, [NI_pred >= 0])
            results = prob.solve(solver = cvx.SCS)
            N_pred[ilc,:] = NI_pred.value[sp2] / I_norm[sp2]
            C_pred = C.value
        
        return N_pred, C_pred

def accessible_bands(nu_binedges, juse, jtarg):
    '''
    for a given jtarg, return in which bands we can reconstructed.
    The band is defined reconstructable only if the corresponding z source of whole bandwidth
    has other lines falls in the observable range [min(nu_binedges), max(nu_binedges)]
    
    Inputs:
    =======
    nu_binedges[arr, Nnu+1]: obs freq bin edges [GHz]
    juse[list]: list of lines considered
    jtarg[int]: targeting line
    
    Ouputs:
    =======
    inband[boolean, Nnu]: return if we can probe jtarg in the band,
                        i.e. there exist other lines in the whole freq range
    z_binedges[float, Nnu+1]: the source redshift of the jtarg 
    '''
    juse = np.array(juse)
    np.delete(juse, np.argwhere(juse==jtarg))
    nu_max = max(nu_binedges)
    nu_min = min(nu_binedges)
    if jtarg !=0:
        nu0_targ = spec_lines.CO(jtarg).to(u.GHz, equivalencies=u.spectral()).value
    else:
        nu0_targ = spec_lines.CII.to(u.GHz, equivalencies=u.spectral()).value

    z_binedges = (nu0_targ /nu_binedges) - 1

    inband = np.zeros(len(nu_binedges))
    for j in juse: 
        if j != 0:
            nu0 = spec_lines.CO(j).to(u.GHz, equivalencies=u.spectral()).value
        else:
            nu0 = spec_lines.CII.to(u.GHz, equivalencies=u.spectral()).value

        nu_obs_vec = nu0 / (1 + z_binedges)
        inband += np.asarray([1 if nu_min <= nu_obs <= nu_max else 0 for nu_obs in nu_obs_vec])
    inband = inband[1:] + inband[:-1]
    inband = [inband >= 4]
    inband = inband[0]
    return inband, z_binedges

def survey_band_bins(jtarg, survey_name):
    '''
    Define the binned broad bands for reconstructed CO lines.
    
    Inputs:
    =======
    jtarg: J(upper) CO, only take 3,4,5,6
    survey: name of survey - 'TIME', 'survey1', 'survey2'
    
    Ouputs:
    =======
    idx_vec[list of list]: freq bin index use in each broad band.
    z_min_vec[list]: z_min for each broad band bin.
    z_max_vec[list]: z_max for each broad band bin.
    '''
    if survey_name == 'TIME':
        survey_param = TIME_param()
        nu_binedges = survey_param.nu_binedges_science
    elif survey_name == 'survey1':
        survey_param = survey1_param()
        nu_binedges = survey_param.nu_binedges
    elif survey_name == 'survey2':
        survey_param = survey1_param()
        nu_binedges = survey_param.nu_binedges
    else:
        raise ValueError('Survey name invalid.')
        
    juse = [0,2,3,4,5,6]
    inband, z_binedges = accessible_bands(nu_binedges, juse, jtarg)
    in_idx = np.where(inband)[0]
    z_bins = (z_binedges[1:] + z_binedges[:-1]) / 2
    
    if jtarg == 3:
        idx_vec = np.where(z_bins > 0.4)[0]
        idx_vec = np.intersect1d(idx_vec, in_idx)
        z_min = z_binedges[idx_vec[0]]
        z_max = z_binedges[idx_vec[-1] + 1]
        idx_vec = [idx_vec]
        z_min_vec = [z_min]
        z_max_vec = [z_max]
        name_vec = ['J3 high']
        
    elif jtarg == 4:
        idx_vec1 = np.where(z_bins < 0.8)[0]
        idx_vec1 = np.intersect1d(idx_vec1, in_idx)
        idx_vec2 = np.where(z_bins > 0.8)[0]
        idx_vec2 = np.intersect1d(idx_vec2, in_idx)
        z_min1 = z_binedges[idx_vec1[0]]
        z_max1 = z_binedges[idx_vec1[-1] + 1]
        z_min2 = z_binedges[idx_vec2[0]]
        z_max2 = z_binedges[idx_vec2[-1] + 1]
        idx_vec = [idx_vec1, idx_vec2]
        z_min_vec = [z_min1, z_min2]
        z_max_vec = [z_max1, z_max2]
        name_vec = ['J4 low', 'J4 high']
        
    elif jtarg == 5:
        idx_vec1 = np.where(z_bins < 1.3)[0]
        idx_vec1 = np.intersect1d(idx_vec1, in_idx)
        idx_vec2 = np.where(z_bins > 1.3)[0]
        idx_vec2 = np.intersect1d(idx_vec2, in_idx)
        z_min1 = z_binedges[idx_vec1[0]]
        z_max1 = z_binedges[idx_vec1[-1] + 1]
        z_min2 = z_binedges[idx_vec2[0]]
        z_max2 = z_binedges[idx_vec2[-1] + 1]
        idx_vec = [idx_vec1, idx_vec2]
        z_min_vec = [z_min1, z_min2]
        z_max_vec = [z_max1, z_max2]
        name_vec = ['J5 low', 'J5 high']
        
    elif jtarg == 6:
        idx_vec = in_idx
        z_min = z_binedges[idx_vec[0]]
        z_max = z_binedges[idx_vec[-1] + 1]
        idx_vec = [idx_vec]
        z_min_vec = [z_min]
        z_max_vec = [z_max]
        name_vec = ['J6 low']
    else:
        raise ValueError("Only jco = 3,4,5,6 can be reconstructed.")
        
        
    return idx_vec, z_min_vec, z_max_vec, name_vec



def survey_band_values(I_targ, jtarg_vec, survey_name):
    
    jtarg_vec1 = [jtarg for jtarg in jtarg_vec if jtarg in [3,4,5,6]]
    
    avg_list = []
    se_list = []
    Ndat_list = []
    zmin_list = []
    zmax_list = []
    name_list = []
    for jtarg in jtarg_vec1:
        jidx = int(np.where(np.array(jtarg_vec)==jtarg)[0])
        idx_vec, z_min_vec, z_max_vec, name_vec = survey_band_bins(jtarg, survey_name)
        for _, (idx,z_min,z_max,name) in enumerate(zip(idx_vec, z_min_vec, z_max_vec, name_vec)):
            Ndat = I_targ[jidx,:,idx].size
            avg = np.mean(I_targ[jidx,:,idx])
            sig = np.std(I_targ[jidx,:,idx]) / np.sqrt(Ndat)
            avg_list.append(avg)
            se_list.append(sig)
            Ndat_list.append(Ndat)
            zmin_list.append(z_min)
            zmax_list.append(z_max)
            name_list.append(name)
            
    return np.array(avg_list), np.array(se_list), np.array(Ndat_list), np.array(zmin_list), np.array(zmax_list), np.array(name_list)

def prediction_pvalue(Ipred_targ, Itrue_targ, jtarg_vec, survey_name):
    Itrue_avg, Itrue_se, _, zmin_list, zmax_list, name_list = TIME_band_values(Itrue_targ, jtarg_vec)
    Ipred_avg, Ipred_se, _, _, _, _ = survey_band_values(Ipred_targ, jtarg_vec, survey_name)
    t_value = (Ipred_avg - Itrue_avg) / np.sqrt(Itrue_se**2 + Ipred_se**2)
    # p(>x) = 1 - erf(x/sqrt(2))
    # https://math.stackexchange.com/questions/37889/why-is-the-error-function-defined-as-it-is
    p_value = special.erfc(abs(t_value) / np.sqrt(2))
    return p_value, t_value, zmin_list, zmax_list, name_list
