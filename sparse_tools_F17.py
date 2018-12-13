from lightcone_sim_F17 import *
from sklearn import preprocessing
import cvxpy as cvx

def sparse_dict(dth, nu_binedges, line_use, dz = 0.0005):

    z_coords_all = np.arange(dz,10,dz)
    z_coords_all_list = z_coords_all.reshape(len(z_coords_all),-1).tolist()
    I_coords_all,_ = Ivox_from_zsrc(z_coords_all_list, dth, nu_binedges, line_use, "Lya", verbose=0)
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

def run_lasso(A, I_norm, Iobs_all, alpha, fit_bg = False):
    
    N_nu, N_z = A.shape
    N_lc = Iobs_all.shape[0]
    N_pred = np.zeros([N_lc, N_z])
    A_raw = A_raw = A @ np.diag(I_norm)
    Imean = np.mean(I_norm) 
    
    def norm2_term(A, I, N):
        return cvx.pnorm(cvx.matmul(A, N) - I, 2)**2 / N_nu

    def norm1_term(N):
        return cvx.pnorm(N, 1)

    def loss2(A, I, N):
        return norm2_term(A, I, N).value / N_nu
    
    def loss1(N):
        return norm1_term(N).value
    
    if not fit_bg:
        for ilc in range(N_lc):
            N_var = cvx.Variable(N_z)
            obj_func = (norm2_term(A_raw, Iobs_all[ilc], N_var) / Imean**2) + (alpha * norm1_term(N_var))
            objective = cvx.Minimize(obj_func)
            prob = cvx.Problem(objective, [N_var >= 0])
            results = prob.solve(solver = cvx.SCS, verbose = False)
            N_pred[ilc,:] = N_var.value
            #norm2_term_value = loss2(A_raw, Iobs_all[ilc], N_var) / Imean**2
            #norm1_term_value = alpha * loss1(N_var)
            #print('norm2 term = %.3e, norm1 term = %.3e'%(norm2_term_value,norm1_term_value))

#             NI_var = cvx.Variable(N_z)
#             obj_func = (norm2_term(A, Iobs_all[ilc], NI_var) / Imean**2) + \
#                         (alpha * norm1_term(NI_var) / Imean)            
#             objective = cvx.Minimize(obj_func)
#             prob = cvx.Problem(objective, [NI_var >= 0])
#             results = prob.solve(solver = cvx.SCS, verbose =False)
#             N_pred[ilc,:] = NI_var.value / I_norm
#             norm2_term_value = loss2(A, Iobs_all[ilc], NI_var) / Imean**2
#             norm1_term_value = alpha * loss1(NI_var) / Imean
#             print('norm2 term = %.3e, norm1 term = %.3e'%(norm2_term_value,norm1_term_value))
            
            
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