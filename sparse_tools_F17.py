from lightcone_sim_F17 import *
import cvxpy as cvx

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
    

def run_MP(A, I_norm, Iobs_all, e_th, iter_max = 10):
    
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
            if (R < e_th) or iter_count == iter_max:
                break
            iter_count += 1
            gamma = np.argmax(np.dot(R_arr.reshape(1,-1), A)[0])
            amp = np.sum(A[:,gamma] * R_arr)
            u = amp * A[:,gamma]
            NI_arr[gamma] += amp
            R_arr -= u
            f_arr += u
            R = np.sqrt(np.mean(R_arr**2))

        N_pred[ilc,:] = NI_arr / I_norm
        #print('Light cone %d MP end in %d iterations.'%(ilc, iter_count))
    return N_pred

def run_MP_sig(A, I_norm, Iobs_all, sigI, sig_th, iter_max = 100):
    
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
        #print('Light cone %d MP end in %d iterations.'%(ilc, iter_count))
    return N_pred