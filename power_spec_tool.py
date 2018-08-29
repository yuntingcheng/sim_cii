import numpy as np
import scipy
from scipy import signal

def power_spec_2D(map1, map2, dx, dy, window_func = None):
    '''
    calculate cross power spectrum of a 2-dim Nx x Ny map1 and map2. 
    Set map1 = map2 for auto spectrum.
    
    Inputs:
    ======
    map1, map2: input 2D maps
    dx, dy: the grid size in the 0th, 1st dimension
    window_func: [None, 'blackman'] apply window function. Default not apply (None)
    
    Outputs:
    =======
    P2D: 2D power spectrum, 
        dimension: (Nx+1)/2 if Nx odd; Nx/2 + 1 if Nx even (same for Ny)
    kx_vec, ky_vec: corresponding kx, ky vector
    '''
    
    if map1.shape != map2.shape:
        raise ValueError('two input maps do not have the same shape')

    Nx, Ny = map1.shape
    
    # Window function
    if window_func == 'blackman':
        W = _blackman2D(Nx,Ny)

        map1w = map1 * W
        map2w = map2 * W
    elif window_func== None:
        map1w = map1.copy()
        map2w = map2.copy()
    else:
        raise ValueError('window function name must be None or "blackman". ')
    
    kx_vec_all = np.fft.fftfreq(Nx) * 2 * np.pi / dx
    ky_vec_all = np.fft.fftfreq(Ny) * 2 * np.pi / dy
    ftmap1 = np.fft.fftn(map1w) * dx * dy
    ftmap2 = np.fft.fftn(map2w) * dx * dy
    
    V = Nx * Ny * dx * dy
    P2D_all = np.real(ftmap1 * np.conj(ftmap2)) / V

    Nuse = _kvec_Nuse(Ny)
    ky_vec_all= abs(ky_vec_all[:Nuse])
    P2D_all = P2D_all[:,:Nuse]
    N2D_all = np.ones_like(P2D_all)
    
    # extract only the positive kx part
    N_use = _kvec_Nuse(Nx)
    N_dup = _kvec_Ndup(Nx)
    kx_vec = abs(kx_vec_all[:N_use])
    ky_vec = ky_vec_all.copy()
    P2D = P2D_all[: N_use, :]
    N2D = N2D_all[: N_use, :]
    Pdup = P2D_all[-N_dup:, :]
    Ndup = N2D_all[-N_dup:, :]
    P2D[1:1+N_dup,:] = (P2D[1:1+N_dup,:] + np.flip(Pdup, axis=0)) /2 
    N2D[1:1+N_dup,:] = N2D[1:1+N_dup,:] + np.flip(Ndup, axis=0)
    
    return P2D, kx_vec, ky_vec, N2D


def power_spec_3D(map1, map2, dx, dy, dz, window_func = None):
    '''
    calculate cross power spectrum of a 3-dim Nx x Ny x Nz map1 and map2. 
    Set map1 = map2 for auto spectrum.
    
    Inputs:
    ======
    map1, map2: input 3D maps
    dx, dy, dz: the grid size in the 0th, 1st dimension
    window_func: [None, 'blackman'] apply window function. Default not apply (None)
    
    Outputs:
    =======
    P3D: 3D power spectrum, 
        dimension: (Nx+1)/2 if Nx odd; Nx/2 + 1 if Nx even (same for Ny, Nz)
    kx_vec, ky_vec, kz_vec: corresponding kx, ky vector
    '''
    
    if map1.shape != map2.shape:
        raise ValueError('two input maps do not have the same shape')

    Nx, Ny, Nz = map1.shape
    
    # Window function
    if window_func == 'blackman':
        W = _blackman3D(Nx,Ny,Nz)
        map1w = map1 * W
        map2w = map2 * W
    elif window_func== None:
        map1w = map1.copy()
        map2w = map2.copy()
    else:
        raise ValueError('window function name must be None or "blackman". ')
    
    kx_vec_all = np.fft.fftfreq(Nx) * 2 * np.pi / dx
    ky_vec_all = np.fft.fftfreq(Ny) * 2 * np.pi / dy
    kz_vec_all = np.fft.fftfreq(Nz) * 2 * np.pi / dz
    ftmap1 = np.fft.fftn(map1w) * dx * dy * dz
    ftmap2 = np.fft.fftn(map2w) * dx * dy * dz
    
    V = Nx * Ny * Nz * dx * dy * dz
    P3D_all = np.real(ftmap1 * np.conj(ftmap2)) / V
    
    Nuse = _kvec_Nuse(Nz)
    kz_vec_all= abs(kz_vec_all[:Nuse])
    P3D_all = P3D_all[:,:,:Nuse]
    N3D_all = np.ones_like(P3D_all)
    
    # extract only the positive kx, ky part
    N_use = _kvec_Nuse(Ny)
    N_dup = _kvec_Ndup(Ny)
    ky_vec = abs(ky_vec_all[:N_use])    
    P3D = P3D_all[:,: N_use, :]
    N3D = N3D_all[:,: N_use, :]
    
    Pdup = P3D_all[:,-N_dup:, :]
    Ndup = N3D_all[:,-N_dup:, :]
    P3D[:,1:1+N_dup,:] = (P3D[:,1:1+N_dup,:] + np.flip(Pdup, axis=1))/2 
    N3D[:,1:1+N_dup,:] = N3D[:,1:1+N_dup,:] + np.flip(Ndup, axis=1)
    
    N_use = _kvec_Nuse(Nx)
    N_dup = _kvec_Ndup(Nx)
    kx_vec = abs(kx_vec_all[:N_use])    
    P3D = P3D[: N_use,:, :]
    N3D = N3D[: N_use,:, :]
    
    Pdup = P3D[-N_dup:, :, :]
    Ndup = N3D[-N_dup:, :, :]
    P3D[1:1+N_dup,:,:]  = (P3D[1:1+N_dup,:,:] + np.flip(Pdup, axis=0))/2 
    N3D[1:1+N_dup,:,:]  = N3D[1:1+N_dup,:,:] + np.flip(Ndup, axis=0)
    
    kz_vec = kz_vec_all.copy()
    return P3D, kx_vec, ky_vec, kz_vec, N3D

def PS2D_to_PS1D(P2D, kx_vec, ky_vec, N2D, binedges = [], nbins = 30, logbin = True):
    '''
    Calculate the 1D circular averaged power spectrum
    '''
    kx_arr,ky_arr = np.meshgrid(kx_vec, ky_vec)
    kx_arr = np.swapaxes(kx_arr,1,0)
    ky_arr = np.swapaxes(ky_arr,1,0)
    kr_arr = np.sqrt(kx_arr**2 + ky_arr**2)
    kmin = min(kr_arr[kr_arr>0])
    kmax = max(kr_arr[kr_arr>0])
    
    if len(binedges)==0:
        if logbin: 
            binedges = np.logspace(np.log10(kmin),np.log10(kmax), nbins+1)
            bins = (binedges[:-1] + binedges[1:]) / 2
        else: 
            binedges = np.linspace(kmin, kmax, nbins+1)
            bins = np.sqrt(binedges[:-1] * binedges[1:])
    else:
        bins = (binedges[:-1] + binedges[1:]) / 2

    # make sure the boundary points are not excluded by numerical error
    binedges[0] *=0.99
    binedges[-1] *= 1.01

    P1D = np.histogram(kr_arr, bins=binedges, weights=P2D)[0] \
         / np.histogram(kr_arr, bins=binedges)[0]
    
    N1D = np.histogram(kr_arr, bins=binedges, weights=N2D)[0]
    return P1D, bins, N1D

def PS3D_to_PS1D(P3D, kx_vec, ky_vec, kz_vec, N3D, binedges = [], nbins = 30, logbin = True):
    '''
    Calculate the 1D circular averaged power spectrum
    '''
    kx_arr,ky_arr,kz_arr = np.meshgrid(kx_vec, ky_vec, kz_vec)
    kx_arr = np.swapaxes(kx_arr,1,0)
    ky_arr = np.swapaxes(ky_arr,1,0)
    kz_arr = np.swapaxes(kz_arr,1,0)
    kr_arr = np.sqrt(kx_arr**2 + ky_arr**2 + kz_arr**2)
    kmin = min(kr_arr[kr_arr>0])
    kmax = max(kr_arr[kr_arr>0])
    
    if len(binedges)==0:
        if logbin: 
            binedges = np.logspace(np.log10(kmin),np.log10(kmax), nbins+1)
            bins = (binedges[:-1] + binedges[1:]) / 2
            
        else: 
            binedges = np.linspace(kmin, kmax, nbins+1)
            bins = np.sqrt(binedges[:-1] * binedges[1:])
            
    else:
        bins = (binedges[:-1] + binedges[1:]) / 2
        
    # make sure the boundary points are not excluded by numerical error
    binedges[0] *=0.99
    binedges[-1] *= 1.01
    
    P1D = np.histogram(kr_arr, bins=binedges, weights=P3D)[0] \
         / np.histogram(kr_arr, bins=binedges)[0]
    
    N1D = np.histogram(kr_arr, bins=binedges, weights=N3D)[0]
    
    return P1D, bins, N1D

def PS3D_to_PS2D(P3D, kx_vec, ky_vec, kz_vec, N3D, \
                 binedges_p = [], binedges_l = [], nbins_p = 30, nbins_l = 30, logbin = True):
    '''
    Calculate the 1D circular averaged power spectrum
    '''
    kx_arr,ky_arr,kz_arr = np.meshgrid(kx_vec, ky_vec, kz_vec)
    kx_arr = np.swapaxes(kx_arr,1,0)
    ky_arr = np.swapaxes(ky_arr,1,0)
    kz_arr = np.swapaxes(kz_arr,1,0)
    
    kl_arr = kz_arr.flatten()
    klmin = min(kl_arr[kl_arr>0])
    klmax = max(kl_arr[kl_arr>0])

    kp_arr = np.sqrt(kx_arr**2 + ky_arr**2).flatten()
    kpmin = min(kp_arr[kp_arr>0])
    kpmax = max(kp_arr[kp_arr>0])
    
    if len(binedges_l)==0:
        if logbin: 
            binedges_l = np.logspace(np.log10(klmin),np.log10(klmax), nbins_l+1)
            bins_l = (binedges_l[:-1] + binedges_l[1:]) / 2
        
        else: 
            binedges_l = np.linspace(klmin, klmax, nbins_l+1)
            bins_l = np.sqrt(binedges_l[:-1] * binedges_l[1:])
            
    else:
        bins_l = (binedges_l[:-1] + binedges_l[1:]) / 2

    binedges_l[0] *= 0.99
    binedges_l[-1] *= 1.01

    if len(binedges_p)==0:
        if logbin: 
            binedges_p = np.logspace(np.log10(kpmin),np.log10(kpmax), nbins_p+1)
            bins_p = (binedges_p[:-1] + binedges_p[1:]) / 2
            
        else: 
            binedges_p = np.linspace(kpmin, kpmax, nbins_p+1)
            bins_p = np.sqrt(binedges_p[:-1] * binedges_p[1:])
            
    else:
        bins_p = (binedges_p[:-1] + binedges_p[1:]) / 2
    
    binedges_p[0] *= 0.99
    binedges_p[-1] *= 1.01
        
    
    P2D = np.histogram2d(kp_arr,kl_arr, bins=[binedges_p,binedges_l], weights=P3D.flatten())[0] \
         / np.histogram2d(kp_arr,kl_arr, bins=[binedges_p,binedges_l])[0]
    
    N2D = np.histogram2d(kp_arr,kl_arr, bins=[binedges_p,binedges_l], weights=N3D.flatten())[0]
    
    return P2D, bins_p, bins_l, N2D

def _blackman1D(N):
    W = np.blackman(N//2)
    W = scipy.signal.fftconvolve(W,W,mode='full')
    W = np.append(W,0)
    if N%2==1: W = np.append(0,W)
    W = W / np.sqrt(np.mean(W**2.))
    W = W.reshape(-1,1)
    return W

def _blackman2D(Nx, Ny):
    Wx = _blackman1D(Nx)
    Wy = _blackman1D(Ny)
    W = np.matmul(Wx,Wy.T)
    return W

def _blackman3D(Nx, Ny, Nz):   
    Wx = _blackman1D(Nx)
    Wy = _blackman1D(Ny)
    Wz = _blackman1D(Nz)

    Wxx,Wyy,Wzz = np.meshgrid(Wx,Wy,Wz)
    W = Wxx*Wyy*Wzz
    W = np.swapaxes(W,0,1)    
    
    return W

def _kvec_Nuse(N):
    # number of useful k space values
    if N % 2 == 1: Nuse = int((N - 1) / 2) + 1
    else: Nuse = int((N - 2) / 2) + 2
    return Nuse

def _kvec_Ndup(N):
    # number of duplicated (negative) k values
    if N % 2 ==1: Ndup = int((N - 1) / 2)
    else: Ndup = int((N - 2) / 2)
    return Ndup

def map2D_from_PS1D(PS1D, k_vec, Nx, Ny, dx, dy):
    kx_vec = np.fft.fftfreq(Nx) * 2 * np.pi / dx
    ky_vec = np.fft.fftfreq(Ny) * 2 * np.pi / dy
    kxx,kyy = np.meshgrid(kx_vec,ky_vec)
    kxx = np.swapaxes(kxx,0,1)
    kyy = np.swapaxes(kyy,0,1)
    k_arr = np.sqrt(kxx**2 + kyy**2)
    P_arr = np.interp(k_arr,k_vec,PS1D)
    V = Nx * Ny * dx * dy
    
    real_part = np.random.normal(size=k_arr.shape)
    im_part = np.random.normal(size=k_arr.shape)

    ft_map = (real_part + im_part*1.0j) * np.sqrt(abs(P_arr * V)) / dx / dy
    map2D = np.fft.ifftn(ft_map)
    map2D = np.real(map2D)
    return map2D

def map3D_from_PS1D(PS1D, k_vec, Nx, Ny, Nz, dx, dy, dz):
    kx_vec = np.fft.fftfreq(Nx) * 2 * np.pi / dx
    ky_vec = np.fft.fftfreq(Ny) * 2 * np.pi / dy
    kz_vec = np.fft.fftfreq(Nz) * 2 * np.pi / dz
    kxx,kyy,kzz = np.meshgrid(kx_vec,ky_vec,kz_vec)
    kxx = np.swapaxes(kxx,0,1)
    kyy = np.swapaxes(kyy,0,1)
    kzz = np.swapaxes(kzz,0,1)
    k_arr = np.sqrt(kxx**2 + kyy**2 + kzz**2)
    P_arr = np.interp(k_arr,k_vec,PS1D)
    V = Nx * Ny * Nz * dx * dy * dz
    
    real_part = np.random.normal(size=k_arr.shape)
    im_part = np.random.normal(size=k_arr.shape)

    ft_map = (real_part + im_part*1.0j) * np.sqrt(abs(P_arr * V)) / dx / dy / dz
    map3D = np.fft.ifftn(ft_map)
    map3D = np.real(map3D)
    return map3D

'''
def map2D_from_PS1D(PS1D, k_vec, Nx, Ny, dx, dy):
    kx_vec = np.fft.fftfreq(Nx) * 2 * np.pi / dx
    ky_vec = np.fft.fftfreq(Ny) * 2 * np.pi / dy
    kx_vec = np.arange(0, Nx//2 + 1) * 2 * np.pi / (Nx * dx)
    ky_vec = np.fft.fftfreq(Ny) * Ny * 2 * np.pi / (Ny * dy)
    k_arr = np.sqrt(kx_vec[np.newaxis,:]**2 + ky_vec[:,np.newaxis]**2)
    P_arr = np.interp(k_arr,k_vec,PS1D)
    V = Nx * Ny * dx * dy
    
    real_part = np.random.normal(size=k_arr.shape)
    im_part = np.random.normal(size=k_arr.shape)

    ft_map = (real_part + im_part*1.0j) * np.sqrt(abs(P_arr * V / 2)) / dx / dy
    map2D = np.fft.irfft2(ft_map).T
    
    return map2D
'''
