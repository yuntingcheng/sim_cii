from cosmo_tools import *
from sympy import Symbol, solve, exp

class CIB_model:
    def __init__(self, alpha = 0.36, beta = 1.75, gamma = 1.7, delta = 3.6,\
                 T0 = 24.4, Meff = 10**12.6, sig2_lm = 0.5, L0 = 0.02):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.T0 = T0
        self.Meff = Meff
        self.sig2_lm = sig2_lm
        self.L0 = L0

    def LIR(self, M, z):
        if isinstance(M, list) or isinstance(M, np.ndarray):
            lir = np.asarray([self._Sigma(Mi) for Mi in M])*\
                    self.L0 * self._Phi(z) * self._Theta_int(z)
        else:
            lir = self.L0 * self._Phi(z) * self._Sigma(M) * self._Theta_int(z)
        return lir * u.L_sun
    
    def SFR(self, M, z):
        return self.LIR(M,z) * 1.7e-10 * u.M_sun / u.L_sun / u.yr

    def SFRD(self, z, Mmin = 1e6, Mmax = 1e15):
        
        def _SFRD(z,Mmin, Mmax):
            dndm_arr,M_arr = HMFz(z).dndm(Mmin=Mmin, Mmax = Mmax)
            dlnm = np.log(M_arr[1]) - np.log(M_arr[0])
            SFR_arr = self.SFR(M_arr,z).value
            sfrd = dlnm * np.sum(dndm_arr * M_arr * SFR_arr)
            sfrd *= u.M_sun / u.yr / (u.Mpc / u.h)**3
            sfrd *= cosmo.h**3 / u.h**3
            return sfrd
        
        if isinstance(z, list) or isinstance(z, np.ndarray):
            sfrd_list = [_SFRD(zi, Mmin, Mmax) for zi in z]
            sfrd = np.array([sfrd_list[i].value for i in range(len(z))]) * sfrd_list[0].unit
        else:
            sfrd = _SFRD(z, Mmin, Mmax)
            
        return sfrd
    
    def Lcii(self, M, z):
        return 1.59e-3 * self.LIR(M,z)
    
    def Lco(self, M, z, j):
        L1 = 3.2e4 * u.L_sun * (self.SFR(M,z) / u.M_sun * u.yr)**(3./5)
        return L1 * self._obreschkow_Lj_ratio(z,j)
    
    def I_line(self, z, j, Mmin = 1e6, Mmax = 1e15):
        
        def _Ltot(z,j,Mmin, Mmax):
            dndm_arr,M_arr = HMFz(z).dndm(Mmin=Mmin, Mmax = Mmax)
            dlnm = np.log(M_arr[1]) - np.log(M_arr[0])
            if j==0:
                L_arr = self.Lcii(M_arr,z)
            else:
                L_arr = self.Lco(M_arr,z,j)
            Ltot = dlnm * np.sum(dndm_arr * M_arr * L_arr)
            Ltot = Ltot.value
            Ltot = Ltot * u.Lsun / (u.Mpc/u.h)**3
            return Ltot
        
        def _Itot(z,j, Mmin, Mmax):
            if j==0:
                lambda_rest = spec_lines.CII
            else:
                lambda_rest = spec_lines.CO(j)

            Ltot = _Ltot(z,j, Mmin, Mmax)
            cd = cosmo_dist(z)
            y = lambda_rest* (1+z)**2 / cd.H
            F = Ltot / 4 / np.pi / cosmo.luminosity_distance(z) **2
            I = F * y * cd.comoving_distance**2 
            I = I / u.s / u.Hz / u.sr
            I = I.to(u.Jy / u.sr)

            return I
        
        if isinstance(z, list) or isinstance(z, np.ndarray):
            Iline_list = [_Itot(zi,j, Mmin, Mmax) for zi in z]
            Iline = np.array([Iline_list[i].value for i in range(len(z))]) * Iline_list[0].unit
        else:
            Iline = _Itot(z,j,Mmin,Mmax)
            
        return Iline

    def bias_line(self, z, j, Mmin = 1e6, Mmax = 1e15):
        
        def _bline(z,j, Mmin, Mmax):
            dndm_arr,M_arr = HMFz(z).dndm(Mmin=Mmin, Mmax = Mmax)
            dlnm = np.log(M_arr[1]) - np.log(M_arr[0])
            if j==0:
                L_arr = self.Lcii(M_arr,z).value
            else:
                L_arr = self.Lco(M_arr,z,j).value

            bhalo_arr = HMFz(z).bias(m_arr = M_arr)[0]

            Ltot = dlnm * np.sum(dndm_arr * M_arr * L_arr)
            bLtot = dlnm * np.sum(dndm_arr * M_arr * L_arr * bhalo_arr)
            bline = bLtot / Ltot
            
            return bline
        
        if isinstance(z, list) or isinstance(z, np.ndarray):
            bline = [_bline(zi, j, Mmin, Mmax) for zi in z]
        else:
            bline = _bline(z, j, Mmin, Mmax)
            
        return bline

    def Pshot_line_scalar(self, z, j, Mmin = 1e6, Mmax = 1e15):
        
        def _L2tot(z,j,Mmin, Mmax):
            dndm_arr,M_arr = HMFz(z).dndm(Mmin=Mmin, Mmax = Mmax)
            dlnm = np.log(M_arr[1]) - np.log(M_arr[0])
            if j==0:
                L_arr = self.Lcii(M_arr,z)
            else:
                L_arr = self.Lco(M_arr,z,j)
            Ltot = dlnm * np.sum(dndm_arr * M_arr * L_arr**2)
            Ltot = Ltot.value
            Ltot = Ltot * u.Lsun**2 / (u.Mpc/u.h)**3
            return Ltot
        
        def _Pshot(z,j, Mmin, Mmax):
            if j==0:
                lambda_rest = spec_lines.CII
            else:
                lambda_rest = spec_lines.CO(j)

            L2tot = _L2tot(z,j, Mmin, Mmax)
            cd = cosmo_dist(z)
            y = lambda_rest* (1+z)**2 / cd.H
            DL = cosmo.luminosity_distance(z)
            DA = cd.comoving_distance
            Pshot = L2tot * (y * DA**2 / (4*np.pi*DL**2))**2
            Pshot = Pshot / u.s**2 / u.Hz**2 / u.sr**2
            Pshot = Pshot.to(u.Jy**2 / u.sr**2 * u.Mpc**3 / u.h**3)

            return Pshot
        
        if isinstance(z, list) or isinstance(z, np.ndarray):
            Pshot_list = [_Pshot(zi,j, Mmin, Mmax) for zi in z]
            Pshot = np.array([Pshot_list[i].value for i in range(len(z))]) * Pshot_list[0].unit
        else:
             Pshot= _Pshot(z,j, Mmin, Mmax)
            
        return Pshot

    def Pshot_line(self, z, j, kmin = 1e-3, kmax = 1e1, dlog10k = 0.01, k_arr = [] ,Mmin = 1e6, Mmax = 1e15):
        
        if len(k_arr)==0:
            k_arr = 10**np.arange(np.log10(kmin), np.log10(kmax), dlog10k)
        Pshot = self.Pshot_line_scalar(z, j, Mmin = Mmin, Mmax = Mmax) * np.ones_like(k_arr)
        k_arr *= (u.h / u.Mpc)
        
        return Pshot, k_arr
        
    def Pclus_line(self, z, j, kmin = 1e-3, kmax = 1e1, dlog10k = 0.01, k_arr = [] ,Mmin = 1e6, Mmax = 1e15):
        
        if len(k_arr)==0:
            k_arr = 10**np.arange(np.log10(kmin), np.log10(kmax), dlog10k)
            
        Pshot = self.Pshot_line(z, j, k_arr = k_arr ,Mmin = Mmin, Mmax = Mmax)
        I = self.I_line(z, j, Mmin = Mmin, Mmax = Mmax)
        b = self.bias_line(z, j, Mmin = Mmin, Mmax = Mmax)
        Plin,_ = HMFz(z).P(k_arr = k_arr)
        Pclus = I**2 * b**2 * Plin
        k_arr *= (u.h / u.Mpc)
        Pclus *= (u.Mpc / u.h)**3
        return Pclus, k_arr
    
    def P_line(self, z, j, kmin = 1e-3, kmax = 1e1, dlog10k = 0.01, k_arr = [] ,Mmin = 1e6, Mmax = 1e15):
        
        if len(k_arr)==0:
            k_arr = 10**np.arange(np.log10(kmin), np.log10(kmax), dlog10k)

        Pshot,_ = self.Pshot_line(z, j, k_arr = k_arr ,Mmin = Mmin, Mmax = Mmax)
        Pclus,_ = self.Pclus_line(z, j, k_arr = k_arr ,Mmin = Mmin, Mmax = Mmax)
        P = Pshot + Pclus
        k_arr = k_arr * (u.h / u.Mpc)
        return P, k_arr
    
    def _obreschkow_Lj_ratio(self,z,j):
        # obreschkow et al 2009 eq (16)
        
        def _tauj(j):
            tauc = 2.
            vco = 115 * u.GHz
            Te = 17. * u.K
            x = float(const.h * vco / const.k_B / Te)
            tauj = 7.2 * tauc * np.exp(-x * j**2 / 2)* np.sinh(x * j / 2)
            return tauj
        
        def _ell(z,j):
            vco = 115 * u.GHz
            Te = 17. * u.K
            Tcmb = cosmo.Tcmb(z)
            x = float(const.h * vco / const.k_B / Te)
            xcmb = float(const.h * vco / const.k_B / Tcmb)
            ell = (j**4 / (np.exp(x*j) - 1)) - (j**4 / (np.exp(xcmb*j) - 1))
            return ell

        Lj = (1 - np.exp(- _tauj(j))) * _ell(z,j)
        L1 = (1 - np.exp(- _tauj(1))) * _ell(z,1)

        return Lj/L1        
        
    
    def _Phi(self,z):
        return (1 + z)**self.delta
    
    def _Sigma(self, M):
        return M/np.sqrt(2 * np.pi * self.sig2_lm) \
                * np.exp(-(np.log10(M) - np.log10(self.Meff * cosmo.h))**2 / 2 / self.sig2_lm)
        
    
    def _Theta_int(self,z):
        x = Symbol("x")
        x = float(solve(x/(1 - exp(-x)) - (self.gamma + self.beta + 3))[0])
        
        T = self.T0 * (1 + z)**self.alpha * u.K
        v0 = x * const.k_B * T / const.h
        v0 = float(v0.value)* u.Hz
        vmax,vmin = ([8, 1000] * u.um).to(u.Hz, equivalencies=u.spectral())
        vmin /= v0
        vmax /= v0
        
        def _Theta(v):
            th1 = v**(3 + self.beta) * (np.exp(x) - 1) / (np.exp(x * v) - 1)
            th2 = v**(-self.gamma)
            th = th1
            th[v > np.ones_like(v)] = th2[v > np.ones_like(v)]
            return th
        v_arr = np.logspace(np.log10(1e-2),np.log10(1e2),1000)
        th_arr = _Theta(v_arr)
        int_all = np.sum(v_arr * th_arr)
        int_part = np.sum(v_arr[(v_arr > vmin) & (v_arr < vmax)] \
                          * th_arr[(v_arr > vmin) & (v_arr < vmax)])
        
        return int_part/int_all
