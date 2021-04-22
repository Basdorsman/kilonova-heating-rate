import numpy as np
from astropy import constants as const
from numba import jit


c = const.c.cgs.value
eV = 1.6021766e-12
MeV = 1.0e6*eV
me = const.m_e.cgs.value
e = const.e.esu.value
mu = const.u.cgs.value
E = const.c*const.c*const.m_e
me_MeV = E.to('MeV').value

kappa_beta = 1.0 #MeV cm^2/g
kappa_alpha = -0.15
#alpha_max = 4.
#alpha_min = 1.0
#n = 4.
gam_t = 0.15#0.15
gam_t_sf = -0.98
x_ad = 2.#2.0
#me_MeV = 0.511
#E is kinetic energy in MeV

@jit(nopython=False)
def calc_ad(E):
    p = np.sqrt(((E/me_MeV)+1.)*((E/me_MeV)+1.)-1.)
    return 1. + p*p/3./np.sqrt(p*p+1.)/(np.sqrt(p*p+1.)-1.)

#fitting function of stopping power time beta for Xe 0.01-10MeV
a_kappa = 2.1

@jit(nopython=True) 
def calc_kappa_beta(Etmp):
    E = Etmp*MeV
    Z = 54.
    A = 130.
    Imean =  482.*eV
    gamma = 1.0+E/(me*c*c)
    tmp3 = 1.0-1.0/gamma/gamma
    beta = np.sqrt(tmp3)
    tmp1 = gamma*gamma*me*beta*beta*c*c*E/Imean/Imean/2.
    tmp = np.log(tmp1) - (2./gamma-1.+beta*beta)*np.log(2) + 1. -beta*beta + (1.-1./gamma)*(1.-1./gamma)/8.
    omegap = np.sqrt(4.*np.pi*1.0e5*e*e/me)
    tmp_f = np.log(1.123*me*c*c*c*beta*beta*beta/(e*e*omegap))
    kappabeta = beta*2.*np.pi*Z*np.power(e,4.)*tmp/(me*c*c*beta*beta)/MeV/(A*mu)\
                + beta*4.*np.pi*np.power(e,4.)*tmp_f/(me*c*c*beta*beta)/MeV/(A*mu)
    return kappabeta


@jit(nopython=False) 
def calc_thermalization_time(E0, Mej, vej, Aave, alpha_max, alpha_min, n):
    kappa_beta = calc_kappa_beta(E0)
    rho_inv = (1.-np.power(alpha_max,-n+3.))/(n-3.)+0.5*(1.-alpha_min*alpha_min)
    tmp = ((1.0-np.power(alpha_max,-2.*n+3.))/(2.*n-3.) + 1.-alpha_min)/(rho_inv*rho_inv)
    gamma_ad = calc_ad(E0)
    te2 = tmp*kappa_beta*c*Mej/(4.*np.pi*E0*vej*vej*vej*3.0*(gamma_ad-1.))

    return np.sqrt(te2)


@jit(nopython=False) 
def calc_zero_energy(tau0, args=()):
    tau1, E0 = args
    gamma_ad = calc_ad(E0)
    x_ad = 3.*(gamma_ad-1.)
    tmp = (1.0+gam_t)/(2.-(gam_t+1.)*x_ad)
    tmpp = -2.0 + (1.+gam_t)*x_ad
    x = tau1/tau0
    return 1.0 + tmp*(np.power(x,tmpp)-1.0)/(tau0*tau0)


#atau1 = x_ad
#ntau = -gam_t
#atau1 = x_ad

@jit(nopython=False) 
def epsilon_tau(tau0, tau, E0):
    x = tau/tau0
    gamma_ad = calc_ad(E0)
    atau1 = 3.*(gamma_ad-1.)
    ntau = -gam_t
    pp = -2. + atau1*(1.-ntau)
    ppp = 1./(1.-ntau)
    tmp = np.power(tau0,-atau1*(1.-ntau))*(1.-ntau)/(atau1*(-ntau+1.)-2.)*(np.power(tau,pp)-np.power(tau0,pp))
    tmpp =  np.power(x,-atau1*(1.-ntau))*(1. - tmp)
    #
#    return e_tmp
    if(tmpp > 0.):
        e_tmp =  np.power(tmpp,ppp)
        if(e_tmp > 0.01):
            return np.power(e_tmp,ntau) #note gam_t is chosen to be constant
        else:
            return 0.
    else:
        return 0.

@jit(nopython=False) 
def calc_gamma_deposition(kappa_eff, t, Mej, vej, alpha_min, alpha_max, n):
    w = alpha_min/alpha_max
    k = n - 3.0
    alpha_gam = 0.1*w + 0.003*k/w
    t0 = np.sqrt(alpha_gam*kappa_eff*Mej/(vej*vej))
   # print t0/day
    return 1.0 - np.exp(-t0*t0/(t*t))
