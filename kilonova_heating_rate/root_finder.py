# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 19:17:39 2021

@author: super
"""
from numba import jit


@jit(nopython=False)
def find_root(f, x0, args=(), multip_factor=1.5, iterations=5):
    """Find root, assuming a monotonically increasing function, as well as
    only defined for positive values."""
    # find another point with opposite sign, assuming x0 > 0.
    x1 = x0
    if f(x1, args)>0:
        while f(x1, args)>0:
            x1 /= multip_factor
    elif f(x1, args)<0:
        while f(x1, args)<0:
            x1 *= multip_factor 
    root = secant_method(f, x0, x1, iterations, args)
    return root

@jit(nopython=False)
def secant_method(f, x0, x1, iterations, args):
    """Return the root calculated using the secant method."""
    for i in range(iterations):
        if x0 == x1:
            return x1
        else:
            x2 = x1 - f(x1, args) * (x1 - x0) / float(f(x1, args) - f(x0, args))
            x0, x1 = x1, x2
    return x2

# def f_example(x, args=()):
#     power, offset = args
#     return x**power - offset


# import astropy.constants as const

# c = const.c.cgs.value
# eV = 1.6021766e-12
# MeV = 1.0e6*eV
# me = const.m_e.cgs.value
# e = const.e.esu.value
# mu = const.u.cgs.value
# E = const.c*const.c*const.m_e
# me_MeV = E.to('MeV').value
# kappa_beta = 1.0 #MeV cm^2/g
# kappa_alpha = -0.15
# #alpha_max = 4.
# #alpha_min = 1.0
# #n = 4.
# gam_t = 0.15#0.15
# gam_t_sf = -0.98
# x_ad = 2.#2.0
# #me_MeV = 0.511
# #E is kinetic energy in MeV

# def calc_ad(E):
#     p = np.sqrt(((E/me_MeV)+1.)*((E/me_MeV)+1.)-1.)
#     return 1. + p*p/3./np.sqrt(p*p+1.)/(np.sqrt(p*p+1.)-1.)

# def calc_zero_energy(tau0, args=()):
#     tau1, E0 = args
#     gamma_ad = calc_ad(E0)
#     x_ad = 3.*(gamma_ad-1.)
#     tmp = (1.0+gam_t)/(2.-(gam_t+1.)*x_ad)
#     tmpp = -2.0 + (1.+gam_t)*x_ad
#     x = tau1/tau0
#     return 1.0 + tmp*(np.power(x,tmpp)-1.0)/(tau0*tau0)

# import numpy as np
# tau0 = np.linspace(0.09,3,100)

# y = calc_zero_energy(tau0, args=(0.1,0.1))

# import matplotlib.pyplot as plt
# plt.plot(tau0,y)

# tau0_test = 0.01

# print(find_root(calc_zero_energy, 0.1, args=(0.1, 0.1)))