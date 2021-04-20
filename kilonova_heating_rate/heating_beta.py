import numpy as np
import heat as ht
from astropy import constants as const


c = const.c.cgs.value
day = 86400.
Msun = const.M_sun.cgs.value


##########Start input parameters
Amin = 85
Amax = 209

##########ejecta parameters for thermalization
Mej = 0.05*Msun
vmin = 0.1*c
vmax = 0.4*c
n = 4.5



kappa_effs = np.load('input_files/kappa_effs_A85_238.npy')
ffraction = np.load('input_files/ffraction.npy')


def _heating_rate_beta(t,Mej,vmin,vmax,Amin,Amax,ffraction,kappa_effs,n):
    beta = ht.calc_heating_rate(Mej,vmin,vmax,Amin,Amax,ffraction,kappa_effs,n)
    heat_time = np.array(beta['t'])
    heat_rate = np.array(beta['electron_th'])+np.array(beta['gamma_th'])
    return np.interp(t,heat_time,heat_rate)


t = np.linspace(0.1,1,100)
heating_rate = _heating_rate_beta(t, Mej,vmin,vmax,Amin,Amax,ffraction,kappa_effs,n)


import matplotlib.pyplot as plt
plt.loglog(t, heating_rate)