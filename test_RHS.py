from kilonova_heating_rate import _heating_rate_korobkin, _get_heating_rate_beta, heat
import numpy as np
import astropy.units as u
import astropy.constants as c

# import sys
# sys.path.insert(0,'kilonova_heating_rate')
# import heat as ht

day = 86400.
t = np.linspace(0.01,1,100)*day
mass = (0.05*u.Msun).to_value(u.g)
vmin = (0.1*c.c).to_value(u.cm/u.s)
vmax = 4*vmin


Amin = 85
Amax = 209
kappa_effs = np.load('input_files/kappa_effs_A85_238.npy')
ffraction = np.load('input_files/ffraction.npy')
n = 4.5

heat_korobkin = _heating_rate_korobkin(t)
#heating_time,heating_beta = _get_heating_rate_beta(t,mass,vmin,vmax)
beta=heat.calc_heating_rate(mass,vmin,vmax,Amin,Amax,ffraction,kappa_effs,n)

import matplotlib.pyplot as plt

heat_time = np.array(beta['t'])
heat_rate = np.array(beta['electron_th'])+np.array(beta['gamma_th'])


plt.loglog(t,heat_korobkin)
#plt.loglog(heating_time,heating_beta)
plt.loglog(heat_time,heat_rate)