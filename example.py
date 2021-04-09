from matplotlib import pyplot as plt
import numpy as np
from kilonova import calc_lightcurve
import timeit

# constants
c = 2.99792458e10
day = 86400.
Msun = 1.9885e33
sigma_SB = 5.670373e-5

# parameters
##########ejecta parameters for thermalization
Mej = 0.05*Msun
vej = 0.1*c
n = 4.5 # Is this power law too sharp maybe?
alpha_min = 1.#v_min = alpha_min * vej
alpha_max = 4.0#v_max = alpha_max * vej

   
#input parameters for the calculation of a light curve
kappa_low = 0.5  #opacity [cm^2/g] for v > v_kappa
kappa_high = 3.0 #opacity [cm^2/g] for v < v_kappa
be_kappa = 0.2

dt = 0.01 #days
tmax = 1 #days

LC = calc_lightcurve(Mej,vej,alpha_min,alpha_max,n,kappa_low,kappa_high,be_kappa,dt,tmax)
t = LC['t']
T = LC['T']
L = LC['LC']

# Benchmark it
timing = int(np.round(1e6 * np.median(timeit.repeat('calc_lightcurve(Mej,vej,alpha_min,alpha_max,n,kappa_low,kappa_high,be_kappa,dt,tmax)', globals=globals(), number=1, repeat=10))))

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(4, 4))
fig.suptitle(f'Run time: {timing} µs')
ax1.plot(t, T)
ax2.plot(t, L)
ax2.set_xlabel('Time (d)')
ax1.set_ylabel('Temperature (K)')
ax2.set_ylabel('Luminosity (erg / s)')
ax1.set_xlim(0.01, 1)
ax1.set_ylim(5e3, 2e4)
ax2.set_ylim(1e39, 5e42)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax2.set_yscale('log')
fig.tight_layout()
fig.savefig('example.png', dpi=300)
