import numpy as np
from kilonova import calc_lightcurve

# constants
c = 2.99792458e10
day = 86400.
Msun = 1.9885e33
sigma_SB = 5.670373e-5

# parameters
##########Start input parameters
Amin = 84
Amax = 209 #lanthanide rich

##########ejecta parameters for thermalization
Mej = 0.05*Msun
vej = 0.1*c
n = 4.5 # Is this power law too sharp maybe?
alpha_max = 4.0#v_max = alpha_max * vej
alpha_min = 1.#v_min = alpha_min * vej
   
#input parameters for the calculation of a light curve
kappa_low = 0.5  #opacity [cm^2/g] for v > v_kappa
kappa_high = 3.0 #opacity [cm^2/g] for v < v_kappa
be_kappa = 0.2

dt = 0.01 #days
tmax = 1 #days


LC = calc_lightcurve(Mej,vej,alpha_min,alpha_max,n,kappa_low,kappa_high,be_kappa,dt,tmax)

print(LC['LC'])
