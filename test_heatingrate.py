import pandas as pd
import numpy as np

# fchain = pd.read_csv('input_files/table/85.txt',delim_whitespace=True,header=None)
# testchain = np.loadtxt('input_files/table/85.txt')
# Amax_beta=209
# fchains = np.zeros((Amax_beta+1,7,7))
# chains_length = np.zeros(Amax_beta+1)
# chains_width = np.zeros(Amax_beta+1)

# for A in range(50,Amax_beta+1) :
#     filename = 'input_files/table/'+str(A)+'.txt'
#     fchain = np.loadtxt(filename)
#     try:
#         length, width = fchain.shape
#     except:
#         length = 1
#         width = len(fchain)
#     chains_length[A]=length
#     chains_width[A]=width
#     fchains[A,:length,:width] = fchain
    

    
# fchains_reshaped = fchains.reshape(fchains.shape[0],-1)
# np.savetxt("tables_reshaped.dat", fchains_reshaped)
# np.savetxt("tables_length.dat",chains_length)
# fchains_old_shape = fchains_reshaped.reshape(Amax_beta+1,7,7)

from kilonova_heating_rate import _heating_rate_beta
import astropy.units as u
import astropy.constants as c

mass = (0.05 * u.Msun).to_value(u.g)
velocities = (np.asarray([0.1, 0.2, 0.4]) * c.c).to_value(u.m/u.s)
opacities = (np.asarray([3.0, 0.5]) * u.cm**2 / u.g).value
n = 4.5

ffraction = np.loadtxt('kilonova_heating_rate/ffraction.dat')
kappa_effs = np.loadtxt('kilonova_heating_rate/kappa_effs_A85_238.dat')

heating_time,heating_rate =_heating_rate_beta(mass, velocities[0], velocities[-1], Amin=85, Amax=209,
                        ffraction=ffraction, kappa_effs=kappa_effs, n=n)