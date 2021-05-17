#!/usr/bin/env python
import timeit

from astropy import constants as c
from astropy import units as u
from kilonova_heating_rate import lightcurve
from matplotlib import pyplot as plt
import numpy as np
import synphot
import dorado.sensitivity
bp_dorado = dorado.sensitivity.bandpasses.NUV_D

mass = 0.05 * u.Msun
velocities = np.asarray([0.1, 0.2, 0.4]) * c.c
opacities = np.asarray([3.0, 0.5]) * u.cm**2 / u.g
n = 4.5
t = np.linspace(0.02, 1, 14) * u.day
heating = 'beta'


L, T, r = lightcurve(t, mass, velocities, opacities, n, heating_function=heating)

timing_korobkin = int(np.round(1e3 * np.median(timeit.repeat(
    'lightcurve(t, mass, velocities, opacities, n)',
    globals=globals(), number=1, repeat=10))))

print(f'light curve korobkin = {timing_korobkin} ms')

timing_beta = int(np.round(1e3 * np.median(timeit.repeat(
    'lightcurve(t, mass, velocities, opacities, n, heating_function=heating)',
    globals=globals(), number=1, repeat=10))))

print(f'light curve beta = {timing_beta} ms')

# Evaluate flux in band at 40 Mpc for a few different filters.
DL = 40 * u.Mpc

seds = [
    synphot.SourceSpectrum(synphot.BlackBody1D, temperature=TT)
    * np.pi * (rr / DL).to(u.dimensionless_unscaled)**2
    for TT, rr in zip(T, r)]
    
timing_seds = int(np.round(1e3 * np.median(timeit.repeat(
    '[synphot.SourceSpectrum(synphot.BlackBody1D, temperature=TT)*np.pi*(rr / DL).to(u.dimensionless_unscaled)**2 for TT, rr in zip(T, r)]',
    globals=globals(), number=1, repeat=10))))

print(f'seds = {timing_seds} ms')

abmag_dorado = [synphot.Observation(sed, bp_dorado).effstim(u.ABmag).value for sed in seds]

timing_abmags = int(np.round(1e3 * np.median(timeit.repeat(
    '[synphot.Observation(sed, bp_dorado).effstim(u.ABmag).value for sed in seds]',
    globals=globals(), number=1, repeat=10))))

print(f'abmag = {timing_abmags} ms')
print(f'total time for beta likelihood = {timing_beta+timing_abmags+timing_seds} ms')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(4, 6))
fig.suptitle(f'{heating}')
ax1.plot(t, T)
ax2.plot(t, L)
ax3.plot(t,abmag_dorado,label='dorado')
ax3.set_xlabel(f'Time ({t.unit})')
ax1.set_ylabel(f'Temperature ({T.unit})')
ax2.set_ylabel(f'Luminosity ({L.unit})')
ax3.set_ylabel('Apparent mag (AB)')
ax1.set_xlim(0.01, 1)
ax1.set_ylim(1e3, 1e4)
ax2.set_ylim(1e40, 1e42)
ax3.set_ylim(19, 25)
ax3.invert_yaxis()
ax1.set_yscale('log')
ax2.set_yscale('log')
ax3.legend()
fig.tight_layout()
fig.savefig('example_new.png', dpi=300)
