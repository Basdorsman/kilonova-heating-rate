#!/usr/bin/env python
import timeit

from astropy import constants as c
from astropy import units as u
from kilonova import lightcurve
from matplotlib import pyplot as plt
import numpy as np

mass = 0.05 * u.Msun
velocities = np.asarray([0.1, 0.2, 0.4]) * c.c
opacities = np.asarray([3.0, 0.5]) * u.cm**2 / u.g
n = 4.5
t = np.geomspace(0.02, 0.98, 39) * u.day

L, T = lightcurve(t, mass, velocities, opacities, n)

# Benchmark it
timing = int(np.round(1e6 * np.median(timeit.repeat(
    'bolometric_lightcurve(t, mass, velocities, opacities, n)',
    globals=globals(), number=1, repeat=10))))

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(4, 4))
fig.suptitle(f'Run time: {timing} µs')
ax1.plot(t, T)
ax2.plot(t, L)
ax2.set_xlabel(f'Time ({t.unit})')
ax1.set_ylabel(f'Temperature ({T.unit})')
ax2.set_ylabel(f'Luminosity ({L.unit})')
ax1.set_xlim(0.01, 1)
ax1.set_ylim(5e3, 2e4)
ax2.set_ylim(1e39, 5e42)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax2.set_yscale('log')
fig.tight_layout()
fig.savefig('bolometric_lightcurve.png', dpi=300)
