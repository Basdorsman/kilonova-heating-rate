from importlib import resources

from astropy import constants as c
from astropy import units as u
import numpy as np
from scipy.special import erfc
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

__all__ = ('lightcurve',)

with resources.open_text(__package__, 'Heating_Korobkin2012.dat') as f:
    log_time, log_heating_rate = np.loadtxt(f).T / np.log10(np.e)
_log_heating_interp = interp1d(
    log_time + np.log(u.day.to(u.s)), log_heating_rate, assume_sorted=True)
del log_time, log_heating_rate


def _luminosity(E, t, td, be):
    t_dif = td / t
    tesc = np.minimum(t, t_dif) + be * t
    ymax = np.sqrt(0.5 * t_dif / t)
    return erfc(ymax) * E / tesc


def _rhs(t, E, dM, td, be):
    heat = dM * np.exp(_log_heating_interp(np.log(t)))
    L = _luminosity(E, t, td, be)
    dE_dt = -E / t - L + heat
    return dE_dt


def lightcurve(t, mass, velocities, opacities, n):
    r"""Calculate a kilonova light curve using the Hotokezaka & Nakar model.

    Evolve a the Hotokezaka & Nakar 2020 kilonova heating light curve model
    (:doi:`10.3847/1538-4357/ab6a98`). This model assumes a density profile
    that is a power-law function of velocity, given by:

    .. math:: \rho(v) = \rho_0 \left(\frac{v}{v_0}\right)^{-n}

    The opacity is a piecewise constant function of velocity. The velocities
    must be an array of length >= 2, :math:`(v_0, \dots, v_\mathrm{max})`. The
    opacities must be an array of length >= 1 (one less than the length of the
    velocities array), forming a lookup table of velocities to opacities.

    The output is the effective blackbody luminosity, temperature, and radius
    as a function of time.

    Parameters
    ----------
    time : :class:`astropy.units.Quantity`
        Rest-frame time(s) at which to evaluate the light curve, in units
        compatible with `s`. May be given as an array in order to evaluate
        at multiple times.
    mass : :class:`astropy.units.Quantity`
        Ejected mass in units compatible with `g`.
    velocities : :class:`astropy.units.Quantity`
        Array of ejecta velocities in units compatible with `cm/s`.
        Length must be >= 2.
    opacities : :class:`astropy.units.Quantity`
        Array of opacities in units compatible with `cm**2/g`.
        Lenght must be >= 1, and 1 less than the length of `vej`.
    n : int, float
        Power-law index of density profile.

    Returns
    -------
    luminosity : :class:`astropy.units.Quantity`
        Luminosity in units of `erg/s`.
    temperature : :class:`astropy.units.Quantity`
        Temperature in units of `K`.
    radius : :class:`astropy.units.Quantity`
        Blackbody radius in units of `cm`.

    """
    # Validate arguments
    t0 = 0.01 * u.day
    if np.any(t <= t0):
        raise ValueError(f'Times must be > {t0}')
    if len(velocities) != len(opacities) + 1:
        raise ValueError('len(velocities) must be len(opacities) + 1')

    # Convert to internal units.
    t = t.to_value(u.s)
    t0 = t0.to_value(u.s)
    mej = mass.to_value(u.g)
    bej = (velocities / c.c).to_value(u.dimensionless_unscaled)
    vej_0 = velocities[0].to_value(u.cm / u.s)
    kappas = opacities.to_value(u.cm**2 / u.g)

    # Prepare velocity shells.
    n_shells = 100
    be_0 = bej[0]
    be_max = bej[-1]
    rho_0 = mej * (n - 3) / (4*np.pi*vej_0**3) / (1 - (be_max/be_0)**(3 - n))
    dbe = (be_max - be_0) / n_shells
    bes = np.arange(be_0, be_max, dbe)

    # Calculate optical depths from each shell to infinity.
    i = np.searchsorted(bej, bes)  # Determine which shell we are in.
    # Integrate across whole shells.
    tau_accum = -np.cumsum((kappas * np.diff((bej/be_0)**(1-n)))[::-1])[::-1]
    tau_accum = np.concatenate((tau_accum, [0]))
    taus = tau_accum[i]
    # Integrate across the remainder of the shell that we are in.
    taus += kappas[i - 1] * ((bes/be_0)**(1-n) - (bej[i]/be_0)**(1-n))
    taus *= vej_0 * rho_0 / (n - 1)

    dMs = 4*np.pi*vej_0**3*rho_0*(bes/be_0)**(2-n)*dbe/be_0
    tds = taus * bes

    # Evolve in time.
    out = solve_ivp(
        _rhs, (t0, t.max()), np.zeros(n_shells), first_step=t0,
        args=(dMs[:, None], tds[:, None], bes[:, None]), vectorized=True)

    # Find total luminosity.
    LL = _luminosity(out.y, out.t[None, :], tds[:, None], bes[:, None]).sum(0)
    log_L_interp = interp1d(
        np.log(out.t[1:]), np.log(LL[1:]), kind='cubic', assume_sorted=True)

    # Do cubic interpolation in log-log space to evaluate at sample times.
    # Note that solve_ivp could do this in principal, but it does interpolation
    # in linear-linear space, which causes some minor artifacts.
    L = np.exp(log_L_interp(np.log(t))) * (u.erg / u.s)

    # Effective radius
    be = np.exp(np.interp(2*np.log(t), np.log(taus[::-1]), np.log(bes[::-1])))
    r = be * t * (c.c * u.s)

    # Effective temperature
    T = ((L / (4 * np.pi * c.sigma_sb * np.square(r)))**0.25).to(u.K)

    # Done!
    return L, T, r.to(u.cm)
