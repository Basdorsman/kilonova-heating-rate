from astropy import constants as c
from astropy import units as u
import numpy as np
from scipy.special import erfc
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

__all__ = ('lightcurve',)


def _luminosity(E, t, td, be):
    t_dif = td / t
    tesc = np.minimum(t, t_dif) + be * t
    ymax = np.sqrt(0.5 * t_dif / t)
    return erfc(ymax) * E / tesc


def _rhs(t, E, dM, td, be):
    heat = dM * _heating_rate(t)
    L = _luminosity(E, t, td, be)
    dE_dt = -E / t - L + heat
    return dE_dt


def _heating_rate(t, eth=0.5):
    """Calculate nuclear specific heating rate as a function of time.

    This function is a fit calculated in Korobkin et al. 2012
    (:doi:`10.1111/j.1365-2966.2012.21859.x`), based on a set of simulations
    of nucleosynthesis in the dynamic ejecta of compact binary mergers. This
    fit contains the following fit parameters: eps0 = 2e18, t0 = 1.3,
    sig = 0.11, alpha = 1.3.

    Parameters
    ----------
    time : float, numpy.ndarray
        Rest-frame time(s) at which to evaluate the light curve. May be given
        as an array in order to evaluate at multiple times.
    eth : float, numpy.ndarray
        Heating efficiency parameter, introduced by Korobkin et al. 2012,
        which measures the fraction of nuclear power which is retained in the
        matter. They use eth = 0.5.

    Returns
    -------
    heating_rate : float, numpy.ndarray
        The nuclear specific heating rate has units (erg/g/s) which is
        implied but not explicitly used in this function.

    """
    eps0 = 2e18  # units: erg/g/s
    t0 = 1.3  # units: s
    sig = 0.11  # units: s
    alpha = 1.3  # units: -
    brac = 0.5 - 1. / np.pi * np.arctan((t-t0) / sig)
    return eps0 * brac**alpha * eth / 0.5


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

    Examples
    --------

    Evaluate luminosity, temperature, and radius at a given time:

    >>> from astropy.constants import c
    >>> from astropy import units as u
    >>> t = 1 * u.day  # Time
    >>> mass = 0.05 * u.Msun  # Ejected mass
    >>> velocities = np.asarray([0.1, 0.2]) * c  # Inner and outer velocity
    >>> opacities = 3.0 * u.cm**2 / u.g  # Constant gray opacity
    >>> n = 4.5  # Velocity profile
    >>> L, T, r = lightcurve(t, mass, velocities, opacities, n)
    >>> print(L)
    7.205179189890574e+40 erg / s
    >>> print(T)
    4435.734488631963 K
    >>> print(r)
    511070140219816.5 cm

    Evaluate the flux at a given distance in a given band using synphot:

    >>> import synphot
    >>> DL = 100 * u.Mpc  # Luminosity distance
    >>> spectrum = synphot.SourceSpectrum(synphot.BlackBody1D, temperature=T)
    >>> spectrum *= np.pi * (r / DL).to_value(u.dimensionless_unscaled)**2
    >>> bandpass = synphot.SpectralElement.from_filter('johnson_j')
    >>> apparent_mag = synphot.Observation(spectrum, bandpass).effstim(u.ABmag)
    >>> print(apparent_mag)
    21.031737072570557 mag(AB)

    """
    # Validate arguments
    t0 = 0.01 * u.day
    opacities = np.atleast_1d(opacities)
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

    # Use inverse log spacing for velocity steps.
    bes = np.flipud(be_max + be_0 - np.geomspace(be_0, be_max, n_shells))
    dbe = np.diff(bes)
    bes = bes[:-1]

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
        _rhs, (t0, t.max()), np.zeros(len(bes)), first_step=t0,
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
