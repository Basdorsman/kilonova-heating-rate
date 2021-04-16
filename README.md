# kilonova-heating-rate

This is a Python package to calculate kilonova light curves using the
Hotokezaka & Nakar (2019) model, which assumes radioactive heating, a power-law
velocity profile, and gray opacities that are a piecewise-constant function of
velocity.

This Python package is based on the original source code release from the 2019
paper (https://github.com/hotokezaka/HeatingRate), but includes the following
enhancements:

* **Easy to install** with [Pip], the Python package manager.
* **Physical units** are integrated with [Astropy], the community Python
  package for astronomy.
* **Flexible** specification of opacities: either constant, or piecewise
  constant as a function of ejecta velocity.
* **Fast** due to the use of [Numpy] to evaluate the right-hand side of the
  system of ordinary differential equations that is solved to evalute the light
  curve.

## To cite

If you use this work to produce a peer-reviewed journal article, please cite
the following papers:

* Korobkin, O., Rosswog, S., Arcones, A., & Winteler, C. 2012, "On the
  astrophysical robustness of the neutron star merger r-process," *Monthly
  Notices of the Royal Astronomical Society*, 426, 1940.
  https://doi.org/10.1111/j.1365-2966.2012.21859.x
* Hotokezaka, K. & Nakar, E. 2020, "Radioactive Heating Rate of *r*-process
  Elements and Macronova Light Curve," *Astrophysical Journal*, 891, 152.
  https://doi.org/10.3847/1538-4357/ab6a98

## To install

Installation is easy with [Pip]:

    $ pip install kilonova-heating-rate

## To use

See example code in [example.py].

![Example plot](https://github.com/dorado-science/kilonova-heating-rate/raw/main/example.png)

[Pip]: https://pip.pypa.io
[Astropy]: https://www.astropy.org
[Numpy]: https://github.com/numpy/numpy
[example.py]: https://github.com/dorado-science/kilonova-heating-rate/blob/main/example.py
