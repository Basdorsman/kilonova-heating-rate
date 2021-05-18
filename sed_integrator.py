from astropy import constants as const
from astropy import units as u
import numpy as np
import synphot


def get_abmag(T,r,distance,bandpass):
    '''calculate AB magnitude faster.
    
    This code is largely copied from Synphot, using: synphot.BlackBody1D,
    synphot.SourceSpectrum and synphot.Observation. However, it's vectorized.
    Redshift has not yet been implemented here.
    
    Parameters
    ----------
    T : float 1darray
        Blackbody Temperature
    r : float 1darray
        Blackbody radius
    D : float
        Luminosity distance
    bandpass : object synphot.spectrum.SpectralElement
        Bandpass
    
    Returns
    -------
    abmag : float 1darray 
        AB magnitudes
    '''
    h = const.h.cgs
    c = const.c.cgs
    k_B = const.k_B.cgs
    
    wav = bandpass.waveset
    wav_nounits = wav.value
    B_l = 2*h*c**2/np.multiply(wav**5,np.exp(h*c/(k_B*np.outer(T,wav)))-1) #spectral radiance
    f_l = B_l.T*np.pi*(r / distance)**2 # scale spectral density for flux
    f_PHOTLAM = (f_l.T/((h*c/wav))).to(1/u.cm**2/u.s/u.Angstrom)*bandpass(wav) #convert to photlam and apply bandpass
    num = abs(np.trapz(f_PHOTLAM.value * wav_nounits, x=wav_nounits)) 
    den = np.ones(len(T))*abs(np.trapz(bandpass(wav) * wav_nounits, x = wav_nounits))
    val = (num / den)*synphot.units.PHOTLAM #convolve with bandpass for total photon flux
    abmag = synphot.units.convert_flux(bandpass.pivot(),val,u.ABmag).value #return ABmag
    return abmag

def get_abmag_synphot(T, r, distance, bandpass):
    seds = [synphot.SourceSpectrum(synphot.BlackBody1D, temperature=TT) * np.pi * (rr / distance).to(u.dimensionless_unscaled)**2 for TT,rr in zip(T, r)]
    abmag_synphot = [synphot.Observation(sed, bandpass).effstim(u.ABmag).value for sed in seds]
    return abmag_synphot

#plt.plot(t,abmag_byhand,label='byhand')
#plt.plot(t,abmag_synphot,label='synphot')
#plt.plot(t,abmag_byhand-abmag_synphot,label='ratio')

#idx = 0
#plt.plot(wav,synphot.Observation(seds[idx],bp_dorado)(wav))
#plt.plot(wav,f_convolve[idx])

#plt.plot(wav,f_PHOTLAM[idx],label='byhand')
#plt.plot(wav,seds[idx](wav),label='synphot')

#plt.legend()
