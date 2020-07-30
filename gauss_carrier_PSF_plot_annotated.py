from scipy import *
from scipy.constants import speed_of_light
from matplotlib.pyplot import *

def gauss(w,w0,sigma):
    return exp(-((w-w0)**2) / (2*sigma**2))

def gaussian_FWHM():
    N = 20000  # buffer size
    SR = 20e6  # sample rate (seconds)
    tmr = linspace(0, N/SR, N) # time range
    wavelength = 1330e-9 # meter
    wavelengthBW = 60e-9

    FWHM = 2*log(2)/pi * wavelength**2/wavelengthBW  #[m]
    print('FWHM',FWHM,'m')
    sigma = FWHM/2/sqrt(2*log(2)) #[m]
    print('sigma',sigma,'m')

    v_M = 0.04 # [m/s]
    spr = tmr*v_M # spatial range [m]
    dist = max(spr)
    print('scan distance',dist,'m')
    f_D = 2 * v_M / wavelength #[1/s]
    print('f_D',f_D,'Hz')

    spPerLen = N/dist # [sp / m]
    print('[sp/m]',spPerLen)
    FWHMsp = FWHM * spPerLen
    print('FWHMsp',FWHMsp)
    sigmaSp = FWHMsp/2/sqrt(2*log(2))

    # How do we incorporate the group velocity and DeltaTau_g?
    # DeltaTau_g tells us the time the wave bunch requires to travel a distance Delta_l.
    # This is of course much faster than we can actually measure.
    # We could in theory plot the Gaussian envelope in terms of the time again which should be it
    #
    # However, we measure actually the time of either the scanning mirror or
    # the delay time due to the sample refractional index.
    v_g = speed_of_light
    Delta_l = spr
    DeltaTau_g = 2 * Delta_l / v_g
    Delta_l = DeltaTau_g / 2 * v_g

    w = spr
    w0 = 0
    w0 = 18e-6 # must be close to the um dimensions
    # G_in_samples = gauss(w=w,w0=w0,sigma=sigma)
    G_in_samples = exp(-(w - w0)**2/(2*sigma**2))
    line_FWHMsp = where(G_in_samples >= 0.5)[0]
    line_FWHM = line_FWHMsp/SR*v_M
    s0 = min(line_FWHM)
    s1 = max(line_FWHM)
    print('sample FHWM',abs(s0-s1))
    print('time FHWM',abs(s0-s1)/SR)
    print('spatial FHWM',abs(s0-s1)/SR*v_M)

    # plot(tmr, sin(2*pi*f_D*tmr))
    plot(spr*1e6,G_in_samples)
    print(line_FWHM)
    plot(line_FWHM*1e6,ones(len(line_FWHM))/2)
    # xlabel('time (s)') # tmr
    # xlabel('$\Delta l$ ($\mu$m)') # spr
    # ylabel('Amplitude (arb.)')
    gca().set_yticks(linspace(0,1.0,11))
    grid(True)
    show()

    # Conceptionally we need always to distinguish between the true propagation time of the light
    # and the scanning time.

    # As long as our scanning time is much less than the speed of light we obtain an representation
    # of the interference characteristics vs the scan time.

    # The Delta_tau_g in this case is the time the light requires to pass the distance mismatch.
    # But we generate a mismatch by moving the reference mirror much slower!

    # For FD-OCT the mismatch is indeed as fast as by the speed of light vs. the delay in the sample.

def gaussian_FWHM_B():
    '''
    Proof showing for a particular Gaussian centered around zero we can calculate the sigma
    for some FWHM matching the expected width in the plot.
    :return:
    '''
    xr = linspace(-3,3,100)
    FWHM = 1
    sigma = FWHM/(2*sqrt(2*log(2)))
    y = exp( -(xr-0)**2/(2*sigma**2))
    print(min(where(y>=0.5)))
    # Interpolation could improve the accuracy to determine the FWHM from the data points in the plot.
    # However, as the plot itself is not the source of data this is not essential or even misleading.
    # The plot is just the representation of the true data with limited resolution (sample rate).
    yi_min = where(y>=0.5)[0].min()
    yi_max = where(y>=0.5)[0].max()
    print('yi_min',yi_min,'yi_max',yi_max,'diff',diff((xr[yi_min],xr[yi_max])))
    plot(xr, y)
    xlabel('units in pi')
    gca().set_yticks(linspace(0, 1.0, 11))
    gca().set_xticks(linspace(min(xr), max(xr), 13))
    grid(True)

    show()
gaussian_FWHM()