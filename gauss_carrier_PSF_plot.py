from scipy import *
from scipy.signal import gaussian
from matplotlib.pyplot import *
N = 20000  # buffer size
SR = 20e6  # sample rate (seconds)
tmr = linspace(0, N/SR, N) # time range
wavelength = 1330e-9 # meter
wavelengthBW = 60e-9

FWHM = 2*log(2)/pi * wavelength**2/wavelengthBW  #[m]
print('FWHM',FWHM,'m')
sigma = FWHM/2/sqrt(2*log(2)) #[m]
print('sigma',sigma,'m')

v_M = 0.02 # [m/s]
spr = tmr*v_M # spatial range [m]
print('scan distance',max(spr),'m')
f_D = 2 * v_M / wavelength #[1/s]
print('f_D',f_D,'Hz')

spPerLen = N/max(spr) # [sp / m]
print('[sp/m]',spPerLen)
FWHMsp = FWHM * spPerLen
print('FWHMsp',FWHMsp)
sigmaSp = FWHMsp/2/sqrt(2*log(2))
# sigma = 30000/2/sqrt(2*log(2))
G_in_samples = gaussian(M=N,std=sigmaSp)
s0 = (array(where(G_in_samples > 0.5)).min())
s1 = (array(where(G_in_samples > 0.5)).max())
print('sample FHWM',abs(s0-s1))

# plot(tmr, sin(2*pi*f_D*tmr))
plot(spr*1e6,G_in_samples)
# xlabel('time (s)') # tmr
# xlabel('$\Delta l$ ($\mu$m)') # spr
# ylabel('Amplitude (arb.)')
grid(True)
show()