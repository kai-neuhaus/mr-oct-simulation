from scipy import *
from scipy.signal import gaussian
from scipy.constants import speed_of_light
from matplotlib.pyplot import *
# not clear if we still use it under this heading (filename).
# Because what does it mean anyway?

N = 20000  # buffer size
SR = 20e6  # sample rate [s]
tau_scan = linspace(0, N / SR, N) # time range [s]
wavelength = 1330e-9 # [m]
wavelengthBW = 60e-9 # [m]
FWHM = 2*log(2)/pi * wavelength**2/wavelengthBW  #[m]
FWHM_tau = 2 * log(2) / pi * (wavelength * speed_of_light) ** 2 / (wavelengthBW * speed_of_light)  # [1/s]
sigma_tau = FWHM_tau/2/sqrt(2*log(2)) #[m]
v_M = 0.02 # [m/s]
spr = tau_scan * v_M # spatial range [m]
f_D = 2 * v_M / wavelength #[1/s]
t0 = 0.5e-3 #[s]
w = 2*pi*f_D
I_t = exp(-1j * w * tau_scan)#
I_t = exp(-(tau_scan-t0)**2*(2*sigma_tau**2))

plot(tau_scan,I_t)

print('FWHM',FWHM,'m')
print('sigma',sigma_tau,'m')
print('scan distance',max(spr),'m')
print('f_D',f_D,'Hz')
s0 = tau_scan[(array(where(I_t > 0.5)).min())]
s1 = tau_scan[(array(where(I_t > 0.5)).max())]
FWHM = abs(s0 - s1)
print('measured FHWM', FWHM, 's')

# xlabel('time (s)') # tmr
# xlabel('$\Delta l$ ($\mu$m)') # spr
# ylabel('Amplitude (arb.)')
grid(True)
show()