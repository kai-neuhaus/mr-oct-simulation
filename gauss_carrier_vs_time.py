from scipy import *
from scipy.signal import gaussian, hilbert
from scipy.constants import speed_of_light
from matplotlib.pyplot import *


N = 20000  # buffer size
SR = 20e6  # sample rate [1/s]
tau_scan = linspace(0, N / SR, N) # time range [s]
wavelength = 1330e-9 # [m]
wavelengthBW = 60e-9 # [m]
L_FWHM = 2*log(2)/pi * wavelength**2 / wavelengthBW
L_sigma = L_FWHM/2/sqrt(2*log(2))
v_M = 0.05 # [m/s]
T_FWHM = L_FWHM / v_M
T_sigma = L_sigma /  v_M
f_D = 2 * v_M / wavelength #[1/s]
t0 = 0.5e-3 # sample location [s]
w = 2*pi*f_D
I_t = exp(-1j * w * tau_scan) * exp(-(tau_scan-t0)**2 / (2*T_sigma**2))
plot(tau_scan,I_t)

print('FWHM',T_FWHM**2,'s')
# print('sigma',sigma_tau,'s')
# print('scan distance',max(spr),'m')
print('f_D',f_D,'Hz')
spr = linspace(0, max(tau_scan)*v_M,N)
s0 = spr[(array(where(abs(hilbert(real(I_t))) > 0.5)).min())]
s1 = spr[(array(where(abs(hilbert(real(I_t))) > 0.5)).max())]
FWHM = abs(s0 - s1)
print('measured FHWM', FWHM, 'm')

xlabel('time (s)') # tau
# xlabel('$\Delta l$ ($\mu$m)') # spr
ylabel('Amplitude (arb.)')
# grid(True)
show()

# pretty much confirmed