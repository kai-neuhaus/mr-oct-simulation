from scipy import *
from matplotlib.pyplot import *
N = 20000  # buffer size
SR = 20e6  # sample rate (seconds)
tau_scan = linspace(0, N/SR, N).astype(complex) # [s]
wavelength = 1330e-9 # [m]
v_M = 0.005 # [m/s]
f_D = 2 * v_M / wavelength # [1/s], [Hz]
k = 2*pi/wavelength #[1/m]
D_L = linspace(0, f_D*wavelength*N/SR, N) # [m]
w = 2*pi*f_D
# coherent_wave = exp(-1j*w*tau_scan)
coherent_wave = exp(-1j*2*k*D_L)
D_L_um = D_L*1e6
plot(tau_scan, coherent_wave)
show()
