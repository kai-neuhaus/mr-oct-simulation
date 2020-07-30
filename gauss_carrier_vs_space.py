from scipy import *
from scipy.signal import hilbert
from matplotlib.pyplot import *
import my_format_lib as mylib
mylib.format_plot()
# manuscript:lst:gausscarrspace
N = 20000  # buffer size
SR = 20e6  # sample rate [s]
tau_scan = linspace(0, N / SR, N) # time range [s]
wavelength = 1330e-9 # [m]
wavelengthBW = 60e-9 # [m]
L_FWHM = 2*log(2)/pi * wavelength**2 / wavelengthBW
L_sigma = L_FWHM/2/sqrt(2*log(2))
D_k = pi/sqrt(log(2))*wavelengthBW/wavelength**2
v_M = 0.05 # [m/s]
D_L = tau_scan * v_M # spatial range [m]
D_L_um = D_L*1e6
f_D = 2 * v_M / wavelength #[1/s]
L_0 = 23e-6 #[m]
K = 2*pi / wavelength
I_t = exp(-1j * 2 * K * D_L) * exp(-(D_L-L_0)**2 * (D_k**2))
plot(D_L_um,I_t)
# manuscript:lst:gausscarrspace

plot(D_L_um,abs(hilbert(real(I_t))))
arrow_x = D_L_um[where(abs(hilbert(real(I_t))) >= 0.5+0j)[0]]
print(arrow_x)
gca().annotate('', # leave empty as otherwise arrow is scaled to text
               textcoords='data',
               xycoords='data',
               xy=(arrow_x[0],0.5),
               xytext=(arrow_x[0]-10,0.5),
               arrowprops=dict(arrowstyle='->'))
gca().annotate('', # leave empty as otherwise arrow is scaled to text
               textcoords='data',
               xycoords='data',
               xy=(arrow_x[-1],0.5),
               xytext=(arrow_x[-1]+10,0.5),
               arrowprops=dict(arrowstyle='->'))
gca().annotate('FWHM',
               xycoords='data',
               textcoords='data',
               xy=(arrow_x[-1]+1,0.55)
               )

# grid(True)
print('L_FWHM',L_FWHM,'m')
print('L_sigma',L_sigma,'m')
print('scan distance',max(D_L),'m')
print('f_D',f_D,'Hz')
s0 = D_L[(array(where(abs(hilbert(real(I_t))) > 0.5)).min())]
s1 = D_L[(array(where(abs(hilbert(real(I_t))) > 0.5)).max())]
FWHM = abs(s0 - s1)
print('measured FHWM', FWHM, 'm')

xlabel('Space ($\mathrm{\mu m}$)') # Dk
# xlabel('$\Delta l$ ($\mu$m)') # spr
ylabel('Amplitude (arb.)')
# grid(True)
tight_layout()
savefig('gauss_carrier_vs_space.pdf')
show()