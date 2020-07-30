from scipy import *
from scipy.signal import gaussian, hilbert
from scipy.constants import speed_of_light
from matplotlib.pyplot import *
from my_format_lib import *
format_plot()
# manuscript:lst:mrovstime
SN = 20000  # buffer size
SR = 20e6  # sample rate [1/s]
tau_scan = linspace(0, SN / SR, SN) # time range [s]
wavelength = 1330e-9 # [m]
wavelengthBW = 60e-9 # [m]
L_FWHM = 2*log(2)/pi * wavelength**2 / wavelengthBW #[m]
L_sigma = L_FWHM/2/sqrt(2*log(2)) #[m]
v_M = 0.1 # [m/s]
D = 20e-6 # spacing [m]
D_T = D/v_M # spacing [s]
N_O = 5 # amount of orders of reflection
T_FWHM = L_FWHM / v_M #[s]
T_sigma = L_sigma /  v_M #[s] sigma_w
f_D = 2 * v_M / wavelength #[1/s]
t0 = 0.1e-3 # sample location [s]
w = 2*pi*f_D
I_t = []
for N in range(1,N_O+1):
    I_t.append( exp(-1j * N * w * tau_scan) * exp(-(tau_scan-t0-(N-1)*D_T)**2 / (2*(T_sigma/N)**2)) )
plot(tau_scan*1e3,sum(I_t,axis=0))
# manuscript:lst:mrovstime

#####
ax = gca()
ax.set_xlabel('time (ms)') # tau
ax.set_ylabel('Amplitude (arb.)')
ax.set_xlim([0,max(tau_scan)*1e3])
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
tick_space = 6
ax2.set_xticks(range(tick_space))
D_L = v_M * tau_scan #[m]
D_L_um = D_L * 1e6
ax2.set_xticklabels(['{:1.0f}'.format(real(nr)) for nr in D_L_um[linspace(0,SN-1,tick_space).astype(int)]])
ax2.set_xlabel('space ($\mu$m)')

print('FWHM',T_FWHM**2,'s')
print('sigma',T_sigma,'s')
print('f_D',f_D,'Hz')
spr = linspace(0, max(tau_scan)*v_M,SN)
# f = figure('plot spatial')
# plot(spr*1e6,I_t)
# plot(spr*1e6,abs(hilbert(real(I_t))))
# grid(True)
# xlabel('space (um)')
print('scan distance',max(spr),'m')
s0 = spr[(array(where(abs(hilbert(real(I_t))) > 0.5)).min())]
s1 = spr[(array(where(abs(hilbert(real(I_t))) > 0.5)).max())]
FWHM = abs(s0 - s1)
print('measured FHWM', FWHM, 'm')
tight_layout()
savefig('mro_vs_time.pdf')
show()

# pretty much confirmed

# Please appreciate one problematic confusion if we use time as x-axis.
# If the mirror speed is increased the time-axis does not change as it depends
# merely on the digitizer.
# Only due to the Doppler effect we get more finges visible.
# Also if the sample is delayed as absolute value it depends on the relative time position
# of the digitizer again and remains constant in position.
# If the