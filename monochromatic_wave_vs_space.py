from scipy import *
from matplotlib.pyplot import *
import my_format_lib
my_format_lib.format_plot()
# manuscript:lst:monowavespace
N = 20000  # buffer size
SR = 20e6  # sample rate [Hz] [1/s]
tau_scan = linspace(0, N/SR, N).astype(complex) # [s]
wavelength = 1330e-9 # [m]
v_M = 0.003 # [m/s]
f_D = 2 * v_M / wavelength # [1/s], [Hz]
D_L = tau_scan*f_D*wavelength # [m]
D_L_um = D_L*1e6
k = 2*pi/wavelength #[1/m]
coherent_wave = exp(-1j*k*2*D_L)
plot(D_L_um, coherent_wave)
# manuscript:lst:monowavespace

# title('Monochromatic wave in space')
xlabel('Space ($\mathrm{\mu m}$)')
ylabel('Amplitude (arb.)')
arrow_x = D_L_um[where(coherent_wave >= 0.999999+0j)[0]]
gca().annotate('', # leave empty as otherwise arrow is scaled to text
               textcoords='data',
               xycoords='data',
               xy=(arrow_x[2],1),
               xytext=(arrow_x[3],1.0),
               arrowprops=dict(arrowstyle='<->'))
gca().annotate('$\lambda_D$',
               xycoords='data',
               textcoords='data',
               xy=(arrow_x[3]-mean(diff(arrow_x))/1.3,0.8)
               )
tight_layout()
savefig('monowave_vs_space.pdf')
show()
# There is still 2*D_L which so far appears to be plausible but is somewhat double related to the Doppler.
# Now on the other hand, D_L has nothing realy to do with the Doppler or yes it has if a) Doppler effec is
# included and the double pass of D_L in which case the 2 cancels. But this need to be tested.
def mono_wave_vs_time():
    '''
    Test run showing only time
    :return:
    '''
    N = 20000  # buffer size
    SR = 20e6  # sample rate (seconds)
    tau_scan = linspace(0, N / SR, N).astype(complex)  # [s]
    wavelength = 1330e-9  # [m]
    v_M = 0.005  # [m/s]
    f_D = 2 * v_M / wavelength  # [1/s], [Hz]
    w = 2 * pi * f_D
    coherent_wave = exp(-1j*w*tau_scan)
    plot(tau_scan, coherent_wave)

    title('Monochromatic wave in space')
    xlabel('space (um)')
    tight_layout()
    show()

def mono_wave_vs_space():
    '''
    Test run showing time and space
    :return:
    '''
    N = 20000  # buffer size
    SR = 20e6  # sample rate (seconds)
    tau_scan = linspace(0, N / SR, N).astype(complex)  # [s]
    wavelength = 1330e-9  # [m]
    v_M = 0.005  # [m/s]
    f_D = 2 * v_M / wavelength  # [1/s], [Hz]
    k = 2 * pi / wavelength  # [1/m]
    D_L = tau_scan * f_D * wavelength  # linspace(0, f_D*wavelength*N/SR, N) # [m]
    w = 2 * pi * f_D
    # coherent_wave = exp(-1j*w*tau_scan)
    coherent_wave = exp(-1j * 2 * k * D_L)
    D_L_um = D_L * 1e6
    plot(D_L_um, coherent_wave)

    title('Monochromatic wave in space')
    xlabel('space (um)')
    tight_layout()
    show()
