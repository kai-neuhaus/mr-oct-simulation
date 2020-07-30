from scipy import *
import matplotlib.pyplot as pp
from my_format_lib import * # run from my_format_lib !
format_plot()
# manuscript:lst:monowavetime
N = 20000  # buffer size
SR = 20e6  # sample rate [Hz] [1/s]
wavelength = 1330e-9 # [m]
v_M = 0.003 # [m/s]
tau_scan = linspace(0, N/SR, N).astype(complex) # [s]
f_D = 2 * v_M / wavelength # [1/s], [Hz]
w = 2*pi*f_D
coherent_wave = exp(-1j*w*tau_scan)
tau_scan_ms = tau_scan*1e3 # x-axis: time
D_L_um = v_M * tau_scan *1e6 # x-axis: space
plot_(xaxes=[tau_scan_ms, D_L_um], yaxis=coherent_wave)
# manuscript:lst:monowavetime

# call this r
# def plot(xaxes, yaxis):
#     '''
#     Copy and paste this into my_format_lib and run from there instead of here
#     :param xaxis:
#     :param yaxis:
#     :return:
#     '''
#     pp.plot(xaxes[0],yaxis)
#     ax = pp.Axes
#     ax = pp.gca()
#     ax2 = ax.twiny()
#     D_L = v_M * max(tau_scan)
#     print('D_L_um',D_L_um,'um')
#     # title('Monochromatic wave in time')
#     ax.set_xlabel('time (ms)')
#     ax.set_xlim([0,max(tau_scan_ms)])
#     ax2.set_xlim(ax.get_xlim())
#     tick_space = 12
#     ax2.set_xticks(range(tick_space))
#     ax2.set_xticklabels(['{:1.1f}'.format(real(nr)) for nr in D_L_um[linspace(0,N-1,tick_space).astype(int)]])
#     ax2.set_xlabel('space ($\mu$m)')
#     pp.grid(True)
#     pp.tight_layout()
#     pp.savefig('monowave_vs_time.pdf')
#     pp.show()

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
