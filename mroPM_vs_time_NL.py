from scipy import *
from scipy.fftpack import *
from scipy.signal import gaussian, hilbert
from scipy.constants import speed_of_light
from matplotlib.pyplot import *
from my_format_lib import *
format_plot()

# def stop():
# manuscript:lst:mronoattnvstimeNL
SN = 60000  # buffer size
SR = 20e6  # sample rate [1/s]
tau_scan = linspace(0, SN / SR, SN)  # time range [s]
wavelength = 1330e-9 # [m]
wavelengthBW = 60e-9 # [m]
R_PM = 0.8 # reflectivity fraction of PM
L_FWHM = 2*log(2)/pi * wavelength**2 / wavelengthBW #[m]
L_sigma = L_FWHM/2/sqrt(2*log(2)) #[m]
v_M = 0.03 # [m/s]
D = 16e-6 # spacing [m]
D_T = D/v_M # spacing [s]
N_O = 5 # amount of orders of reflection
T_FWHM = L_FWHM / v_M #[s]
T_sigma = L_sigma /  v_M #[s] sigma_w
f_D = 2 * v_M / wavelength #[1/s]
v_M_phs = f_D*SN/SR*sin(linspace(-pi,pi,SN)) # distortion due to scanning.
t0 = 0.6e-3 # sample location [s]
w = 2*pi*f_D
I_t = []
def G(N): return exp(-(tau_scan-t0-((N-1)*(D_T)))**2 / (2*(T_sigma/N)**2)) # envelope
def O(N): return exp(-1j * N * ( w * tau_scan + v_M_phs))  # carrier
def T_PM(N): return (1-R_PM)**2 * R_PM**(N-1) # attenuation PM splitting ratio
for N in range(1,N_O):
    I_t.append( T_PM(N) * G(N) * O(N) )
plot(tau_scan*1e3,sum(I_t,axis=0))
# manuscript:lst:mronoattnvstimeNL


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
# spr = linspace(0, max(tau_scan)*v_M,SN)
# f = figure('plot spatial')
# plot(spr*1e6,I_t)
# plot(spr*1e6,abs(hilbert(real(I_t))))
# grid(True)
# xlabel('space (um)')
# print('scan distance',max(spr),'m')
# s0 = spr[(array(where(abs(hilbert(real(I_t))) > 0.5)).min())]
# s1 = spr[(array(where(abs(hilbert(real(I_t))) > 0.5)).max())]
# FWHM = abs(s0 - s1)
# print('measured FHWM', FWHM, 'm')
tight_layout()
savefig('mroPM_vs_time_NL.pdf')
# show()


class Real_data(object):
# The distance between the order in the data buffer does change according to some measurements
# /media/kai/sdb5/PS_MRO/NF2117/Sensitivity_MRO_bal_170904a/ChA-20170904-174046MD_I450_G300_R1000_SAA180RAA060_r1.npy
    locs = [
    [[21500,61900],[15800,57000],[7900, 52000]],
    [[30700,58000],[17900,48100],[6100, 39500]],
    [[39500,59700],[23200,46000],[4330, 31500]],
    [[46000,61900],[31500,49100],[9800, 31500]],
    [[49100,62300],[31500,46900],[2870, 25700]],
    [[46900,58200],[25700,41900],[17100, 34600]],
    [[41900,54000],[34600,47500],[23400, 38100]]]

    def list_diffloc_items(self):
        l_diff = []
        for l in self.locs:
            i_mean = []
            for i in l:
                # print(diff(i))
                i_mean.append(diff(i))
            print(mean(i_mean))
            l_diff.append(mean(i_mean))
        print('l_diff',diff(l_diff))
rd = Real_data()
rd.list_diffloc_items()

class My_verification():
    # The reduction of frequency to the left occurs due to the negativity of the sin(-pi ...) !
    # The full sine goes from zero to -1 and back to zero the first half.
    # With increasing steps for w * tm the phase progression is attenuated by each value of sin.
    # That means as the w * tm progresses faster than the sin is reducing to -1 the actual frequency does increase
    # slowly already.
    # Towards the center the frequency is increased due to increasing sin values.
    # Towards the end a reduction occurs due to the shifting of higher frequencies towards the center.
    # Meaning the phase in the center is original phase + sin_distortion, and as the phase is accumulating
    # at the end we have phase_in - orig_phase + 0.
    # The orig_phase was basically added to the center and is now missing at the end.
    # This causes also that the actual frequency is now increased by sqrt(2*pi)/2.
    # The second half must do the same by
    NN = 10000
    tm = linspace(-1,1,NN)
    f = 20 #/sqrt(2*pi)/2
    w = 2*pi*f
    # Due to the accumulation of the phase we need a negative part coming from sin(-pi ...).
    #
    ph = 2*f* sin(linspace(-pi, pi, NN))

    def plot_conventional(self):
        figure()
        tm = self.tm
        w = self.w
        plot( exp(1j * w * tm))

    def plot_distortion(self):
        # The distortion is independent of the acquisition buffer length and only from the scanning cycle.
        # That means if we assume one buffer length shown is one cycle then it is 2pi.
        # For real buffers this needs to be tuned.
        figure()
        plot( self.ph )

    def plot_wave_distorted(self):
        # Although the distortion is independent on the buffer length it depends on the frequency strength.
        # See here that the strength must be 2 * f.
        figure()
        NN = self.NN
        f = self.f
        tm = self.tm
        w = self.w
        ph = self.ph
        subplot(211)
        sig = exp(1j * (w * tm + (ph)))
        plot( sig )
        title('signal')
        subplot(212)
        fr = linspace(-NN/2,NN/2,NN)
        plot(fr, abs(fftshift(fft(real(sig)))) )
        title('FFT')
        xlim([0, 10*f])

    def plot_gauss_distortion(self):
        # The gauss distortion is not known yet and we can only evaluate on the signal.
        # To some degree we should expect that with reduced Doppler the width of the Gaussian increases.
        # Because with increased Doppler the width does decrease.
        NN = self.NN
        tm = self.tm
        def dist_gauss(D):
            offset = 1/(D+0.5)
            return cos(linspace(-pi / 2+offset, pi / 2-offset, NN))
        self.dist_gauss = dist_gauss
        figure()
        subplot(211)
        plot(self.dist_gauss(D=0.0))

    def plot_gauss_conventional(self):
        tm = self.tm
        D = 1
        sigma = 0.5/(self.dist_gauss(D))
        subplot(212)
        plot(exp(-(tm-D)**2/0.5**2))
        plot(exp(-(tm-D)**2/sigma**2))

# ph = -10*imag(exp(-1j*linspace(-pi,pi,1000)))
    # phg = 1/(1*sin(linspace(-pi/105,pi/105,NN)))
    # plot( exp(1j * (w * tm + ph))) # + 1/2*exp(-1j*ph)*exp(-1j * wt))
    # plot( 1/2*exp(1j * wt -pi/2 +ph) - 1/2*exp(-1j * wt +pi/2 +ph))
    # gauss = exp(-(tm-pi)**2/1**2)
    # plot(gauss)
    # gaussm = exp(-(tm-pi+1.5)**2/phg**2)
    # plot(gaussm)

    # figure()
    # plot(unwrap(angle(hilbert(real(exp(1j * (w * tm + ph)))))))
mv = My_verification()
# mv.plot_conventional()
# mv.plot_distortion()
# mv.plot_wave_distorted()
# mv.plot_gauss_distortion()
# mv.plot_gauss_conventional()

show()