from scipy import *
from scipy.fftpack import *
from scipy.signal import gaussian, hilbert, windows
from scipy.constants import speed_of_light
from matplotlib.pyplot import *
from my_format_lib import *
format_plot()

SN = 65580  # buffer size
SR = 10e6  # sample rate [1/s]
tau_scan = linspace(-SN/2/SR, SN/2/SR, SN) # time range [s]
wavelength = 1330e-9 # [m]
wavelengthBW = 60e-9 # [m]
K = 2*pi/wavelength # [1/m]
R_PM = 0.8 # reflectivity fraction of PM
L_FWHM = 2*log(2)/pi * wavelength**2 / wavelengthBW #[m]
# see /home/kai/Documents/00_00_Thesis/thesis_v0-0-0-1_bitbucket/appendix/scripts/gauss_carrier_PSF_plot_annotated.py
# the gaussian requires the sigma value to obtain the right FWHM
# brezinski_2006: possibly p108(111), p111(115) eq. 5.45, p114(118) eq. 5.63, p120(124) eq. 5.95
L_sigma = L_FWHM / sqrt(8 * log(2)) #[m]
# welch gemert 2011 optical-thermal response, p735
# L_c == L_FWHM
# l_c coherence length is synonym with sigma / STD of Gaussian
# 2 L_c = 8 * log(2) / DK -> 2 DK = 8 * log(2) / L_c
L_c = L_FWHM
DK =  1/sqrt(2)/L_sigma # sqrt(2) is the 2 for the sigma unsquared
v_M = 90e-6/max(tau_scan)/2 # [m/s]
D = 140e-6 # spacing [m]
L_0 = 0 * 1e-6 # [m] sample mirror e.g. 0
N_O = 1 # amount of orders of reflection
f_D = 2 * v_M / wavelength #[1/s]
L = tau_scan * v_M # [m]
dL = abs(max(L)-min(L)) # length of range [m]
v_M_phs = 0 #f_D*SN/SR*sin(linspace(-pi,pi,SN)) # distortion due to scanning.

I_t = []

def G(N,L,L_0):
    C_1 = 0.55 # smaller -> smaller spacing 0.55, 2.3
    C_2 = 2.3
    W_0 = 1.0
    # print(C_1*C_2)
    sN = (N * dL / 2 + (N - 1) * D * C_1  + dL / 2 - N * dL)
    # return (exp(-(L - (L_0 - sN + (N - 1) * D * C_1 * C_2) / N) ** 2 / (2 * ( L_sigma / N) ** 2)))
    return (exp(-(L - L_0 ) ** 2 / (2 * ( L_sigma ) ** 2)))
    # return (exp( -(( L - L_0 ) ** 2) * (DK ** 2)))


def O(N,k,L): return exp(-1j * N * (K * 2 * L))  # carrier
def T_PM(N): return (1-R_PM)**2 * R_PM**(N-1) # attenuation PM splitting ratio
fig = figure(figsize=(5.65*2,5),num='using L_c')
counter=0
mirror_start = 200e-6 #2 3
mirror_start = 300e-6 #3 4
mirror_start = 400e-6 #3 4 5p
mirror_start = 600e-6 #3 4 5 6
mirror_start = 10e-6 # 1st
mirror_stop = mirror_start # stay
mirror_steps = 1 #
sum_sig = []
for L_0 in linspace(L_0-mirror_start, L_0-mirror_stop, 1):
    for N in range(1,N_O+1):
        I_t.append( T_PM(N) * G(N, L, L_0)  * O(N,K,L))
    sum_sig = sum(I_t,axis=0)
    tau_scan_ms = (tau_scan + SN/2/SR) * 1e3
    # plot(tau_scan_ms,sum(I_t,axis=0))
    subplot(121)
    gca().cla()
    plot(tau_scan,sum_sig)
    ylim((-0.05,0.05))
    subplot(122)
    gca().cla()
    tukey_win = windows.tukey(SN,alpha=0.5)
    sum_sig_win = tukey_win * sum_sig
    semilogy(fftfreq(n=len(tau_scan),d=1/SR*1e3),abs((fft(real(sum_sig_win)))))
    xlabel('Frequency (kHz)')
    xlim((0,200))
    ylim((1e-2,1e3))
    tight_layout()
    pause(0.001)
    # savefig('movie_{:04d}.jpg'.format(counter))
    counter+=1
    # while waitforbuttonpress() == 0: pass
    # I_t=[]

# By deduction we concluded that the FWHM must reduce for each higer order by N.
# As a note of care, the apparent FWHM can be changed also by changing D_L, but this
# is a mere geometrical or sampling problem and does not reproduce the true effects at work.

# Furthermore the displacement is non-constant / non-linear.
# Because in the time space the light travels of course between the spacing which is constant.
# According to dsouza_2016_towards it was empirically shown that the total scan range D is
# D = s/2 + d(n-1) + ns/2 with s for the length of the first order, respectivel sn len of the n-th order
# "small" d the displacement between two orders.
# If we need to consider now that multiple reflections only occur if a sample mirror is in the range of
# multiple orders.

# We can of course play with D_L but then the FWHM should not change.
# See eq. 2.34 which actually shows that either the FWHM does change or the apparent tau_g.
# In the case of tau_g we assume that it is tau_scan and that tau_g = speed_of_light.
# Because tau_g must be constant or the speed of light we encounter a larger mirror velocity which
# virtually reduces tau_scan, or reduces FWHM, or increases the scanning range.

# Consequently the problem here will be to decide what does reduce the FWHM?
# Because in true time space the FWHM is constant, though the range is large which is however only
# scanned at higher speed and the


#####
# Attention again. We use linear mirror motion and larger velocity does large scan range!
# For real data the range is given voice coil!
#####

def set_axes_special():
    # ax = Axes()
    ax = gca()
    # ax.set_xlabel('time (ms)') # tau
    # ax.set_ylabel('Amplitude (arb.)')
    max_xrng = max(tau_scan)*v_M*1e6
    min_xrng = min(tau_scan)*v_M*1e6
    ax.set_xlim([min_xrng,max_xrng])
    # ax.set_xticks(arange(min_xrng,max_xrng,15))
    grid(True)
    # ax.set_xticklabels()
    # ax2 = ax.twiny()0000
    # ax2.set_xlim(ax.get_xlim())
    # tick_space = 6
    # ax2.set_xticks(range(tick_space))
    # D_L = v_M * tau_scan #[m]
    # D_L_um = D_L * 1e6
    # ax2.set_xticklabels(['{:1.0f}'.format(real(nr)) for nr in D_L_um[linspace(0,SN-1,tick_space).astype(int)]])
    # ax2.set_xlabel('space ($\mu$m)')

def measure_gauss():
    '''
    Measure the FWHM mostly to assure that the input parameters are applied correctly.
    :return:
    '''
    print('FWHM',L_FWHM,'m')
    print('L_c', L_sigma, 'm')
    print('f_D',f_D,'Hz')
    print('v_M',v_M,'m/s')
    print('L',max(L)*1e6,'um')
    print('dL', dL * 1e6, 'um')

    # disable I_t = [] above if we want to plot here!
    gauss_hilb = abs(hilbert(real(sum(I_t,axis=0))))
    subplot(121) # see what we use above
    # measure FWHM
    plot(tau_scan,gauss_hilb)
    gauss_hilb_max = max(gauss_hilb)
    print('sig max',gauss_hilb_max)
    s0 = L[array(where(gauss_hilb > gauss_hilb_max/2)).min()]
    s1 = L[array(where(gauss_hilb > gauss_hilb_max/2)).max()]
    FWHM = abs(s0 - s1)
    print('measured FHWM', FWHM, 'm')
    grid(True)

    # Measure frequency
    freq1_max_samples = argmax(abs((fft(real(sum(I_t,axis=0))))))
    freq2_max_samples = argmax(abs((fft(real(sum(I_t,axis=0)))))[freq1_max_samples+1:])
    print('max at', freq1_max_samples,freq2_max_samples,'samples')
    freq_array = fftfreq(n=len(tau_scan), d=1 / SR * 1e3)
    freq_array_r = freq_array[::-1]
    print('max freq',freq_array[freq1_max_samples],freq_array[freq1_max_samples+freq2_max_samples],'kHz')
    # subplot(122)
    # tukey_win = windows.tukey(SN,alpha=0.5)
    # sum_sig_win = tukey_win * sum_sig[::-1]
    # semilogy(fftfreq(n=SN,d=1), (abs(fft(sum_sig_win))), '.')
    # xlim((0,0.04))
    # ylim((1e-2,1e3))


measure_gauss()
tight_layout()
savefig('mroPM_vs_space_NL.pdf')
show()


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


# measured values from MRO_SampleMirror_raw_adjoined_graphically in pixels
# path = '/media/kai/sdb5/PS_MRO/NF2117/' + myname + '/'
# myname = 'Sensitivity_MRO_bal_170904a'
# file_selected = 'ChA*954MD*r1.npy'

    dist = [418,268,209,128,134,96,73]
    fwhm = [77,40,29,22,20,15,14,12]
    def plot_dist(self):
        figure()
        plot(self.dist,'o',label='data')
        #
        plot(range(-1,9,1),max(self.dist)/arange(10),'+',markersize=14,label='D/N')
        xlabel('order')
        ylabel('Distance (pixel)')
        legend()
        tight_layout()
        savefig('distance_simu_vs_data.pdf')

    def plot_fwhm(self):
        figure()
        plot(self.fwhm,'o',label='data')
        # confirm law of FWHM
        plot(max(self.fwhm)/arange(1,10),'+',label='FWHM/N')
        xlabel('order')
        ylabel('FWHM (pixel)')
        legend()
        tight_layout()
        savefig('fwhm_simu_vs_data.pdf')

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

    def evaluate_buffer_shift(self,N=2):
        close('all')
        print('='*20)
        D = 20 # should not be larger than dL/2
        dL = 100
        L_0 = 0
        print('N',N,'dL',dL,'D',D,'L_0',L_0)
        sN = N*dL/2 + (N-1)*D + dL/2 - N*dL
        print('sN',sN/N)
        # bf_pos_sN = L_0 + sN
        # print('bf_pos_sN',bf_pos_sN)
        # dd =  dL/2 - (N-1)*D + dL/2*(1-N) + L_0
        # print('dd',dd)
        # ddn = dd/N
        # print('dd/N',ddn)
        # ldd = sN + ddn
        # print('ldd',ldd)
        # print('bf_pos', bf_pos_sN - ddn)
        # return ldd
        return None
# rd = Real_data()
# rd.plot_fwhm()
# rd.plot_dist()
# rd.list_diffloc_items()
# rd.evaluate_buffer_shift(N=5)

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
# mv = My_verification()
# mv.plot_conventional()
# mv.plot_distortion()
# mv.plot_wave_distorted()
# mv.plot_gauss_distortion()
# mv.plot_gauss_conventional()

