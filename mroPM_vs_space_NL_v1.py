import os
from scipy import *
from scipy.fftpack import *
from scipy.signal import gaussian, hilbert, windows
from scipy.constants import speed_of_light
from matplotlib.pyplot import *
from my_format_lib import *
from appendix.scripts.my_format_lib import MPLHelper
format_plot()

SN = 65580  # buffer size
SR = 20e6  # sample rate [1/s]
# tau_scan = linspace(-SN/2/SR, SN/2/SR, SN) # time range [s]
tau_scan = linspace(0, SN/SR, SN) # time range [s]
wavelength = 1330e-9 # [m]
wavelengthBW = 60e-9 # [m]
K = 2*pi/wavelength # [1/m]
R_PM = 0.8 # reflectivity fraction of PM
L_FWHM = 2*log(2)/pi * wavelength**2 / wavelengthBW #[m]
L_sigma = L_FWHM / sqrt(8 * log(2)) #[m]
DK =  1/sqrt(2)/L_sigma #
L_S = 90e-6
v_M = L_S/max(tau_scan) # [m/s]
D = 145e-6 # spacing [m]
L_0 = 0 * 1e-6 # [m] sample mirror e.g. 0
N_O = 10 # amount of orders of reflection
f_D = 2 * v_M / wavelength #[1/s]
L = tau_scan * v_M # [m]
dL = abs(max(L)-min(L))/2 # length of range [m]
f_S = 152
I_t = []
v_M_phs = f_D / f_S * sin(linspace(-pi / 2, pi / 2, SN))  # distortion due to scanning.
CD = 1.02 # spcaing larger -> larger
CDK = 0.75 # FWHM larger -> narrower
CK = 0.970 # Frequency larger -> larger
def G(N,L,L_0):
    L_B = L_S - (L_0 + (N-1) * CD*D - L_S/2 + N*L_S/2)/N
    return  exp( -((L - L_B ) * CDK*DK*N)**2 )
def O(N,k,L): return exp(-1j * N*(CK * K * 2 * L))  # carrier
def T_PM(N):  return (1-R_PM)**2 * R_PM**(N-1) # attenuation PM splitting ratio
# mirror_stop = CD * 1000e-6 # travel of translational stage [m]
# mirror_start = -45e-6 # match position of acquired data
# mirror_steps = 200 #
# sum_sig = []
# all_sigs = [] # for export and comparison
# match_amplitude = 20350 / T_PM(1)
# fig = figure()
# for L_0 in linspace(L_0-mirror_start, L_0-mirror_stop, mirror_steps):
#     I_t = []
#     for N in range(1,N_O+1):
#         temp_sig =  G(N, L, L_0)  * O(N,K,L) * T_PM(N) * match_amplitude
#         I_t.append( temp_sig )
#     sum_sig = sum(I_t,axis=0)
#     fig.clf()
#     plot(L*1e3,sum_sig)
#     ylim((-25000,25000))
#     pause(0.1)


######### cut-off here

# fig = figure(figsize=(5.68*2,5),tight_layout=True)
figure( figsize=(11, 4), tight_layout=True) # match for comparison

counter=0
mirror_stop = CD * 1000e-6
# offset: larger -> shift to left (but depends on the direction)
mirror_start = -45e-6 + 0e-6 # + 50*5e-6 # show for some higher orders
# mirror_start = 160e-6 #2 3
# mirror_start = 300e-6 #3 4
# mirror_start = 400e-6 #3 4 5p
# mirror_start = 600e-6 #3 4 5 6
# mirror_start = 0e-6 # 1st
# mirror_stop = mirror_start # stay
mirror_steps = 200 #
sum_sig = []
all_sigs = [] # for export and comparison
match_amplitude = 20350 / T_PM(1)
if not os.path.exists('movie_jpgs'):
    os.mkdir('movie_jpgs')
for L_0 in linspace(L_0-mirror_start, L_0-mirror_stop, mirror_steps):
    I_t=[]
    for N in range(1,N_O+1):
        temp_sig =  G(N, L, L_0)  * O(N,K,L) * T_PM(N) * match_amplitude
        I_t.append( temp_sig )
    sum_sig = sum(I_t,axis=0)
    tau_scan_ms = (tau_scan + SN/2/SR) * 1e3
    # plot(tau_scan_ms,sum(I_t,axis=0))
    subplot(221)
    gca().cla()
    # plot(tau_scan,sum_sig)
    plot(L*1e3,sum_sig[::-1])
    xlabel('Space (mm)')
    # xlabel('Time (s)') # tau_scan
    # ylim((-0.05,0.05))
    ylim((-25000,25000))
    subplot(222)
    gca().cla()
    tukey_win = windows.tukey(SN,alpha=0.5)
    sum_sig_win = tukey_win * sum_sig
    semilogy(fftfreq(n=len(tau_scan),d=1/SR*1e3),abs((fft(real(sum_sig_win)))))
    xlabel('Frequency (kHz)')
    xlim((0,500))
    # ylim((1e-5,1e3)) # for unmatched amplitude
    ylim((1e4, 1e8))
    # pause(0.001)
    # savefig('movie_jpgs/movie_{:04d}.jpg'.format(counter))
    counter+=1
    print(counter)
    all_sigs.append(sum_sig)
    pause(0.1)
    while waitforbuttonpress() == 0: pass

# show()
# save('movie_jpgs/simu_vs_15595MD_CD1.02CDK0.75CK0.970.npy',all_sigs)
# save('constants_vs_data_nl_vp1.npy',all_sigs)

# Paul's approach
ml_tau_scan = linspace(0, 1 / (2 * f_S), SN)
def G_p(N,L,L_0): # Paul's approach
    C_1 = 1.0
    C_2 = 1.0
    W_0 = 0.7 # smaller -> small FWHM
    C_s = 0.92
    range_order = D*(N-1) - C_s*N*dL* (1/2 - 2*f_S*ml_tau_scan)
    # print('range_order',range_order)
    return (exp( -(( range_order + L_0 )**2  * ((0.8*DK ) ** 2))))
def O_p(N,k,L):
    C_f = 0.95
    range_order = D*(N-1) - C_f*N*dL*( 1/2 - 2*f_S*ml_tau_scan)
    # range_order = D*(N-1) - C_f*N*dL/2 * cos(6*f_S*ml_tau_scan)
    v_M_phs = f_D/f_S*sin(linspace(-pi/2,pi/2,SN))  # distortion due to scanning.
    return exp(-1j * (K * 2 * range_order))  # carrier

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

