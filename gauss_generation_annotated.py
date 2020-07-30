from scipy import *
from scipy.constants import speed_of_light
from scipy.signal import hilbert
from scipy.fftpack import *
from numpy.random import normal
from matplotlib.pyplot import *

# Generate Gaussian by summing large amount of frequency components
class MPLHelper(object):
    mpl_key = None
    rcParams['font.family'] = 'DejaVu Serif' # look this up on the console by typing rcParams and find the right section
    # or rcParams.serif to see which serif families are available. However not all families are installed
    rcParams['font.size'] = 14
    rcParams['mathtext.fontset'] = 'dejavuserif'
    # Emmh. Didn't do this myself.
    # This guy https://stackoverflow.com/questions/2537868/sans-serif-math-with-latex-in-matplotlib
    # did this Answer 4.
    rcParams['text.latex.preamble'] = [
        r'\usepackage{siunitx}',  # i need upright \micro symbols, but you need...
        r'\sisetup{detect-all}',  # ...this to force siunitx to actually use your fonts
        # r'\usepackage{helvet}',  # set the normal font here
        # r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
        # r'\sansmath' # <- tricky! -- gotta actually tell tex to use
    ]

    def __init__(self):
        # We keep this as a placeholder to show that we do not initialize anything
        pass

    def getCycMarkers(self, name=None):
        assert name, 'Please give a name! I.e. name = ''marker'' which can be accessed by cycler[''marker'']'
        mks = list(matplotlib.markers.MarkerStyle.markers.keys())
        del mks[1] # remove one that is barely visibile
        single_cycle = cycler(name,mks) # cycler is from matplotlib. Or use itertools.cycle
        cyclic_mks = single_cycle()
        return cyclic_mks

    def list_of_markers(self):
        mks = list(matplotlib.markers.MarkerStyle.markers.keys())
        del mks[1] # remove one that is barely visibile
        return mks

    def mpl_button_handler(self, evt):
        print(evt.key)
        if evt.key in 'qQ': close('all')

    def register_close_key(self, key='q'):
        self.mpl_key = key
        gcf().canvas.mpl_connect('key_press_event', self.mpl_button_handler)

def gauss(w,w0,sigma):
    return exp(-((w-w0)**2) / (2*sigma**2))

def task_A():
    N = 20000  # buffer size
    SR = 20e6  # sample rate (seconds)
    tmr = linspace(0, N/SR, N) # time range
    wavelength = 1330e-9 # meter
    wavelengthBW = 60e-9

    freqs = normal(loc=40,scale=2,size=100)
    sig = []
    for f in freqs:
        sig.append(cos(linspace(-f*pi,f*pi,N)))
        # print(shape(sig))
        cla()
        plot(array(sig).sum(axis=0))
        pause(0.01)
    show()
# task_A()

def gauss_FFTgauss():
    N = 20000
    SR = 20e6

    zr = linspace(-N,N,2*N)
    kr = linspace(-1/N,1/N,1*N)
    print(1/N)
    sampleBW = 250 # samples
    dk = 2*sqrt(2*log(2))/sampleBW
    print(dk,dk**2)
    def G(zr): return exp(-((zr)**2) *2*dk**2)
    subplot(311)
    plot(zr,G(zr)),grid(True)
    gca().set_yticks([0.5])
    gca().set_xticks([-sampleBW/2,sampleBW/2])
    title('Spatial space [m]')
    subplot(312)
    plot(abs(fftshift(fft(exp(-((zr)**2)*2*dk**2))))),grid(True)
    title('Apply fft -> k space [1/m]')
    subplot(313)
    title('Theory fft -> k space [1/m]')
    plot( exp( -(kr**2) / ((dk/200)**2) )),grid(True)
    tight_layout()
    show()
# gauss_FFTgauss()

def measure_FWHM(gauss_shape, x_range, unit = ''):
    s0 = x_range[(array(where(gauss_shape > 0.5)).min())]
    s1 = x_range[(array(where(gauss_shape > 0.5)).max())]
    FWHM = abs(s0-s1)
    print('measured FHWM', FWHM,unit)
    return FWHM

def from_gaus_generation():
    N = 40000  # buffer size
    SR = 20e6  # sample rate (seconds)
    tmr = linspace(0, N / SR, N)  # time range [s]
    print('scan time:',max(tmr),'s','dt:',1/SR,'s',1/SR*1e9,'ns')
    wavelength = 1330e-9  # meter
    wavelengthBW = 60e-9

    FWHM = 2 * log(2) / pi * wavelength ** 2 / wavelengthBW  # [m]
    FWHM_tau = 2 * log(2) / pi * (wavelength*speed_of_light) ** 2 / (wavelengthBW * speed_of_light) #[1/s]
    print('FWHM', FWHM, 'm')
    print('1/FWHM_tau', 1/FWHM_tau, 's')
    sigma = FWHM / 2 / sqrt(2 * log(2))  # [m]
    print('sigma', sigma, 'm')

    v_M = 0.2  # [m/s]
    spr = tmr * v_M  # spatial range [m]
    sigma_tau = (1/FWHM_tau) / 2 / sqrt(2 * log(2))
    print('sigma_tau',sigma_tau,'s')
    print('scan distance', max(spr), 'm', max(spr)*1e6,'um')
    f_D = 2 * v_M / wavelength  # [1/s]
    print('f_D', f_D, 'Hz')

    t0 = -0.0*N/SR/2
    z0 = 0* N/SR/2 * v_M
    all_Gs = []
    for O in range(1,19):
        D = 10e-6 #[m] spacing
        Dt = (D / v_M) #[s] spacing in tau
        # G_in_samples = exp(-(spr-z0-D)**2 / 2/sigma**2 * O )
        w = 2 * pi * f_D * (O)
        osc = exp(-1j*w*tmr)
        G_in_samples = exp(-(tmr-t0-(O*Dt))**2 / 2/ sigma_tau**2 * O) * osc
        all_Gs.append(G_in_samples)

    # using spr to confirm FWHM
    measure_FWHM(all_Gs[0],spr,'m')
    measure_FWHM(all_Gs[0],tmr,'s')
    print(shape(all_Gs))

    subplot(211)
    # plot(spr*1e6, array(all_Gs).T)
    plot(tmr, real(array(all_Gs).sum(axis=0)))
    xlabel('time (s)') # tmr
    # xlabel('$\Delta l$ ($\mu$m)') # spr
    ylabel('Amplitude (arb.)')
    title('SP:{:d}, FWHM:{:2.1f} um,\n WL:{:1.2f} um, BW:{:1.2f} um,v_M:{:1.3f} m/s'.format(N,FWHM*1e6,wavelength*1e6,wavelengthBW*1e6,v_M))
    grid(True)
    subplot(212)
    # for G in all_Gs:
    #     plot(linspace(-SR/2,SR/2,N),abs(fftshift(ifft(G))))
    Gsum = sum(all_Gs,axis=0)
    plot(linspace(-SR/2,SR/2,N)*1e-3,abs(fftshift(ifft(Gsum))),'.-')
    xlabel('Frequency (kHz)')
    xlim([0,10000])
    tight_layout()
    show()

# from_gaus_generation()

import gauss_generation as gaussGen
class GaussGen(MPLHelper):

    def plotGG(self):
        print(gaussGen.f_D, gaussGen.f_DBW)
        print(gaussGen.FWHM)
        figure()
        plot(gaussGen.spr*1e6, gaussGen.spectral_wave_sum)
        plot(gaussGen.spr*1e6, abs(hilbert(real(gaussGen.spectral_wave_sum))))
        grid(True)
        show()
GaussGen().plotGG()