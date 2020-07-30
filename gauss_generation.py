import os
from scipy import *
from scipy.constants import speed_of_light
from scipy.fftpack import * # overwrite hilbert here!
from scipy.signal import hilbert, tukey, convolve, correlate, resample, resample_poly, chirp
from scipy.optimize import fsolve
from numpy.random import normal,randn
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.pyplot import *
from matplotlib.path import Path
# from pprint import pprint
import warnings
import time
from scipy.interpolate import UnivariateSpline
from scipy import interpolate
import scipy as sy

# set_printoptions(precision=3)


# Generate Gaussian by summing large amount of frequency components

class SimulationHelpers(object):
    def measure_FWHM(self, x, g):
        '''
        The g parameter needs to be a Gaussian or envelope
        :param x:
        :param g:
        :return:
        '''
        spline = UnivariateSpline(x, g - max(g) / 2, s=0)
        rts = spline.roots()
        r1, r2 = rts.min(), rts.max()
        return abs(r1-r2)

    def measure_FWHM_h(self,x,y):
        '''
        If the signal is oscillatory with Gaussian shape the Hilbert needs to be applied first.
        :param x:
        :param y:
        :return:
        '''
        env = abs(hilbert(real(y)))
        spline = UnivariateSpline(x,env-max(env)/2,s=0)
        rts = spline.roots()
        r1, r2 = rts.min(), rts.max()
        return abs(r1-r2)

    def measure_FWHM_plot(self,x,y):
        '''
        Y needs to be the a Hilbert if the envelope is fitted by a spline.
        In other cases it can be fitted over y**2.
        :param x:
        :param y:
        :return:
        '''
        spline = UnivariateSpline(x,y,s=0)
        plot(x,spline(x))
        return spline

class GaussTheory(SimulationHelpers):
    '''

    '''
    def __init__(self, run=False):
        '''
        Call GaussTheory(run=True)
        @param run:
        '''

        # self.std_freq_vs_time_dom()
        # self.integral_gauss()
        self.mixing_artifacts()
        # self.test_cursor()
        show()

    def std_freq_vs_time_dom(self):
        '''
        Show the inverse relationship of tha STD in the freq-domain vs. time-domain
        @return:
        '''
        N = 100000
        rng_min = -pi
        rng_max = pi
        z = linspace(rng_min,rng_max,N)
        dz = 1/N/rng_max
        # sig_sq =  (2*(0.01/sqrt(2*log(2))))**2
        sig_sq =  (2*(0.1*2*pi))**2
        G = exp(-z**2/sig_sq)
        Gfft = abs((fft(G)))
        Gfft = Gfft / Gfft.max()
        plot(z,G)
        plot(fftfreq(N,dz), Gfft)
        grid(True)

    def integral_gauss(self):
        '''
        Show that int(Gauss) = sqrt(pi)
        http://mathworld.wolfram.com/GaussianIntegral.html
        @return:
        '''
        N = 10000
        bd_up  =  2
        bd_low = -bd_up
        sigma = 1
        x = linspace(bd_low,bd_up,N)
        G = exp(-x**2/(2*sigma**2))
        Gn = G / (N/bd_up)
        plot(x,Gn,'.-',linewidth=0.5,ms=5.5)
        self.measure_FWHM_plot(x,Gn)
        xlim((-5,5))
        print('sqrt(pi)       =',sqrt(pi))
        print('trapz(G,x)     =',trapz(G,x))
        print('trapz(G**2,x)  =',trapz(G**2,x))
        print('trapz_diff     =',trapz(G**2,x) - sqrt(pi))
        print('sqrt(pi)/2     =',sqrt(pi)/2)

        print('\n')
        fwhm_measure = self.measure_FWHM(x,Gn)/2
        print('FWHM_spline/2  =',fwhm_measure)
        print('FWHM_theo      =',sigma*sqrt(2*log(2)))
        grid(True)

    def mixing_artifacts(self):
        '''
        Hypothesis:
        * In the optical domain before digitizing the chirped frequencies generate a beat that
        is within a frequency band of one order.
        * The simulation here tries to recreate this

        Limits:
        * The simulation heavily depends on the frequncy components and the sample rate
        * The sample rate has no effect in the optical domain as it is not yet digitized

        Observation:
        * Some components are within the frequency band.
        * With more frequencies or sample points the parasitic frequencies reduce or shift
        * No significant peak can be observed.
        @return:
        '''
        T = 50 #cycles to acquire
        N = 10000*T
        win = tukey(N,alpha=0.0)
        t_up = T*1e-3 #s Acquisition time
        t_low = 0
        t = linspace(t_low,t_up,N)
        f1l = 30.0e3 #1/s
        f2l = 45.0e3
        mt1 = ((sin(linspace(-pi/2,pi/2,N)))) #linspace(0,1.0,N)
        mt2 = ((sin(linspace(-pi/2,pi/2,N)))) #linspace(0,1.1,N)
        # print(trapz(mt1+1.0))
        # sig1 = win*sin(2*pi*f1l*t)
        sig1 = win*sin(((3.21)/N*T**2)*f1l*2*pi*mt1 )
        sig2 = win*sin(((3.21)/N*T**2)*f2l*2*pi*mt2 )
        sig1l = win*sin(2*pi*f1l*t)
        sig2l = win*sin(2*pi*f2l*t)
        sigMix = sig1+sig2
        sigMixl = sig1l+sig2l

        figure(tight_layout=True)
        plot(t, sigMix)
        # plot(t, sig1l)
        plot(t,mt1)

        figure(tight_layout=True)
        ax = gcf().add_subplot(111)
        plot(fftfreq(N,t_up/N), abs(fft(sigMix)))
        plot(fftfreq(N,t_up/N), abs(fft(sigMixl))/5.2)
        xlim((0,f2l+f2l/2))
        from matplotlib.widgets import Cursor
        cursor = Cursor(ax,lw=2)
        show()

    def test_cursor(self):
        fig = figure(figsize=(8, 6))
        ax = fig.add_subplot(111, facecolor='#FFFFCC')

        x, y = 4 * (random.rand(2, 100) - .5)
        ax.plot(x, y, 'o')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

        from matplotlib.widgets import Cursor
        # set useblit = True on gtkagg for enhanced performance
        cursor = Cursor(ax, useblit=True, color='red', linewidth=2)



class GaussGenerator(SimulationHelpers):
    '''
    Generate Gauss and evaluate or measure FWHM.
    '''
    N = 20000  # buffer size
    SR = 20e6  # sample rate (seconds)
    f_D = 61e3 # measured Doppler frequency 1st order non-linear
    f_D_lin = f_D/pi
    f_S = 152  # measured scanning frequency [1/s]
    L_S = 90e-6 # scanning range mirror [m]
    v_M = L_S * f_S # about 0.014 m/s
    # print('v_M',v_M,'m/s')
    # tau_p must be replaced with some expression that relates tau_p to the scan speed of the mirror
    # and the speed of light. Because tau_p is the phase velocity of the em-wave!
    space_smp = linspace(-N/2,N/2,N).astype(complex)
    smp_rng = linspace(-N/2/SR, N/2/SR, N).astype(complex) # time range
    space = linspace(-100e-6, 100e-6, N).astype(complex) # spatial range in [m]
    tau_p = linspace(0,200e-6/v_M,N).astype(complex) # scan time for 200 um based on mirror speed
    # print('tau_p',tau_p,'s')
    # space = space_smp
    wavelength = 1330e-9 # meter
    wavelengthBW = 60e-9

    FWHM = 2*log(2)/pi * wavelength**2/wavelengthBW  #[m]
    # print('FWHM',FWHM*1e6,'um')

    spr = linspace(0, f_D*wavelength*N/SR, N) # spatial range
    number_freq_components = 100
    number_wl_components=number_freq_components


    # generate all frequencies terms
    # w_N = 2 * pi * normal(loc=f_D, scale=f_DBW, size=number_freq_components)
    # w_N = w_N + abs(min(w_N))

    def __init__(self,run=False):
        if run:
            # GaussGenerator().convolution_of_GaussOsci()
            # GaussGenerator().plot_Gauss_accurate()
            # GaussGenerator().accurate_Gauss_fft()
            # GaussGenerator().tutorial_fft_mathematica()
            # GaussGenerator().tutorial_fft_mathematica_adding_frequency()
            # GaussGenerator().tutorial_gen_wave_with_fft_and_interference() # **
            # GaussGenerator().spectrum_to_complex() # ***
            # GaussGenerator().multi_sample_reflectors() # ****
            # GaussGenerator().phase_and_spectrum()
            # GaussGenerator().FD_E_fields_principle() # ****
            # GaussGenerator().FD_calibration_line() # ****
            # GaussGenerator().plot_wvl_freq_ranges() # **
            # GaussGenerator().sum_stepwise_plot_with_fit()
            GaussGenerator().compare_sum()
            # GaussGenerator().compare_errors()
            # GaussGenerator().run_comparison()
            # GaussGenerator().simple_wave_summation_gauss_distribution()
            # GaussGenerator().generate_carrier_Gauss_conventionally()
            # GaussGenerator().generate_carrier_Gauss_by_FFT()
        pass

    def convolution_of_GaussOsci(self):
        '''
        Demonstrate convolution of oscillation with Gaussian envelope.
        There is no evidence yet that this is valid for TD-OCT.
        :return:
        '''
        rng_max = 100*pi
        rng = linspace(0,rng_max, 10000)
        sigma = 10
        sig = exp(1j*rng) * exp(-(rng_max/2-rng)**2 / (2*sigma**2))
        sigc = convolve(sig,sig,mode='same')
        print(self.measure_FWHM_h(rng,sig))
        print(self.measure_FWHM_h(rng,sigc))
        figure()
        plot(rng,sig)
        plot(rng,abs(hilbert(real(sig))))
        plot(rng,sigc/(1000/(pi/2)))
        plot(rng,abs(hilbert(real(sigc)))/(1000/(pi/2)))
        show()

    def plot_Gauss_accurate(self):
        '''
        The meaning of accurate is here related to the accurate amplitude and use of parameters.

        Besides of mathworld the wikipedia provides some additional informations
        https://en.wikipedia.org/wiki/Gaussian_function
        e.g. like higher order Gaussians and the sampled Gaussian kernel.
        If we need the latter I don't know yet.
        :return:
        '''
        x = linspace(-500,500,35000)
        u = 0
        # setting sigma directly relates to the STD of the Gaussian.
        sigma = 1000
        # http://mathworld.wolfram.com/GaussianFunction.html

        # The deduction of the FWHM is given at the webpage as well
        FWHM = pi*1
        sigma = FWHM/2/sqrt(2*log(2))
        # The scaling factor assures the area under the curve is one.
        # Otherwise the amplitude is one.
        A = 1/sigma/sqrt(2*pi)
        G = exp(-(x-u)**2 / (2*sigma**2))
        figure(num='Gauss accurate', tight_layout=True)
        plot(x, G,'.-',lw=0.5)
        plot(x,abs(fftshift(fft(G)))/sqrt(2*pi*sigma**2)/35,'.-',lw=0.5)
        ax = gca()
        ax.set_yticks(arange(0,1,0.1))
        grid(True)
        show()

    def accurate_Gauss_fft(self):
        '''
        I expand on plot_Gauss_accurate using fft to see if I can understand the maths behind it.

        We know that the Gaussian bandwidth for a spectral bandwidth calculates to
        psf = 2*log(2)/pi * f_0**2 / df

        For the initial plot all values need to be converted into 1/m!

        Besides of mathworld the wikipedia provides some additional informations
        https://en.wikipedia.org/wiki/Gaussian_function
        e.g. like higher order Gaussians and the sampled Gaussian kernel.
        If we need the latter I don't know yet.
        :return:
        '''
        N = self.N
        x = linspace(0,1/20000,N) # 20000 nm
        u = 1/1330 # 1330 nm center wavelength

        # The deduction of the FWHM is given at the webpage as well
        FWHM = 1/60 # 60 nm bandwidth
        print('FWHM_spectrum:',FWHM,'nm')
        print('FWHM_td:',2*log(2)/pi * 1330**2 / 60,'nm')
        sigma = FWHM/2/sqrt(2*log(2))
        # The scaling factor assures the area under the curve is one.
        # Otherwise the amplitude is one.
        A = 1/sigma/sqrt(2*pi)
        G = exp(-(x-u)**2 / (2*sigma**2))

        figure(num='Gauss accurate', tight_layout=True)
        subplot(211)
        plot(x, G)
        ax = gca()
        # ax.set_yticks(arange(0,1,0.1))
        grid(True)
        xlabel('Wavelength (nm)')

        subplot(212)
        # to avoid the array reversal do not use fftfreq
        # x_inv = fftfreq(n=self.N, d=self.N)
        x_inv = linspace(0,20000,N)
        G_fft = fftshift(fft(G)) # do fftshift to reverse arrays
        G_fft = G_fft/max(G_fft)
        plot(x_inv, G_fft, '.-')
        plot(x_inv, abs(hilbert(real(G_fft))))
        print(self.measure_FWHM_h(x, G_fft))
        grid(True)
        ax = gca()
        ax.set_yticks(arange(0,1.1,0.1))
        show()

    def tutorial_fft_mathematica(self):
        '''
        See tutorial text
        /home/kai/zotero/zotero_private/storage/XTF29L24/Tutorial8_FFT.pdf
        The bin ratio N allows to adjust the amount of samples FFT vs time/space.
        :return:
        '''
        N = 0.2 # bin ratio
        c = 300 # nm/fs
        wl_0 = 800 # nm center wavelength
        w_0 = 2*pi*c/wl_0 # center frequency rad/s

        print('w_0',w_0/N,'rad/s')

        num = 2**12 # number of sample points
        print('num',num)
        T = 2000 # time [fs] arbitrary acquisition time
        dt = T/num/N # sample rate
        print('dt',dt,'fs/S')
        dw = 2*pi/num/dt/N # the same as below
        dw = 2*pi/T*N #[1/fs] # frequency range
        print('dw',dw,'PHz',dw*1e3,'THz')

        #
        time = linspace(-num*dt/2, num*dt/2, num).astype(complex)
        freq = linspace(-num/2*dw, num/2*dw, num).astype(complex) # relative frequency range
        print('freq rng:',freq[0],freq[-1])

        # around the centre frequency the relative frequency is determining the visible or sample range
        # due to the time range.
        freq_abs = linspace(w_0-dw*num/2,w_0+dw*num/2,num)
        print('freq abs:',freq_abs[0], freq_abs[-1])

        # in our case the fwhm in fs correlates due to the heterodyne mixing to a spatial range.
        # The spatial range is based on the frequency 800 nm.
        fwhm = 100 # fs
        P_avg = 0.001 # W
        PRR = 1/100e6 *1e15 # pulse repetition rate
        P_0 = P_avg * PRR / fwhm
        def e(t): return sqrt(P_0)*exp(-2*log(2)*((t)/fwhm)**2)
        def I(t): return e(t) * conj(e(t))

        figure(num='tutorial FFT',tight_layout=True)
        subplot(211)
        plot(time, I(time),'.-')
        xlim((-250,250))
        xlabel('Time (fs)')
        ylabel('Intensity (W)')

        e_w = fftshift(fft(e(time)))
        # the python fft does not implicitly cancel the bin number.
        I_w = e_w * conj( e_w )/num
        wavel_x = 2*pi*c/freq_abs

        ax = subplot(212)
        ax2 = ax.twiny()
        ax2.set_xlabel('1/fs')
        ax2.plot(freq, I_w[::-1])
        ax2.set_xlim((-0.088,0.088))

        ax.plot(wavel_x, (I_w),'.-')
        ax.set_xlim((770,830))
        # ylabel('Spectral intensity')
        # xlabel('nm')
        show()

    def tutorial_fft_mathematica_adding_frequency(self):
        '''
        We should acknowledge that at this stage the resulting time-domain signal is the true
        high frequency signal which is not yet the interference signal.

        Consequently, the use of 'wavenumber' on the generated waven would be meaningless!
        (see further discussion here)

        Answer the question of the target range
        =======================================
        If we work with linear frequencies (https://en.wikipedia.org/wiki/Wavenumber) then
        the fft on wvl_spec [nm] --> then [1/m] and v = 1/wvl_spec.

        Wavenumber?
        ===========
        The problem is however that we obtain a wave and not a spectrum.
        So the wavenumber on a wave is meaningless!

        Construction of the x-range of the target wave: wavenumber for a wave is meaningless !!
        =======================================================================================
        So a wave can have time or length.

        Time is currently implicit and unknown but space should be according to the wavelength.

        However how many waves are there?

        The frequency of the light source is 225 THz meaning it would take for one cycle to travel
        a distance of 1330 nm in 1/225 THz or 4.4e-15s (4.4e-3 fs) (m-3,u-6,n-9,f-12).

        Due to sampling and the selected frequency range 4.508e+15 / 4096 = df = 1.1e+12 [1/s].

        Then the time delta is 1/1.1e+12 = 9.086e-13 (0.91 fs).

        If we have time then we can also calculate the length light travels during this time
        z = c * t = 300 000 km/s * 9.1e-13 s = 300e6 * 9.1e-13 = 2.724e-4 m.

        See also tutorial text
        /home/kai/zotero/zotero_private/storage/XTF29L24/Tutorial8_FFT.pdf
        The bin ratio N allows to adjust the amount of samples FFT vs time/space.

        '''
        nr = 0.05 # bin ratio -> change in concert with N and
        print('nr: {}'.format(nr))
        c = speed_of_light #* 1e-15 / 1e-9 # nm/fs
        wvl_0 = 1330e-9 # m center wavelength
        print('f_0: {:0.3e} 1/s'.format(c/wvl_0))
        print('wl_0: {:0.3e} m'.format(wvl_0) )
        w_0 = 2*pi*c/wvl_0 # center frequency rad/s
        f_0 = c/wvl_0 # center frequency 1/s
        print('w_0: {:0.3e} rad/s'.format(w_0/nr))

        N = 2**12 # number of sample points
        print('N',N)

        # in our case the fwhm in fs correlates due to the heterodyne mixing to a spatial range.
        # The spatial range is based on the frequency 800 nm.
        fwhm_wl = 60e-9
        print('fwhm_wl: {:0.3e}'.format(fwhm_wl))
        sigma_wl = fwhm_wl/2/sqrt(2*log(2))
        print('sigma_wl: {:0.3e}'.format(sigma_wl))

        # 2 times wl_0 / f_0 is arbitrary and serves only to adjust the samples.
        wvl_spec = linspace(0, 2*wvl_0,N)/nr
        print('wvl_spec: {:0.3e}'.format(wvl_spec[-1]))
        freq_spec = linspace(0, 2*f_0, N)/nr
        print('freq_spec: {:0.3e}'.format(freq_spec[-1]))
        dt = 1/(freq_spec[-1]/N)
        time_rng = linspace(0,dt,N)
        print('time_rng: {:0.3e}'.format(time_rng[-1]))
        spac_rng = linspace(0,speed_of_light*time_rng[-1],N)

        # around the centre frequency the relative frequency is determining the visible or sample range
        # due to the time range.
        # freq_abs = linspace(w_0-dw*N/2,w_0+dw*N/2,N)
        # print('freq abs:',freq_abs[0], freq_abs[-1])

        def S(wl): return exp(-((wl-wvl_0)**2/(2*sigma_wl**2)))
        print('fwhm_wl_out: {:0.3e}'.format(self.measure_FWHM(wvl_spec,S(wvl_spec))))
        # def e(t): return sqrt(P_0)*exp(-2*log(2)*((t)/fwhm)**2)
        def I(t): return e(t) * conj(e(t))
        # print('fwhm_wl_m:',self.measure_FWHM(wvlen,S(wvlen)*conj(S(wvlen))))

        figure(num='tutorial FFT freq to time',tight_layout=True)
        ax = subplot(311)
        ax.set_yticks(arange(0,1.1,0.1))
        wvl_spec_nm = wvl_spec*1e9
        plot(wvl_spec_nm,S(wvl_spec),'.-')
        xlabel('Wavelength (nm)')
        grid(True)
        xlim((0,wvl_spec_nm[-1]))

        def freq_axis():
            ax2 = ax.twiny()
            # ax2.plot(freq_Phz*1e3,S(wvlen),'.',ms=1)
            # ax2.set_xticks(linspace(0,freq_Phz[-1]*1e3,3))
            import matplotlib.ticker as plticker
            loc = plticker.AutoLocator() #MultipleLocator()  # this locator puts ticks at regular intervals
            ax2.xaxis.set_major_locator(loc)
            freq_spec_THz = freq_spec*1e-12
            ax2.set_xlim(0,freq_spec_THz[-1]) # from Hz to THz
            ax2.set_xlabel('Frequency (THz)')
        freq_axis()

        ax = subplot(312)

        # Linear wavenumber
        v = 1/wvl_0/2
        print('v:{:0.3e} 1/m {:0.3} 1/nm'.format(v,v*1e-9))
        # At this point the high frequency light wave

        # plot(s_rng,fftshift(fft(convolve(S(wvlen),(S(wvlen))))),'.-')
        S_fft = fftshift(fft(S(wvl_spec)))
        S_fft = S_fft/max(S_fft)
        # spline = self.UnivariateSpline(s_rng,S_fft,s=0)
        # print(spline.roots())
        # plot(s_rng/1e3,spline(s_rng),'.-')

        # plot(time_rng*1e12,S_fft,'.-')
        # xlabel('Time (fs)')
        plot(spac_rng*1e6,real(S_fft),'.-')
        xlabel('Space (um)')
        plot(spac_rng*1e6,abs(hilbert(real(S_fft))),'.-')
        print('fwhm_s: {:0.3f} (um)'.format(self.measure_FWHM_h(spac_rng*1e6,S_fft)))
        print('fwhm_s/2: {:0.3f} (um)'.format(self.measure_FWHM_h(spac_rng*1e6,S_fft)/2))
        # xlabel('Range (um)')
        grid(True)

        subplot(313)
        # Consquently applying the FFT again should result to be about 225 THz again.
        S_ifft = abs(fftshift(fft(real(S_fft))))
        import warnings
        warnings.filterwarnings('error') # make warnings try except able
        try:
            # We expect only here a division by zero and accept this minor disonance.
            # But in any other cases we should check for this.
            f_rng = fftfreq(n=N,d=freq_spec)
        except RuntimeWarning as rw:
            pass

        f_rng = linspace(-freq_spec[-1]/2,freq_spec[-1]/2,N)
        plot(f_rng/1e12,S_ifft,'.-')
        xlabel('Frequency (THz)')


        show()

    def plot_wvl_freq_ranges(self):
        '''
        This is just to illustrate how the scales need to be matched.
        :return:
        '''
        wvl_0 = 1330
        wvl_rng = linspace(1e-9,wvl_0*2)
        frq_rng = 300/wvl_rng

        fig,ax = subplots()

        plot(wvl_rng,wvl_rng,label='wvl')
        plot(wvl_rng,ones(len(wvl_rng))*wvl_0,label='wvl_0')
        print(argwhere(wvl_rng-wvl_0 < 1/500).max())
        legend()

        # Axes().set_yticklabels()

        ax2=ax.twinx()
        axy=ax.twiny()
        ax2.semilogx(frq_rng)
        xt1 = ax.get_xticks()
        axy.set_xticks(xt1[1:-1])
        # axy.set_xticklabels(['{:0.3f}'.format(i) for i in frq_rng[range(len(xt1[1:-1]))]])
        # plot(frq_rng,label='frq')
        # plot(ones(len(frq_rng))*300/wvl_0,label='frq_0')
        # print(argwhere(300/wvl_0-frq_rng < 1/500).max())
        # ax2.legend()
        show()

    def tutorial_gen_wave_with_fft_and_interference(self):
        '''
        TODO Since the sample wave passes twice the sample (or sample arm)
        Fercher (2003), p268, "Since backscattered light travels twice through the sample, ..."
        What is the underlying physical effect.

        (S1 + S2)**(1/4) provides a 13 um PSF. Why?
        ===========================================
        This may be some artifact of the relation of the halfing of the source spectrum

        High level idea of using interference
        =====================================
        The light spectrum of 1330 nm will have a managable wavelength of 1.3 um alright, that can be easily
        detected but not the temporal oscillation.
        The spatial detection is exactly the interferrometry probing one wave with a reference wave or overlay
        a sample spectrum with a reference spectrum.
        If the sample reflection returns some coherent waves those will superimpose with the reference waves
        crating a signal of either the spatial frequency of the coherent waves or the frequency of the scanning
        mirror.

        The produced time-domain wave based on the spectrum may be possibly the wave that is encountered
        during interference.

        Because, the interferometer does nothing else than probing the spatial structure of the sample
        beam with the reference beam by scanning.

        Basically the scanning just reduces the wave fluctuations to a more managable frequency.

        The only reason to perform interference is to investigate the spectral distortion due to a distorted
        sample spectrum.

        Show that superposition does half the PSF.
        ==========================================
        Superposition does not half the PSF.
        Convolution or multiplication in the frequency domain does widen the spectrum, but this is not the
        effect that creates the PSF with the right bandwidth.
        However, the correct bandwidth of the PSF must be by obeying the double pass of the interferrometer arms.

        Now the superposition itself would widen the PSF either convolving the TD or multiplying the spectrum.

        However, due to the mirror motion the Doppler effect does in space just double the frequencies.

        This most easily achieved by just halving the frequency range of the source spectrum or doubling the
        bandwidth.

        How to half the PSF by summing two source spectra?
        ==================================================
        According to theory the summing does not change the spectrum.
        The sum creates the superposition and does not change anything except if the sample spectrum has changed.


        See also tutorial_fft_mathematica_adding_frequency to study the generation of the frequency range
        according to fundamental physical properties.
        See also tutorial text
        /home/kai/zotero/zotero_private/storage/XTF29L24/Tutorial8_FFT.pdf
        The bin ratio N allows to adjust the amount of samples FFT vs time/space.

        '''
        nr = 0.05  # bin ratio -> change in concert with N and
        print('nr: {}'.format(nr))
        c = speed_of_light  # * 1e-15 / 1e-9 # nm/fs
        wvl_0 = 1330e-9  # m center wavelength
        print('wl_0: {:0.3e} m'.format(wvl_0))
        f_0 = c / wvl_0
        print('f_0: {:0.3e} 1/s'.format(f_0))
        w_0 = 2 * pi * c / wvl_0  # center frequency rad/s
        print('w_0: {:0.3e} rad/s'.format(w_0))

        N = 2 ** 12  # number of sample points
        print('N', N)

        # in our case the fwhm in fs correlates due to the heterodyne mixing to a spatial range.
        # The spatial range is based on the frequency 800 nm.
        fwhm_wl = 60e-9
        print('fwhm_wl: {:0.3e}'.format(fwhm_wl))
        sigma_wl = fwhm_wl / 2 / sqrt(2 * log(2)) # * 2
        # The most plausible way to add the double pass
        # sigma_wl = fwhm_wl / 1 / sqrt(2 * log(2))
        print('sigma_wl: {:0.3e}'.format(sigma_wl))

        # 2 times wl_0 / f_0 is arbitrary and serves only to adjust the samples.
        # TODO: evaluate if this can be ommitted with the right nr vs N settings.
        rng_f = 2
        wvl_spec = linspace(0, rng_f * wvl_0, N) / nr
        print('wvl_spec: {:0.3e}'.format(wvl_spec[-1]))
        freq_spec = linspace(0, rng_f * f_0, N) / nr
        print('freq_spec: {:0.3e}'.format(freq_spec[-1]))
        dt = 1 / (freq_spec[-1] / N)
        time_rng = linspace(0, dt, N)
        print('time_rng: {:0.3e}'.format(time_rng[-1]))
        spac_rng = linspace(0, speed_of_light * time_rng[-1], N)
        # TODO: spac_rng can also be calculate as below
        # wvl_rng =  wvl_0*N/2
        # spac_rng = linspace(-wvl_rng, wvl_rng, N)*nr
        # c * t/2 == c * dt * N/2 == c * 1/(f_0) * N /2 == c * 1/(c/wvl_0) * N / 2
        # c * wvl_0 *N / c  / 2 == wvl_0 * N / 2



        def calc_wavelength():
            def S1(wl): return exp(-((wl-wvl_0)**2/(2*sigma_wl**2)))
            # def S1(wl): return exp(-(4*log(2)*(wl-wvl_0)**2/(fwhm_wl**2)))
            print('fwhm_wl_out: {:0.3e}'.format(self.measure_FWHM(wvl_spec,S1(wvl_spec))))
            def S2(wl): return exp(-((wl-wvl_0)**2/(2*sigma_wl**2)))
            # def S2(wl): return exp(-(4*log(2)*(wl-wvl_0)**2/(fwhm_wl**2)))
            print('fwhm_wl_out2: {:0.3e}'.format(self.measure_FWHM(wvl_spec,S2(wvl_spec))))

            figure(num='tutorial FFT freq to time',tight_layout=True)
            ax = subplot(311)
            ax.set_yticks(arange(0,1.1,0.1))
            wvl_spec_nm = wvl_spec*1e9
            plot(wvl_spec_nm,S1(wvl_spec),'.-')
            plot(wvl_spec_nm,S2(wvl_spec),'r+-',ms=1)
            xlabel('Wavelength (nm)')
            grid(True)
            xlim((0,wvl_spec_nm[-1]/10))

            def freq_axis():
                ax2 = ax.twiny()
                # ax2.plot(freq_Phz*1e3,S(wvlen),'.',ms=1)
                # ax2.set_xticks(linspace(0,freq_Phz[-1]*1e3,3))
                import matplotlib.ticker as plticker
                loc = plticker.AutoLocator() #MultipleLocator()  # this locator puts ticks at regular intervals
                ax2.xaxis.set_major_locator(loc)
                dummy = ones(len(freq_spec))*NaN
                plot(freq_spec,dummy)
                freq_spec_THz = freq_spec*1e-12
                ax2.set_xlim(freq_spec_THz[0],freq_spec_THz[-1]) # from Hz to THz
                ax2.set_xlabel('Frequency (THz)')
            freq_axis()

            ax = subplot(312)
            # halving the wavelength spectral range causes the doubling of all frequency components.
            # The same could be achieved by doubling the bandwidth.
            # S = S(wvl_spec/2)
            S1 = S1(wvl_spec)
            S2 = S2(wvl_spec)
            I1 = S1+S2
            # I1 = convolve(S1,S2)[range(0,len(S1)+len(S2),2)] # mode='same','valid' fail. Why?
            # I1 = convolve(S1,S2)[0:len(S1)]
            # I1 = (S1+S2)**(1/4) #OK
            # I1 = (S1**(1/2)+S2**(1/2))**(1/2)
            # for shift in range(len(S2)):
                # widens
                # I1 = (S1 + roll(S2,shift=shift))
            # I1 = 2*sqrt(S1*S2)
            # I1 = abs(S1)**(1/4) + abs(S2)**(1/4)
            # I1 = S1
            # I2 = S2
            # Linear wavenumber
            v = 1/wvl_0/2
            print('v:{:0.3e} 1/m {:0.3} 1/nm'.format(v,v*1e-9))

            S_fft1 = fftshift(fft(I1))
            S_fft1 = S_fft1/max(S_fft1)
            # S_fft2 = fftshift(fft(I2))
            # S_fft2 = S_fft2/max(S_fft2)
            I_fft = S_fft1
            # I_fft = convolve(S_fft, S_fft, mode='same') # wider
            # I_fft = S_fft1**2 * S_fft2**2 #PSF OK, but freq wrong
            # I_fft = real(S_fft1*conj(S_fft2)) # creates envelope
            # I_fft = (S_fft1 + S_fft2)
            # I_fft = real(S_fft1) * real(S_fft2)
            # spline = self.UnivariateSpline(s_rng,S_fft,s=0)
            # print(spline.roots())
            # plot(s_rng/1e3,spline(s_rng),'.-')

            # plot(time_rng*1e12,S_fft,'.-')
            # xlabel('Time (fs)')
            plot(spac_rng*1e6,real(I_fft),'.-')
            xlabel('Space (um)')
            plot(spac_rng*1e6,abs(hilbert(real(I_fft))),'.-')
            print('fwhm_s: {:0.3f} (um)'.format(self.measure_FWHM_h(spac_rng*1e6,I_fft)))
            print('fwhm_s/2: {:0.3f} (um)'.format(self.measure_FWHM_h(spac_rng*1e6,I_fft)/2))
            # xlabel('Range (um)')
            grid(True)
            xlim((40,100))

            subplot(313)
            # Consquently applying the FFT again should result to be about 225 THz again.
            S_ifft = abs(fftshift(fft(real(I_fft))))
            import warnings
            warnings.filterwarnings(action='error',lineno=168) # make warnings try except able
            try:
                # We expect only here a division by zero and accept this minor disonance.
                # But in any other cases we should check for this.
                f_rng = fftfreq(n=N,d=freq_spec)
            except RuntimeWarning as rw:
                pass

            f_rng = linspace(-freq_spec[-1]/2,freq_spec[-1]/2,N)
            plot(f_rng/1e12,S_ifft,'.-')
            print('fwhm_test: {:0.3f} (um)'.format(self.measure_FWHM_h(f_rng/1e12/10,S_ifft)))
            xlabel('Frequency (THz)')
            xlim((-500,500))

            # savefig('tutorial_gen_wave_with_fft_and_interference.pdf')
        calc_wavelength()
        show()


    def spectrum_to_complex(self):
        '''
        Demonstrate the summation of the complex field on the detector (line camera).

        Two spectral fields are generated and by introducing a path-length mismatch showing different frequency
        carriers.

        Furthermore we add gradually the DC build up and the auto correlation terms.

        '''
        c = speed_of_light
        nr = 0.1  # bin ratio -> change in concert with N and
        print('nr: {}'.format(nr))
        wvl_0 = 1330e-9  # m center wavelength
        print('wl_0: {:0.3e} m'.format(wvl_0))
        f_0 = c / wvl_0
        print('f_0: {:0.3e} 1/s'.format(f_0))
        w_0 = 2 * pi * c / wvl_0  # center frequency rad/s
        print('w_0: {:0.3e} rad/s'.format(w_0))

        N = 2 ** 12  # number of sample points
        print('N', N)

        # in our case the fwhm in fs correlates due to the heterodyne mixing to a spatial range.
        # The spatial range is based on the frequency 800 nm.
        fwhm_wl = 60e-9
        print('fwhm_wl: {:0.3e}'.format(fwhm_wl))
        sigma_wl = fwhm_wl / 2 / sqrt( 2 * log(2) )
        print('sigma_wl: {:0.3e}'.format(sigma_wl))

        wvl_spec = linspace(-wvl_0/2, wvl_0/2, N) / nr
        print('wvl_spec: {:0.3e}'.format(wvl_spec[-1]))
        # Because the complex wave must have spatial frequency according to wavelength or one cycle must have wvl_0!
        # We divide by two here to allow symmetry around zero.
        wvl_rng =  wvl_0*N/2
        spac_rng = linspace(-wvl_rng, wvl_rng, N)*nr

        def S1(wl): return exp(-((wl-0e-9)**2/(2*sigma_wl**2)))
        print('fwhm_wl_out: {:0.3e}'.format(self.measure_FWHM(wvl_spec,S1(wvl_spec))))
        def S2(wl): return exp(-((wl-500e-9)**2/(2*sigma_wl**2)))
        print('fwhm_wl_out2: {:0.3e}'.format(self.measure_FWHM(wvl_spec,S2(wvl_spec))))
        def F(wl):
            # actually we assume that S2 is the sample wave.
            # So currently we do not explicitly simulate the generation of the sample wave.
            delta = zeros(len(wl))
            delta[100] = 1
            return delta

        figure(num='tutorial FFT freq to time',tight_layout=True,figsize=(9,8))
        ax = subplot(411)
        # ax.set_yticks(arange(0,1.1,0.1))
        wvl_spec_nm = wvl_spec*1e9
        plot(wvl_spec_nm,S1(wvl_spec),'.',ms=1)
        plot(wvl_spec_nm,S2(wvl_spec),'+',ms=1)
        # plot(wvl_spec_nm,S1(wvl_spec)+S2(wvl_spec),'r',ms=1.5)
        xlabel('Wavelength (nm)')
        grid(True)
        # xlim(((wvl_0-2*fwhm_wl)*1e9,(wvl_0+2*fwhm_wl)*1e9)) # zoom
        xlim((-3000,3000)) # semi zoom

        S1 = S1(wvl_spec)
        S2 = S2(wvl_spec)

        # I1 = S1

        ax = subplot(412)

        S_fft1 = (fft(S1))
        S_fft1 = S_fft1/max(S_fft1)
        S_fft2 = (fft(S2))
        S_fft2 = S_fft2/max(S_fft2)
        # I_fft = real(S_fft1) + roll(imag(S_fft1),1000)*1j
        spac_rng_um = spac_rng*1e6
        # plot(spac_rng_um,real(S_fft1),'-',lw=1)
        # plot(spac_rng_um,real(S_fft2),'-',lw=1)
        # f_fft = fft(F(wvl_spec))
        # We assume that the second Gaussian spectrum is the reflected wave.
        #TODO: However, the combination operation does not produce yet the right results.
        #correlation alone create nevertheless a high frequency content.
        # S_fftS = ((correlate((S_fft1),S_fft2,mode='same'))**2)
        # plot((S_fftS),'.',ms=1,lw=1)



        #TODO: This step is only the cross correlation term of S1 and S2
        #see multi_sample_reflectors()
        #TODO: Summing first the two spectra is what arrives at the detector and evaluate ...
        # S_sum: The two spectra are separate before the grating but we can treat the also as a sum already.
        # f_sum: The grating performs FFT on the sum of the spectra.
        # I_sum: Because the camera can only detect intensity we have to convert the S1 and S2 fields to power.
        S_sum = S1+S2
        f_sum = fftshift(ifft(S_sum))
        I_sum = abs(f_sum)**2
        #TODO: Important! Point out the difference of the resolution of the simulation and the camera pixels!
        # I_sum_spread is the resolution based on the amount of simulated wavelengths!
        # The camera resolution is independent, and we could in theory simulate an infinity of wavelength components.
        I_sum_spread = I_sum[1800:2300]
        cam_rng = linspace(0,1024,len(I_sum_spread))
        plot(cam_rng,I_sum_spread,'.-',ms=1,lw=0.5)
        xlabel('Camera pixel')
        # xlabel('Space (um)')
        # plot(spac_rng_um,abs(hilbert(real(S_fftS))),'-',lw=2)
        # print('fwhm_s: {:0.3f} (um)'.format(self.measure_FWHM_h(spac_rng_um,S_fftS)))
        # print('fwhm_s/2: {:0.3f} (um)'.format(self.measure_FWHM_h(spac_rng_um,I_fft)/2))
        # xlabel('Range (um)')
        grid(True)
        # xlim((-60,60))
        # xlim()

        ax = subplot(413)
        ax.set_yticks(arange(0,1.1,0.1))
        wvl_spec_nm = linspace(-wvl_spec_nm[-1]/2,wvl_spec_nm[-1]/2,N)
        # S_ifft = abs(fftshift(fft(real(S_fftS))))
        # S_ifft = S_ifft/S_ifft.max()
        # plot(wvl_spec_nm,S_ifft,'.-')
        I_ifft = fftshift(ifft(I_sum))
        semilogy(spac_rng_um,abs(I_ifft))
        # print('fwhm_psf: {:0.3f} (um)'.format(self.measure_FWHM_h(spac_rng_um,I_fft)))
        xlabel('Space (um)')
        # xlim(((wvl_0-2*fwhm_wl)*1e9,(wvl_0+2*fwhm_wl)*1e9))
        xlim((-300,300))
        grid(True)

        ax = subplot(414)
        # plot(angle(S_ifft))

        # savefig('tutorial_gen_wave_with_fft_and_interference.pdf')
        show()

    show_list = ['N','nr','wvl_0','fwhm_wl','sigma_wl','f_0','w_0','wvl_spec','fwhm_r','E_ssf']
    def multi_sample_reflectors(self):
        '''
        Demonstrate the summation of the complex field on the detector (line camera).

        Right now we obtain only some summed signal.

        However, simulate now the situation of path-length mismatch.
        That means if d_L = 0 then the carrier should have f = 0.

        P3(2520) the set of equations 1 to 6 are important.
        Eq. 6 especially relating phase to delta z.
        But not much more related to modelling.
        Tomlins, P. H., & Wang, R. K. (2005). Theory, developments and applications of optical coherence tomography. Journal of Physics D: Applied Physics, 38(15), 2519. doi:10.1088/0022-3727/38/15/002

        '''
        c = speed_of_light
        nr = 0.1  # bin ratio -> change in concert with N and
        wvl_0 = 1330e-9  # m center wavelength
        f_0 = c / wvl_0
        w_0 = 2 * pi * c / wvl_0  # center frequency rad/s

        N = 2 ** 14  # number of sample points

        # in our case the fwhm in fs correlates due to the heterodyne mixing to a spatial range.
        # The spatial range is based on the frequency 800 nm.
        fwhm_wl = 60e-9
        sigma_wl = fwhm_wl / 2 / sqrt( 2 * log(2) )

        wvl_spec = linspace(-wvl_0/2, wvl_0/2, N) / nr
        # Because the complex wave must have spatial frequency according to wavelength or one cycle must have wvl_0!
        # We divide by two here to allow symmetry around zero.
        wvl_rng =  wvl_0*N/2
        spac_rng = linspace(-wvl_rng, wvl_rng, N)*nr

        def S_r(wl): return real(0.1*exp(-((wl-0e-9)**2/(2*sigma_wl**2))))
        fwhm_r = self.measure_FWHM(wvl_spec,S_r(wvl_spec))
        # TODO: create multiple reflectors.
        def S_s(wl,d_wl): return real(exp(-((wl-d_wl)**2/(2*sigma_wl**2))))

        d_wls = array([])*1e-9 #,5100,5200,5000,5400])*1e-9
        pws = array([0.2,0.18,0.16,0.14]) #,0.01,0.02,0.08,0.1])
        S_ssn = []
        for d_wl,pw in zip(d_wls,pws):
            S_ssn.append(pw*S_s(wvl_spec,d_wl))

        figure(num='multi reflectors',tight_layout=True,figsize=(9,8))
        rowcols = (3,1)
        ax = subplot(*rowcols,1)
        # ax.set_yticks(arange(0,1.1,0.1))
        wvl_spec_nm = wvl_spec*1e9
        plot(S_r(wvl_spec),'.',ms=1)
        plot(sum(S_ssn,axis=0),'+',ms=1)
        # xlabel('Wavelength (nm)')
        xlabel('Wavelength (pixel)')
        grid(True)
        # xlim((-3000,3000)) # semi zoom

        S_rs = S_r(wvl_spec)
        S_ss = sum(S_ssn,axis=0)

        ax = subplot(*rowcols,2)

        spac_rng_um = spac_rng*1e6

        # We assume that the second Gaussian spectrum is the reflected wave.

        ### Summing first the two spectra is what arrives at the detector and evaluate ...
        # Recall, that the time-domain signal from the light source is not shown here.
        # Although, we can simply perform this by setting the spectrum at 1330 nm and the fft should create the
        # TD-signal of the source.
        ### It is important, however, to recall that the two TD signals do not interfere before hitting the camera!
        #   Consequently, we can use the spectrum --> saying it contains a number of frequencies
        #   and pass those through the grating towards the camera.
        #   The only thing is we sum the spectra here before the FFT but we may confirm that this should work also
        #   the other way around.
        # S_sum: The two spectra are separate before the grating but we can treat the also as a sum already.
        # f_sum: The grating performs FFT on the sum of the spectra.
        # I_sum: Because the camera can only detect intensity we have to convert the S1 and S2 fields to power.
        i_title = '...'
        def sum_by_Spectrum():
            ### It is interesting that the summing before the grating provides the same effects
            i_title = 'Operation: $|\mathcal{F\ }\{S_r + sum(S_s[i])\}|^2$'
            S_sum = S_rs + S_ss # spectra summed before grating
            E_sum = fftshift(ifft(S_sum)) # grating
            return i_title,(abs(E_sum))**2 # intensity
        # i_title,I_sum= sum_by_Spectrum()

        def sum_by_E_field():
            ### It is interesting that the summing before the grating provides the same effects
            i_title = 'Operation: $|\mathcal{F}\{S_r\} + \mathcal{F}\{sum(S_s[i])\}|^2$ = \n $|E_r + \sum E_s|^2$'
            E_r = fftshift(fft(S_rs))
            if S_ss.any():
                E_s = fftshift(fft(S_ss)) # spectra summed before grating
                E_sum = E_r + E_s # grating
            else:
                E_sum = E_r
            return i_title,(abs(E_sum))**2 # intensity
        i_title,I_sum= sum_by_E_field()

        def auto_cross_DC():
            ### Complete intensity including DC and auto correlation artifacts.
            # Grating nonlinearity should not occur but the spectral offset.
            # Perhaps this reduces the auto corr terms.
            i_title = '$|\sum \mathcal{F}\{S_r} + F(S_si))|^2$'
            E_rs = fftshift(ifft(S_rs))
            E_ss = sum([E_rs + fftshift(ifft(s_n)) for s_n in S_ssn],axis=0)
            E_sum = E_ss
            return i_title,(abs(E_sum))**2
        # i_title,I_sum = auto_cross_DC()

        def cross_corr_only():
            ### Not summing the E_rsf is like having removed DC and auto correlation artifacts
            i_title = 'Operation: $\mathfrak{R}\{E_r^*\ \sum E_s[i]\}$\nBackground removed.'
            E_rs = fftshift(ifft(S_rs))
            E_ss = sum([fftshift(ifft(s_n)) for s_n in S_ssn],axis=0)
            return i_title,  real(conj(E_rs)* E_ss) # + abs(E_rsf)**2
        # i_title,I_sum = cross_corr_only()

        def remove_DC_spectrum():
            i_title = '$|\mathfrak{F}\{S_r\} + \sum (\mathfrak{F}(S_si)) - \mathfrak{F}\{S_r\}|^2$'
            E_rs = fftshift(ifft(S_rs))
            E_ss = sum([ fftshift(ifft(s_n)) for s_n in S_ssn],axis=0)
            E_sum =  E_ss + E_rs
            background = (abs(E_sum - E_rs))**2 + abs(E_rs)**2#+ abs(E_ss)**2
            return i_title, abs(E_sum)**2,background
            # return (abs(f_sum - conj(E_rsf)))**2
        # i_title,I_sum,background = remove_DC_spectrum()

        #TODO: Important! Point out the difference of the resolution of the simulation and the camera pixels!
        # I_sum_spread is the resolution based on the amount of simulated wavelengths!
        # The camera resolution is independent, and we could in theory simulate an infinity of wavelength components.
        I_sum_spread = I_sum
        # I_sum_spread = resample(I_sum_spread,1024)
        cam_rng = linspace(0,1024,len(I_sum_spread))

        plot(cam_rng,I_sum_spread,'.-',ms=1,lw=0.5)
        # plot(cam_rng,background,lw=1,color='red')
        title(i_title)
        xlabel('Camera pixel')
        grid(True)
        xlim((500,525))
        # xlim()

        ax = subplot(*rowcols,3)
        # ax.set_yticks(arange(0,1.1,0.1))
        wvl_spec_nm = linspace(-wvl_spec_nm[-1]/2,wvl_spec_nm[-1]/2,N)
        I_ifft = abs(fftshift(ifft((I_sum_spread))))
        # backf = abs(fftshift(ifft(background)))
        plot(I_ifft)
        # plot(backf)
        # semilogy((I_ifft ))
        # semilogy(backf)

        xlabel('Space (pixel)')
        # xlim(((wvl_0-2*fwhm_wl)*1e9,(wvl_0+2*fwhm_wl)*1e9))
        # xlim((-300,300))
        grid(True)

        # ax = subplot(*rowcols,4)
        # ax.text(x=0,y=0,s='Hello')
        remove = ['*','$','|',' ','+','^','\\mathfrak','\\mathcal','\\','{','}','(',')','[',']','\\sum','\n','=','Operation:']
        for r in remove:
            i_title = i_title.replace(r,'')
        print('save as',i_title)
        savefig('simu_FD_{}.pdf'.format(i_title))
        self.show_values(self.show_list)
        show()

    def phase_and_spectrum(self):
        '''
        This is a playground to investigate when we have a real valued vs. a complex valued signal.

        Key points here are:
        * generate by sinusoidal function
        * generate by exponential function:
            take note that this requires a negative range and + pi/2 to match with the hilbert
        * use hilber for conversion
        :return:
        '''
        rng = linspace(0,10*2*pi,1000)
        sig = sin(rng) # real valued signal
        sig = hilbert(sig) # make real to a complex signal.
        fig = figure(num='hilbert')
        fig.canvas.manager.window.move(0,300)

        plot(real(sig),'.',label='real')
        plot(imag(sig),'.',label='imag')
        legend()

        ph = pi/2
        sige = exp( 1j*(rng - ph) ) # complex valued signal
        fig = figure(num='exp {}'.format(ph))
        print(fig.canvas.manager.window.geometry())
        print(fig.canvas.manager.window.x())

        fig.canvas.manager.window.move(700,300)
        plot(real(sige),'.',label='real')
        plot(imag(sige),'.',label='imag')
        legend()

        # sigFFT = fft(sig)
        # figure()
        # plot(real(sigFFT),'.')
        # plot(imag(sigFFT)/2/pi/10,'.')
        # xlim((980,1000))
        show()

    show_list = ['N','nr','wvl_0','spectrum','fwhm_wl','sigma_wl','f_0','w_0','wvl_spec',('fwhm_r','{:1.3e}'),'E_ssf']
    def FD_E_fields_principle(self):
        '''
        Demonstrate the summation of the complex field on the detector (line camera).

        Right now we obtain only some summed signal.

        However, simulate now the situation of path-length mismatch.
        That means if d_L = 0 then the carrier should have f = 0.

        P3(2520) the set of equations 1 to 6 are important.
        Eq. 6 especially relating phase to delta z.
        But not much more related to modelling.
        Tomlins, P. H., & Wang, R. K. (2005). Theory, developments and applications of optical coherence tomography. Journal of Physics D: Applied Physics, 38(15), 2519. doi:10.1088/0022-3727/38/15/002

        '''
        c = speed_of_light
        nr = 0.013  # bin ratio -> smaller -> increase TD samples -> decrease spectral samples
        rng = int(70 * 0.1/nr) # attempt to recalculate the spectrum to camera illumination for each nr

        wvl_0 = 1330e-9  # m center wavelength
        f_0 = c / wvl_0
        w_0 = 2 * pi * c / wvl_0  # center frequency rad/s

        bit = 16
        N = 2 ** bit  # number of sample points

        # in our case the fwhm in fs correlates due to the heterodyne mixing to a spatial range.
        # The spatial range is based on the frequency 800 nm.
        fwhm_wl = 60e-9
        sigma_wl = fwhm_wl / 2 / sqrt( 2 * log(2) )

        # Eq. 18 Carvalho
        n_avg = 1 # avg refractive index
        z_max = 1/4/n_avg * wvl_0**2 / fwhm_wl * N
        # Eq. 2.27 Liu 2008
        z_max = log(2)/2/pi * wvl_0**2 / fwhm_wl * N

        wvl_spec = linspace(0, wvl_0/2, N) / nr
        # wvl_spec = roll(wvl_spec, N//2)
        # Because the complex wave must have spatial frequency according to wavelength or one cycle must have wvl_0!
        # We divide by two here to allow symmetry around zero.
        wvl_rng =  wvl_0*N/2
        spac_rng = linspace(-wvl_rng, wvl_rng, N)*nr

        def S_r(wl): return real(exp(-((wl-wvl_0)**2/(2*sigma_wl**2))))
        spectrum = S_r(wvl_spec)
        fwhm_r = self.measure_FWHM(wvl_spec,spectrum)
        def S_s(wl,d_wl): return real(exp(-((wl-wvl_0-d_wl)**2/(2*sigma_wl**2))))

        # Create multiple sample reflectors
        pos = 1000*1e-9
        # d_wls = array(concatenate((arange(0,2200,400),arange(8000,10200,400))))*1e-9 + pos#,5100,5200,5000,5400])*1e-9
        # d_wls = array([0,50,100,150])*1e-9 + pos #,250,350,450,550])*1e-9 + pos
        n_layers = 2
        d_wls = linspace(0,1000,n_layers)*1e-9 + pos
        # pws = array(ones(len(d_wls))*0.9)
        pws = ones(len(d_wls))*0.7
        S_ssn = [] # array of sample reflector spectra
        for d_wl,pw in zip(d_wls,pws):
            S_ssn.append(pw*S_s(wvl_spec,d_wl))

        S_rs = S_r(wvl_spec) # reference spectrum
        E_r = fftshift(fft(S_rs)) # reference field
        S_ss = sum(S_ssn,axis=0) # sum of sample spectra
        # E_s = fftshift(fft(S_ss)) # sample field
        # I_sum = abs( E_r + E_s ) ** 2

        E_sn = fftshift(fft(S_ssn))

        I_sum = abs(sum([E_r + E_s for E_s in E_sn],axis=0))**2

        figure(num='multi reflectors',tight_layout=True,figsize=(9,8))
        rowcols = (2,1)
        ax = subplot(*rowcols,1)

        wvl_spec_nm = wvl_spec*1e9

        plot(wvl_spec_nm,S_rs,'.-',ms=4)
        plot(wvl_spec_nm,S_ss,'+-',ms=4)
        # xlabel('Wavelength (nm)')
        xlabel('Wavelength (nm)')
        grid(True)
        # xlim((0,10000)) # semi zoom

        ax = subplot(*rowcols,1)
        cla()
        spac_rng_um = spac_rng*1e6
        #TODO: Important! Point out the difference of the resolution of the simulation and the camera pixels!
        # I_sum_spread is the resolution based on the amount of simulated wavelengths!
        # The camera resolution is independent, and we could in theory simulate an infinity of wavelength components.
        cam_pixel = 1024
        # z_max = 1/4/n_avg * wvl_0**2 / fwhm_wl * cam_pixel
        # Eq. 2.27 Liu 2008
        z_max = log(2)/2/pi * wvl_0**2 / fwhm_wl * cam_pixel

        # I_sum_spread = I_sum
        lz = len(I_sum)

        I_sum_spread = resample(I_sum[lz//2-rng:lz//2+rng],cam_pixel)
        #TODO: The maximum depth for FD-OCT is determined by the pixel resolution of the line camera
        # Here again, in relation to the source spectrum the actual illumination width is determined by the grating.
        #
        # cam_rng = slice(len(I_sum_spread)//2-cam_pixel//2,len(I_sum_spread)//2+cam_pixel//2)
        I_sum_spread = np.diff(I_sum_spread,1)
        plot( I_sum_spread ,'.-',ms=2,lw=1)
        # plot(cam_rng,background,lw=1,color='red')
        #TODO: Although the camera pixel are important equally, show also time
        # I am not sure if there is a suitable representation but try to find out.
        title('Intensity: $|E_r + E_s|^2$')
        xlabel('Camera pixel')
        grid(True)
        # xlim((500,525))
        # xlim()

        ax = subplot(*rowcols,2)
        # ax.set_yticks(arange(0,1.1,0.1))
        wvl_spec_nm = linspace(-wvl_spec_nm[-1]/2,wvl_spec_nm[-1]/2,N)
        I_ifft = abs(fftshift(ifft((I_sum_spread))))
        # backf = abs(fftshift(ifft(background)))
        z_rng = linspace(-z_max,z_max,len(I_ifft))
        semilogy(z_rng*1e3,I_ifft,'.-',ms=2,lw=1)

        xlabel('Space (mm)')
        # xlim(((wvl_0-2*fwhm_wl)*1e9,(wvl_0+2*fwhm_wl)*1e9))
        # xlim((-300,300))
        grid(True)

        # ax = subplot(*rowcols,4)
        # ax.text(x=0,y=0,s='Hello')
        remove = ['*','$','|',' ','+','^','\\mathfrak','\\mathcal','\\','{','}','(',')','[',']','\\sum','\n','=','Operation:']

        save_tag = 'nr{1:2.0f}_bit{2}_pos{0:2.0f}'.format(pos*1e9,nr*1e3,bit) #'samples'+time.strftime('%Y%m%d%H%M%S')
        print('save as',save_tag)
        savefig('simu_FD_{}.png'.format(save_tag))
        self.show_values(self.show_list)
        show()

    def FD_calibration_line(self):
        '''
        Demonstrate the creation of a calibration line

        See additional literature in masterThesis_Patricia_Carvalho
        '''
        c = speed_of_light
        nr = 0.1  # bin ratio -> change in concert with N and
        wvl_0 = 1330e-9  # m center wavelength
        f_0 = c / wvl_0
        w_0 = 2 * pi * c / wvl_0  # center frequency rad/s

        N = 2 ** 10  # number of sample points

        # in our case the fwhm in fs correlates due to the heterodyne mixing to a spatial range.
        # The spatial range is based on the frequency 800 nm.
        fwhm_wl = 60e-9
        sigma_wl = fwhm_wl / 2 / sqrt( 2 * log(2) )

        wvl_spec = linspace(0, wvl_0/2, N) / nr
        # wvl_spec = roll(wvl_spec, N//2)
        # Because the complex wave must have spatial frequency according to wavelength or one cycle must have wvl_0!
        # We divide by two here to allow symmetry around zero.
        wvl_rng =  wvl_0*N/2
        spac_rng = linspace(-wvl_rng, wvl_rng, N)*nr

        def S_r(wl): return real(0.1*exp(-((wl-wvl_0)**2/(2*sigma_wl**2))))
        fwhm_r = self.measure_FWHM(wvl_spec,S_r(wvl_spec))
        def S_s(wl,d_wl): return real(exp(-((wl-d_wl)**2/(2*sigma_wl**2))))

        # Create stack of reflectors at wavelength position units
        #TODO: the wavelength position unit relates to the reflected spectrum vs distance
        # The only way to simulate this right now is the mathematical way.

        # Create a sample stack for inital position in wavelength units.
        offset_stack = 3000e-9
        stack = array([0])*1e-9 + offset_stack
        n_alines = 200
        layers = repeat(stack,n_alines).reshape(len(stack),n_alines) # or use numpy.matlib.repmat

        # move layers
        steps = linspace(0,10000,n_alines)*1e-9
        layers = layers + steps

        # The power (actually should be based on the refr-index) is currently assumed one for all layers
        S_rs = S_r(wvl_spec) # reference spectrum
        E_r = fftshift(fft(S_rs)) # reference field
        S_ssn = []
        for aline in layers.T:
            S_ssn.append( sum([S_s(wvl_spec, layer) for layer in aline],axis=0))

        # For the calibration line each sample reflector position is a separate A-line
        alines = []
        for S_sn in S_ssn:
            E_s = fftshift(fft(S_sn))
            I_n = abs( E_r + E_s ) ** 2 # interference
            envelope = abs(fft(I_n))
            alines.append( envelope )

        # img_matrix = 20*log10(array(alines).T)
        img_matrix = array(alines).T
        i_max = img_matrix.max()
        print(shape(img_matrix))
        figure(num='calibration line',tight_layout=True,figsize=(9,8))
        ish = imshow(img_matrix,
                     cmap='CMRmap',
                     # cmap='Accent_r',
                     interpolation='none',
                     aspect='equal',
                     extent=(0,6000,0,6000),
                     vmin=10,vmax=100, # show sensitive tip
                     # vmin=-2.5e2, vmax=130, # show digital noise floor
                     # vmin=0, vmax=i_max/8,
                     # vmin=-i_max//2, vmax=i_max//2, # for linear intensity
                     )
        # ylim((45,55))
        colorbar(mappable=ish,ax=gca(),pad=0.01,shrink=0.9,label='$dB_{20}$')
        save_tag = 'layers_move' + time.strftime('%Y%m%d%H%M%S')
        print('save as',save_tag)
        savefig('simu_FD_calibration_{}.pdf'.format(save_tag))


        figure(num='profile',tight_layout=True)
        plot(linspace(0,6000,len(img_matrix[:,0])),img_matrix[:,0])
        grid(True)

        self.show_values(self.show_list)
        show()

    def show_values(self,show_list):
        '''
        Inspect a function at the point of call and display all parameters.
        This is not supposed to be debugging or logging.
        This is to evaluate the function status and member values.
        We need this in the case to understand how an algorithm is operating.
        We usually would generate more print output or even plotting for such purpose,
        but occassionally log or print statements clutter too much the code itself.
        Decorators are too coarse to achieve sufficient access to the memberes.
        
        Furthermore this function is not completely remote as it can be placed at any point
        :return:
        '''
        import inspect
        # import traceback
        # keys = ['nr','wl_0','f_0','w_0','N','fwhm_wl','sigma_wl','wvl_spec','fwhm_wl_out','fwhm_wl_out2']
        # keys = ['N','nr','wvl_0','fwhm_wl','sigma_wl','f_0','w_0','wvl_spec','fwhm_r','E_ssf']
        # keys = [{'N','{:3e'}]
        keys = show_list
        for key in keys:
            if type(key) is tuple:
                mplier = 1.0
                if len(key) == 4: # has unit, multiplier
                    format_str = '{}: ' + key[1] + ' ' + key[2]
                    mplier = key[3]
                elif len(key) == 3: # has units
                    format_str = '{}: ' + key[1] + ' ' + key[2]
                else:
                    format_str = '{}: ' + key[1]
                var_name = key[0]
                func_frame = sys._getframe(1) #may need to be changed depending on function level
                func_members = inspect.getmembers(func_frame)
                func_locals = dict(func_members)['f_locals']
                value = func_locals.get(var_name)
                if value is not None and var_name is not None:
                    value = value * mplier
                    print(format_str.format(var_name,value))
            else:
                print('{}: {}'.format(key,dict(inspect.getmembers(sys._getframe(1)))['f_locals'].get(key)))

    def generate_carrier_Gauss_conventionally(self):
        '''
        Generation of the Gaussian with carrier by conventional means using center wavelength
        and bandwidth.
        Convert by FFT and reverse to demonstrate we can obtain the Gaussian alone by FFT much faster.
        :return:
        '''
        space = self.space
        FWHM = self.FWHM
        measure_FWHM = self.measure_FWHM
        wavelength = self.wavelength
        k = 2*pi/wavelength
        wavelengthBW = self.wavelengthBW
        tau_p = self.tau_p
        sigma_fwhm = FWHM / sqrt(4 * log(2))
        gauss_theory = exp(-(space+70e-6)**2/(sigma_fwhm)**2)
        carrier_theory = exp(-1j*pi*k*space)

        figure('Gauss+Carrier conventional')
        sig = gauss_theory * carrier_theory
        subplot(411)
        title('original signal')
        plot(sig)
        subplot(412)
        title('imag/real from fft')
        sig_fft = fftshift(fft(real(sig)))
        sig_fft_abs = abs(sig_fft)
        print(sig_fft)
        # plot(sig_fft_abs)
        plot(real(sig_fft),'b')
        plot(imag(sig_fft),'r')
        subplot(413)
        title('inv fft')
        sig_rev = ifft(fftshift(real(sig_fft)))
        plot(sig_rev)
        # plot(unwrap(angle(sig_fft)))
        subplot(414)
        plot(sig_rev-sig)
        title('digital error')
        tight_layout()
        show()

    def generate_carrier_Gauss_by_FFT(self):
        '''
        Construction of signal based on eq. 6
        Hellmuth, T. (1996). Contrast and resolution in optical coherence tomography (Vol. 2926, pp. 228–237). doi:10.1117/12.260800

        :return:
        '''
        space = self.space
        space_smp_fft = linspace(-int(self.N/2),int(self.N/2),(self.N)).astype(complex)
        FWHM = self.FWHM
        measure_FWHM = self.measure_FWHM
        wavelength = self.wavelength
        k = 2*pi/wavelength
        wavelengthBW = self.wavelengthBW
        tau_p = self.tau_p
        sigma_fwhm = FWHM / sqrt(4 * log(2))
        print('sigma_fwhm',sigma_fwhm)
        reco_r = exp(-((space_smp_fft-9900)/5)**2)
        reco_i = exp(-((space_smp_fft+9900)/5)**2)
        reco = reco_i + reco_r
        print(reco)

        figure(num='Gauss+Carrier by FFT')
        n=2
        subplot(n,1,1)
        plot(space_smp_fft,reco)
        subplot(n,1,2)
        plot(fftshift(fft((real(reco)))),'.',lw=1,markersize=1)
        show()


    def sum_stepwise_plot_with_fit_manuscript(self, n=None):
        # manuscript:lst:spectrumsumnormal
        space = linspace(-100e-6, 100e-6, N).astype(complex)  # spatial range in [m]
        tau_p = linspace(0, 200e-6 / v_M, N).astype(complex)  # scan time for 200 um based on mirror speed
        wavelength = 1330e-9  # meter
        wavelengthBW = 60e-9
        FWHM = 2 * log(2) / pi * wavelength ** 2 / wavelengthBW  # [m]
        wavelength = 1330e-9
        gauss_ideal = exp(-(space)**2/(FWHM / sqrt(4 * log(2)))**2)
        number_wl_components = 1000
        # use normal distribution for random frequencies
        wl_N = normal(loc=wavelength, scale= wavelengthBW/sqrt(2*log(2)) , size=number_wl_components)

        for wl,i in zip(wl_N,range(len(wl_N))):
            k_n = 2*pi/wl
            t_0 = 100e-6/self.v_M
            spectral_wave = (exp(-1j*k_n*space))
            spectral_wave_sum += spectral_wave
        # manuscript:lst:spectrumsumnormal


    def sum_stepwise_plot_with_fit(self, n=None):
        # generate all wavelength terms
        space = self.space
        FWHM = self.FWHM
        measure_FWHM = self.measure_FWHM
        wavelength = self.wavelength
        wavelengthBW = self.wavelengthBW
        tau_p = self.tau_p
        gauss_ideal = exp(-(space)**2/(FWHM / sqrt(4 * log(2)))**2)
        print('FWHM measured:',measure_FWHM(space, gauss_ideal)*1e6,'um')
        #normrnd for ML
        if not any(n):
            number_wl_components = 1000
        else:
            number_wl_components = n
        wl_N = normal(loc=wavelength, scale= wavelengthBW/sqrt(2*log(2)) , size=number_wl_components)
        print(shape(wl_N))
        ylim((-1,1))
        grid(True)
        xlabel('range ($\mu m$)')
        ylabel('Amplitude (arb.)')
        spectral_wave_sum = zeros(len(tau_p)).astype(complex)
        space_um = space*1e6
        line_c, = plot(space_um, spectral_wave_sum)
        line_e, = plot(space_um, abs(hilbert(real(spectral_wave_sum))),color='orange',label='sum signal')
        line_g, = plot(space_um, gauss_ideal,'r',label='theory')

        t_0 = time.clock()
        for wl,i in zip(wl_N,range(len(wl_N))):
            k_n = 2*pi/wl
            t_0 = 100e-6/self.v_M
            spectral_wave = (exp(-1j*k_n*space))
            # spectral_wave = cos(2 * pi / wl * space)
            spectral_wave_sum += spectral_wave
            if mod(i,100)==0:
                line_c.set_ydata((spectral_wave_sum)/max(spectral_wave_sum))
                line_e.set_ydata(abs(hilbert(real(spectral_wave_sum/max(spectral_wave_sum)))))
                pause(0.0001)
                # if waitforbuttonpress() == 0: pass

        t_s = time.clock()
        print(t_s-t_0,'s')
        print('FWHM sim:',measure_FWHM(space,abs(hilbert(real(spectral_wave_sum)))),'m')

        legend()
        savefig('sum_stepwise_plot_with_fit_{}.pdf'.format(number_wl_components))
        save('sum_{:d}.npy'.format(number_wl_components),stack((space,spectral_wave_sum)))
        show()

    def compare_sum(self):
        '''
        Generate according files with sum_stepwise_plot_with_fit
        :return:
        '''
        space = self.space
        FWHM = self.FWHM
        measure_FWHM = self.measure_FWHM

        # gauss_ideal = 1 / (FWHM / sqrt(4 * log(2))) / pi * exp(-(space) ** 2 / (FWHM / sqrt(4 * log(2))) ** 2)
        gauss_ideal = exp(-(space) ** 2 / (FWHM / sqrt(4 * log(2))) ** 2)
        # gauss_ideal = gausqqs_ideal/max(gauss_ideal)
        FWHM_gauss_ideal = measure_FWHM(space, (gauss_ideal))
        figure(tight_layout=True)
        for n in [500,1000,5000,10000,15000]:
            print('n={}'.format(n))
            filename = 'sum_{}.npy'.format(n)
            if not os.path.exists(filename):
                self.sum_stepwise_plot_with_fit(n=n)
                sum = load(filename)
            else:
                sum = load(filename)
            space = sum[0]
            space_um = space*1e6
            sig = real(sum[1])
            env = abs(hilbert((sig)))/max(sig)
            subplot(211)
            plot(space_um,env,label='N={}'.format(n))
            legend()

            subplot(212)
            plot(space_um,gauss_ideal-env,label='N={}'.format(n))

            metrics = {'med':median(env)}
            # plot(space,metrics['med']*ones(len(space)))
            metrics['max'] = max(env)
            metrics['SNR'] = metrics['max']/metrics['med']
            metrics['SNR_dB'] = 20*log10(metrics['SNR'])
            metrics['FWHM'] = measure_FWHM(space,env)
            metrics['FWHM_e'] = abs(metrics['FWHM']-FWHM_gauss_ideal)/FWHM_gauss_ideal
            metrics['med_err'] = median(gauss_ideal-env)
            for k in metrics.keys():
                print(k,metrics[k])
        subplot(211)
        plot(space_um,gauss_ideal,label='theoretical')
        xlabel('z ($\mathrm{\mu m}$)')
        subplot(212)
        title('error')

        print('theoretical')
        legend(loc='upper right')
        # savefig('gauss_sum_simu_compare.pdf')
        show()

    def compare_errors(self):
        errors = array(((500,0.030), (1000,0.020), (5000, 0.011), (10000, 0.009), (15000, 0.007) ))
        fwhm_es = array(((500,0.077),(1000,0.030),(5000,0.0047),(10000,0.0067),(15000,0.012)))
        snrs = array(((500,25),(1000,30),(5000,37),(10000,38),(15000,42)))

        def three_plots():
            rcParams['font.family'] = 'Serif'
            rcParams['text.usetex'] = True
            rcParams['text.latex.preamble'] = ['\\usepackage{siunitx}']
            rcParams['font.size'] = 16
            rcParams['lines.linewidth'] = 2.0
            rcParams['lines.markersize'] = 7.0

            figure(figsize=(12,4))
            subplot(131)
            errorbar(errors.T[0]/1e3,errors.T[1],yerr=(0.005),capsize=3,marker=4,linestyle='')
            ylabel('Noise level vs. unity')
            xlabel(r'(a) Frequency components (N$\times 1000$)')
            title('Noise power')

            subplot(132)
            errorbar(fwhm_es.T[0]/1e3,fwhm_es.T[1]/self.N*1e6,yerr=(0.002/self.N*1e6),capsize=3,marker=5,linestyle='')
            ylabel(r'Sampl fraction (\SI{1E-6}{})')
            xlabel(r'(b) Frequency components (N$\times 1000$)')
            title('FWHM error')

            subplot(133)
            errorbar(snrs.T[0]/1e3,snrs.T[1],yerr=(0.00),capsize=3,marker='o',linestyle='')
            ylabel('SNR (dB)')
            xlabel(r'(c) Frequency components (N$\times 1000$)')
            title('SNR')

            subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.2)
            # tight_layout(h_pad=0.5,w_pad=0.5)
        three_plots()

        def twin_ax_plot():
            '''
            This is more difficult to plot all in one
            :return:
            '''
            ax = subplot(111)
            axSNR = ax.twinx()
            axSNR.set_ylabel('SNR (dB)')
            errorbar(errors.T[0],errors.T[1],yerr=(0.005),capsize=3,marker=4,linestyle='',label='noise power')
            # plot(snrs.T[0]),snrs.T[1],'+',label='SNR')
            errorbar(fwhm_es.T[0],fwhm_es.T[1],yerr=(0.002),capsize=3,marker=5,linestyle='',label='FWHM error')
            # plot(fwhm_es.T[0], fwhm_es.T[1],'+',label='FWHM error')
            title('Error vs. number of iterations')
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Error')
            legend()

        savefig('compare_errors.pdf')
        show()

    def run_comparison(self):
        # self.sum_stepwise_plot_with_fit()
        # self.compare_sum()
        self.compare_errors()

    def sum_source_spectrum_coarse(self):
        '''
        Plot only a few wavelengths to visualize the principle for the manuscript.
        :return:
        '''
        space = self.space
        FWHM = self.FWHM
        measure_FWHM = self.measure_FWHM
        wavelength = self.wavelength
        wavelengthBW = self.wavelengthBW
        tau_p = self.tau_p
        # generate all wavelength terms
        gauss_ideal = exp(-(space)**2/(FWHM / sqrt(4 * log(2)))**2)
        print('FWHM measured:',measure_FWHM(space, gauss_ideal)*1e6,'um')
        #normrnd for ML
        number_wl_components = 100
        wl_N = normal(loc=wavelength, scale= wavelengthBW/sqrt(2*log(2)) , size=number_wl_components)
        # wl_N = arange(wavelength,wavelength+10e-6,100e-9 )
        print(shape(wl_N))
        ylim((-1,1))
        grid(True)
        xlabel('range ($\mu m$)')
        ylabel('Amplitude (arb.)')
        spectral_wave_sum = zeros(len(tau_p)).astype(complex)
        space_um = space*1e6
        # line_c, = plot(space_um, spectral_wave_sum)
        # line_e, = plot(space_um, abs(hilbert(real(spectral_wave_sum))))
        # line_g, = plot(space_um, gauss_ideal,'r')
        wl=0
        t_0 = time.clock()
        for wl,i in zip(wl_N,range(len(wl_N))):
            # spectral_wave = (exp(-1j*wl*tau_p))
            spectral_wave = cos(2 * pi / wl * space)
            spectral_wave_sum += spectral_wave
            if mod(i,10)==0:
                plot(space_um, spectral_wave,lw=1.0)
                xlim((-10, 10))
                # line_c.set_ydata((spectral_wave_sum)/max(spectral_wave_sum))
                # line_e.set_ydata(abs(hilbert(real(spectral_wave_sum/max(spectral_wave_sum)))))
                # pause(0.0001)
                if waitforbuttonpress() == 0: pass

        plot(space_um, (spectral_wave_sum)/max(spectral_wave_sum),'k',lw=3)
        xlim((-10, 10))
        # savefig('sum_source_spectrum_coarse_v1.pdf')
        t_s = time.clock()
        print(t_s-t_0,'s')
        print('FWHM sim:',measure_FWHM(space,abs(hilbert(real(spectral_wave_sum)))),'m')


    def sum_stepwise_plot(self):
        # generate all wavelength terms
        number_wl_components = 10000 #self.number_wl_components
        space = self.space
        FWHM = self.FWHM
        measure_FWHM = self.measure_FWHM
        wavelength = self.wavelength
        wavelengthBW = self.wavelengthBW
        tau_p = self.tau_p

        wl_N = normal(loc=wavelength, scale=1 * wavelengthBW, size=number_wl_components)
        ylim((-1,1))
        grid(True)
        xlabel('range ($\mu m$)')
        ylabel('Amplitude (arb.)')
        spectral_wave_sum = zeros(len(tau_p)).astype(complex)
        space_um = space*1e6
        line, = plot(space_um, spectral_wave_sum)

        print(shape(line))
        wl=0
        t_0 = time.clock()
        for wl,i in zip(wl_N,range(len(wl_N))):
            # spectral_wave = (exp(-1j*wl*tau_p))
            spectral_wave = cos(2 * pi / wl * space)
            spectral_wave_sum += spectral_wave
            if mod(i,100)==0:
                line.set_ydata((spectral_wave_sum)/max(spectral_wave_sum))
                pause(0.0001)
            if waitforbuttonpress() == 0: pass
        t_s = time.clock()
        print(t_s-t_0,'s')

    def simple_wave_summation_rec_distribution(self):
        # This small experiment demonstrates if we have emission of all frequencies with equal probability
        # The spectrum is a sinc function due to the rectangular probability distribution.
        t = linspace(-2,2,20000)
        f_n = arange(10,11,0.01)
        # print(f_n)
        wave_all = zeros(len(t))
        for f in f_n:
            w = 2*pi*f
            wave = sin(w*t-pi/2)
            wave_all += wave
            cla()
            plot(wave_all/max(wave_all))
            if waitforbuttonpress() == 0: pass

    def simple_wave_summation_gauss_distribution(self):
        # This small experiment demonstrates if we have emission of all frequencies with equal probability
        # The spectrum is a sinc function due to the rectangular probability distribution.
        N = 20000
        t = linspace(-3,3,N)
        f_n = normal(loc=10,scale=0.25,size=500)
        # print(f_n)
        wave_all = zeros(len(t))
        figure()
        line, = plot(wave_all)
        ylim((-1,1))
        for f,i in zip(f_n,range(len(f_n))):
            w = 2*pi*f
            wave = sin(w*t-pi/2)
            wave_all += wave *tukey(N)
            # cla()
            # plot(wave_all/max(wave_all))
            line.set_ydata(wave_all/max(wave_all))

            pause(0.0001)
            # if waitforbuttonpress() == 0: pass
            savefig('sum_simu_{:03d}.jpg'.format(i))

    def run(self):
        # self.compare_sum()
        # print('done')
        # show()
        print('run1 non active')

    def run2(self):
        # self.sum_source_spectrum_coarse()
        # self.sum_stepwise_plot()
        # self.sum_stepwise_plot_with_fit()
        # self.simple_wave_summation_rec_distribution()
        self.simple_wave_summation_gauss_distribution()
        print('run2 done')
        show()
        # print('none active')

class Tomlins(GaussGenerator):
    def masterThesis_Patricia_Carvalho(self):
        '''
        original approach

        Rawat, C. S., & Gaikwad, V. S. (2014). Signal analysis and image simulation for optical coherence tomography (OCT) systems. In 2014 International Conference on Control, Instrumentation, Communication and Computational Technologies (ICCICCT) (pp. 626–631). doi:10.1109/ICCICCT.2014.6993037

        A plain application of equations ... may be good for comparison
        Patrícia Miguel da Silva Carvalho. (n.d.). OPTICAL COHERENCE TOMOGRAPHY Layout Simulation using MATLAB. Retrieved from https://estudogeral.sib.uc.pt/bitstream/10316/31107/1/MasterThesis_PatriciaCarvalho.pdf

        Demonstration of some unique TD delayline using static mirror!
        Jansz, P., Wild, G., Richardson, S., & Hinckley, S. (2011). Simulation of optical delay lines for Optical Coherence Tomography. In Quantum Electronics Conference Lasers and Electro-Optics (CLEO/IQEC/PACIFIC RIM), 2011 (pp. 1400–1402). doi:10.1109/IQEC-CLEO.2011.6194128

        Literature collection discussing applications of TDFD methods in wide area including biophotonics,
        and mentions MEEP from MIT.
        Taflove, A., Oskooi, A., & Johnson, S. G. (2013). Advances in FDTD Computational Electrodynamics: Photonics and Nanotechnology. Artech House.

        See this presentation with some examples showing the fringe pattern for FD OCT without Gaussian offset.
        Tromberg, D. B. (n.d.). What is Biomedical Optics?, 101.

        Theory complementary. But not very detailed. More application.
        Splinter, R., & Hooper, B. A. (2006). An Introduction to Biomedical Optics. CRC Press.

        :return:
        '''
        SN = 1024 * 1
        c = speed_of_light
        cwl = 800e-9
        fwhm = 50e-9
        sigma = fwhm / sqrt(8*log(2))
        w_0 = 2*pi*c / cwl
        w = linspace(w_0-(w_0/3),w_0+(w_0/3), SN)
        fwhm_w = 2*pi*c*fwhm / cwl**2
        sigma_w = fwhm_w / sqrt(8*log(2))

        self.show_values(['SN','c','cwl','fwhm','sigma','w_0','w','fwhm_w','sigma_w'])

        y = exp(-(w_0-w)**2/(2*sigma_w**2))

        self.show_values(['y'])

        do_source = True
        if do_source:
            figure(num='Source')
            plot(w,y)
            title('Light Source Spectrum')
            xlabel('Optical Frequency (rad/s)')
            ylabel('Amplitute (a.u.)')

        #sample properties

        n = [1.00, 1.30, 1.50, 1.30, 1.00]
        z = array([5.00, 15.00, 30.00, 100.0, 1.00])*1e-6

        s1 = 0
        h = 0

        for i in range(0,3): # 0,1,2
            rj = ( n[i+1] - n[i] ) / (n[i+1] + n[i])
            s1 = s1 + n[i] * z[i]
            h = h + rj * exp( 1j * 2 * (w / c ) * s1)


        # time-domain
        def time_domain():
            x = linspace(0, 100e-6, 1024)
            T1 = zeros(len(x))

            for j in range(len(x)):
                for jj in range(len(w)):
                    ph = cos( 2 * x[j] * w[jj] / c)
                    T1[j] = T1[j] + real( 0.5 * ( y[jj] * h[jj] * ph))

            figure()
            plot(x/1e-6, T1, lw=1)
            title('TD-OCT interferogram')
            xlabel('Mirror displacement ($\mathrm{\mu m}$)')
            ylabel('Amplitute (a.u.)')
        # time_domain()

        # fourier domain
        I = zeros(len(w))
        for j in range(len(w)):
          # I(j) = ( 0.25 * y(j).* (abs( h(j) ).^2 )) + (0.25 * y(j)) + (0.5 * real( y(j)*h(j)));
            I[j] = ( 0.25 * y[j] * (abs( h[j] )**2 )) + (0.25 * y[j]) + (0.5 * real( y[j]*h[j]))

        N = len(I)
        I2 = abs(fftshift(ifft(I)))
        I2 = I2[0:N//2][::-1]
        # wrange = w[len(w)] - w[0]
        wrange = w[-1] - w[0]
        zz = linspace(0, N*pi*c/2/wrange, N//2)

        figure()
        plot(w,I)
        title('FD-OCT: Spectral intensity')
        xlabel('Optical frequency (rad/s')
        ylabel('Amplitude (a.u.)')
        xlim((2.1*1e15,2.6*1e15))

        figure()
        plot(zz/1e-6, I2)
        title('FD-OCT: A-scan')
        xlabel('Depth (um)')
        ylabel('Amplitude (a.u.)')
        xlim((0,100))

        tight_layout()
        show()

    def masterThesis_Patricia_Carvalho_analysis(self):
        '''
        analyse signals

        Conclusion: The simulation is inaccurate due to the application of the FFT of the delta function
        directly, which causes basically two sinusoidals to be multiplied.
        Also, two sinusoidals overlap slightly which is to see in the simulation results.

        Key points so far:
        1) The spectrum should be convoluted with the delta.
           But the FFT of the delta is a waveform.
           !!!! However, the delta position (z-pos of the layer) is a different frequency!!!!
           This frequency change is the encoded depth position!

        * Generation of spectrum OK. Plot as digital components.
        '''
        SN = 2**10
        c = speed_of_light
        cwl = 800e-9
        fwhm = 50e-9 # 50 nm
        sigma = fwhm / sqrt(8*log(2))
        w_0 = 2 * pi * c / cwl
        # frequency range
        w_rng = linspace( w_0 - (w_0/3), w_0 + (w_0/3), SN)
        fwhm_w = 2 * pi * c * fwhm / cwl**2
        sigma_w = fwhm_w / sqrt(8*log(2))


        def S(w):
            '''Generation of all frequencies with Gaussian distribution according
            to sigma_w and center frequency w_0.'''
            return exp(-(w_0-w)**2/(2*sigma_w**2))

        S = S(w_rng)
        self.show_values(['S'])

        do_source = False
        if do_source:
            figure(num='Source')
            plot(w_rng,S,'.')
            title('Light Source Spectrum')
            xlabel('Optical Frequency (rad/s)')
            ylabel('Amplitute (a.u.)')

        #sample properties

        # nn = [1.00, 1.30, 1.50, 1.30, 1.00]
        # zz = array([5.00, 15.00, 30.00, 100.0, 1.00])*1e-6

        # Note, the relative thickness of the layers determine the path delay
        # which changes the all frequency components of the spectrum

        nn = array([1.00,1.30,1.50])
        rj = [(n2 - n1)/(n2 + n1) for n2,n1 in zip(nn[1:],nn[0:-1])]
        print(rj)
        zz = array([50.00,30.00,50.00])*1e-6

        def sample_response_h(run=False):
            s1 = 0
            h = 0
            if run:
                for i in range(0,len(nn)-1): # 0,1,2
                    rj = ( nn[i+1] - nn[i] ) / (nn[i+1] + nn[i])
                    s1 = s1 + nn[i] * zz[i]
                    h = h + rj * exp( 1j * 2 * (w_rng / c ) * s1)
                figure(num='sample response H')
                plot(h,'.-',lw=0.5)

        sample_response_h()

        def sample_response_h_2(run=True):
            '''
            According to the array nn if the first layer has n = 1.0 then along the thickness
            zz[0] = z this would relate to the beam travelling in air.

            Note here that h is the FFT of the delta function consequently generating a spatial frequency
            and not so much a single position.
            This is required to
            :param run:
            :return:
            '''
            # This array should be the same as for the original function for the 1st layer.
            win_fun = tukey(M=len(w_rng))
            return exp(1j * 2 * w_rng / c * nn[0] * zz[0]) * win_fun
        my_h = sample_response_h_2()

        figure()
        n_air = 1.0
        phi_z = 2*w_rng/c*n_air*80e-6
        # plot(fftshift(ifft(S)),'.-',lw=0.5)
        plot(fftshift(ifft(my_h)) * S,'.-',lw=0.5)
        # plot(convolve(abs(ifft(my_h)),S,mode='same')*cos(phi_z),'.-',lw=0.5)

        # The sample response function is calculate as




        # time-domain
        time_domain = False
        if time_domain:
            x = linspace(0, 100e-6, 1024)
            T1 = zeros(len(x))

            for j in range(len(x)):
                for jj in range(len(w_rng)):
                    ph = cos( 2 * x[j] * w_rng[jj] / c)
                    T1[j] = T1[j] + real( 0.5 * ( S[jj] * h[jj] * ph))

            figure()
            plot(x/1e-6, T1, lw=1)
            title('TD-OCT interferogram')
            xlabel('Mirror displacement ($\mathrm{\mu m}$)')
            ylabel('Amplitute (a.u.)')

        # fourier domain
        fourier_domain = False
        if fourier_domain:
            I = zeros(len(w_rng))
            for j in range(len(w_rng)):
              # I(j) = ( 0.25 * y(j).* (abs( h(j) ).^2 )) + (0.25 * y(j)) + (0.5 * real( y(j)*h(j)));
                I[j] = ( 0.25 * S[j] * (abs( h[j] )**2 )) + (0.25 * S[j]) + (0.5 * real( S[j]*h[j]))

            N = len(I)
            I2 = abs(fftshift(ifft(I)))
            I2 = I2[0:N//2][::-1]
            # wrange = w[len(w)] - w[0]
            wrange = w_rng[-1] - w_rng[0]
            z_rng = linspace(0, N*pi*c/2/wrange, N//2)

            figure()
            plot(w_rng,I)
            title('FD-OCT: Spectral intensity')
            xlabel('Optical frequency (rad/s')
            ylabel('Amplitude (a.u.)')
            xlim((2.1*1e15,2.6*1e15))

            figure()
            plot(z_rng/1e-6, I2)
            title('FD-OCT: A-scan')
            xlabel('Depth (um)')
            ylabel('Amplitude (a.u.)')
            xlim((0,100))


        tight_layout()
        show()

    def tomlins_simulation(self):
        '''
        This simulations is a step-by step generation of each signal.
        Possibly I will provide multiple functions named tomlins_...
        to accomodate special separate aspects of the simulation.
        This function may accomodate perhaps the complete simulation.
        :return:
        '''
        pass

    def tomlins_light_source_spectrum(self, do_plot=False, cwl_shift=0):
        '''
        Plot and compare different representation of the source spectrum.

        Note, although, we could directly convert the spectrum to frequency and Gaussian envelope
        it would be somewhat diverging to calculate the PSF properly.
        Because, the source alone is not yet the superposition.
        Strictly speaking we would need to calculate the auto correlation of the source,
        however, the question remains about the accurate spatial range without having any reference
        to a path-length mismatch.


        :return:
        '''
        #todo -3 sample ratio factor determinse the relative depth range z and PSF
        # The relative sample ration also determines the depth range for a raw light source spectrum.
        # If no OPL is available then the depth and the PSF directly depends on this sample ratio.
        # At this point this appears to be more like an arbitrary selection of some value.
        spectral_width_factor = 100 # this is equal to our ratio factor
        swf = spectral_width_factor
        SN = 2**10
        c = speed_of_light
        cwl = 800e-9 + cwl_shift
        fwhm = 50e-9 # 50 nm
        sigma = fwhm / sqrt(8*log(2))
        wl_rng = linspace( cwl - sigma*swf, cwl + sigma*swf, SN) # wavelength range to use for calculation

        def S(w,w_0,s):
            return exp( -(w - w_0)**2 / (2*s**2))

        if cwl_shift == 0:
            w_0 = 2 * pi * c / cwl
            fwhm_w = 2 * pi * c * fwhm / cwl**2
            sigma_w = fwhm_w / sqrt(8*log(2))
            w_rng = linspace( w_0 - sigma_w*swf, w_0 + sigma_w*swf, SN) # frequency range to use for calculation
            source_spectrum_w = S(w_rng, w_0, sigma_w)
        else:
            print('CWL shifted. Disabled calculation of frequency.')
            w_rng = None
            source_spectrum_w = None


        source_spectrum_wl = S(wl_rng, cwl, sigma)

        if do_plot:
            if cwl_shift == 0:
                figure(num='source spectrum (w)',tight_layout=True)
                plot(w_rng, source_spectrum_w,'.')
                mimx = (min(source_spectrum_w),max(source_spectrum_w))
                plot([w_0]*2, mimx)
                xlabel('Frequency (rad/s)')
                title('Source spectrum in circular frequency.')

            figure(num='source spectrum (WL)',tight_layout=True)
            plot(wl_rng*1e9, source_spectrum_wl,'.')
            spline = UnivariateSpline(wl_rng*1e9, source_spectrum_wl, s=0)
            plot(wl_rng*1e9, spline(wl_rng*1e9))
            mimx = (min(source_spectrum_wl),max(source_spectrum_wl))
            plot([cwl*1e9]*2, mimx)
            gca().set_yticks(linspace(0,1,11))
            grid(True)
            xlabel('Wavelength (nm)')
            title('Source spectrum in wavelength.')
            print('FWHM (measured): ', self.measure_FWHM(wl_rng*1e9, source_spectrum_wl),' nm')

        self.show_values([('SN','{:g}'),
                          ('c','{}','m/s',1),
                          ('cwl','{:3.3f}','nm',1e9),
                          ('fwhm','{:3.3f}','nm',1e9),
                          ('sigma','{:3.3f}','nm',1e9),
                          ('w_0','{:3.3f}','rad/s'),
                          ('w_rng','{}'),
                          ('fwhm_w','{}','rad/s'),'sigma_w'])
        # show()
        return (cwl,fwhm,w_rng,wl_rng,source_spectrum_w,source_spectrum_wl)

    def problem_FFT_samples(self):
        '''
        "The FFT of a Gaussian spectrum is a Gaussian again."
        See Green's function.

        :return:
        '''
        pass

    def tomlins_source_freq_Gauss_envelope(self):
        '''
        Plot and analyse the FFT of the spectrum.
        We can the calculate the PSF.

        The tomlins_light_source spectrum is shifted by the CWL!
        Consequently the FFT will generate a frequency related to the CWL which has however here
        no meaning.

        So we need to move the center wavelength to zero or ignore it!

        :return:
        '''
        c = speed_of_light
        (cwl,bw,w_rng,wl_rng,spectrum_w,spectrum_wl) = self.tomlins_light_source_spectrum(do_plot=True, cwl_shift=0)

        fwhm = 2*log(2)/pi * cwl **2 / bw
        print('FWHM: ', fwhm, 'm')
        # frng =
        print('wl_max: ',wl_rng[0]*1e9, 'nm')
        print('wl_min: ',wl_rng[-1]*1e9, 'nm')
        # frng = 1/abs(wl_rng[0]-wl_rng[-1])
        # print('frng: ',frng*1e6, 'um')
        uuu = 0.1
        z_rng = linspace(-uuu*len(wl_rng)//2,uuu*len(wl_rng)//2,len(wl_rng))

        #todo z_rng directly depends on the

        # If
        # FWHM = 2*log(2)/pi * CWL**2 / BW
        # then
        # FRNG = 2*log(2)/pi * CWL**2 / CWRNG

        # The k_rng can not be used as it describes the frequency content.
        # Only the CWL is the determining factor vs. the speed of light.

        # Then for a sufficient scanning length the PSF could be reproduced.
        # In the case of the raw light source the detector would neet to scan very fast.

        # The only way to determine the scanning time based on the samples used is to relate the samples
        # of the spectrum to some time.

        # Assuming that the dt is 1/dt from the spectrum one could further assume that the changed of frequency
        # component df could return the sample rate in S/s.



        source_fft = fftshift(fft(spectrum_wl))
        source_fft = source_fft/source_fft.max()
        figure()
        #todo showning the frequency of light demonstrates that the practical resolution or detector speed
        # is insufficient to capture it.
        plot(fftshift(fft(roll(spectrum_wl,-500))),'.-',lw=0.5)
        # plot(imag(source_fft),'.-',lw=0.5)
        # plot(z_rng,abs(hilbert(real(source_fft))),'-',lw=0.5)
        # spline = UnivariateSpline(z_rng, abs(hilbert(real(source_fft))), s=0)
        # plot(z_rng, spline(z_rng))

        # plot(z_rng,imag(source_fft))
        # plot(z_rng,abs(hilbert(real(source_fft))))
        grid(True)
        xlabel('Frequency k (1/m) or Depth z (m)')
        print('PSF: ',self.measure_FWHM_h(z_rng,source_fft),' um')

        source_abssq = abs(source_fft)**2
        source_abssq = source_abssq/source_abssq.max()
        # figure()
        # plot(k_rng,source_abssq)
        # plot(z_rng,source_abssq)
        # xlabel('Frequency k (1/m)')
        show()


class TomlinsSimulation_v0(Tomlins):
    SN = 2 ** 12
    c = speed_of_light
    CWL = 1330e-9
    BW = 60e-9  # 50 nm
    FWHM_psf = 2*log(2)/pi * CWL**2 / BW
    sigma = BW / sqrt(8 * log(2))
    plot_sigma_width = 5
    WL_rng = linspace(CWL - sigma*plot_sigma_width , CWL + sigma*plot_sigma_width , SN)  # wavelength range to use for

    f_0 = c / CWL

    w_0 = 2 * pi * c / CWL
    FWHM_w = 2 * pi * c * BW / CWL ** 2
    sigma_w = FWHM_w / sqrt(8 * log(2))
    # w_rng = linspace(w_0 - sigma_w*5, w_0 + sigma_w*5, SN)
    w_rng_max = w_0 + w_0*1.0
    w_rng = linspace(0, w_rng_max, SN)

    def __init__(self):
        '''Show values if object is initialized'''
        print('FWHM_z', self.FWHM_psf)
        print('f_0', self.f_0 * 1e-12, 'THz')
        print('FWHM_w', self.FWHM_w, 'rad/s', self.FWHM_w / 2 / pi * 1e-12)

    def source_FD(self,):
        '''
        Source in the frequency domain.

        Liu 2008, p25
        :return:
        '''
        CWL = self.CWL
        BW = self.BW
        c = self.c
        SN = self.SN
        w_rng = self.w_rng
        w_0 = self.w_0
        sigma_w = self.sigma_w

        def plot_power_spectrum():
            def Sw( w, w0, s_w ):
                return sqrt(2*pi/s_w**2) * exp(-(w-w0)**2 / (2*s_w**2))

            S_w_w0 = Sw(w_rng, w_0, sigma_w)
            figure(num='frequency')
            plot(w_rng/2/pi*1e-12, S_w_w0,'.-',lw=0.5,label='$\lambda_0$={:1.0f} nm, $\Delta\lambda$ = {:1.0f} nm'.format(CWL*1e9,BW*1e9))
            xlim(array([w_0-sigma_w*5,w_0+sigma_w*5])/2/pi*1e-12)
            grid(True)
            xlabel('Frequency (THz)')
            ylabel('Power vs. frequency (a.u.)')
            title('Plotted with {} sigma frequency width'.format(self.plot_sigma_width))
            legend(loc='upper right')
            # savefig('source_{:1.0f}nm_freq.pdf'.format(CWL))
        plot_power_spectrum()

        # The double pass is related to the relative change of the sample arm to the distance change of a reflector.
        # For the frequency domain this means the relative change from the zero-path length differenc l_R == l_S
        # to a reflector distance l_S - l_r.
        # The relative distance difference causes a the light to travel a distance 2 (l_S + l_r) and the effect of
        # change of frequency is therefore 2 x l_r.
        # In effect a single frequency component increases in frequency twice or the FWHM_psf is twice the source.
        # In other words if the sample layer is scanning over a distance Z, the light would see a scanning distance
        # 2 x Z and the PSF would occur as FWHM/2, and a correction must be applied.
        def plot_TD_from_power_spectrum():
            f_max = 2*w_rng[-1]/2/pi # note here that we use 2 x the w_rng due to the double pass.
            print('f_max',f_max,'Hz')
            ST = 1/f_max*self.SN #s
            print('ST',ST,'s')
            z = ST * c # s * m/s == m
            zC = z/2*1e6
            print('zC',zC)
            print('z',z,'m')
            z_rng = linspace(0,z,self.SN)*1e6
            figure(num='fft of freq.')
            plot(z_rng,fftshift(fft(S_w_w0)),'.-',lw=0.5)

            def envelope_hilbert():
                spline=UnivariateSpline(z_rng,abs(hilbert(real(fftshift(fft(S_w_w0))))),s=0)
                plot(z_rng,spline(z_rng))
            envelope_hilbert()

            def envelope_on_absolute_values():
                spline=UnivariateSpline(z_rng,abs(fftshift(fft(S_w_w0))),s=0)
                plot(z_rng,spline(z_rng))
                print(self.measure_FWHM(z_rng,abs(fftshift(fft(S_w_w0)))))
            envelope_on_absolute_values()

            xlabel('z (um)')
            xlim(array([zC-20,zC+20]))

        def plot_S_vs_WL():
            def Sw( w, w0, s_w ):
                return sqrt(2*pi/s_w**2) * exp(-(w-w0)**2 / (2*s_w**2))

            S_w_w0 = Sw(w_rng, w_0, sigma_w)
            figure(num='frequency',tight_layout=True)
            plot(w_rng/2/pi*1e-12, S_w_w0,'.-',lw=0.5,label='$\lambda_0$=800 nm, $\Delta\lambda$ = 50 nm')
            # stem(w_rng/2/pi*1e-12, real(S_w_w0),basefmt=' ',linefmt='-.b', label='$\lambda_0$=800 nm, $\Delta\lambda$ = 50 nm')
            # fit = interpolate.splrep(w_rng/2/pi*1e-12, S_w_w0,s=0)
            # plot(w_rng/2/pi*1e-12,interpolate.splev(w_rng/2/pi*1e-12,fit,der=0))
            # spline = UnivariateSpline(w_rng/2/pi*1e-12, S_w_w0,s=0)
            # plot(w_rng/2/pi*1e-12, spline(w_rng/2/pi*1e-12))
            grid(True)
            xlabel('Frequency (THz)')
            ylabel('Power vs. frequency (a.u.)')
            title('Plotted with 5 sigma frequency width')
            legend()
            # savefig('source_800nm_freq.pdf')
            return S_w_w0
        # S_w_w0 = plot_FD_in_freq()

        def plot_TD_from_FD():
            # The highest frequency bin determines the sample rate.
            # With the sample rate the sample time is then SR * SN.
            print('f_max',w_rng[-1]/2/pi*1e-12,'THz')
            print('ST: {:1.3e} s, {:1.3e} fs'.format(1/(w_rng[-1]/2/pi)*SN, 1/(w_rng[-1]/2/pi)*SN*1e12))
            print('dz:',(1/(w_rng[-1]/2/pi)*SN)*c*1e6,'um') # c * t == [m/s] * [s] == [m]
            # t = fftshift(fftfreq(n=SN,d=w_rng[-1]))
            # z = 1 / (w_rng[-1] / 2 / pi) * SN * c
            # zr = linspace(0,z,SN)
            figure(num='fft of freq.')
            plot(fftshift(fft(S_w_w0)),'.-',lw=0.5)
            # # plot(t*c*1e9,abs(hilbert(real(fftshift(fft(S_w_w0))))),'-',lw=1.5)
            # spline = UnivariateSpline(zr*1e3,abs(hilbert(real(fftshift(fft(S_w_w0))))),s=0)
            # plot(zr*1e3,spline(zr*1e3))
            # print('FWHM_meas:',self.measure_FWHM_h(zr*1e3,real(fftshift(fft(S_w_w0)))),'um')
            # grid(True)
            # xlabel('Distance (um)')
            # savefig('source_800nm_freq.pdf')
        # plot_TD_from_FD()

        def plot_vs_WL():
            WLR = self.WL_rng
            CWL = self.CWL
            s = self.sigma

            def SWL( WLR, CWL, s ):
                return sqrt(2*pi/s**2) * exp(-(WLR-CWL)**2 / (2*s**2))

            S_WLR_CWL = SWL( WLR, CWL, s)
            figure(num='wavelength')
            plot(WLR*1e9, S_WLR_CWL,'.-',lw=0.5,label='$\lambda_0$={:1.0f} nm, $\Delta\lambda$ = {:1.0f} nm'.format(CWL*1e9,BW*1e9))
            grid(True)
            xlabel('Wavelength (nm)')
            ylabel('Power vs. wavelength (a.u.)')
            title('Plotted with {} sigma wavelength width'.format(self.plot_sigma_width))
            legend(loc='upper right')
            # savefig('source_{:1.0f}nm_WL.pdf'.format(CWL*1e9))
        plot_vs_WL()

        show()


    def source_TD(self):
        pass

class TomlinsSimulation_v1(Tomlins):
    '''
    This contains mainly all aspects of the error of using the wrong sample range and amount of samples.
    '''
    SN = 2 ** 16
    c = speed_of_light
    CWL = 1330e-9
    BW = 60e-9  # 50 nm
    FWHM_psf = 2*log(2)/pi * CWL**2 / BW
    sigma = BW / sqrt(8 * log(2))
    plot_sigma_width = 5.0
    range_factor = 1200

    WL_rng_max = CWL + CWL*range_factor
    WL_rng = linspace(0, WL_rng_max, SN)  # wavelength range to use for

    f_0 = c / CWL

    w_0 = 2 * pi * c / CWL
    FWHM_w = 2 * pi * c * BW / CWL ** 2
    sigma_w = FWHM_w / sqrt(8 * log(2))
    # w_rng = linspace(w_0 - sigma_w*5, w_0 + sigma_w*5, SN)
    w_rng_max = w_0 + w_0*range_factor
    w_rng = linspace(0, w_rng_max, SN)

    def __init__(self):
        '''Print values only if object is used and initialized.'''
        print('FWHM_z {:1.3f} um'.format(self.FWHM_psf * 1e6))
        print('f_0 {} THz'.format(self.f_0 * 1e-12))
        print('FWHM_w {:1.3e} rad/s ({:1.0f} THz)'.format(self.FWHM_w, self.FWHM_w / 2 / pi * 1e-12))

        rcParams['font.size'] = 16

        self.plot_freq_wl_rng_vs_error()


    def plot_freq_wl_rng_vs_error(self):
        '''
        Plot the deviation (error) that occurs depending on the selected frequency range
        around the generated source spectrum.
        :return:
        '''
        f_wl_vs_rng = dict({})
        # measured Hilbert deviation vs max freq/wl range for SN = 2 ** 10
        f_wl_vs_rng[1024]         ={'rng':[2.0, 2.02, 2.04, 2.06, 2.08, 2.1, 2.2],#, 5.0, 10],
                                 'err':[78,  54,   21,   6,    1.2,  0.2, 0.2]}#, 0.2, 0.2]
        # measured Hilbert deviation vs max freq/wl range for SN = 2 ** 11
        f_wl_vs_rng[2048]         ={'rng':array([1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 5.0, 25, 50]) + 1.0,
                                'err':array([78,   56,   23,   6.6,  1.2,  0.1,  0.1, 0.1,0.2])}
        # 2 ** 12
        f_wl_vs_rng[4096]         ={'rng':array([1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 5.0, 25,  50,  75,  80]) + 1.0,
                                 'err':array([78,   57,   24,   7.0,  1.2,  0.1,  0.05,0.05,0.05,0.05,0.05])}
        # 2 ** 14
        f_wl_vs_rng[16384]         ={'rng':array([1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 5.0, 25,  50,  75,  100, 200]) + 1.0,
                                  'err':array([78,   57,   24,   7.0,  1.2,  0.04, 0.01,0.01,0.01,0.01,0.01,0.01])}
        # 2 ** 16
        f_wl_vs_rng[65536]         ={'rng':array([1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 5.0,  25,   50,   100,  200, 500,1200]) + 1.0,
                                  'err':array([78,   57,   24,   7.0,  1.2,  0.03, 0.003,0.003,0.003,0.003,0.003,0.003,0.003])}
        rcParams['font.family']='Serif'
        rcParams['font.size']=14
        figure(num='error',figsize=(7,6),tight_layout=True)
        for SN,m in zip(f_wl_vs_rng.keys(),['s','o','^','v','+']):
            freq_wl_rng = f_wl_vs_rng[SN]['rng']
            percent_diff_hilbert = f_wl_vs_rng[SN]['err']
            loglog(freq_wl_rng, percent_diff_hilbert,marker=m,linestyle=' ',label='SN={}'.format(SN))
        # for inset
        # xlim((2.0, 2.22))
        ylim((1e-3,1e2))
        # gca().set_xticks([2.1,2.2]) #  linspace(2,2.2,10))
        # from matplotlib import ticker
        # gca().xaxis.set_major_formatter(ticker.MultipleLocator(1.0))
        title('Deviation of ideal Gaussian FWHM vs.\nfrequency/wavelength range')
        xlabel(r'Frequency/wavelength range $\times$ center value')
        ylabel('Percentage deviation from ideal Gaussian FWHM')
        legend()

        # plot inset
        from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                          mark_inset)
        ax1 = gca()
        ax2 = axes([0.3, 0.45, 0.4, 0.45]) #pre-plot
        ip = InsetPosition(ax1, [0.25,0.45,0.4,0.45]) #resize
        mark_inset(ax1,ax2,loc1=2,loc2=3,fc='None',ec='0.5')
        ax2.set_axes_locator(ip)
        for SN,m in zip(f_wl_vs_rng.keys(),['s','o','^','v','+']):
            freq_wl_rng = f_wl_vs_rng[SN]['rng']
            percent_diff_hilbert = f_wl_vs_rng[SN]['err']
            ax2.loglog(freq_wl_rng, percent_diff_hilbert,marker=m,linestyle=' ',label='SN={}'.format(SN))
        ax2.set_xlim((1.95, 2.25))
        ax2.set_ylim((1e-2,1e2))
        ax2.set_xticks([2,2.2],minor=True)
        ax2.set_xticklabels(['2','2.2'],minor=True)
        tight_layout()
        savefig('PSF_error_inset_plt.pdf')
        show()

    def source_FD(self,):
        '''
        Source in the frequency domain.

        Liu 2008, p25
        :return:
        '''

        CWL = self.CWL
        BW = self.BW
        c = self.c
        SN = self.SN
        w_rng = self.w_rng
        w_0 = self.w_0
        sigma_w = self.sigma_w
        sw = self.plot_sigma_width

        def plot_power_spectrum():
            def Sw( w, w0, s_w ):
                return sqrt(2*pi/s_w**2) * exp(-(w-w0)**2 / (2*s_w**2))

            S_w_w0 = Sw(w_rng, w_0, sigma_w)
            figure(num='frequency',tight_layout=True)
            plot(w_rng/2/pi*1e-12, S_w_w0/S_w_w0.max(),'.-',lw=0.5,label='$\lambda_0$={:1.0f} nm, $\Delta\lambda$ = {:1.0f} nm'.format(CWL*1e9,BW*1e9))
            # stem(w_rng/2/pi*1e-12, S_w_w0/S_w_w0.max(),basefmt=' ',label='$\lambda_0$={:1.0f} nm, $\Delta\lambda$ = {:1.0f} nm'.format(CWL*1e9,BW*1e9))
            xlim(array([w_0-sigma_w*sw,w_0+sigma_w*sw])/2/pi*1e-12)
            grid(True)
            xlabel('Frequency (THz)')
            ylabel('Power vs. frequency (a.u.)')
            title('Plotted with {} sigma frequency width,\n $f_{{max}}$={:1.0f} THz'.format(self.plot_sigma_width,w_rng[-1]/2/pi*1e-12))
            legend(loc='upper right')
            savefig('source_{:1.0f}nm_rf{:3.0f}_SN{}_freq.pdf'.format(CWL*1e9,self.range_factor*100,SN))
            return S_w_w0
        S_w_w0 = plot_power_spectrum()

        def plot_TD_from_power_spectrum():
            f_max = 2*w_rng[-1]/2/pi # note here that we use 2 x the w_rng due to the double pass.
            print('f_max',f_max,'Hz')
            ST = 1/f_max*self.SN #s
            print('ST',ST,'s')
            z = ST * c # s * m/s == m
            zC = z/2*1e6
            print('zC',zC)
            print('z',z,'m')
            z_rng = linspace(0,z,self.SN)*1e6
            figure(num='fft of freq.',tight_layout=True)
            plot(z_rng,fftshift(fft(S_w_w0/S_w_w0.max())),'.-',lw=0.5,label='TD signal of source')
            title('Time-domain wave of source.')
            def envelope_hilbert():
                spline=UnivariateSpline(z_rng,abs(hilbert(real(fftshift(fft(S_w_w0/S_w_w0.max()))))),s=0)
                plot(z_rng,spline(z_rng),'+',label='Hilbert')
                psf_h = self.measure_FWHM_h(z_rng, real(fftshift(fft(S_w_w0/S_w_w0.max()))))
                print(psf_h)
                print( abs(psf_h*1e-6 - self.FWHM_psf)/self.FWHM_psf*100 )
            envelope_hilbert()

            def envelope_on_absolute_values():
                spline=UnivariateSpline(z_rng,abs(fftshift(fft(S_w_w0/S_w_w0.max()))),s=0)
                plot(z_rng,spline(z_rng),label='Univariate spline')
                psf_s = self.measure_FWHM(z_rng, abs(fftshift(fft(S_w_w0/S_w_w0.max()))))
                print(psf_s)
                print( abs(psf_s*1e-6 - self.FWHM_psf)/self.FWHM_psf*100 )
            envelope_on_absolute_values()

            xlabel('z (um)')
            xlim(array([zC-25,zC+25]))
            legend()
            savefig('source_{:1.0f}nm_rf{:3.0f}_SN{}_space_z.pdf'.format(CWL*1e9,self.range_factor*100,SN))

        # plot_TD_from_power_spectrum()

        def plot_vs_WL():
            WLR = self.WL_rng
            CWL = self.CWL
            s = self.sigma

            def SWL( WLR, CWL, s ):
                return sqrt(2*pi/s**2) * exp(-(WLR-CWL)**2 / (2*s**2))

            S_WLR_CWL = SWL( WLR, CWL, s)
            figure(num='wavelength',tight_layout=True)
            plot(WLR*1e9, S_WLR_CWL/S_WLR_CWL.max(),'.-',lw=0.5,label='$\lambda_0$={:1.0f} nm, $\Delta\lambda$ = {:1.0f} nm'.format(CWL*1e9,BW*1e9))
            # stem(WLR*1e9, S_WLR_CWL/S_WLR_CWL.max(),basefmt=' ',label='$\lambda_0$={:1.0f} nm, $\Delta\lambda$ = {:1.0f} nm'.format(CWL*1e9,BW*1e9))
            xlim(array([CWL-s*sw,CWL+s*sw])*1e9)
            grid(True)
            xlabel('Wavelength (nm)')
            ylabel('Power vs. wavelength (a.u.)')
            title('Plotted with {} sigma wavelength width,\n $\lambda_{{max}}$={:1.0f} nm'.format(self.plot_sigma_width,WLR[-1]*1e9))
            legend(loc='upper right')
            savefig('source_{:1.0f}nm_rf{:3.0f}_SN{}_WL.pdf'.format(CWL*1e9,self.range_factor*100,SN))
        # plot_vs_WL()

        show()

    def source_TD(self):
        pass

class TomlinsSimulation(SimulationHelpers):
    '''
    This contains mainly all aspects of the error of using the wrong sample range and amount of samples.
    '''
    
    SN = 2 ** 16
    c = speed_of_light
    CWL = 1330e-9
    BW = 60e-9  # 50 nm
    FWHM_psf = 2*log(2)/pi * CWL**2 / BW
    sigma = BW / sqrt(8 * log(2))
    #todo this is due for a cleanup.
    plot_sigma_width = 5.0 # to scale the x-range based on sigma
    sw = plot_sigma_width
    range_factor = 50 # how much more relative to the CWL of range should be generated?
    # this has inverse impact on the freq range or spatial range.
    # 

    WL_rng_max = CWL + CWL*range_factor
    WL_rng = linspace(0, WL_rng_max, SN)  # wavelength range to use for
    WLR = WL_rng

    f_0 = c / CWL

    w_0 = 2 * pi * c / CWL
    FWHM_w = 2 * pi * c * BW / CWL ** 2
    sigma_w = FWHM_w / sqrt(8 * log(2))
    # w_rng = linspace(w_0 - sigma_w*5, w_0 + sigma_w*5, SN)
    w_rng_max = w_0 + w_0*range_factor
    w_rng = linspace(0, w_rng_max, SN)

    def __init__(self,run=False):
        '''Print values only if object is used and initialized.'''
        print('FWHM_z {:1.3f} um'.format(self.FWHM_psf * 1e6))
        print('f_0 {} THz'.format(self.f_0 * 1e-12))
        print('FWHM_w {:1.3e} rad/s ({:1.0f} THz)'.format(self.FWHM_w, self.FWHM_w / 2 / pi * 1e-12))

        if run:
            rcParams['font.size'] = 16

            # self.representation_freq_vs_WL()
            # self.sample_points_vs_noise()
            # S_w_w0 = self.plot_power_spectrum(do_save=False)
            # self.plot_TD_from_power_spectrum(S_w_w0,do_save=False,
            #                                  do_envelope_hilbert=True,
            #                                  do_envelope_on_absolute_values=True)
            # self.plot_SourceSpectrum_vs_WL(do_save=False)
            # self.calc_reflectivities()
            # self.plot_relative_reflectivities()
            # self.plot_kronecker_delta(do_plot=True,do_save=True)
            # self.plot_kronecker_delta_manuscript(do_plot=True,do_save=False)
            self.plot_kronecker_field(do_plot=True,do_save=False) # use plot simulation_OCT_1D
            # self.sample_response_function(do_plot=True,do_save=False)
            # self.sample_response_function_analysis(do_plot=True,do_save=False)
            # self.compare_kron_sample_response()
            self.plot_srf_field(do_plot=True, do_save=False) # use plot simulation_OCT_1D
    #         self.compare_kronecker_vs_srf_field(do_plot=True,do_save=True)
            # self.analyze_SRF_H()
            # self.skew_or_not()
            # self.test_convolution()

            show()

    def representation_freq_vs_WL(self):
        '''
        Simply show spectrum for WL, rad/s, and 1/s.
        :return:
        '''

        c = speed_of_light
        SN = 10000
        nm = 1e-9
        CWL = 1330*nm
        WL_rng = linspace(0,CWL*2,SN)
        BW = 60*nm
        sig_BW = BW/sqrt(8*log(2))
        S = exp(-(CWL-WL_rng)**2/ sig_BW**2)

        # WL to circular frequency
        wc = 2*pi*c/CWL
        w_rng = linspace(0, wc*2, SN)
        w_BW = 2 * pi * c * BW / CWL ** 2
        sig_wBW = w_BW / sqrt(8 * log(2))
        S_w = exp(-(wc-w_rng)**2/sig_wBW**2)

        # WL to linear frequency
        f_c = c/CWL
        f_rng = linspace(0,f_c*2,SN)
        f_BW = c * BW / CWL ** 2
        sig_fBW =  f_BW / sqrt(8*log(2))
        S_f = exp(-(f_c-f_rng)**2 / sig_fBW**2)

        subplot(311)
        plot(WL_rng*1e6, S)
        xlabel('$\lambda$ ($\mathrm{\mu m}$)')

        # plot(w_rng,f_spec)

        subplot(312)
        plot(w_rng*1e-12,S_w)
        xlabel('$\omega$ (rad/fs)')

        subplot(313)
        plot(f_rng*1e-12, S_f)
        xlabel('f (THz)')

        tight_layout()
        show()

    def sample_points_vs_noise(self):
        '''
        For manuscript
        :return:
        '''
        rcParams['font.family']='Serif'
        rcParams['font.size']=12

        c = speed_of_light
        SN = 1000

        ratios = [8,4,2,1.25]
        for ratio in ratios:
            figure(num='{}'.format(ratio.__str__()))
            chop_fraction = int(SN//2)
            chop = slice(SN//2-chop_fraction,SN//2+chop_fraction)
            print(chop)
            nm = 1e-9
            CWL = 1330*nm
            CWL_rng_max = CWL * ratio
            print('max',CWL_rng_max, 'cwl', CWL_rng_max-CWL, )
            WL_rng = linspace(0,CWL_rng_max,SN)
            # BW = 60*nm #
            BW = 60*nm*ratio
            sig_BW = BW/sqrt(8*log(2))
            S = exp(-(CWL-WL_rng)**2/ sig_BW**2)
            # S = S[chop]

            # WL to circular frequency
            wc = 2*pi*c/CWL
            w_rng = linspace(0, wc*ratio, SN)
            w_BW = 2 * pi * c * BW / CWL ** 2
            sig_wBW = w_BW / sqrt(8 * log(2))
            S_w = exp(-(wc-w_rng)**2/sig_wBW**2)


            subplot(221)
            suptitle('N/f = {}'.format(ratio),)
            plot(S,'.-',lw=0.5)
            title('Source spectrum')
            ylabel('Power (a.u.)')
            xlabel('Frequency components (a.u.)')

            subplot(222)
            plot(S,'.-',lw=0.5)
            xlim((SN//ratio-ratio/sig_BW*1e-6,SN//ratio+ratio/sig_BW*1e-6))
            title('Source spectrum (zoom)')
            ylabel('Power (a.u.)')
            xlabel('Frequency components (a.u.)')

            subplot(223)
            plot((fftshift(fft(S))),'.',lw=0.5)
            title('Field intensity')
            ylabel('Intensity (a.u.)')
            xlabel('spatial or time (a.u.)')
            SN = chop.stop - chop.start
            # xlim((SN//2-2/sig_BW*1e-6,SN//2+2/sig_BW*1e-6))

            subplot(224)
            plot((fftshift(fft(S))),'.-',lw=0.5)
            title('Field intensity (zoom)')
            ylabel('Intensity (a.u.)')
            xlabel('spatial or time (a.u.)')
            SN = chop.stop - chop.start
            xlim((SN//2-0.5*ratio/sig_BW*1e-6,SN//2+0.5*ratio/sig_BW*1e-6))

            def plot_inset():
                # currently not used here. But I keep it maybe for later
                from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                                  mark_inset)
                ax1 = gca()
                ax2 = axes([0.3, 0.45, 0.4, 0.45]) #pre-plot
                ip = InsetPosition(ax1, [0.45,0.6,0.6,0.6]) #resize
                # mark_inset(ax1,ax2,loc1=2,loc2=3,fc='None',ec='0.5')
                ax2.set_axes_locator(ip)
                ax2.plot((fftshift(fft(S))),'.',lw=0.5)
                xlim((SN//2-2/sig_BW*1e-6,SN//2+2/sig_BW*1e-6))
            # plot_inset()

            tight_layout()
            gcf().subplots_adjust(top=0.88)
            savefig('sample_point_vs_noise_ratio{}.pdf'.format(str(ratio).replace('.','_')))


    def skew_or_not(self,percent_BW = 4.5):
        '''
        This function can plot different situations of a spectrum vs wavelength and frequency
        to demonstrate when a skew becomes significant.
        A skew does exist but depends on the size of the bandwidth.
        A bandwidth of 5% of the CWL does not produce any visible skew.
        At a BW = 20% of the CWL a skew becomes notable.
        :return:
        '''
        c = speed_of_light
        pos = 1330e-9*+0.0
        scale = 1e-9
        CWLr = array([1330*scale])
        WL_rng = linspace(1*scale, CWLr[-1]*2, 10000)
        # BW = 60*10*scale

        def S(wlr, CWL, sig_BW): return exp(-(CWL - wlr) ** 2 / (2 * sig_BW ** 2))

        for CWL in CWLr:
            percentage = percent_BW/100
            BW = CWL * percentage
            sig_BW = BW / sqrt(8 * log(2))


            subplot(211)
            plot(WL_rng*1e6,S(WL_rng,CWL,sig_BW),'-',lw=0.5,label='BW={:1.3f} um'.format(BW*1e6))
            gca().set_yticks([0.5])
            grid(True)
            legend()
            xlabel('WL (um)')
            subplot(212)
            plot(c/WL_rng*1e-12, S(WL_rng,CWL,sig_BW), '-',lw=0.5)
            gca().set_yticks([0.5])
            grid(True)
            xlim((c/CWL*0.45e-12,c/CWL*4e-12))
            xlabel('f THz')
        tight_layout()

    def plot_power_spectrum(self,do_save=False):
        '''
        Source in the frequency domain.

        Liu 2008, p25
        :return:
        '''
        _ = self

        def Sw( w, w0, s_w ):
            return sqrt(2*pi/s_w**2) * exp(-(w-w0)**2 / (2*s_w**2))

        S_w_w0 = Sw(_.w_rng, _.w_0, _.sigma_w)
        figure(num='frequency',tight_layout=True)
        plot(_.w_rng/2/pi*1e-12, S_w_w0/S_w_w0.max(),'.-',lw=0.5,label='$\lambda_0$={:1.0f} nm, $\Delta\lambda$ = {:1.0f} nm'.format(_.CWL*1e9,_.BW*1e9))
        # stem(_.w_rng/2/pi*1e-12, S_w_w0/S_w_w0.max(),basefmt=' ',label='$\lambda_0$={:1.0f} nm, $\Delta\lambda$ = {:1.0f} nm'.format(_.CWL*1e9,_.BW*1e9))
        xlim(array([_.w_0-_.sigma_w*_.sw,_.w_0+_.sigma_w*_.sw])/2/pi*1e-12)
        grid(True)
        xlabel('Frequency (THz)')
        ylabel('Power vs. frequency (a.u.)')
        title('Plotted with {} sigma frequency width,\n $f_{{max}}$={:1.0f} THz'.format(_.sw,_.w_rng[-1]/2/pi*1e-12))
        legend(loc='upper right')
        if do_save:
            savefig('source_{:1.0f}nm_rf{:3.0f}_SN{}_freq.pdf'.format(_.CWL*1e9,self.range_factor*100,SN))
        return S_w_w0


    def plot_TD_from_power_spectrum(self, S_w_w0,do_save=False, do_envelope_hilbert=False, do_envelope_on_absolute_values=False):
        '''
        Plot the FFT(S_w_w0).
        This plot takes the member values of the light source specs such as WL, BW, CWL to calculate
        the right spatial distribution and can measure the PSF.
        Please take note that the position of the Gaussian envelope is based here merely on the speed of light
        and has no relevance to any sample position yet.
        :param S_w_w0:
        :param do_save:
        :param do_envelope_hilbert:
        :param do_envelope_on_absolute_values:
        :return:
        '''
        _ = self
        c = speed_of_light
        f_max = 2 * _.w_rng[-1]/2/pi # note here that we use 2 x the _.w_rng due to the double pass.
        print('f_max',f_max,'Hz')
        ST = 1/f_max*self.SN #s
        print('ST',ST,'s')
        z = ST * c # s * m/s == m
        zC = z/2*1e6
        print('zC',zC)
        print('z',z,'m')
        z_rng = linspace(0,z,self.SN)*1e6
        TD_field = fftshift(fft(S_w_w0))
        figure(num='fft of freq.',tight_layout=True)
        plot(z_rng,TD_field,'.-',lw=0.5,label='TD signal of source')
        title('Time-domain wave of source.')
        def envelope_hilbert():
            spline=UnivariateSpline(z_rng,abs(hilbert(real(fftshift(fft(S_w_w0/S_w_w0.max()))))),s=0)
            plot(z_rng,spline(z_rng),'+',label='Hilbert')
            psf_h = self.measure_FWHM_h(z_rng, real(fftshift(fft(S_w_w0/S_w_w0.max()))))
            print(psf_h)
            print( abs(psf_h*1e-6 - self.FWHM_psf)/self.FWHM_psf*100 )
        if do_envelope_hilbert: envelope_hilbert()

        def envelope_on_absolute_values():
            spline=UnivariateSpline(z_rng,abs(fftshift(fft(S_w_w0/S_w_w0.max()))),s=0)
            plot(z_rng,spline(z_rng),label='Univariate spline')
            psf_s = self.measure_FWHM(z_rng, abs(fftshift(fft(S_w_w0/S_w_w0.max()))))
            print(psf_s)
            print( abs(psf_s*1e-6 - self.FWHM_psf)/self.FWHM_psf*100 )
        if do_envelope_on_absolute_values: envelope_on_absolute_values()

        xlabel('z (um)')
        xlim(array([zC-25,zC+25]))
        legend()
        if do_save:
            savefig('source_{:1.0f}nm_rf{:3.0f}_SN{}_space_z.pdf'.format(CWL*1e9,self.range_factor*100,SN))

        return TD_field,z_rng


    def plot_SourceSpectrum_vs_WL(self,do_save=False):
        '''
        This is the power_spectrum in the wavelength domain.
        :param do_save:
        :return:
        '''
        _ = self

        def SWL( WLR, CWL, s ):
            return sqrt(2*pi/s**2) * exp(-(WLR-CWL)**2 / (2*s**2))

        S_WLR_CWL = SWL( _.WLR, _.CWL, _.sigma)
        figure(num='wavelength',tight_layout=True)
        plot(_.WLR*1e9, S_WLR_CWL/S_WLR_CWL.max(),'.-',lw=0.5,label='$\lambda_0$={:1.0f} nm, $\Delta\lambda$ = {:1.0f} nm'.format(_.CWL*1e9,_.BW*1e9))
        # stem(WLR*1e9, S_WLR_CWL/S_WLR_CWL.max(),basefmt=' ',label='$\lambda_0$={:1.0f} nm, $\Delta\lambda$ = {:1.0f} nm'.format(CWL*1e9,BW*1e9))
        xlim(array([_.CWL-_.sigma*_.sw,_.CWL+_.sigma*_.sw])*1e9)
        grid(True)
        xlabel('Wavelength (nm)')
        ylabel('Power vs. wavelength (a.u.)')
        title('Plotted with {} sigma wavelength width,\n $\lambda_{{max}}$={:1.0f} nm'.format(self.plot_sigma_width,_.WLR[-1]*1e9))
        legend(loc='upper right')
        if do_save:
            savefig('source_{:1.0f}nm_rf{:3.0f}_SN{}_WL.pdf'.format(_.CWL*1e9,self.range_factor*100,_.SN))

        return

    def plot_kronecker_delta(self,do_plot=False, do_save=False):
        '''
        Relatet to plot in manuscript fig:kroneckerdeltas
        Plot the Kronecker delta based on positions z_widths and refractive indexes 'ns'.

        :param do_plot:
        :param do_save:
        :return:
        '''
        air = 1.0
        ns = array([air, 1.001, 1.002, 1.003])
        micrometer = 1e-6
        z_widths = array([15,60,90])*micrometer
        z_widths = z_widths * ns[0:-1]

        # manuscript:2712:lst:kroneckerarrayconstruction
        z_locs = z_widths.cumsum()
        cum_width = z_widths.sum()
        z_rng_max = cum_width*2
        z_rng = linspace(0,z_rng_max,self.SN)
        getlocidx = interpolate.interp1d( [0,z_rng_max], [0, self.SN])

        rs_kd = zeros(self.SN) # create empty Kronecker delta array -> all zeros
        rjs = array([(nn-np)/(np+nn) for np,nn in zip(ns[0:-1],ns[1:])]).squeeze()

        rs_kd[getlocidx(z_locs).astype(int)] = 1 * rjs # To be explicit we multiply unity with reflectivity
        # manuscript:2721:lst:kroneckerarrayconstruction
        if do_plot:
            figure('kroneckre deltas',tight_layout=True)
            plot(z_rng*1e6,rs_kd,'.-',lw=0.5)
            xlim((0,z_rng_max*1e6))
            title('Kronecker delta')
            xlabel('z ($\mathrm{\mu m}$)')
            ylabel('field reflectivity $r_j$')
            if do_save: savefig('kronecker_deltas.pdf')

        _ = self
        _.z_rng = z_rng
        _.kd = rs_kd

    def plot_kronecker_delta_manuscript(self,do_plot=False, do_save=False):
        '''
        Related to plot fig:kroneckerdeltas
        This code is suitable for appendix
        Plot the Kronecker delta based on positions z_widths and refractive indexes 'ns'.

        :param do_plot:
        :param do_save:
        :return:
        '''
        # manuscript:2747:lst:kroneckerarrayconstruction
        air = 1.0
        ns = array([air, 1.001, 1.002, 1.003])
        z_widths = array([5,15,30])*1e3 # milli meter
        z_widths = (z_widths * ns[0:-1]).round().astype(int)
        z_total = z_widths.sum()*2
        rs_kd = zeros(z_total) # create empty Kronecker delta array -> all zeros
        # manuscript:2752:lst:kroneckerarrayconstruction
        # manuscript:2755:lst:kroneckerarrayfresnelvalues
        rjs = array([(nn-np)/(np+nn) for np,nn in zip(ns[0:-1],ns[1:])]).squeeze()
        # manuscript:2755:lst:kroneckerarrayfresnelvalues
        # manuscript:2758:lst:kroneckerarrayassignfresnel
        z_locs = z_widths.cumsum()
        rs_kd[z_locs] = 1 * rjs
        # manuscript:2759:lst:kroneckerarrayassignfresnel

        z_rng = arange(z_total)
        if do_plot:
            figure('kroneckre deltas',tight_layout=True)
            plot(z_rng*1e-3,rs_kd,'.-',lw=0.5)
            # xlim((0,60))
            title('Kronecker delta')
            xlabel('z ($\mathrm{\mu m}$)')
            ylabel('reflectivity')
            if do_save: savefig('kronecker_deltas.pdf')

        _ = self
        _.z_rng = z_rng
        _.kd = rs_kd

        # keep for manuscript
        # lst: kroneckerarrayfresnelvalues
        rjs = []
        for np,nn in zip(ns[0:-1],ns[1:]):
            rjs.append((np-nn)/(np+nn))
        # lst:kroneckerarrayfresnelvalues

    def plot_kronecker_field(self,do_plot=False, do_save=False):
        '''
        Related to plot:

        Plot the Kronecker delta convoluted with the field.

        :param do_plot:
        :param do_save:
        :return:
        '''
        air = 1.0
        ns = array([air, 1.001, 1.002, 1.003])
        z_widths = array([15,60,90]) # micro meter
        z_widths_delay = (z_widths * ns[0:-1]).round().astype(int) #correct depth with ref index
        z_locs = z_widths_delay.cumsum()
        z_widths_max = z_locs[-1]
        S_w_w0 = self.plot_power_spectrum()
        # TD field is E_in = s(w)*exp(2ikz_s) = FFT(S(w))
        E_in, z_rng_field = self.plot_TD_from_power_spectrum(S_w_w0)
        SN = len(E_in) # need now to use sample len of source
        rs_kd = zeros(SN) # create empty Kronecker delta array -> all zeros
        z_rng_max = z_rng_field.max()
        z_rng_interp = interpolate.interp1d( [0,z_rng_max], [0, SN]) # interpolate z_widths to fit into SN elements
        z_indexes = z_rng_interp( z_locs ).astype(int) # get index for each layer location
        rjs = array([(nn-np)/(np+nn) for np,nn in zip(ns[0:-1],ns[1:])]).squeeze()
        rs_kd[ z_indexes ] = 1 * rjs
        E_s = convolve( abs(E_in), rs_kd, mode='same')
        I_s = abs( E_s )
        #todo How is the source field calibrated with the z_rng of the kroneckre deltas ?
        # z_rng = arange(z_total)
        z_rng = z_rng_field

        if do_plot:
            figure(num='kroneckre fields',tight_layout=True)
            plot(z_rng,E_s,'-',lw=1.5)
            xlim((0,200))
            title('Kronecker field\n$FFT(S(\omega)) \otimes r_j\delta(z_s - z_{s_n})$')
            xlabel('z ($\mathrm{\mu m}$)')
            ylabel('reflectivity')
            if do_save: savefig('kronecker_fields.pdf')

        _ = self
        _.z_rng = z_rng
        _.kd = rs_kd
        _.kronecker_sample_fields = E_s

    def test_convolution(self):
        # problem This is to confirm that the convolution is not equal to mere multiplication.
        # only in the frequency domain the operation changes.
        a1 = zeros(20)
        a2 = zeros(20)
        a1[[5,7,14]] = 1
        a2[[5,7,14]] = 0.6
        figure()
        subplot(211)
        plot(a1)
        plot(a2)
        subplot(212)
        plot(convolve(a1,a2,mode='same'))


    def fresnel_reflection(self, n1, n2):
        '''
        See manuscript is referenced to label lst:kroneckerarrayfresnelvalues
        :param n1:
        :param n2:
        :return:
        '''
        r = (n1-n2)/(n1+n2)
        print(r)
        return r

    def calc_reflectivities(self):
        '''
        Call this separately by providing values for the array ns.
        :return:
        '''
        air = 1.0
        ns = array([air, 1.001, 1.002, 1.003]) # refractive index
        print(ns)
        nsd = sy.diff(ns)
        nss = [n1+n2 for n1,n2 in zip(ns[0:-1],ns[1:])]
        print(nsd/nss)
        #print for latex
        print('expected: ')
        [print('\SI{{{:1.3e}}}{{}} &'.format(val),end='',flush=True) for val in nsd/nss]
        print('\n')

    def plot_relative_reflectivities(self):
        '''
        Call this separately from measured values of reflectivity at different boundaries alas values of
        the array ns after simulation.
        :return:
        '''
        reflectivities={
        'Fresnel'     :([4.998e-04 , 4.993e-04 , 4.988e-04]),
        'Kronecker'     :([4.998e-04 , 4.993e-04 , 4.988e-04]),
        'SRF(no win)'  :([4.284e-04 , 4.451e-04 , 4.199e-04]),
        'SRF(Tukey $\\alpha$=0.8)' :array([2.379e-04 , 2.412e-04 , 2.358e-04])}

        figure(num='reflectivities',tight_layout=True)
        fresnel = array(reflectivities['Fresnel'])*1e3
        for key,m in zip(reflectivities.keys(),['s','o','v','^']):
            if 'Fresnel' not in key:
                plot(fresnel,array(reflectivities[key])*1e3,marker=m, linestyle='-',label=key)
        legend()
        title('Reflectivity vs Fresnel\n')
        ylabel('Simulated (see legend)')
        xlabel('Fresnel calculated')
        savefig('reflectivity_vs_fresnel.pdf')

        # rel_change_refs = []
        #
        figure(num='relative refs',tight_layout=True)
        for key,m in zip(reflectivities.keys(),['s','o','v','^']):
            plot(sy.diff(reflectivities[key]),marker=m, linestyle=' ',label=key)
        legend()
        title('Relative change of reflectivity\nbetween boundaries')
        ylabel('Deviation')
        # # Axes.set_xt
        gca().set_xticks([0,1.0])
        gca().set_xticklabels(['$r_1-r_2$','$r_2-r_3$'])
        savefig('reflectivity_relative_change.pdf')

        figure(num='recalibrated',tight_layout=True)
        plot(fresnel,array(reflectivities['Kronecker'])*1e3,'s',label='Kronecker')
        plot(fresnel,reflectivities['SRF(Tukey $\\alpha$=0.8)']*1e3*2.1,'o',label='SRF re-calibrated')
        title('Re-calibrated SRF')
        xlabel('Fresnel')
        ylabel('Simulated (see legend)')
        legend()
        savefig('reflectivity_re_calibrated.pdf')

    def sample_response_function(self,do_plot=False,do_save=False):
        '''
        Please take note that the SRF alone is only of limited use although the FFT can be used
        if it consistent with the Kronecker deltas.
        :param do_plot:
        :param do_save:
        :return:
        '''
        r_j_f = self.fresnel_reflection
        _ = self
        c = speed_of_light
        f_max = 2 * _.w_rng[-1]/2/pi # note here that we use 2 x the _.w_rng due to the double pass.
        ST = 1/f_max*_.SN #s
        z = ST * c # s * m/s == m
        z_rng = linspace(-z/2,z/2,_.SN)*1e6

        air = 1.0
        ns = array([air, 1.001, 1.002, 1.003]) # refractive index
        z_widths = array([5,15,30]) * 1e-6 #
        alpha = None
        Hj = []
        nm, zm = [],[]
        for n1,n2,zj in zip(ns[0:-1],ns[1:],z_widths):
            # for each slab j the wavefront is delayed and accumulates -> Sum_m_j
            r_j = r_j_f(n1,n2)
            nm.append(n1)
            zm.append(zj)
            Sum_m_j = sum([n*z for n,z in zip(nm,zm)])
            Hj.append(r_j * exp( 1j * 2 * _.w_rng / c * Sum_m_j))

        alpha = None
        tw = None
        # Sum all response frequencies into one signal
        if alpha is not None:
            tw = tukey(M=self.SN, alpha=alpha)
            H = sum(Hj,axis=0)*tw
        else:
            H = sum(Hj,axis=0)

        # H= sum([r_j * exp( 1j * 2 *_.w_rng / c * n * loc )

        if do_plot:
            figure(num='sample resp fun',tight_layout=True)
            plot(z_rng, fftshift(abs(fft(H)))/self.SN,'.-',lw=0.5)
            xlim((0,60))
            title('Sample response function H')
            xlabel('z ($\mathrm{\mu m}$)')

            if do_save:
                savefig('sample_response_function_tw{}.pdf'.format(alpha))
        _.tw = tw
        _.z_rng = z_rng
        _.H = H

    def sample_response_function_analysis(self,do_plot=False,do_save=False):
        '''
        We add in some intermediate plots to demonstrate the properties of the sample response function.
        The point here is that the sample response function describes reflective layer boundaries with
        layer thickness of z_widths and each layer has a refractive index n stored in an array ns.

        The theory describes H(w) =

        Is there anybody who did compute this?

        :param do_plot:
        :param do_save:
        :return:
        '''
        r_j_f = self.fresnel_reflection
        _ = self
        c = speed_of_light
        f_max = 2 * _.w_rng[-1]/2/pi # note here that we use 2 x the _.w_rng due to the double pass.
        ST = 1/f_max*_.SN #s
        z = ST * c # s * m/s == m
        z_rng = linspace(-z/2,z/2,_.SN)*1e6

        air = 1.0
        ns = array([air, 1.001, 1.002, 1.003]) # refractive index
        z_widths = array([5,15,30]) * 1e-6 #
        alpha = None
        Hj = []
        nm, zm = [],[]
        for n1,n2,zj in zip(ns[0:-1],ns[1:],z_widths):
            # for each slab j the wavefront is delayed and accumulates -> Sum_m_j
            r_j = r_j_f(n1,n2)
            nm.append(n1)
            zm.append(zj)
            Sum_m_j = sum([n*z for n,z in zip(nm,zm)])
            Hj.append(r_j * exp( 1j * 2 * _.w_rng / c * Sum_m_j))

        alpha = None
        tw = None
        # Sum all response frequencies into one signal
        if alpha is not None:
            tw = tukey(M=self.SN, alpha=alpha)
            H = sum(Hj,axis=0)*tw
        else:
            H = sum(Hj,axis=0)

        # H= sum([r_j * exp( 1j * 2 *_.w_rng / c * n * loc )

        if do_plot:
            figure(num='sample resp fun',tight_layout=True)
            plot(H)
            # plot(z_rng, fftshift(abs(fft(H)))/self.SN,'.-',lw=0.5)
            # xlim((0,60))
            # title('Sample response function H')
            # xlabel('z ($\mathrm{\mu m}$)')

            if do_save:
                savefig('sample_response_function_tw{}.pdf'.format(alpha))
        _.tw = tw
        _.z_rng = z_rng
        _.H = H

    def compare_kron_sample_response(self):
        _ = self
        _.plot_kronecker_delta(do_plot=False, do_save=False)
        print('kronecker values:')
        [print('\SI{{{:1.3e}}}{{}} & '.format(val[0]),end='',flush=True) for val in _.kd[argwhere(_.kd>4*1e-4)]]
        print('\n')
        fig = figure(tight_layout=True)
        # fig,ax=subplots(2,1,sharey=True,tight_layout=True)
        subplot(211)
        plot(_.z_rng * 1e6, _.kd, '.-', lw=0.5)
        ylabel('Reflectivity')
        ylim((0,5e-4))
        xlim((0, 60))
        title('Kronecker delta')
        xlabel('z ($\mathrm{\mu m}$)')
        a1 = axes([0.5, 0.67, 0.2, 0.2])
        a1.plot(_.z_rng*1e6,_.kd,'.-',lw=0.5)
        a1.set_xlim((20-0.03,20+0.05))
        a1.set_xticks([])
        a1.set_yticks([])

        _.sample_response_function(do_plot=False,do_save=False)
        kdH = fftshift(abs(fft(_.H)))/self.SN
        print('SRF values:')
        [print('\SI{{{:1.3e}}}{{}} & '.format(val[0]),end='',flush=True) for val in kdH[argwhere(kdH>2.3*1e-4)]]
        print('\n')
        subplot(212)
        plot(_.z_rng, kdH, '.-', lw=0.5)
        ylabel('Reflectivity')
        ylim((0,5e-4))
        xlim((0, 60))
        title('Sample response function H')
        xlabel('z ($\mathrm{\mu m}$)')
        a2 = axes([0.5, 0.2, 0.2, 0.2])
        a2.plot(_.z_rng,kdH,'.-',lw=0.5)
        a2.set_xlim((20-0.03,20+0.05))
        a2.set_xticks([])
        a2.set_yticks([])
        savefig('compare_kron_sample_response.pdf')

        # figure(),plot(_.tw)

    def plot_srf_field(self,do_plot=False,do_save=False):
        '''
        Product of the source spectrum with the SRF and the FFT to make it comparable with the Kronecker field.
        :param do_plot:
        :param do_save:
        :return:
        '''
        r_j_f = self.fresnel_reflection # get the function
        _ = self # use _ to access self
        
        # lst:plotsrffield
        c = speed_of_light
        f_max = 2 * _.w_rng[-1]/2/pi # note here that we use 2 x the _.w_rng due to the double pass.
        ST = 1/f_max*_.SN #s
        z = ST * c # s * m/s == m
        z_rng = linspace(-z/2,z/2,_.SN)*1e6

        air = 1.0
        ns = array([air, 1.001, 1.002, 1.003]) # refractive index
        z_widths = array([15,60,90]) * 1e-6 #
        alpha = None
        Hj = []
        nm, zm = [],[]
        for n1,n2,zj in zip(ns[0:-1],ns[1:],z_widths):
            # for each slab j the wavefront is delayed and accumulates -> Sum_m_j
            r_j = r_j_f(n1,n2)
            nm.append(n1)
            zm.append(zj)
            Sum_m_j = sum([n*z for n,z in zip(nm,zm)])
            Hj.append(r_j * exp( 1j * 2 * _.w_rng / c * Sum_m_j))

        H = sum(Hj,axis=0)
        S_w_w0 = self.plot_power_spectrum()
        E_S = fftshift((fft(H * S_w_w0)))
        # lst:plotsrffield
        
        if do_plot:
            figure(num='sample resp fun',tight_layout=True)
            plot(z_rng, E_S ,'-',lw=1.5)
            xlim((-0,200))
            title('   Sample field ($E_S = S(\omega)H(\omega)$)')
            xlabel('z ($\mathrm{\mu m}$)')

            if do_save:
                savefig('srf_field.pdf')

        _.z_rng = z_rng
        _.H = H
        _.srf_sample_fields = E_S
        
    def compare_kronecker_vs_srf_field(self, do_plot=True,do_save=False):
        _ = self
        self.plot_kronecker_field(do_plot=True)
        self.plot_srf_field(do_plot=True)
        krf = _.kronecker_sample_fields.max()
        srf = _.srf_sample_fields.max()
        max_avg_ref = mean([krf, srf])
        if do_plot:
            figure()
            for r in range(1):
                field_diff = _.kronecker_sample_fields - roll(_.srf_sample_fields,_.SN//2+r)
                fd = field_diff/max_avg_ref * 100
                plot(_.z_rng,fd,label='roll {}'.format(r))
                xlabel('z ($\mu m$)')
                ylabel('Error value (%)')
                title('Variation between SRF and Kronecker fields.')
    #        legend()
                tight_layout()
        if do_save:
            savefig('compare_kronecker_vs_srf_field.pdf')            

class Theory_Gauss_signal(GaussGenerator):
    '''
    Demonstrate the generation of Gauss and different use of FFT
    to optimize / speed up the generation of ideal signal.
    '''
    def __init__(self):
        pass

    def double_pass(self):
        pass

    def gauss(self,w,w0,sigma):
        return exp(-((w-w0)**2) / (2*sigma**2))

    def sin_phase(self,beg,end,n):
        #
        return sin(linspace(beg,end,n))

    def gaussian_FWHM(self):
        N = 20000  # buffer size
        SR = 20e6  # sample rate (seconds)
        tmr = linspace(0, N/SR, N) # time range
        wavelength = 1330e-9 # meter
        wavelengthBW = 60e-9

        FWHM = 2*log(2)/pi * wavelength**2/wavelengthBW  #[m]
        print('FWHM',FWHM,'m')
        sigma = FWHM/2/sqrt(2*log(2)) #[m]
        print('sigma',sigma,'m')

        v_M = 0.04 # [m/s]
        spr = tmr*v_M # spatial range [m]
        dist = max(spr)
        print('scan distance',dist,'m')
        f_D = 2 * v_M / wavelength #[1/s]
        print('f_D',f_D,'Hz')

        spPerLen = N/dist # [sp / m]
        print('[sp/m]',spPerLen)
        FWHMsp = FWHM * spPerLen
        print('FWHMsp',FWHMsp)
        sigmaSp = FWHMsp/2/sqrt(2*log(2))

        # How do we incorporate the group velocity and DeltaTau_g?
        # DeltaTau_g tells us the time the wave bunch requires to travel a distance Delta_l.
        # This is of course much faster than we can actually measure.
        # We could in theory plot the Gaussian envelope in terms of the time again which should be it
        #
        # However, we measure actually the time of either the scanning mirror or
        # the delay time due to the sample refractional index.
        v_g = speed_of_light
        Delta_l = spr
        DeltaTau_g = 2 * Delta_l / v_g
        Delta_l = DeltaTau_g / 2 * v_g

        w = spr # lin space
        # w = spr*
        w0 = 0e-6
        w0 = 18e-6 # must be close to the um dimensions
        # G_in_samples = gauss(w=w,w0=w0,sigma=sigma)
        # simple Gauss
        G_in_samples = real(exp(-(w - w0 )**2/(2*sigma**2)))
        # modulated Gauss
        # G_in_samples = real(exp(-(w - w0 + max(spr)*real(sin_phase(-pi,pi,n=N)))**2/(2*sigma**2)))
        line_FWHMsp = where(G_in_samples >= 0.5)[0]
        line_FWHM = line_FWHMsp/SR*v_M
        s0 = min(line_FWHM)
        s1 = max(line_FWHM)
        print('sample FHWM',abs(s0-s1))
        print('time FHWM',abs(s0-s1)/SR)
        print('spatial FHWM',abs(s0-s1)/SR*v_M)

        # plot(tmr, sin(2*pi*f_D*tmr))
        plot(spr*1e6,G_in_samples)
        xlabel('space ($\operatorname{\mu m}$)')
        print(line_FWHM)
        plot(line_FWHM*1e6,ones(len(line_FWHM))/2)
        # xlabel('time (s)') # tmr
        # xlabel('$\Delta l$ ($\mu$m)') # spr
        # ylabel('Amplitude (arb.)')
        gca().set_yticks(linspace(0,1.0,11))
        grid(True)
        show()

        # Conceptionally we need always to distinguish between the true propagation time of the light
        # and the scanning time.

        # As long as our scanning time is much less than the speed of light we obtain an representation
        # of the interference characteristics vs the scan time.

        # The Delta_tau_g in this case is the time the light requires to pass the distance mismatch.
        # But we generate a mismatch by moving the reference mirror much slower!

        # For FD-OCT the mismatch is indeed as fast as by the speed of light vs. the delay in the sample.

    def gaussian_FWHM_B(self):
        '''
        Proof showing for a particular Gaussian centered around zero we can calculate the sigma
        for some FWHM matching the expected width in the plot.
        :return:
        '''
        xr = linspace(-3,3,100)
        FWHM = 1
        sigma = FWHM/(2*sqrt(2*log(2)))
        y = exp( -(xr-0)**2/(2*sigma**2))
        print(min(where(y>=0.5)))
        # Interpolation could improve the accuracy to determine the FWHM from the data points in the plot.
        # However, as the plot itself is not the source of data this is not essential or even misleading.
        # The plot is just the representation of the true data with limited resolution (sample rate).
        yi_min = where(y>=0.5)[0].min()
        yi_max = where(y>=0.5)[0].max()
        print('yi_min',yi_min,'yi_max',yi_max,'diff',diff((xr[yi_min],xr[yi_max])))
        plot(xr, y)
        xlabel('units in pi')
        gca().set_yticks(linspace(0, 1.0, 11))
        gca().set_xticks(linspace(min(xr), max(xr), 13))
        grid(True)

        show()

    def carrier(self):
        t = linspace(0,1,1000)
        f = 10
        carrier = (exp(1j*2*pi*f*t))
        plot(carrier)
        show()

    def carrier_frequ_change(self):
        '''This was for the response letter to the Performance analysis.'''
        scn_rng = linspace(-pi/2,2*pi-pi/2,1000)
        scn_rng2 = linspace(0,1,1000)
        f_chng = (sin(scn_rng) + 1.0)/2*60
        print(abs(arcsin(pi/4)))
        plot(scn_rng2,f_chng)
        xlabel('Scanning mirror position')
        ylabel('Carrier frequency (kHz)')
        grid(True)
        show()

    def loss_due_to_modulation(self):

        f_max = 60
        f_lim = 10
        x = 0.133
        f = (1+cos(-2*pi*x - pi))/2*f_max
        print(f,x)

        print('='*10)
        print(f_lim/f_max*2)
        print(1+cos(-2*pi*x -pi))

        print('='*10)
        print(f_lim/f_max*2 -1)
        print(cos(-2*pi*x -pi))

        print('='*10)
        print((arccos(deg2rad(f_lim/f_max*2 -1))))
        print((-2*pi*x -pi))

    def carrier_modulated(self):
        # Some effects occuring here:
        # 1) if the f_s is increased then this means that less fringes can be captured and the carrire f_D
        # reduces
        # This neglects here that at higher scanning frequencies f_s the actual mirror speed increases and
        # the Doppler effect will increase the fringe frequencies.
        # But at the moment this part is usefule to investigate the direct relations of frequency modulation.
        NS = 2**16
        f_D = 30000
        f_s = 152 # scan frequency
        t_s = linspace(0,1/f_s,NS) # scan time
        print(f_s*max(t_s))
        d_f = abs(0 - f_s)
        C = f_D/d_f
        ph = C * sin(2*pi* (f_s) * t_s - pi)

        carrier = exp(1j * (2*pi*f_D*t_s ) )
        carrier_mod = (exp(1j*(2*pi*f_D*t_s + ph) ))

        figure(tight_layout=True,figsize=(8,9))

        subplot(411)
        plot(t_s, carrier,label='carrier linear phase',linewidth=1)
        # title('Linear carrier frequency')
        xlabel('Time (s)')
        legend()

        subplot(412)
        plot(t_s,carrier_mod,label='carrier mod. phase',linewidth=1)
        plot(t_s,ph/C,'--',label='mod. phase')
        # title('Modulated carrier frequency')
        xlabel('Time (s)')
        legend()

        subplot(413)
        plot(t_s,unwrap(angle((carrier_mod))),label='carrier phase')
        plot(t_s,NS/2*gradient(unwrap(angle((carrier_mod)))),'--',label='carrier diff')
        x_norm = linspace(0,1,len(carrier_mod))
        carrier_freq = gradient(unwrap(angle((carrier_mod))))
        carrier_freq = carrier_freq/carrier_freq.max() * 60
        # plot(x_norm, carrier_freq,label='60 kHz')
        # plot(x_norm, carrier_freq*2,label='120 kHz')
        # plot(x_norm, carrier_freq*5,label='300 kHz')
        # plot([10,10]) # some level
        # title('Carrier frequency vs. SRM position')
        xlabel('Time (s)')
        # ylabel('Frequency (kHz)')
        ylabel('(rad)')
        grid(False)
        legend()

        subplot(414)
        carrier_fft = abs((fft(carrier)))
        max_carr_fft = max(carrier_fft)
        plot(fftfreq(n=NS,d=max(t_s)/NS*1e3),carrier_fft/max_carr_fft,label='carrier spectrum linear phase')
        mod_fft =  abs((fft(carrier_mod)))
        scale_mod = max(mod_fft)
        plot(fftfreq(n=NS,d=max(t_s)/NS*1e3),mod_fft/scale_mod,'--',label='carriers spectrum mod. phase')
        title('Frequency spectrum of modulated carrier.')
        # ylabel('P')
        xlabel('Frequency (kHz)')
        xlim((0,60)) # kHz
        legend(loc='upper left')
        savefig('/Users/kai/Documents/Acer_mirror/Documents/00_00_Thesis/thesis_v0-0-0-1_bitbucket/ch2/ch2_images/phase_modulation_demo.pdf')
        # savefig('/home/kai/Documents/00_ZoteroManaged/MRO_Sensitivity_2017/text_ceri-life-R500_IEEE_PJ/journal_review/modulated_carrier.png')
        show()

    def carrier_space_modulated(self):
        # Some effects occuring here:
        # 1) if the f_s is increased then this means that less fringes can be captured and the carrire f_D
        # reduces
        # This neglects here that at higher scanning frequencies f_s the actual mirror speed increases and
        # the Doppler effect will increase the fringe frequencies.
        # But at the moment this part is usefule to investigate the direct relations of frequency modulation.
        warnings.warn('This is not yet fully developed.')
        NS = 10000
        f_D = 60e3
        f_s = 152 # scan frequency
        t_s2 = linspace(0,1/f_s/2,NS) # scan time half cycle
        t_s2_mod = cos(2*pi*f_s*linspace(0,1/f_s,NS)/2)
        print(f_s*max(t_s2))
        A = f_D
        d_f = abs(0 - f_s)
        ph = A/f_s * sin(2*pi*f_s*t_s2 - pi)

        carrier = (exp(1j * (2 * pi * f_D * t_s2)))
        carrier_mod = (exp(1j * (2 * pi * f_D * t_s2_mod)))

        figure(tight_layout=True)

        subplot(411)
        plot(carrier,label='carrier')
        legend()

        subplot(412)
        plot(carrier_mod,label='carrier mod')
        plot(t_s2_mod,label='mod phase')
        legend()

        subplot(413)
        plot(unwrap(angle((carrier_mod))),label='carrier phase')
        legend()

        subplot(414)
        plot(fftfreq(n=NS,d=1/NS/2),abs((fft(real(carrier)))))
        plot(fftfreq(n=NS,d=1/NS/2),abs((fft(real(carrier_mod)))))
        xlim((0,1500))
        show()

    def gauss_heterodyne_demo(self):
        N = 20000  # buffer size
        SR = 20e6  # sample rate (seconds)
        wavelength = 1330e-9  # meter
        wavelengthBW = 60e-9
        k = 2*pi/wavelength

        space = linspace(-0e-6, 2*1.33e-6, N).astype(complex)  # spatial range in [m]

        figure(tight_layout=True)
        subplot(211)
        title('Source carrier')
        xlabel('Space ($\mu m$)')
        ylabel('Amplitude (arb)')
        sig1 = exp( 1j * ( k * space - pi/2))
        sig2 = exp( 1j * ( 1.5 * k * space ) )
        plot(space*1e6,(sig1))
        plot(space*1e6,(sig2))

        subplot(212)
        title('Heterodyne mixed')
        xlabel('Space ($\mu m$)')
        ylabel('Amplitude (arb)')
        # plot(space*1e6, (abs(sig1+sig2))**2, lw=1.0)
        det_sig = real(sig1*conj(sig2))
        plot(space*1e6, det_sig, lw=1.0)

        savefig('gauss_heterodyne_demo.pdf')
        show()

    def gauss_heterodyne_demo_2(self):
        '''
        Some experiments to play around to sum, multipyl, or convolute to waves.
        :return:
        '''
        N = 20000  # buffer size
        SR = 20e6  # sample rate (seconds)
        wavelength = 1330e-9  # meter
        wavelengthBW = 60e-9
        k = 2*pi/wavelength
        FWHM = 2 * log(2) / pi * wavelength ** 2 / wavelengthBW  # [m]
        # Currently taking care by guessing the source BW.
        sigma_fwhm = FWHM / sqrt(2 * log(2))
        # sigma_fwhm = FWHM / sqrt(2 * log(2))
        dk = 1/sigma_fwhm
        l0 = 60e-6
        space = linspace(-0e-6, 100*1.33e-6, N).astype(complex)  # spatial range in [m]

        figure(tight_layout=True)
        subplot(211)
        title('Source carrier+Gauss')
        xlabel('Space ($\mu m$)')
        ylabel('Amplitude (arb)')
        sig1 = exp( 1j * ( k * space - pi/2)) * exp( - ((space - l0) * dk)**2)
        print('FWHM (meas, m)',self.measure_FWHM(space,abs(hilbert(real(sig1)))))
        sig2 = exp( 1j * ( 2* k * space - pi/2) ) * exp( - ((space - l0 - 0e-6) * dk)**2)
        plot(space*1e6,(sig1),lw=1,label='source')
        plot(space*1e6,(sig2),lw=0.5,label='refrence')
        legend()

        subplot(212)
        title('Heterodyne mixed')
        xlabel('Space ($\mu m$)')
        ylabel('Amplitude (arb)')
        det_sig = real(sig2*conj(sig1))
        # det_sig =convolve(sig1,sig2,mode='same')
        print('FWHM (meas, m)',self.measure_FWHM(space,abs(hilbert(real(det_sig)))))
        # Bouma 2001: p43(53) eq.4
        print(sig1)
        print(sig2)
        plot(space*1e6, det_sig, lw=1.0)
        # plot(fftfreq(n=N,d = 1/space*1e6), abs((fft(abs((sig1)+(sig2)))**2)), lw=1.0)

        savefig('gauss_heterodyne_demo.pdf')
        show()

    def gauss_heterodyne_demo_3(self):
        '''
        Generate a wave based on the FFT using BW = 60 nm that should come out with 13 or 6.5 um.
        Please acknowledge that the increase of N alone does not per se improve the sample rate
        of the time-domain signal converted from spectrum to time.
        To compensate it would be required to reduce the bandwidth and shift the frequency and rescale
        the time base of the time domain signal to the time corresponding to the frequency.
        :return:
        '''
        class FFT_sample_rate_experiment(Theory_Gauss_signal):
            def __init__(self,N=1000):

                self.N = N # buffer size
                self.SR = 20e6  # sample rate (seconds)
                self.scale = 1e6
                self.wavelength = 1330e-9 #* scale  # meter
                self.wavelengthBW = 60e-9
                self.k = 2*pi/self.wavelength
                self.k_rng = linspace(0,2*self.k/self.N,self.N)
                self.FWHM = 2 * log(2) / pi * self.wavelength ** 2 / self.wavelengthBW  # [m]
                # Currently taking care by guessing the source BW.
                self.sigma_fwhm = self.FWHM / sqrt(2 * log(2))
                # sigma_fwhm = FWHM / sqrt(2 * log(2))
                self.dk = 1/self.sigma_fwhm
                self.space = linspace(-0e-6, 100*1.33e-6, self.N).astype(complex)  # spatial range in [m]

            def gauss_in_range_2pi(self):
                '''
                Key points to acknowledge in this experiment
                a)
                :return:
                '''
                figure(tight_layout=True,num='2 pi',figsize=(12,8))
                ax = Axes
                ax = subplot(211)
                self.__init__(N=1000)
                N = self.N
                fwhm = 0.1/N #0.1 # 0.05
                k =    1/N #1.0 # 0.5
                title('Spectrum of source fwhm: {:g}, k: {:g}, N: {:g}'.format(fwhm,k,N))
                xlabel('k (1/.)')
                grid(True)
                ax.set_yticks(arange(0,1.1,0.1))


                k_range_len = 0.02 # 2*pi
                k_range = linspace(0,k_range_len,N)
                sigma = fwhm/2/sqrt(2*log(2))
                S = exp(-(k_range-k)**2 / (2*sigma**2) )
                # print('fwhm',self.measure_FWHM(wn_r,S))
                plot(k_range,S,'.')

                ax = subplot(212)
                grid(True)
                ax.set_yticks(arange(0,1.1,0.1))

                I = (fft(S))

                inv_rng = fftfreq(N,1/k_range[-1])
                A = sqrt(2*pi*N)/2 # this is not accurate, a guess. Bette find the right relation.
                plot(inv_rng[1:-1], (I/max(I))[1:-1],'.',lw=1,ms=1)

                savefig('Spectrum_vs_sr_fwhm{:g}_k{:g}_N{:g}.jpg'.format(fwhm,k,N))
                # Conclusion:
                # It should be noted that for extreme cases the input sample rate is little and the sample rate
                # after FFT is huge or vice versa.
                # E.g. the FFT here generates a huge sample space, while the actual signal is squeezed onto only
                # a few samples.
                # For example if the original sample range is 100 and fwhm = 1 then the fft range is -50 ... 50
                # and the

            def gauss_in_range_2pi_measure(self):
                '''
                As in gauss_in_range_2pi but we add measurements for spectrum and TD signal.
                1) if we plot in sample space source and results we can observe the change of the sample rate but
                due to conversion factors the bandwidth and frequency is kept constant.
                2) Observe the same condition as in 1 using a defined k_range and t_range!
                3) The FWHM in k_space is sigma_k = FWHM/2/sqrt(2*log(2))
                4) The FWHM measured in t_space confirms.
                5) Measure FWHM on Gauss in k_space!
                WORK
                6) Calculate the FWHM by constructing envelope in t_space!
                What must be k to obtain BW in TD?
                :return:
                '''
                self.__init__(N=20000)
                N = self.N
                k_range_unit  = 'mm'
                k_range_scale = {'m':1, 'mm':1e3}[k_range_unit]

                # fwhm =  self.wavelengthBW*k_range_scale
                fwhm = self.wavelengthBW*1e6
                # print('fwhm:',fwhm)
                # fwhm = 10 c [m/s] wl [m] --> f [1/s] = c/wl [m/s/m]->[1/s]
                # sigma_k = fwhm/2/sqrt(2*log(2))
                sigma_k = 2*pi/fwhm
                sigma_k = 2.54
                f_0 = speed_of_light/self.wavelength   # frequency [100/m] [100/s]
                print('f_0:',f_0,'1/s',' (',f_0/1e12,'THz)')
                k_0 = 2*pi/self.wavelength/k_range_scale
                print('k_0',k_0,'1/{}'.format(k_range_unit))
                k= k_0

                def correct_fwhm_and_k(sigma_k,k,N):
                    '''
                    Correct the fwhm and k according to the chosen sample rate.
                    :param fwhm:
                    :param k:
                    :param N:
                    :return:
                    '''
                    return sigma_k/N, k/N

                # Currently with scaling the x axis with k_range the correction may not be required.
                # TODO test if the sampling rate still is valid.
                # sigma_k,k = correct_fwhm_and_k(sigma_k,k,N)



                # k_range_len = 1 # matches sample length
                k_range_len = 2*2*pi/self.wavelength/k_range_scale
                k_range = linspace(0,k_range_len,N)
                t_range = linspace(0,1/k_range_len,N)
                print('k_rng:',k_range[-1],'1/',k_range_unit)

                figure(tight_layout=True,num='2 pi',figsize=(12,8))
                ax = Axes

                ax = subplot(211)
                title('Spectrum of source fwhm: {:g}, k: {:g}, N: {:g}'.format(fwhm,k,N))
                xlabel('k (1/{})'.format(k_range_unit))
                grid(True)
                ax.set_yticks(arange(0,1.1,0.1))

                S = exp( -(k_range - k) ** 2 / ( 2 * sigma_k ** 2) )
                print('fwhm_k_meas',self.measure_FWHM(k_range,S),'1/{}'.format(k_range_unit))
                plot(k_range, S,'.')


                ax = subplot(212)
                grid(True)
                ax.set_yticks(arange(0,1.1,0.1))
                ax.set_xlabel('space ({})'.format(k_range_unit))
                I = fftshift(fft(S))

                inv_rng = fftfreq(N,1/k_range[-1])
                A = sqrt(2*pi*N)/2 # this is not accurate, a guess. Bette find the right relation.
                plot(t_range, (I/max(I)),'.-',lw=1,ms=1)
                print('fwhm_td_meas',self.measure_FWHM(t_range,abs(hilbert(real(I/max(I))))),'m')

            def gauss_in_by_tutorial(self):
                pass
        # FFT_sample_rate_experiment().gauss_in_range_2pi()
        # FFT_sample_rate_experiment().gauss_in_range_2pi_measure()
        show()


class Runner(object):

    # GaussTheory(run=True)
    # GaussGenerator(run=False)
    # GaussGenerator().compare_errors()
    # GaussGenerator().compare_sum()


    # Tomlins().masterThesis_Patricia_Carvalho() # *
    # Tomlins().masterThesis_Patricia_Carvalho_analysis() # **
    # Tomlins().tomlins_light_source_spectrum() # ***
    # Tomlins().tomlins_source_freq_Gauss_envelope() # ***
    # TomlinsSimulation_v1() #psf_error
    # TomlinsSimulation(run=True)
    TomlinsSimulation().sample_points_vs_noise()

    # Theory_Gauss_signal().gaussian_FWHM()
    # Theory_Gauss_signal().gaussian_FWHM_B()
    # Theory_Gauss_signal().carrier()
    # Theory_Gauss_signal().loss_due_to_modulation()
    # Theory_Gauss_signal().carrier_modulated() # ***
    # Theory_Gauss_signal().carrier_frequ_change() # ***
    # Theory_Gauss_signal().carrier_space_modulated()
    # Theory_Gauss_signal().gauss_heterodyne_demo()
    # Theory_Gauss_signal().gauss_heterodyne_demo_2()
    # Theory_Gauss_signal().gauss_heterodyne_demo_3()
    pass

Runner()
