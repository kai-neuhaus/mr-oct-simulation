"""
Created on 14/05/2018; 23:17
@author: Kai Neuhaus
@email: k.neuhaus2@nuigalway.ie
IDE: PyCharm Community edition

To run some script go to the bottom and uncomment the desired line.

The script was tested with Python 3.6.3.

"""
from scipy import *
from scipy.constants import speed_of_light
from scipy.fftpack import *
from scipy.signal import *
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

class SimulationHelpers(object):
    def measure_FWHM(self, x, g):
        '''
        Measure FWHM by fitting a spline on any shape in g.
        Usually the shape should be a Gaussian envelope but other envelopes may
        produce some results as well, though then it is not a FWHM anymore.
        :param x:
        :param g:
        :return:
        '''
        spline = UnivariateSpline(x, g - max(g) / 2, s=0)
        rts = spline.roots()
        r1, r2 = rts.min(), rts.max()
        plt = UnivariateSpline(x, g , s=0)(x)
        return abs(r1-r2), plt, x

    def measure_FWHM_hr(self,x,y):
        '''
        Measure FWHM but make y real: y' = abs(hilbert(real(y))
        @param x:
        @param y:
        @return:
        '''
        env = abs(hilbert(real(y)))
        spline = UnivariateSpline(x,env-max(env)/2,s=0)
        rts = spline.roots()
        r1, r2 = rts.min(), rts.max()
        plt = UnivariateSpline(x,env,s=0)(x)
        return abs(r1-r2),plt,x

    def measure_FWHM_h(self,x,y):
        '''
        Measure FWHM by y' = abs(hilbert( y ))
        @param x:
        @param y:
        @return:
        '''
        env = abs(hilbert(y))
        spline = UnivariateSpline(x,env-max(env)/2,s=0)
        rts = spline.roots()
        r1, r2 = rts.min(), rts.max()
        plt = UnivariateSpline(x,env,s=0)(x)
        return abs(r1-r2),plt,x

class Sample(object):
    '''
    The Sample object is providing all functions to generate reflecting sample layers.
    E.g. initializing a sample by calling

    smp = Sample()

    creates by default a three layer structure with widts = [15, 60, 90] and refractive
    index = [1.001, 1.002, 1.003].

    To create your own structure you would call

    smp = Sample(ns = [1.001,1.1], z_widths = [100, 200])

    To plot the response of the sample response function along the sample z-axis you need
    also a light source

    src = Source()

    which generates by default a center wavelength of 1330 nm and a bandwidth of 60 nm.

    smp.plot_H_td(src)
    '''
    air = 1.0
    ns = None
    micrometer = 1e-6
    z_widths = None

    def __init__(self,ns=[1.001, 1.002, 1.003],z_widths=[15, 60, 90]):
        '''
        Initialize a sample layer structure
        @param ns:
        @param z_widths: in micrometer
        '''
        self.ns = array([self.air, *ns])
        self.z_widths = array(z_widths) * self.micrometer

    def __mul__(self, other):
        assert False, 'This is not implemented yet!'
        pass

    def kronecker_deltas(self,source,new_z_rng=None,verbose=False):
        '''
        The Kronecker deltas are actually just the r_j array - the reflectivities.
        **Return the Kronecker deltas**.\n\n
        FFT( Kronecker delta) = int( r_s * exp(2ikz_s)) = H.\n
        So FFT(Kronecker deltas) == H.\n
        so\n
        H = r * exp(2ikz_n) \n
        and\n
        FFT(H) = convolve( r , exp(2ikz_n)).\n
        \\\n
        According to Izatt, Choma p52\n
        r = sum( r * kd(z_S - z_Sn)) where kd(...) is the \delta for Kronecker delta.\n
        E_s = E_i * convolve(r , exp(2*i*k*z_Sn))\n
        then\n
        E_s = E_i * H

        :param z_rng: If given this range is used to place the Kronecker deltas into
        :return:  rs_kd, z_rng, z_rng_max
        '''
        _ = self
        SN = source.SN
        z_widths = _.z_widths * _.ns[0:-1]
        z_locs = z_widths.cumsum()
        # print('z_locs',z_locs)
        if verbose:
            print('kronecker_deltas.ns',_.ns)
            print('kronecker_deltas.z',_.z_widths)
            print('kronecker_deltas.z_locs',z_locs)
        cum_width = z_widths.sum()
        if type(new_z_rng) is list or type(new_z_rng) is ndarray:
            # assert cum_width * 4 <= max(new_z_rng), 'The new_z_rng is too small cum_width*2={}, new_z_rng={}'.format(cum_width*2,max(new_z_rng))
            z_rng_max = max(new_z_rng)
        else:
            z_rng_max = cum_width*4
        z_rng = linspace(-z_rng_max,z_rng_max,SN)
        getlocidx = interpolate.interp1d( [-z_rng_max,z_rng_max], [-SN, SN])

        rs_kd = zeros(SN) # create empty Kronecker delta array -> all zeros
        rjs = array([(np-nn)/(np+nn) for np,nn in zip(_.ns[0:-1],_.ns[1:])]).squeeze()
        # create a double sided Kronecker delta
        # rs_kd[getlocidx(-z_locs).astype(int)] = 1 * rjs
        try:
            rs_kd[getlocidx(z_locs).astype(int)] = 1 * rjs
        except ValueError as ve:
            print('z_locs',z_locs)
            print('z_rng_max',z_rng_max)
            raise(ValueError('z_locs',z_locs,'z_rng_max',z_rng_max))


        rs_kd = fftshift(rs_kd) # reverse to correct orientation.

        return rs_kd, z_rng, z_rng_max

    def plot_kronecker(self,source,do_plot=True,do_save=False):
        '''
        Plot the Kronecker delta based on positions z_widths and refractive indexes 'ns'.\s
        See more details below.

        :param do_plot:
        :param do_save:
        :return:

        **To get the Kronecker deltas call kronecker_deltas**.\n\n
        Principly FFT(Kronecker deltas) == H.\n
        so\n
        H = r * exp(2ikz_n) \n
        and\n
        FFT(H) = convolve( r , exp(2ikz_n)).\n
        \\\n
        According to Izatt, Choma p52\n
        r = sum( r * kd(z_S - z_Sn)) where kd(...) is the \delta for Kronecker delta.\n
        E_s = E_i * convolve(r , exp(2*i*k*z_Sn))\n
        then\n
        E_s = E_i * H
        '''
        rs_kd,z_rng,z_rng_max = self.kronecker_deltas(source)
        if do_plot:
            figure('kroneckre deltas',tight_layout=True)
            plot(z_rng*1e6,abs(rs_kd),'.-',lw=0.5)
            # xlim((0,z_rng_max*1e6))
            xlim((0,200))
            title('Kronecker delta')
            xlabel('z ($\mathrm{\mu m}$)')
            ylabel('field reflectivity $r_j$')
            if do_save: savefig('kronecker_deltas.pdf')

    def generate_H(self, source):
        '''
        Compute the sample response function H and return.\n
        **Please take note that FFT(H) = Kronecker deltas.**\n

        :return: H, z_rng

        The SRF is defined for one layer as (Ch2: eq 1.12) as\n
        H = r * exp(2*i*k*z_s) = r * exp(2*i*w/c*z_s).\n
        \\\n
        According to Izatt, Choma p53
        the sample field is calculated as\n
        E_S = E_i * convolve( r , exp(2*i*k*z_z) )\n
        E_S = E_i * H
        \\\n
        This function computes H for multiple layers and in principle means
        to track all interface positions z and the layers with refractive index n between.
        Therefore, a layer boundary at z[n] needs to account for all n[n-1]:\n
        n[n-1] = sum(n[0..n-1]*z[0..n] (see code).\n
        Hj = r_j * exp( 2*i*w/c * sum( n*z ) for all covering layers n, and z.\n
        H = sum( Hj )
        '''
        _ = self
        print('generate_H.ns',_.ns)
        print('generate_H.z',_.z_widths)

        r_j_f = lambda n1,n2: (n1-n2)/(n1+n2)
        src = source
        w_rng = src.w_rng
        SN = src.SN
        c = speed_of_light
        f_max = 2 * w_rng[-1]/2/pi # note here that we use 2 x the _.w_rng due to the double pass.
        ST = 1/f_max*SN #s
        z = ST * c # s * m/s == m
        z_rng = linspace(-z/2,z/2,SN)

        z_widths = _.z_widths * _.ns[0:-1] # correct with ref. index
        Z_j = z_widths.cumsum()
        rjs = array([(n1-n2)/(n1+n2) for n1,n2 in zip(_.ns[0:-1],_.ns[1:])])
        Hj = []
        for r_j,z_j in zip(rjs,Z_j):
            Hj.append(r_j * exp( 1j * 2 * w_rng / c * z_j))
        H = sum(Hj,axis=0)

        return H, z_rng

    def generate_SRM_PM(self,source,spacing=[]):
        '''
        Generate multiple reflecting scanning layers.
        In principle the same as the sample response function, with the addition
        to allow to set a spacing.
        @param source:
        @param spacing: array to zero out values to create a spacing.
        @return: H, z_rng
        '''
        assert not any(spacing), 'Spacing is not used yet!'
        _ = self
        # print('generate_SRM_PM.ns',_.ns)
        # print('generate_SRM_PM.z',_.z_widths)

        r_j_f = lambda n1,n2: (n1-n2)/(n1+n2)
        src = source
        w_rng = src.w_rng
        SN = src.SN
        c = speed_of_light
        f_max = 2 * w_rng[-1]/2/pi # note here that we use 2 x the _.w_rng due to the double pass.
        ST = 1/f_max*SN #s
        z = ST * c # s * m/s == m
        z_rng = linspace(-z/2,z/2,SN)

        z_widths = _.z_widths * _.ns[0:-1] # correct with ref. index
        Z_j = z_widths.cumsum()
        rjs = array([(n1-n2)/(n1+n2) for n1,n2 in zip(_.ns[0:-1],_.ns[1:])])
        Hj = []
        for r_j,z_j in zip(rjs,Z_j):
            Hj.append(r_j * exp( 1j * 2 * w_rng / c * z_j))
        H = sum(Hj,axis=0)

        return H, z_rng


    def plot_H_td(self, source, do_plot=True, do_save=False, tukey_alpha=False):
        '''
        Generate the H according to Tomlins 2005, eq.11.\n
        H(w) is the spectral modulation depending on depth!\n
        \\\n
        more notes below.

        :param do_plot:
        :param do_save:
        :return:

        Tomlins states for a TD-OCT\n
        I(d_z) = 1/4 * int( S * (H**2 + 1)) + 1/2* \int(S * H * exp( PHI(z) ))\n
        Take note that the integrals convert the spectrum into fields and then intensity fractions\n
        are the results.
        '''

        H, z_rng = self.generate_H(source)
        SN = source.SN
        if tukey_alpha is type(number):
            apply_tukey = True
        else:
            apply_tukey = False
        if apply_tukey:
            # tukey_alpha = 0.9 was tested but the
            print('Apply Tukey window for plot_H.')
            tukwin = tukey(len(H),alpha=tukey_alpha,sym=False)
            H = tukwin*H
        if do_plot:
            figure(num='sample resp fun',tight_layout=True)
            plot(z_rng, abs(fftshift(fft(H)))/SN,'.-',lw=0.5)
            xlim((0,200))
            title('Sample response function FFT(H)')
            xlabel('z ($\mathrm{\mu m}$)')
            ylabel('reflectivity $r_j$')

            if do_save:
                savefig('sample_response_function.pdf')

    def plot_H_freqDom(self,source,do_plot=True,do_save=True):
        '''
        Please take note that the SRF alone is only of limited use although the FFT can be used
        if it consistent with the Kronecker deltas.
        :param do_plot:
        :param do_save:
        :return:
        '''

        H, z_rng = self.generate_H(source)
        if do_plot:
            figure(num='sample resp fun FD',tight_layout=True)
            subplot(211)
            plot( H )#,'.',ms=1.5)
            title('$H(\omega)$')
            subplot(212)
            plot( H )#,'.',ms=1.5)
            xlim((0,200))
            title('$H(\omega)$(zoom)')

            if do_save:
                savefig('sample_response_function_freqDom.pdf')

class Source(SimulationHelpers):
    '''
    Source describes the spectral content of the light source.
    '''
    SN = 2 ** 16
    c = speed_of_light

    range_factor = 50 # How much more relative to the CWL of range should be generated?
    # This has inverse impact on the freq range or spatial range.
    # This is somewhat similar to zero padding sufficient sample points.

    # Initialize all values to None to make them to exist here.
    # Values are set below within __init__.
    CWL = None
    BW = None  # 50 nm
    FWHM_psf = None
    sigma = None

    WL_rng_max = None
    WL_rng = None  # wavelength range to use for
    WLR = None

    f_0 = None

    w_0 = None
    FWHM_w = None
    sigma_w = None
    w_rng_max = None
    w_rng = None

    SRF = 'SRF'
    KRN = 'Kron'

    def __init__(self,center_wavelength=1330e-9,bandwidth=60e-9):
        '''Print values only if object is used and initialized.'''
        self.CWL = center_wavelength
        self.BW = bandwidth
        self.FWHM_psf = 2 * log(2) / pi * self.CWL ** 2 / self.BW
        # note here that we use 2 x the _.w_rng due to the double pass.
        # todo Perform this for frequency here as well.
        self.sigma = self.BW / sqrt(8 * log(2))

        self.WL_rng_max = self.CWL + self.CWL * self.range_factor
        self.WL_rng = linspace(0, self.WL_rng_max, self.SN)  # wavelength range to use for
        self.WLR = self.WL_rng

        self.f_0 = self.c / self.CWL

        self.w_0 = 2 * pi * self.c / self.CWL
        self.FWHM_w = 2 * pi * self.c * self.BW / self.CWL ** 2
        # note here that we use 2 x the _.w_rng due to the double pass.
        self.sigma_w = self.FWHM_w / sqrt(8 * log(2))
        # w_rng = linspace(w_0 - sigma_w*5, w_0 + sigma_w*5, SN)
        self.w_rng_max = self.w_0 + self.w_0 * self.range_factor
        self.w_rng = linspace(0, self.w_rng_max, self.SN)

        print('CWL',self.CWL)
        print('BW',self.BW)
        print('FWHM_z {:1.3f} um'.format(self.FWHM_psf * 1e6))
        print('f_0 {} THz'.format(self.f_0 * 1e-12))
        print('FWHM_w {:1.3e} rad/s ({:1.0f} THz)'.format(self.FWHM_w, self.FWHM_w / 2 / pi * 1e-12))
        print('w_0 {} rad/s'.format(self.w_0))
        print('sigma_w {} rad/s'.format(self.sigma_w))

        print('PSF th:',2*log(2)/pi*self.CWL**2/self.BW)

    def spectrum(self, w, w0, s_w, normalize='spectrum'):
        '''
        Generate spectrum.\n
        According to Izatt, Choma p52 \n
        E_i = s(k,w) * exp(i(kz - wt)).\n
        \\\n
        This function returns\n
        S(k,w) = FFT(E_i)
        The parameters must be set in the class header Source.
        :param w: Frequency range in rad/s
        :param w0: Center frequency in rad/s
        :param s_w: Sigma in rad/s
        :param mode: SRF or Kron for SRF: sample response function or Kron: Kronecker delta.
        The Kronecker delta requires the source field to be converted into the spatial domain by sigma/sqrt(2*pi).
        :return: S, w, w0, s_w
        '''

        if 'spectrum' in normalize:
            norm_value = 1/sqrt(2 * pi / (s_w ** 2))
        elif 'probability' in normalize:
            # see http://mathworld.wolfram.com/GaussianFunction.html
            print('Normalize for probability.')
            norm_value= 1/s_w/sqrt(2*pi/2)
        elif 'liu' in normalize:
            print('Normalize acc. to Liu p28.')
            norm_value = sqrt(2 * pi / (s_w ** 2)) # see Liu p28
        else:
            print('No normalization.')
            norm_value = 1  # sqrt(s_w ** 2 / 2 / pi)
        # normalize = 1/sqrt(2*pi/s_w**2) # this generates a spectrum exactly 1

        # S = sqrt(2 * pi / s_w ** 2) * exp(-(w - w0) ** 2 / (2 * s_w ** 2)) * norm_value
        S = sqrt(2 * pi / s_w ** 2) * exp(-(w - w0) ** 2 / (2 * (s_w / sqrt(6 * log(pi))) ** 2)) * norm_value

        return S, w, w0, s_w


    def get_E_w_i(self):
        '''
        Return the source field in the spectral domain.
        :param mode: Source.SRF or Source.KRN this is currently required due to FFT double sided changing the z_rng.
        :return: E_w_i, z_rng
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

        S_w_w0 = self.spectrum(_.w_rng, _.w_0, _.sigma_w*2*sqrt(2) )[0]
        E_i = S_w_w0
        return E_i, z_rng

    def get_E_i_td(self,sigma_x = 1,w_x =1):
        '''
        Return the field from the source.
        :return: E_i, z_rng
        '''
        _ = self
        c = speed_of_light
        f_max = 2 * _.w_rng[-1]/2/pi # note here that we use 2 x the _.w_rng due to the double pass.
        print('f_max',f_max,'Hz')
        ST = 1/f_max*self.SN #s
        print('ST',ST,'s')
        z = ST * c # s * m/s == m
        zC = z/2
        print('zC',zC*1e6)
        print('z',z,'m')
        z_rng = linspace(0,z,self.SN)

        S_w_w0 = self.spectrum(_.w_rng, _.w_0*w_x, _.sigma_w*sigma_x )[0]
        E_i_td = fftshift(fft(S_w_w0))
        return E_i_td, z_rng

    def plot_E_i_td(self, do_plot=False, do_save=False, do_envelope_hilbert=False, do_envelope_absolute=False):
        '''
        According to Izatt, Choma p52\n
        E_i = s(k,w) * exp(i(kz - wt))\n
        this is also\n
        E_i = FFT( S(k,w) )
        Please take note that in this case the spatial spacing must be adapted by 2*pi
        or the bandwidth to obtain the right width of the PSF.
        Otherwise and currently the sigma = FWHM / sqrt( 2*pi * 8 log(2)) the 2*pi conversion
        is required to account for the FFT operation.

        :param mode: Possibly we should use src.SRF if the KRD is now double sided.
        :param do_save: Save a plot.
        :return: E_i_td, z_rng
        '''
        print('plot_E_i_td')
        _ = self
        c = speed_of_light
        f_max = 2 * _.w_rng[-1]/2/pi # note here that we use 2 x the _.w_rng due to the double pass.
        print('f_max',f_max,'Hz')
        ST = 1/2/f_max*self.SN #s
        print('ST',ST,'s')
        z = ST * c # s * m/s == m
        zC = z/2*1e6
        print('zC',zC)
        print('z',z,'m')
        z_rng = linspace(0,z,self.SN)
        um = 1e6

        S_w_w0,w_rng,w_0,s = self.spectrum(_.w_rng, _.w_0, _.sigma_w)
        E_i_td = fftshift(fft(S_w_w0))
        if do_plot:
            figure(num='fft of freq.',tight_layout=True)
            plot(z_rng*um,E_i_td,'.-',lw=0.5,label='TD signal of source')
            title('Time-domain wave of source.')
            def envelope_hilbert():
                spline=UnivariateSpline(z_rng,abs(hilbert(real(fftshift(fft(S_w_w0/S_w_w0.max()))))),s=0)
                plot(z_rng*um,spline(z_rng),'+',label='Hilbert')
                psf_h = self.measure_FWHM_h(z_rng*um, real(fftshift(fft(S_w_w0/S_w_w0.max()))))
                print('PSF',psf_h,'\nbased on Hilbert')
                print('PSF % error {:1.4f}'.format(abs(psf_h*1e-6 - self.FWHM_psf)/self.FWHM_psf*100 ))
            if do_envelope_hilbert:
                envelope_hilbert()

            def envelope_on_absolute_values():
                spline=UnivariateSpline(z_rng*um,abs(fftshift(fft(S_w_w0/S_w_w0.max()))),s=0)
                plot(z_rng*um,spline(z_rng*um),label='Univariate spline')
                psf_s = self.measure_FWHM(z_rng*um, abs(fftshift(fft(S_w_w0/S_w_w0.max()))))
                print(psf_s)
                print( abs(psf_s*1e-6 - self.FWHM_psf)/self.FWHM_psf*100 )
            if do_envelope_absolute:
                envelope_on_absolute_values()

            xlabel('z (um)')
            xlim(array([zC-25,zC+25]))
            legend()
            if do_save:
                savefig('source_{:1.0f}nm_rf{:3.0f}_SN{}_space_z.pdf'.format(_.CWL*1e9,_.range_factor*100,_.SN))

        return E_i_td,z_rng

    def __mul__(self, other):
        assert False, 'This is not implemented yet!'
        pass

    def plot_Freq(self,do_save=False):
        _ = self
        S_w_w0 = self.spectrum(_.w_rng, _.w_0, _.sigma_w)[0]
        figure(num='frequency',tight_layout=True)
        plot(_.w_rng/2/pi*1e-12, S_w_w0/S_w_w0.max(),'.-',lw=0.5,label='$\lambda_0$={:1.0f} nm, $\Delta\lambda$ = {:1.0f} nm'.format(_.CWL*1e9,_.BW*1e9))
        # stem(_.w_rng/2/pi*1e-12, S_w_w0/S_w_w0.max(),basefmt=' ',label='$\lambda_0$={:1.0f} nm, $\Delta\lambda$ = {:1.0f} nm'.format(_.CWL*1e9,_.BW*1e9))
        sw = 5
        xlim(array([_.w_0-_.sigma_w*sw,_.w_0+_.sigma_w*sw])/2/pi*1e-12)
        grid(True)
        xlabel('Frequency (THz)')
        ylabel('Power vs. frequency (a.u.)')
        title('Plotted with {} sigma frequency width,\n $f_{{max}}$={:1.0f} THz'.format(sw,_.w_rng[-1]/2/pi*1e-12))
        legend(loc='upper right')
        if do_save:
            savefig('source_{:1.0f}nm_rf{:3.0f}_SN{}_freq.pdf'.format(_.CWL*1e9,self.range_factor*100,_.SN))

    def plot_WL(self,do_save=False):
        _ = self
        S_WLR_CWL = self.spectrum(_.WLR, _.CWL, _.sigma/2)[0]
        figure(num='wavelength',tight_layout=True)
        plot(_.WLR*1e9, S_WLR_CWL ,'.-',lw=0.5,label='$\lambda_0$={:1.0f} nm, $\Delta\lambda$ = {:1.0f} nm'.format(_.CWL*1e9,_.BW*1e9))
        # stem(WLR*1e9, S_WLR_CWL/S_WLR_CWL.max(),basefmt=' ',label='$\lambda_0$={:1.0f} nm, $\Delta\lambda$ = {:1.0f} nm'.format(CWL*1e9,BW*1e9))
        sw = 5 # show times more than the BW
        xlim(array([_.CWL-_.sigma*sw,_.CWL+_.sigma*sw])*1e9)
        grid(True)
        xlabel('Wavelength (nm)')
        ylabel('Power vs. wavelength (a.u.)')
        title('Plotted with {} sigma wavelength width,\n $\lambda_{{max}}$={:1.0f} nm'.format(sw,_.WLR[-1]*1e9))
        legend(loc='upper right')
        if do_save:
            savefig('source_{:1.0f}nm_rf{:3.0f}_SN{}_WL.pdf'.format(_.CWL*1e9,self.range_factor*100,_.SN))

    def plot_circFreq(self,do_save=False):
        _ = self
        S_w = self.spectrum(_.w_rng, _.w_0, _.sigma_w)[0]
        figure(num='circular frequency',tight_layout=True)
        plot(_.w_rng*1e-12, S_w,'.-',lw=0.5,label='$\lambda_0$={:1.0f} nm, $\Delta\lambda$ = {:1.0f} nm'.format(_.CWL*1e9,_.BW*1e9))
        # stem(WLR*1e9, S_WLR_CWL/S_WLR_CWL.max(),basefmt=' ',label='$\lambda_0$={:1.0f} nm, $\Delta\lambda$ = {:1.0f} nm'.format(CWL*1e9,BW*1e9))
        sw = 5 # show times more than the BW
        xlim(array([_.w_0-_.sigma_w*sw,_.w_0+_.sigma_w*sw])*1e-12)
        grid(True)
        xlabel('$\omega$ (T rad/s)')
        ylabel('Power vs. $\omega$ (a.u.)')
        title('Plotted with {} sigma wavelength width,\n range {}$\\times${:1.0f} T rad/s'.format(sw,_.range_factor,_.w_0*1e-12))
        legend(loc='upper right')
        if do_save:
            savefig('source_{:1.0f}nm_rf{:3.0f}_SN{}_omega.pdf'.format(_.CWL*1e9,self.range_factor*100,_.SN))

class Simulate_TD(SimulationHelpers):
    def __init__(self):

        rcParams['text.usetex']=True
        rcParams['font.size']=24
        rcParams['lines.linewidth']=0.5

    def run_simu_TD(self):
        um = 1e6
        c = speed_of_light
        # Simulate for Tomlins data
        smp = Sample(ns=[1.3,1.5,1.0],z_widths=[5,15,30])
        src = Source(center_wavelength=800e-9, bandwidth=50e-9)

        E_t_i, ez_rng = src.get_E_i_td( sigma_x=(4*sqrt(2)), w_x=2)
        E_t_i /= E_t_i.max()

        figure(tight_layout=True)
        plot(ez_rng*um, real(E_t_i)),title('Source field')
        rs_kd, kz_rng, z_rng_max = smp.kronecker_deltas(src,new_z_rng=ez_rng)
        H, hz_rng = smp.generate_H(src)
        # correct the SRF frequency powers
        alpha = 1.0 # 0.48 is only required if we compare the delta function
        win = tukey(len(H), alpha=alpha, sym=False)
        SN = src.SN
        H = win * H

        figure(tight_layout=True)
        plot(kz_rng*um, abs(rs_kd),label='KRN')
        plot(hz_rng*um, abs(fftshift(fft(H)))/src.SN,label='SRF')
        plot(hz_rng*um,win)
        ylabel('Intensity (arb.)')
        xlabel('Displacement in air (\\textmu m)')
        # xlim((0, 80))

        E_t_s = convolve(E_t_i, rs_kd, mode='same')

        figure(tight_layout=True)
        title('$E(t)_S$ Kronecker delta')
        kz_rng = kz_rng*um
        plot(kz_rng, real(E_t_s)) #position OK
        ylabel('Reflectivity (n.u.)')
        xlabel('Displacement in air (\\textmu m)')
        xlim((0,80))
        # ylim((-40,40))

        # measure one layer PSF
        def measure():
            E_t_s[0:36500] = 0
            E_t_s[37800:] = 0
            plot(kz_rng, real(E_t_s))
            psf, plt, x = self.measure_FWHM_hr(kz_rng, E_t_s)
            print('PSF kr', psf) # PSF OK, Frequ. OK
            plot(x, plt, label='psf_kr= {:1.6} $\mathrm{{\mu m}}$'.format(psf)), \
            grid(True)
        measure()

        E_t_i2, ez_rng = src.get_E_i_td( sigma_x=2*sqrt(2), w_x=1)
        E_t_i2 /= E_t_i2.max()

        # divide by SN to normalize getting reflectivities
        E_t_s2 = convolve(E_t_i2, abs(fftshift(fft(H)))/src.SN, mode='same')
        figure(tight_layout=True)
        title('$E(t)_S$ Sample response function')
        plot(hz_rng*um, real(E_t_s2))
        ylabel('Intensity (arb.)')
        xlabel('Displacement in air (\\textmu m)')
        xlim((0,80))
        # ylim((-40,40))

        # measure one layer PSF
        def measure():
            E_t_s2[0:40000] = 0
            E_t_s2[43000:] = 0
            plot(hz_rng*um, real(E_t_s2))
            psf, plt, x = self.measure_FWHM_hr(hz_rng*um, E_t_s2)
            print('PSF srf', psf) # PSF OK, Frequ. OK
            plot(x, plt, label='psf_h= {:1.6} $\mathrm{{\mu m}}$'.format(psf)), grid(True)
        measure()

        E_w_i3, wz_rng = src.get_E_w_i()
        # E_w_i3 /= E_w_i3.max()
        E_t_i3 = abs(fftshift(fft(E_w_i3*H)))
        figure(tight_layout=True)
        title('FFT\{$E(\omega)_S \cdot H(\omega)$\}')
        plot(hz_rng*um, E_t_i3)
        ylabel('Intensity (arb.)')
        xlabel('Displacement in air (\\textmu m)')
        xlim((0,100))

        def measure():
            figure(tight_layout=True), title('Measure psf')
            plot(z_rng[47000:51000],(E_t_s[47000:51000])) # 47000-51000
            psf,plt,x = self.measure_FWHM_h(ez_rng[47000:51000],(E_t_s[47000:51000]))
            print('PSF',psf)
            plot(x,plt,label='psf= {:1.6} $\mathrm{{\mu m}}$'.format(psf)),grid(True)
            legend()
        # measure()

class Simulate_SD(SimulationHelpers):
    '''
    This basically means to take the E_s(w) and E_r(w) and compute the
    E_d(w) = E_S(w) + E_R(w).

    Because we have the H(w)*E_i(w) = E_S(w) and
    E_i(w) == E_r(w).

    '''
    def __init__(self):
        rcParams['text.usetex'] = True
        rcParams['text.latex.preamble']=['\\usepackage{siunitx}']
        rcParams['font.size'] = 20
        rcParams['lines.linewidth'] = 1.0
        rcParams['lines.markersize'] = 1.0

    def run_simu_SD(self):
        um = 1e6
        nm = 1e9
        c = speed_of_light
        # Simulate for Tomlins data
        smp = Sample(ns=[1.3, 1.5, 1.0], z_widths=[5, 15, 30])
        src = Source(center_wavelength=800e-9, bandwidth=50e-9)
        w_rng   = src.w_rng
        w_0     = src.w_0
        scale_sigma = 4
        sigma_w = src.sigma_w * scale_sigma
        S_w, w, w_0, s_w = src.spectrum(w_rng, w_0*2, sigma_w)

        WL_rng  = src.WL_rng * nm
        CWL     = src.CWL    * nm
        BW      = src.BW     * nm
        figure(tight_layout=True)
        title('Source spectrum $S(\omega)$')
        plot(WL_rng, S_w)
        xlabel('Wavelength (nm)')
        ylabel('Power (arb.)')
        xlim((CWL - 5 * BW, CWL + 5 * BW))

        H_w, hz_rng = smp.generate_H(src)
        E_w_s = S_w * H_w
        E_w_r = S_w
        E_w_d = E_w_r + E_w_s

        k_max = 2*pi/src.WL_rng[-1]
        with warnings.catch_warnings(record=True) as w:
            # Accept division by zero.
            # This has only impact on the image having missing data points where it happens.
            warnings.filterwarnings('always')
            k_rng = 2*pi/src.WL_rng
        k_0   = pi/src.CWL

        figure(tight_layout=True)
        title('Interferogram')
        plot(k_rng/1e3, abs(E_w_d)**2)
        view_rng = (k_0/1e3-.25e3,k_0/1e3+.25e3)
        xlim(view_rng)
        gca().set_xticks([view_rng[0],k_0/1e3,view_rng[1]])
        xlabel('k (1/mm)')
        ylabel('Power (arb.)')

        I_w_d = abs(E_w_d)**2
        I_z = abs(fftshift(ifft(I_w_d)))

        figure(tight_layout=True)
        title('A-line')
        plot(hz_rng*um, I_z, label='A-line')
        # scale the H_w by sqrt( samples ) to get matching powers
        plot(hz_rng*um, abs(fftshift(ifft(real(H_w))))/sqrt(src.SN), label='$r\delta(z)$')
        legend()
        xlim((0,80)),ylim((-0.00001,0.00045))
        xlabel('z (\SI{}{\micro\meter})')
        ylabel('Intestity (arb.)')

class Simulate_MRO(SimulationHelpers):
    '''
    Please be aware that all orders are simulated beginning to scan directly from the PM.
    That means all orders overlap over the full scan range and no segments are considered here.
    To simplify the processing at this stage no order segments were simulated.
    Otherwise a full A-line reconstruction algorithm must be included if desired to do so.

    In this case, within the for loop, the sample creation must be called as many times as
    orders of reflections are considered and the desired z-range must be adapted.

    This does not yet construct overlapping orders for which the Sample() object would need
    to be modified if required.

    Also the simulation can be arbitrarily scaled to any depth or scan width.
    No particular parameter is yet provided to adjust a physical spatial dimension to the
    pixel number.
    This can be improved in future versions.
    Currently, any scaling is done directly during plotting for whatever shape required.

    '''
    def simulate_MRO_O1(self):
        '''
        Simulate a single calibration line with no artifacts.
        @return:
        '''
        um = 1e6
        nm = 1e9
        c = speed_of_light
        src = Source(center_wavelength=1300e-9, bandwidth=60e-9)
        w_rng   = src.w_rng
        w_0     = src.w_0
        scale_sigma = 8
        sigma_w = src.sigma_w*scale_sigma

        S_w = src.spectrum(w_rng, w_0, sigma_w)[0]
        WL_rng  = src.WL_rng * nm
        CWL     = src.CWL    * nm
        BW      = src.BW     * nm

        E_t_i = fftshift(fft(S_w))
        E_t_r = E_t_i/E_t_i.max() # ideal scanning mirror

        # initial z range of TD source field
        f_max = 2*w_rng[-1]/2/pi
        ez = 4/f_max*src.SN * c # set scan range
        ez_rng = linspace(0,ez,src.SN)

        Sw1 = src.spectrum(w_rng, w_0 * 2 , sigma_w )[0] # order 1
        E_t_i1 = fftshift(fft(Sw1))

        bframe = zeros((1,src.SN))

        for z in arange(0,1500,10):
            rs_kd1, zr = Sample(ns=[1.04],z_widths=[0.0+z]).kronecker_deltas(src,new_z_rng=ez_rng)[0:2]

            E_t_s1 = convolve(E_t_i, rs_kd1, mode='same') # sample field
            E_t_d1 = convolve(E_t_r, E_t_s1, mode='same') # scanning
            E_t_d1 /= E_t_d1.max()

            a_line = log(abs((E_t_d1))**2)
            bframe = concatenate((bframe, [a_line]))

        figure(1)
        plot(zr,a_line)

        figure(gcf().number+1)
        imshow(bframe[:,src.SN//2:], aspect=220)#,vmin=-55,vmax=10*src.SN)
        colorbar(ax=gca())

    def simulate_MRO_O2(self):
        '''
        Simulate a single calibration line with artifacts summing additional parasitic sample signals.
        @return:
        '''
        um = 1e6
        nm = 1e9
        c = speed_of_light
        src = Source(center_wavelength=1300e-9, bandwidth=60e-9)
        w_rng   = src.w_rng
        w_0     = src.w_0
        sigma_w = src.sigma_w
        scale = 30

        S_w = src.spectrum(w_rng, w_0, sigma_w*4*scale/10)[0]
        # initial z range of TD source field
        f_max = 2*w_rng[-1]/2/pi
        ez = 100/f_max*src.SN * c # set scan range
        print('ez (um)',ez*1e6)
        ez_rng = linspace(0,ez,src.SN)
        WL_rng  = src.WL_rng * nm
        CWL     = src.CWL    * nm
        BW      = src.BW     * nm

        E_t_i = fftshift(fft(S_w))
        E_t_r = E_t_i/E_t_i.max() # ideal scanning mirror

        Sw1 = src.spectrum(w_rng, w_0, sigma_w * 4 * 2 * scale/10)[0] # order 1
        Sw2 = src.spectrum(w_rng, w_0, sigma_w * 4 * 3 * scale/10)[0]
        E_t_r1 = fftshift(fft(Sw1)) # virtual scanning field
        E_t_i2 = fftshift(fft(Sw2))

        bframe = zeros((1,src.SN))

        i1 = [1,2]
        print('i1',i1)
        i2 = [2,4]
        print('i2',i2)

        for z in arange(0,1500,10): # in um
            zn = [z/i for i in range(0,10)]
            acd1 = zn[i1[0]] - zn[i1[1]]
            acd2 = zn[i2[0]] - zn[i2[1]]

            rs_kd1, zr = Sample(ns=[1+1e-01],z_widths=[0.0+zn[1]*scale]).kronecker_deltas(src,new_z_rng=ez_rng)[0:2]
            rs_ac1, zr = Sample(ns=[1+1e-11],z_widths=[0.0+acd1*scale]).kronecker_deltas(src,new_z_rng=ez_rng)[0:2]
            rs_ac2, zr = Sample(ns=[1+1e-15],z_widths=[0.0+acd2*scale]).kronecker_deltas(src,new_z_rng=ez_rng)[0:2]

            E_t_s1 = convolve(E_t_i, rs_kd1, mode='same') # sample field
            E_t_ac1 = convolve(E_t_i, rs_ac1, mode='same')
            E_t_ac2 = convolve(E_t_i, rs_ac2, mode='same')

            E_t_sc1 = convolve(E_t_r, E_t_s1, mode='same') # scanning
            E_t_sac1 = convolve(E_t_r1, E_t_ac1, mode='same') # scanning 1st inter order
            E_t_sac2 = convolve(E_t_r1, E_t_ac2, mode='same') # scanning 2st inter order

            E_t_d = E_t_sc1 + E_t_sac1 + E_t_sac2

            a_line = log10(abs((E_t_d))**2)
            bframe = concatenate((bframe, [a_line]))


        figure()
        imshow(bframe[0:125,src.SN//2:61700], cmap='CMRmap', aspect=220,vmin=-40,vmax=0)
        colorbar(ax=gca())


# uncomment the desired operation

# Source().plot_E_i_td(do_plot=True)
# Source().plot_WL()
# Source().plot_circFreq()
# Source().plot_Freq()

Simulate_TD().run_simu_TD()

# Simulate_SD().run_simu_SD()

# Simulate_MRO().simulate_MRO_O1()
# Simulate_MRO().simulate_MRO_O2()


show() # keep here to see plots