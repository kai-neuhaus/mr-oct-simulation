"""
Created on 14/05/2018; 23:17
@author: Kai Neuhaus
@email: k.neuhaus2@nuigalway.ie
IDE: PyCharm Community edition

This is based on the file gauss_generation.py
"""

from scipy import *
from scipy.constants import speed_of_light
from scipy.fftpack import * # overwrite hilbert here!
# from scipy.signal import hilbert, tukey, convolve, correlate, resample, resample_poly, nuttall, blackmanharris, blackman, slepian
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

# set_printoptions(precision=3)
rcParams['font.size'] = 16
rcParams['font.family'] = 'Serif'

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

class Source(SimulationHelpers):
    '''
    Source describes the spectral content of the light source and is able to be summed with
    The sample spectrum.
    '''
    SN = 2 ** 16
    c = speed_of_light
    CWL = None
    BW = None  # 50 nm
    FWHM_psf = None
    sigma = None

    range_factor = 50 # how much more relative to the CWL of range should be generated?
    # this has inverse impact on the freq range or spatial range.
    #

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

    def spectrum(self, w, w0, s_w, mode, normalize='spectrum'):
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
        assert self.SRF in mode or self.KRN in mode, 'use mode Source.SRF or Source.KRN please.'
        #stop 0: normalize according to Liu
        #the normalization is not fully understood here because according to Liu this would generate
        # a spectrum with amplitude sqrt(2*pi/sig**2).
        # This would mean S(w-w0) which is the spectrum is normalized by
        # S(w-w0) = sqrt(2*pi/sigma**2) * exp(-(w-w0)**2 / (2*sigma**2))
        # However such operation generates and amplitude that is neither normal for the spectrum nor after FFT.
        # BUT S(w-w0) = 1/sqrt(2*pi/sigma**2) * exp(-(w-w0)**2 / (2*sigma**2)) is normal in the spectrum.
        # Further more according to https://en.wikipedia.org/wiki/Gaussian_function
        # for probability density functions it is given that
        # P(w-w0) = 1/(sigma * sqrt(2*pi) * 1/2 * exp(-(w-w0)**2 / (2*sigma**2))

        if 'spectrum' in normalize:
            warnings.warn('Normalize for spectrum.')
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

        assert self.SRF in mode or self.KRN in mode, 'mode must be either ''SRF'' or ''Kron'''
        #todo the Kron mode requires recalculation of the BW to obtain the right PSF after convolve
        #todo above todo is related to the single vs. double sided preparation of the Kronecker.
        if self.SRF in mode:
            S = sqrt(2 * pi / s_w ** 2) * exp(-(w - w0) ** 2 / (2 * s_w ** 2)) * norm_value
            return S, w, w0, s_w
        elif self.KRN in mode:
            # return sqrt(2 * pi / s_w ** 2) * exp(-(w - w0) ** 2 / (2 * (s_w/sqrt(8*log(2))) ** 2))
            S = sqrt(2 * pi / s_w ** 2) * exp(-(w - w0) ** 2 / (2 * (s_w/sqrt(6*log(pi))) ** 2)) * norm_value
            return S, w, w0, s_w

    # Alias for function name spectrum
    def S(self, w, w0, s_w, mode=SRF, normalize='spectrum'):
        '''
        :param w: w_rng
        :param w0: w_0
        :param s_w: sigma_w
        :param mode: At the moment use SRF always.
        If the Kronecker is double sided no range adjustment is required.
        :param normalize:
        :return: self.spectrum -> S, w, w0, s_w
        '''
        assert self.SRF in mode or self.KRN in mode, 'mode must be either Source.SRF or Source.KRN'
        return self.spectrum(w,w0,s_w,mode,normalize)

    def __S_w_w0(self,mode): # marked as obsolete
        '''
        According to Izatt, Choma p52 \n
        E_i = s(k,w) * exp(i(kz - wt)).\n
        \\\n
        This function returns\n
        S(k,w) = FFT(E_i)

        Generate source spectrum using frequency values in rad/s
        :param mode: SRF or Kron for SRF to correct for the right spatial range.\n
        #todo Check this sometimes if this is still required to correct between SRF and KRN
        #todo Correction for SRF and Kron may only be required if we create one of them single sided and the other double.
        :return:
        '''
        _ = self
        return self.spectrum(_.w_rng, _.w_0, _.sigma_w,mode), _.w_rng, _.w_0, _.sigma_w, mode

    def __S_WL_CWL(self,mode): # marked as obsolete
        '''
        According to Izatt, Choma p52 \n
        E_i = s(k,w) * exp(i(kz - wt)).\n
        \\\n
        This function returns\n
        S(k,w) = FFT(E_i)
        Generate source spectrum using wavelength parameters.
        Not implemented yet
        :param mode: Source.SRF or Source.KRN this is currently required due to FFT double sided changing the z_rng.
        :return:
        '''
        assert False, 'not implemented'

    def __S_f_f0(self,mode): # marked as obsolete
        '''
        According to Izatt, Choma p52 \n
        E_i = s(k,w) * exp(i(kz - wt)).\n
        \\\n
        This function returns\n
        S(k,w) = FFT(E_i)
        Generate source spectrum using frequency in 1/s
        :param mode: Source.SRF or Source.KRN this is currently required due to FFT double sided changing the z_rng.
        :return:
        '''
        assert False, 'not implemented'

    def get_E_w_i(self, mode=SRF):
        '''
        Return the source field in the spectral domain.
        :param mode: Source.SRF or Source.KRN this is currently required due to FFT double sided changing the z_rng.
        :return: E_w_i, z_rng
        '''
        assert self.SRF in mode or self.KRN in mode, 'use mode Source.SRF or Source.KRN please.'
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
        # todo calculation of z_rng here is somewhat odd. Can we relate this to the w_rng better?
        z_rng = linspace(0,z,self.SN)*1e6

        # I am not sure if this correction is still required as we prepare the KRN as
        # double sided signal now.
        # Correct spatial range using SRF mode. We expect PSF of about 15 um (1330,60).
        #See function S_w_w0 doc todo about the reason
        S_w_w0 = self.S(_.w_rng, _.w_0, _.sigma_w ,mode=mode)[0]
        E_i = S_w_w0
        return E_i, z_rng

    def get_E_i_td(self,mode=SRF,sigma_x = 1,w_x =1):
        '''
        Return the field from the source.
        :param mode: Source.SRF or Source.KRN this is currently required due to FFT double sided changing the z_rng.
        :return: E_i, z_rng
        '''
        assert self.SRF in mode or self.KRN in mode, 'use mode Source.SRF or Source.KRN please.'
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
        # todo calculation of z_rng here is somewhat odd. Can we relate this to the w_rng better?
        z_rng = linspace(0,z,self.SN)

        # I am not sure if this correction is still required as we prepare the KRN as
        # double sided signal now.
        # Correct spatial range using SRF mode. We expect PSF of about 15 um (1330,60).
        #See function S_w_w0 doc todo about the reason
        S_w_w0 = self.S(_.w_rng, _.w_0*w_x, _.sigma_w*sigma_x ,mode=mode)[0]
        E_i_td = fftshift(fft(S_w_w0))
        return E_i_td, z_rng

    def plot_E_i_td(self, mode, do_plot=False, do_save=False, do_envelope_hilbert=False, do_envelope_absolute=False):
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
        assert not mode==None and (self.SRF in mode or self.KRN in mode), 'Please use mode Source.SRF or Source.KRN'
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
        z_rng = linspace(0,z,self.SN)
        um = 1e6

        #So far we create a double sided Kronecker in which case all should be based on the SRF mode
        #meaning the time-domain length should create the right PSF.
        #Correct spatial range using SRF mode. We expect PSF of about 15 um (1330,60).
        #See function S_w_w0 doc todo about the reason
        S_w_w0,w_rng,w_0,s = self.S(_.w_rng, _.w_0, _.sigma_w, mode=mode)
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
        S_w_w0 = self.spectrum(_.w_rng, _.w_0, _.sigma_w)
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

    def plot_WL(self,mode,do_save=False):
        _ = self
        S_WLR_CWL = self.spectrum(_.WLR, _.CWL, _.sigma, mode=mode)[0]
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

class Heterodyne_principle(object):
    def __init__(self):
        source = Source()
        source.get_E_w_i()

class Sample(object):
    air = 1.0
    ns = None
    micrometer = 1e-6
    z_widths = None

    def __init__(self,ns=[1.001, 1.002, 1.003],z_widths=[15, 60, 90]):
        '''
        @param ns:
        @param z_widths: in micrometer
        '''
        self.ns = array([self.air, *ns])
        self.z_widths = array(z_widths) * self.micrometer

    def __mul__(self, other):
        assert False, 'This is not implemented yet!'
        pass

    def kronecker_deltas_manuscript(self,source):
        '''
        Manuscript version.
        The Kronecker deltas are actually just the r_j array - the reflectivities.\n
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

        :return:  rs_kd, z_rng, z_rng_max
        '''
        SN = source.SN
        # manuscript:lst:kroneckerarrayconstruction
        air = 1.0
        ns = array([air, 1.001, 1.002, 1.003])
        z_widths = array([15, 60, 90])
        z_widths = z_widths * ns[0:-1] #correct with ref. index
        z_locs = z_widths.cumsum()
        total_width = z_widths.sum()
        z_rng_max = total_width*2
        z_rng = linspace(0,z_rng_max,SN) #use sample length of E_i self.SN
        getlocidx = interpolate.interp1d( [0,z_rng_max], [0, SN])
        rs_kd = zeros(SN) # create empty Kronecker delta array -> all zeros
        rjs = array([(np-nn)/(np+nn) for np,nn in zip(ns[0:-1],ns[1:])]).squeeze()
        rs_kd[getlocidx(z_locs).astype(int)] = 1 * rjs # We indicate the Kron.-Delta by explicitly using the value 1
        # manuscript:lst:kroneckerarrayconstruction

        return rs_kd, z_rng, z_rng_max

    def rjs_manuscript(self):
        air = 1.0
        ns = array([air, 1.001, 1.002, 1.003])
        # manuscript:lst:kroneckerarrayfresnelvalues
        rjs = []
        for np,nn in zip(ns[0:-1],ns[1:]):
            rjs.append((np-nn/(np+nn)))
        # manuscript:lst:kroneckerarrayfresnelvalues


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

    def print_ns(self): print(self.ns)

    def get_z_s(self):
        '''
        Return the list with z positions defined.
        Call print_zs() to show them!
        :return:
        '''
        z_rng = array(self.z_widths * self.ns[0:-1])
        return self.z_widths, z_rng.cumsum()

    def get_r_j(self):
        '''
        Generate array with reflectivity r (small r !) based on given list of refractive indexes ns.
        Call print_ns() to show them!
        :return:
        '''
        r_j_f = lambda n1,n2: (n1-n2)/(n1+n2)
        _ = self
        return [r_j_f(n1,n2) for n1,n2 in zip(_.ns[0:-1],_.ns[1:])]

    def generate_H_manuscript(self, source):
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
        src = source

        _ = self
        w_rng = src.w_rng
        SN = src.SN
        c = speed_of_light
        f_max = 2 * w_rng[-1]/2/pi # note here that we use 2 x the _.w_rng due to the double pass.
        ST = 1/f_max*SN #s
        z = ST * c # s * m/s == m
        z_rng = linspace(-z/2,z/2,SN)*1e6

        # manuscript:lst:sampleresponsefunction
        air = 1.0
        ns = array([air, 1.001, 1.002, 1.003])
        z_widths = array([15, 60, 90])
        z_widths = z_widths * ns[0:-1] # correct with ref. index
        Z_j = z_widths.cumsum()
        rjs = array([(n1-n2)/(n1+n2) for n1,n2 in zip(ns[0:-1],ns[1:])])
        Hj = []
        for r_j, z_j in zip(rjs,Z_j):
            Hj.append(r_j * exp( 1j * 2 * w_rng / c * z_j))
        H = sum(Hj,axis=0)
        # manuscript:lst:sampleresponsefunction
        return H, z_rng

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


class Test_Source(object):
    '''
    This class does collect function calls that provide views on different mathematical representations
    of the source, such as a spectrum and as a field.
    '''
    def __init__(self):
        # Plot all representations of the source
        src = Source()
        # S(w-w0) for lambda / wave length
        src.plot_WL()
        # S(w-w0) for omega (w)
        src.plot_circFreq()
        # E_i = FFT( S )
        src.plot_E_i_td()

        # FFT( S ) should be the same as src.get_E_i()
        S,w_rng = src.S(src.w_rng, src.w_0, src.sigma_w, mode=src.SRF)[0:2]
        E_i,z_rng = src.get_E_i_td()
        figure(tight_layout=True)
        subplot(221)
        plot(w_rng,S),title('Spectrum\n $S(\omega-\omega_0)$')
        subplot(222)
        plot(z_rng, fftshift(fft(S))),title('$E_i$ field\n FFT(S)= s * exp(2ikz_s)')
        subplot(223)
        plot(z_rng,E_i),title('$E_i$ using\n get_E_i()')
# Test_Source()

class Test_Sample(object):
    '''
    Demonstrate the different variants to create the sample.
    '''
    def __init__(self):
        src = Source()
        sample = Sample()
        # use Ctrl + Q to get explanation: Izatt
        # Just show the layers
        sample.plot_kronecker(src)
        # sample.kronecker_deltas() #get the data

        # use Ctrl + Q: see Tomlins to use of H and S
        # Just show the layers by FFT(Kron)
        sample.plot_H_td(src)
        # sample.sample_response_H() #get the data
        # sample.plot_H_freqDom()
# Test_Sample()

class Test_E_s(object):
    '''
    The sample field is created based on the source field
    and based on the defined reflecting interfaces.

    For TD-OCT E_r = conv( E_i, KrD )

    However, the range of E_i is arbitrary and according to Tomlins
    E_r = E_i * exp(-i phi(D z))
    E_s = E_i * H = E_i conv KRD

    For TD-OCT the E_d is the sum of E_s + E_r whereas the frequency of
    E_r is changed due to the Doppler effect.

    This is different to FD-OCT where just both spectra are summed.

    The Doppler frequency can be calculated with the scan speed and range.

    '''

    def __init__(self):
        sample = Sample() #(ns=[1.3,1.5,1.0,1.0],z_widths=[5,15,30,10])
        src = Source()
        src.plot_WL(mode=src.SRF)
        w_rng = src.w_rng
        c = src.c

        S = src.S(src.w_rng, src.w_0, src.sigma_w, mode=src.SRF)[0]

        E_i, E_z_rng = src.get_E_i_td(mode=src.SRF)

        # The E_i has its own range and
        figure(num='Input Field',tight_layout=True)
        plot(E_z_rng, E_i,'.-',lw=0.5)
        title('E_i')
        xlabel('$z (\mathrm{\mu m}$)')

        smp = Sample()
        # smp.plot_H()
        # smp.plot_kronecker()
        rs_kd, z_rng, z_rng_max = smp.kronecker_deltas_manuscript(src)
        figure(num='Kronecker deltas Manuscript',tight_layout=True)
        plot(z_rng, abs(rs_kd))
        title('Kronecker deltas Manuscript')
        xlabel('z ($\mathrm{\mu m}$')

        # creating the deltas in the same E_i TD range using new_z_rng
        rs_kd, z_rng, z_rng_max = smp.kronecker_deltas(src,new_z_rng=E_z_rng*1e-6)

        # If the E_i is reflected it becomes E_r
        # This creates the proper frequency and PSF.
        E_s = convolve(E_i, rs_kd, mode='same')
        figure(num='$E_i \otimes \delta(z - z_n)$ ',tight_layout=True)
        E_d = convolve(E_i,E_s,mode='same')
        plot(z_rng*1e6,E_s)
        title('$(E_i \otimes \delta(z - z_n))^2$ ')
        xlabel('z ($\mathrm{\mu m}$)')

        # (A) Tomlins, Coma recreating the computation of the sample field E_s

class Test_Doppler(object):
    '''
    This is the frequency on the digitizer while the wavelength vs PSF remains that from the source.
    The interesting part here is that we still see the high frequency content of the source.

    This is the integral shown by Tomlins eq.8. but we can also simply apply a simple filter
    removing the high frequency or use FFT to remove the high frequency.
    '''
    def __init__(self):
        SN = 10000
        rng = linspace(-pi,pi,SN)
        w = lambda f: linspace(-f*pi,f*pi,SN)
        f_S = 4000 # source frequency
        f_D = 4005
        smp = sin(w(f_S))
        ref = sin(w(f_D))
        out = smp + ref
        det_high = (out * conj(out))
        figure(num='Doppler test',tight_layout=True)
        plot(det_high)

        # simulate slow detector
        det_freq = fftshift(fft(det_high))
        det_freq[0:4994] = 0
        det_freq[5006:] = 0
        det = ifft(fftshift(det_freq))

        figure(num='Doppler test',tight_layout=True)
        plot(det*2) #squaring is then the intensity spectrum of the TD signal


class Manuscript_kronecker_srf(object):
    '''
    Plots related to the manuscript
    '''
    sample = None
    src = None
    def __init__(self):
        sample = self.sample = Sample() #(ns=[1.3,1.5,1.0,1.0],z_widths=[5,15,30,10])
        src = self.src = Source()

        w_rng = src.w_rng
        c = src.c
        # src.plot_E_i_td(mode=src.KRN)
        # fig:kroneckerdeltas
        # sample.plot_kronecker(src,do_save=True)
        # fig:kroneckerdeltas:end
        self.plot_kronecker_field_manuscript(do_save=True)
        # fig:sampleresponsefunctionFFT
        # sample.plot_H_td(do_save=True)
        # fig:srffield
        # self.plot_H_field_manuscript(do_save=True)

        self.ns = array([1.01, 1.02, 1.03])
        self.z_widths = array([15, 60, 90])
        self.plot_kronecker_field_character_error(do_save=True,name_idx='01')
        self.plot_H_field_character_error(do_save=True,name_idx='01')

    def plot_kronecker_field_manuscript(self,do_save=False):
        # fig:kroneckrefields
        E_i_td, _z_rng = self.src.get_E_i_td(mode=self.src.KRN) # field in TD or spatial domain!
        gamma_z = abs(E_i_td) # E_i_td = FFT(E_i) !
        gamma_z /= gamma_z.max()
        rs_kd, z_rng, z_rng_max = self.sample.kronecker_deltas(self.src)
        E_s = convolve(gamma_z,abs(rs_kd),mode='same')
        figure(num='kroneckre fields', tight_layout=True)
        um = 1e6
        plot(z_rng*um, E_s, '-', lw=1.5)
        xlim((0, 200))
        title('Kronecker field\n$\gamma(z) \otimes r_j\delta(z_s - z_{s_n})$')
        xlabel('z ($\mathrm{\mu m}$)')
        ylabel('reflectivity')
        if do_save: savefig('kronecker_fields.pdf')
        # fig:kroneckrefields:end

    def plot_H_field_manuscript(self,do_save=False):
        E_i_td, _z_rng = self.src.get_E_i_td(mode=self.src.SRF) # field in TD or spatial domain!
        gamma_z = abs(E_i_td)
        gamma_z /= gamma_z.max()
        H, z_rng = self.sample.generate_H(self.src)
        H_td = abs(fftshift(fft(H)))/self.src.SN #correct for power after FFT
        E_s_td = convolve(gamma_z, H_td, mode='same')
        figure(num='SRF fields', tight_layout=True)
        um = 1e6
        plot(z_rng,E_s_td)
        xlim((0,200))
        title('FFT$(S(\omega)H(\omega))$')
        xlabel('z ($\mathrm{\mu m}$)')
        ylabel('reflectivity')
        if do_save: savefig('srf_field.pdf')

    def plot_kronecker_field_character_error(self,do_save=False,name_idx=''):
        ns = self.ns
        z_widths = self.z_widths
        # ns = array([1.001, 1.002, 1.003])
        # z_widths = array([15, 60, 90])
        sample = Sample(ns, z_widths)
        E_i_td, _z_rng = self.src.get_E_i_td(mode=self.src.KRN) # field in TD or spatial domain!
        gamma_z = abs(E_i_td) # E_i_td = FFT(E_i) !
        gamma_z /= gamma_z.max()
        rs_kd, z_rng, z_rng_max = sample.kronecker_deltas(self.src)
        E_s = convolve(gamma_z,abs(rs_kd),mode='same')
        figure(num='kroneckre fields', tight_layout=True)
        um = 1e6
        plot(z_rng*um, E_s, '-', lw=1.5)
        plot(z_rng*um, abs(rs_kd))
        xlim((0, 200))
        title('Kronecker field\n$\gamma(z) \otimes r_j\delta(z_s - z_{s_n})$')
        xlabel('z ($\mathrm{\mu m}$)')
        ylabel('reflectivity')
        if do_save: savefig('kronecker_fields_error_{}.pdf'.format(name_idx))

    def plot_H_field_character_error(self,do_save=False,name_idx=''):
        ns = self.ns
        z_widths = self.z_widths
        # ns = array([1.001, 1.002, 1.003])
        # z_widths = array([15, 60, 90])
        sample = Sample(ns, z_widths)
        sample.air = 1.0
        E_i, _z_rng = self.src.get_E_w_i(mode=self.src.SRF) # field in TD or spatial domain!
        E_i /= E_i.max()
        E_i_td, _z_rng = self.src.get_E_i_td(mode=self.src.SRF) # field in TD or spatial domain!
        # figure()
        gamma_z = abs((E_i_td))
        gamma_z /= gamma_z.max()
        H, z_rng = sample.generate_H()
        alpha = 0.48 # **********************
        tukwin = tukey(len(H), alpha=alpha, sym=False)
        H = tukwin * H

        H_td = abs(fftshift(fft(H)))/sample.SN #correct for power after FFT
        E_s_td = abs(fftshift(fft(E_i*H))) #correct for power after FFT
        # E_s_td = convolve( H_td,gamma_z, mode='same')
        figure(num='SRF fields error', tight_layout=True)
        um = 1e6
        # plot(gamma_z)
        # plot(z_rng,E_i*H)
        plot(z_rng,E_s_td)
        plot(z_rng,H_td)
        xlim((0,200))
        title('Sample response function\nFFT$(S(\omega)H(\omega))$')
        xlabel('z ($\mathrm{\mu m}$)')
        ylabel('reflectivity')
        if do_save: savefig('srf_field_error_{}.pdf'.format(name_idx))

class Test_TD_Doppler(object):
    '''
    Equally like we can now just take the source completly and combine it with the reference.
    The only question with this is that the coherence length in the TD for the reference is 13 um.
    So we assume that this reference is moved along the scan range and creates a flat sum that is
    summing with the sample.
    But this sum is nothing else than exp(-i2k Dz) for Dz the scan range.

    E_r(t) = sum( E_i(t,z))
    E_d(t) = E_s + E_r
    '''
    def __init__(self):
        '''
        Important
        =========
        Due to the slow scanning of the reference mirror a light wave will have passed at the
        point of detection multiple times.
        This means at each S{Delta}t multiple waves from the sample have interfered.
        If the mirror would be as fast as the speed of light, the interference would be only
        by a single wavefront.
        To some degree this is related to the noise rejection if becoming faster but would also
        reduce collection of reflection events.
        The detector itself remains sensitive to each wavefront but if no ballistic photon is
        available then the intensity of interference reduces.

        Other info
        ==========
        The E_r_t is somewhsat arbitrary in the lack of any digitizer.
        The simulation is already digitized and we can use any E_r_t.
        The only difference is that we can change the frequency slightly to simulate the
        Doppler effect.\n
        \\\n

        By theory the detector Doppler frequency is f_D = 2 v_M / \lambda.
        However, as we do not have any v_M the construction would cause only complexity
        to the simulation.\n
        \\\n

        Also, although in theory we can show the high frequency rejection due to the detector
        to simulate with the high frequency content we would need to include a sample range
        accomodating 225 THz vs. maybe 500 kHz of the interference frequency.\\\n
        \\\n

        On the other hand we can just directly use the high-frequency sum in a manner as it exists
        as the simulated signal and pretent we were able to detect it as such.\n

        List of TODOs::
            1. simulate E_r_t = exp( -2i w_0 /c z_rng )\n
            2. create two waves and sum\\\n
            3. confirm the Doppler frequency.

        The effect of scanning can be confirmed if we chose some suitable high vs. low scan rate.\\\n
        However, this is not practical for the general simulation.
        '''
        um = 1e6
        c = speed_of_light
        smp = Sample()
        src = Source()
        # z_scan_rng_max = 400*um
        # z_scan_rng = linspace(0,z_scan_rng_max,src.SN)

        # The Doppler effect
        # The Doppler effect can be demonstrated but is of no relevance for the simulation.
        # Because the simulation directly returns the sum signal, meaning a digital scanning
        # mirror would have not moved at all but captured all samples at once!
        # Therefore the Doppler frequency is exactly that the sum-signal is producing.

        # It would of course be possible to simulate the true scanning as well but this would mean
        # that the computation of the PSF needs to be recomputed based on the expected Doppler effect.
        # There is however not really any new knowledge to be gained out of that and therefore we
        # simulate as is.

        # Interesing nevertheless, is to see how the high-frequency can be converted into the low frequency,
        # which we demonstrate here for a coherent light source (laser).
        # In case the signal is complex the high frequency is removed
        k = 100
        doppler_factor = 1.1
        SN = 10000
        E_z_rng = linspace(0, pi, SN)
        # Create f_1
        E_r1_t = exp( -1j*2* (k+5) * E_z_rng )
        E_r1_tc = cos( 2* (k+5) * E_z_rng )
        figure(num='Source plain',tight_layout=True)
        plot(E_z_rng*1e6,E_r1_t),title('Ref scanned')
        # Create f_2
        E_r2_t = exp( -1j*(2*k * E_z_rng ))
        E_r2_tc = cos( 2* k * E_z_rng )

        # figure(num='Reference scanned',tight_layout=True)
        # plot(z_scan_rng*1e6,E_r2_t),title('Ref scanned')

        # sum sig1 and sig2
        E_rt_t = E_r1_t + E_r2_t
        E_rt_tc = E_r1_tc + E_r2_tc
        figure(num='Sum',tight_layout=True)
        plot(E_z_rng*um,E_rt_t),title('Sum')
        # plot(E_z_rng*um,E_rt_tc),title('Sum')

        # tukwin = tukey(src.SN,alpha=0.74) # avoid sharp frequency cut-off
        E_rt_t_cnj = (E_rt_t * conj(E_rt_t))
        E_rt_tc_cnj = (E_rt_tc * conj(E_rt_tc))
        figure(num='Cnj',tight_layout=True)
        plot(E_z_rng*um,E_rt_t_cnj),title('Cnj')
        # plot(E_z_rng*um,E_rt_tc_cnj),title('Cnj')

        # This is even not directly required if using complex signal.
        # simulate slow detector
        # E_rt_w = fftshift(fft(E_rt_t_cnj))
        # figure(num='S',tight_layout=True)
        # plot(abs(E_rt_w)),title('S')
        # # padding high frequencies to zero -> remove them -> slow detector
        # E_rt_w[0:32704] = 0
        # E_rt_w[32832:] = 0
        # E_rt_t_det = ifft(fftshift(E_rt_w))
        # figure(num='Det',tight_layout=True)
        # plot(abs(E_rt_t_det)),title('Det')


class Simulate_TD(SimulationHelpers):
    def __init__(self):
        # 1. Test the source field: OK
        #   * site note: the range factor is applied twice, CWL and then w_0
        #   * PSF measured manually: OK
        # 2: Test the Kroneckre positions: OK
        #   * Using new_z_rng = ez_rng the positions are wrong! OK
        #   * Using own z_rng positions: OK
        # 3: Test the SRF positions: OK
        #   * Do we need to have a new z_rng?
        # 4: Test convolution with Kronecker:
        #   * Position: OK
        #   * PSF: OK
        #   * Frequency: OK
        # 5: Test convolution with SRF
        #   * Position: OK
        #   * PSF: OK
        #   * Frequency: OK

        rcParams['text.usetex']=True
        rcParams['font.size']=24
        rcParams['lines.linewidth']=0.5
        um = 1e6
        c = speed_of_light
        # Simulate for Tomlins data
        smp = Sample(ns=[1.3,1.5,1.0],z_widths=[5,15,30])
        src = Source(center_wavelength=800e-9, bandwidth=50e-9)

        # OK 1
        # To simulate the superposition with the refrence the width and frequency
        # get a factor of 2 due to the Double pass.
        E_t_i, ez_rng = src.get_E_i_td(mode=src.SRF, sigma_x=2, w_x=2)
        E_t_i /= E_t_i.max()
        figure(tight_layout=True)
        plot(ez_rng*um, real(E_t_i)),title('Source field')

        # OK 2
        rs_kd, kz_rng, z_rng_max = smp.kronecker_deltas(src,new_z_rng=ez_rng)
        # OK 3
        H, hz_rng = smp.generate_H(src)
        # correct the SRF frequency powers
        alpha = 1.0 # 0.48 is only required if we compare the delta function
        win = tukey(len(H), alpha=alpha, sym=False)
        SN = src.SN
        # OK 3-2 Gauss window arbitrary
        # win = exp(-(linspace(0,SN*2,SN)-SN)**2/(2*(SN//exp(1))**2))
        # win = flattop(len(H))
        # compensate_on_data = 1.0 # set to 1.0 if not used
        H = win * H
        # H = win * H

        figure(tight_layout=True)
        plot(kz_rng*um, abs(rs_kd),label='KRN')
        plot(hz_rng*um, abs(fftshift(fft(H)))/src.SN,label='SRF')
        plot(hz_rng*um,win)
        ylabel('Intensity (arb.)')
        xlabel('Displacement in air (\\textmu m)')
        # xlim((0, 80))

        # Stop 3-1 can the Gaussian spectrum do the same than the Hann window?
        # Possibly yes.
        # !!!!!!!!!!!!!
        # But it is difficult, becaus the Gaussian spectral power is distributed
        # over a different range than the SRF and normalization would need to be discussed.
        # !!!!!!!!!!!1!
        test_3_1 = False
        if test_3_1:
            _ = src
            S_w,w_rng,w_0,sigma_w = src.S(_.w_rng,_.w_0,_.sigma_w)
            # S_w /= S_w.max()
            figure(tight_layout=True),title('Test 3-1: Stop')
            # sigma_w /= _.SN
            print('sigma_w',sigma_w)
            plot(hz_rng*um, fftshift(fft(H*S_w)),label='Test 3-1: Stop')
            xlim((0,80))


        # compare KRN and SRF after compensation
        # H2 = zeros(src.SN)
        # H2[0:src.SN//2] = resample(abs(fftshift(fft(H))) / src.SN, src.SN // 2)
        # diff_rs_kd_srf = abs( abs(rs_kd) - H2)
        # figure(tight_layout=True)
        # plot(kz_rng*um, diff_rs_kd_srf)


        # OK 4
        E_t_s = convolve(E_t_i, rs_kd, mode='same')
        figure(tight_layout=True)
        title('$E(t)_S$ Kronecker delta')
        kz_rng = kz_rng*um
        plot(kz_rng, real(E_t_s)) #position OK
        ylabel('Reflectivity (n.u.)')
        xlabel('Displacement in air (\\textmu m)')
        xlim((0,80))
        # ylim((-40,40))
        savefig('E_t_s_KRN.pdf')
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
        # OK 5: PSF is somewhat smaller
        E_t_i2, ez_rng = src.get_E_i_td(mode=src.SRF, sigma_x=1, w_x=1)
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
        savefig('E_t_s_SRF.pdf')
        # measure one layer PSF
        def measure():
            E_t_s2[0:40000] = 0
            E_t_s2[43000:] = 0
            plot(hz_rng*um, real(E_t_s2))
            psf, plt, x = self.measure_FWHM_hr(hz_rng*um, E_t_s2)
            print('PSF srf', psf) # PSF OK, Frequ. OK
            plot(x, plt, label='psf_h= {:1.6} $\mathrm{{\mu m}}$'.format(psf)), grid(True)
        measure()
        # compare KRN and SRF after compensation
        # H2 = zeros(src.SN)
        # H2[0:src.SN//2] = resample(abs(fftshift(fft(H))) / src.SN, src.SN // 2)
        # diff_rs_kd_srf = abs( abs(rs_kd) - H2)
        # figure(tight_layout=True)
        # plot(kz_rng*um, diff_rs_kd_srf)


        # OK SRF first multiplied by the E(w) and then FFT
        E_w_i3, wz_rng = src.get_E_w_i()
        # E_w_i3 /= E_w_i3.max()
        E_t_i3 = abs(fftshift(fft(E_w_i3*H)))
        figure(tight_layout=True)
        title('FFT\{$E(\omega)_S \cdot H(\omega)$\}')
        plot(hz_rng*um, E_t_i3)
        ylabel('Intensity (arb.)')
        xlabel('Displacement in air (\\textmu m)')
        xlim((0,100))
        savefig('FFT_E_w_times_H.pdf')

        # The summing is not further considered due to the fact that the simulation
        # can deal with the frequency of light directly.
        # E_t_r = exp(-2j * src.w_0/c * ez_rng)
        # E_t_d = E_t_s + E_t_r
        # E_t_d = E_t_d * conj(E_t_d)

        def measure():
            figure(tight_layout=True), title('Measure psf')
            plot(z_rng[47000:51000],(E_t_s[47000:51000])) # 47000-51000
            psf,plt,x = self.measure_FWHM_h(ez_rng[47000:51000],(E_t_s[47000:51000]))
            print('PSF',psf)
            plot(x,plt,label='psf= {:1.6} $\mathrm{{\mu m}}$'.format(psf)),grid(True)
            legend()
        # measure()

        # figure(tight_layout=True), title('TD-OCT interferogram\n 800 nm, 50 nm')
        # plot(z_rng*um, abs(rs_kd),lw=0.5)
        # plot(z_rng*um, E_t_r,lw=0.5)
        # plot(z_rng, E_t_d,lw=0.5)
        # xlim((0,80e-6*um)) #max(z_rng*um)))
    def theoretical_vs_measured_reflectivities():
        ns = array([1.0,1.3,1.5,1.0])
        for n1,n2 in zip(ns[0:-1],ns[1:]):
            print((n1-n2)/(n1+n2))


class Simulate_TD_manuscript(SimulationHelpers):
    def __init__(self):

        # manuscript:lst:samplefieldnumericalKR
        um = 1e6
        c = speed_of_light
        smp = Sample(ns=[1.3,1.5,1.0],z_widths=[5,15,30])
        src = Source(center_wavelength=800e-9, bandwidth=50e-9)
        E_t_i, ez_rng = src.get_E_i_td(mode=src.SRF, sigma_x=2, w_x=2)
        rs_kd, kz_rng, z_rng_max = smp.kronecker_deltas(src,new_z_rng=ez_rng)
        E_t_s = convolve(E_t_i, rs_kd, mode='same')
        # manuscript:lst:samplefieldnumericalKR


        # manuscript:lst:samplefieldnumericalSRF
        um = 1e6
        c = speed_of_light
        smp = Sample(ns=[1.3,1.5,1.0],z_widths=[5,15,30])
        src = Source(center_wavelength=800e-9, bandwidth=50e-9)
        E_t_i, ez_rng = src.get_E_i_td(mode=src.SRF, sigma_x=1, w_x=1)
        H, hz_rng = smp.generate_H(src)
        # Tukey window to reduce error of reflectivity
        alpha = 0.48 # 1.0 is a Hann window
        tukwin = tukey(len(H), alpha=alpha, sym=False)
        H = tukwin * H
        E_t_s = convolve(E_t_i, abs(fftshift(fft(H)))/src.SN, mode='same')
        # manuscript:lst:samplefieldnumericalSRF


        # manuscript:lst:samplefieldnumerical*SRF
        um = 1e6
        c = speed_of_light
        smp = Sample(ns=[1.3,1.5,1.0],z_widths=[5,15,30])
        src = Source(center_wavelength=800e-9, bandwidth=50e-9)
        E_w_i, wz_rng = src.get_E_w_i()
        E_t_i = abs(fftshift(fft(E_w_i * H)))
        # manuscript:lst:samplefieldnumerical*SRF

class Simulate_SD_manuscript(SimulationHelpers):
    def __init__(self):

        um = 1e6
        nm = 1e9
        c = speed_of_light

        # manuscript:lst:simulateSD
        smp = Sample(ns=[1.3, 1.5, 1.0], z_widths=[5, 15, 30])
        src = Source(center_wavelength=800e-9, bandwidth=50e-9)

        w_rng   = src.w_rng
        w_0     = src.w_0
        sigma_w = src.sigma_w

        S_w, w, w_0, s_w = src.S(w_rng, w_0, sigma_w*2)

        H_w, hz_rng = smp.generate_H(src)
        E_w_s = S_w * H_w
        E_w_r = S_w

        E_w_d = E_w_r + E_w_s

        I_w_d = abs(E_w_d)**2

        I_z = abs(fftshift(fft(I_w_d)))
        # manuscript:lst:simulateSD



class Simulate_SD(SimulationHelpers):
    '''
    This basically means to take the E_s(w) and E_r(w) and compute the
    E_d(w) = E_S(w) + E_R(w).

    Because we have the H(w)*E_i(w) = E_S(w) and
    E_i(w) == E_r(w) (to some degree this should work out similar tothe TD simulation.

    1) OK: Get spectrum for source and layers and plot
    2) OK: SRF and position -> The SRF is the HF
    3) OK: Multiply with spectrum -> HF + Envelope
    4) OK: Summing sample with reference
    5) OK: Convert summed field E_w_d to intensity
    '''
    def __init__(self):
        rcParams['text.usetex'] = True
        rcParams['text.latex.preamble']=['\\usepackage{siunitx}']
        rcParams['font.size'] = 20
        rcParams['lines.linewidth'] = 1.0
        rcParams['lines.markersize'] = 1.0

        um = 1e6
        nm = 1e9
        c = speed_of_light
        # Simulate for Tomlins data
        smp = Sample(ns=[1.3, 1.5, 1.0], z_widths=[5, 15, 30])
        src = Source(center_wavelength=800e-9, bandwidth=50e-9)
        w_rng   = src.w_rng
        w_0     = src.w_0
        sigma_w = src.sigma_w
        S_w, w, w_0, s_w = src.S(w_rng, w_0*2, sigma_w*2)

        WL_rng  = src.WL_rng * nm
        CWL     = src.CWL    * nm
        BW      = src.BW     * nm
        figure(tight_layout=True)
        title('Source spectrum $S(\omega)$')
        plot(WL_rng, S_w)
        xlabel('Wavelength (nm)')
        ylabel('Power (arb.)')
        xlim((CWL - 5 * BW, CWL + 5 * BW))
        # savefig('simu_SD_source_spectrum.pdf')

        H_w, hz_rng = smp.generate_H(src)
        E_w_s = S_w * H_w
        E_w_r = S_w
        E_w_d = E_w_r + E_w_s

        # Plotting along wavenumber
        # k * \lambda = 2 * pi
        k_max = 2*pi/src.WL_rng[-1]
        with warnings.catch_warnings(record=True) as w:
            # Accept division by zero
            warnings.filterwarnings('always')
            k_rng = 2*pi/src.WL_rng
        k_0   = pi/src.CWL

        figure(tight_layout=True)
        title('Interferogram')
        plot(k_rng/1e3, abs(E_w_d)**2)
        view_rng = (k_0/1e3-.25e3,k_0/1e3+.25e3)
        xlim(view_rng)
        gca().set_xticks([view_rng[0],k_0/1e3,view_rng[1]])
        # gca().set_xticklabels(['\SI{{{:.0E}}}{{}}'.format(k_0-1e6),'\SI{{{:.0E}}}{{}}'.format(k_0),'\SI{{{:.0E}}}{{}}'.format(k_0+1e6)])
        xlabel('k (1/mm)')
        ylabel('Power (arb.)')
        savefig('simu_SD_cam_k_interferogram.pdf')

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
        savefig('simu_SD_A_line.pdf')

    def test_SD_OCT(self):
        rcParams['text.usetex'] = True
        rcParams['font.size'] = 24
        rcParams['lines.linewidth'] = 0.5
        um = 1e6
        c = speed_of_light
        # Simulate for Tomlins data
        smp = Sample(ns=[1.2, 1.1], z_widths=[40, 20])
        src = Source(center_wavelength=800e-9, bandwidth=50e-9)

        _ = src

        # For spectral domain the z_rng depends on the camera
        cz_rng = 1 / (4 * mean(smp.ns)) * src.CWL ** 2 / src.BW * src.SN
        print('cz_rng (m)', cz_rng)

        # OK 1: Spectrum source
        S_w, w, w_0, s_w = src.S(_.w_rng, _.w_0, _.sigma_w*2)
        THz = 1e-12/2/pi
        f,f_0,s_f = (array([w,w_0,s_w])*THz)
        figure(tight_layout=True), title('Test 1:$S(\omega)$')
        plot(f, S_w)
        xlim((f_0 - 5 * s_f, f_0 + 5 * s_f))

        # OK 2: SRF and position
        # This is the HF without envelope
        H_w, hz_rng = smp.generate_H(src)
        figure(tight_layout=True), title('Test 2:$H(\omega)$')
        plot(w, real(H_w))
        # The H_w is a frequency in the frequency domain !!!
        # Therefore it has an frequency axis
        xlim((w_0-5*s_w,w_0+5*s_w))
        figure(tight_layout=True), title('Test 2:$FFT(H(\omega))$ -> $\delta(z)$')
        plot(hz_rng * um, abs(fftshift(ifft(real(H_w)))))

        # OK 3: Multiply with spectrum -> HF + Envelope
        E_w_s = S_w * H_w
        figure(tight_layout=True), title('Test 3:$S(\omega)H(\omega)$')
        plot(w, real(E_w_s))
        # The H_w is a frequency in the frequency domain !!!
        # Therefore it has an frequency axis
        xlim((w_0-5*s_w,w_0+5*s_w))

        # OK 4: Summing sample with reference
        # E_d_w = E_w_r + E_w_s
        # The spectrum here is originating from the grating.
        # The original emission from the light source is in the spatiotemporal domain.
        E_w_r = 0.25*S_w  # ideally the spectrum is reflected
        E_w_d = E_w_r + E_w_s

        figure(tight_layout=True),title('Test 4:$E_D(\omega)$')
        plot(w, real(E_w_d))
        xlim((w_0-5*s_w,w_0+5*s_w))

        # OK 5: Convert summed field E_w_d to intensity
        # The equation below is the mathematical extract for the interference but
        # does not include the reference field and the AC terms.
        # E_w_d = S_w1 * (H_w + 1) ** 2

        figure(tight_layout=True), title('Test 5:$FFT(E_D(\omega))$')
        plot(hz_rng*um, abs(fftshift(fft(abs(E_w_d)**2 ))))
        xlim((0,80))

    def test_autocorrelation_terms(self):
        '''
        See also gauss_generation.py:multi_sample_reflectors
        1) changing SN does not improve sample accuracy in the TD!
        2) Increasing the SN does increase sample length in the TD, but this does not affect the signal!
           See above.
        3) Increasing SN and range in the FD does increase samples in the TD.
           That means the sample frequency reduces in the TD.

        '''
        rcParams['lines.markersize']=1.0
        rcParams['lines.linewidth']=1.0
        mp = 10
        SN = 1000*mp
        x = linspace(-mp*10*pi,mp*10*pi,SN)
        sigma_w = 0.3 # reducing the STD increases the camera pixel ratio
        s0 = 1.0*real(exp(-(x-0)**2/(2*sigma_w**2)))
        s1 = 0.1*real(exp(-(x-0-33)**2/(2*sigma_w**2)))
        s2 = 0.2*real(exp(-(x-0-37)**2/(2*sigma_w**2)))

        figure(tight_layout=True)
        plot(x,s0,'.')
        plot(x,s1,'.')
        plot(x,s2,'.')

        t0 = fftshift(fft((s0)))
        t1 = fftshift(fft((s1)))
        t2 = fftshift(fft((s2)))

        # tr = fftfreq(len(x),d=1/x)
        figure(tight_layout=True)
        plot(real(t0))#,'.')
        plot(real(t1))#,'.')
        plot(real(t2))#,'.')

        s12 = s1+s2
        t12 = t1+t2
        E_r = (fftshift(fft((s0))))
        E_s = (fftshift(fft(s12)))
        E_td = E_r + E_s
        E_o = abs(E_td)**2
        figure(tight_layout=True),title('Spec')
        plot(x, E_o,'.-',ms=0.8,lw=0.5)
        # xlim((SN//2-300,SN//2+300))
        xlim((-30,30))

        E_w = abs(fftshift(ifft(E_o)))
        figure(tight_layout=True),title('TD')
        semilogy(x,E_w)

    def test_multi_sample_reflectors(self):
        '''
        From gauss_generation.py
        Demonstrate the AC terms (See test point (A))
        @return:
        '''
        SN = 2**12
        wlr = linspace(-SN,SN,SN*2)
        sigma_wl = 35
        S_r = 0.4*real(exp(-((wlr-0)**2/(2*sigma_wl**2))))
        S_1 = 0.1*real(exp(-((wlr-250)**2/(2*sigma_wl**2))))
        S_2 = 0.1*real(exp(-((wlr-1550)**2/(2*sigma_wl**2))))

        figure(tight_layout=True),title('Spec')
        plot(wlr, S_r,'.')
        plot(wlr, S_1,'.')
        plot(wlr, S_2,'.')

        # this is the loop with append and then sum
        # S_12 = S_1 + S_2
        S_ss = []
        for s in [S_1,S_2]:
            S_ss.append(s)
        S_12 = sum(S_ss,axis=0)

        # Then we cass sum_by_E_field
        E_r = fftshift(fft(S_r))
        E_12 = fftshift(fft(S_12))
        E_sum = E_r + E_12

        # OK (A-0-0): How to get HF with the right sample frequency?
        # Because we constructed the spectrum based on prior knowledge of the light source,
        # and we did not perform any prior FFT operation to obtain the spectrum we need to
        # emulate the final fftshift we would have otherwise performed.
        figure(tight_layout=True),title('HF')
        # plot(S_r)
        # plot(fftshift(S_r))
        # plot(abs(fftshift(ifft(fftshift(S_r)))))
        # plot(abs(fftshift(fft(fftshift(S_r)))))
        plot(abs(fftshift(fft(fftshift(S_1))))) # real part for the HF

        # OK (A-0) abs()**2 --> ACT occur
        # this is returned
        I_sum = abs(E_sum)**2

        figure(tight_layout=True),title('Interference')
        # OK(A-1): Show I_sum along wlr
        # The wlr is not entierly accurate here because this in the TD now.
        #
        plot(wlr, I_sum)
        xlim((-4*sigma_wl,4*sigma_wl))

        I_sum = abs(fftshift(ifft(I_sum)))
        figure(tight_layout=True),title('A-line')
        plot(I_sum)

    def test_Tomlins_FD_SD(self):
        '''
        Compare gauss_generation.py to Tomlins.
        We demonstrate that the HF + envelope can be generated from the source spectrum
        directly.

        The important point here is that the spectrum must be flipped with fftshift before
        FFT is applied as otherwise a high frequency beating occurs.
        Then the absolute squared field shows the AC terms.

        Task 1   OK: create the high frequency signal of the source (See Tomlins)
        Task 1-1 OK: show spectrum by FFT( signal )
        Task 1-2 OK: revert to signal by applying FFT again.
          Background: Because the signal can be created from the source spectrum.
        Task 1-3 : Generate spectrum directly
        @return:
        '''
        SN = 2**10
        wlr = linspace(-SN,SN,SN*2)
        w_0 = 2*pi*50
        sigma_wl = 200
        tr = linspace(-SN,SN,SN*2)
        kr = linspace(-SN,SN,SN*2)

        # Task 1 OK: The spectrum depends also on k --> s(w,k)
        # Then time and k allow to be together.
        # This works with Tomlins
        s_wk = exp(-((kr-0))**2 / (2*sigma_wl**2))
        E_w_in = s_wk * exp(-1j*w_0*kr)
        # Task 1 OK: Plot
        figure(tight_layout=True),title('Tomlins E_in')
        # Real is the HF content and envelope
        plot(kr, real(E_w_in), '.-',ms=0.5,lw=0.5)
        # plot(kr, (E_w_in), '.-',ms=0.5,lw=0.5) # plot selects only real, shows complex warning

        # Task 1-1 OK: TD spectrum to source spectrum vs. w
        s_wr = abs(fftshift(fft(real(E_w_in)))) # abs to combine imag and real
        figure(tight_layout=True),title('Tomlins Source Spectrum')
        plot(wlr, s_wr, '.-',ms=0.5,lw=0.5)

        # Task 1-2 OK: E_in from the FFT( S )
        # Show that the reverse of the spectrum by FFT creates the HF and envelope
        figure(tight_layout=True),title('Tomlins FFT(Source Spectrum)\nto get E_in back')
        # E_w_inr = (fftshift(fft(fftshift(s_wr))))/(2*SN) # ifft avoids the /(2*SN)
        E_w_inr = (fftshift(ifft(fftshift(s_wr))))
        plot(kr, real(E_w_inr), '.-',markersize=0.5,linewidth=0.5)
        plot(kr, real(E_w_in), '.-',markersize=0.5,linewidth=0.5)

        # Task 1-3 OK: Generate spectrum directly
        w_0 = 2*pi*50 /2/pi
        sigma_wl = .8/log(sqrt(200)) # arbitrarily constructed
        s_w0 = 1*real(exp(-((wlr - w_0))**2 * (2*sigma_wl**2)))
        # This only single sided (real) but the FFT then also generates the HF + env.
        figure(tight_layout=True),title('Spectrum directly')
        plot(wlr, s_w0,'.-',ms=0.5,lw=0.5)
        plot(wlr, s_wr/(SN/4),'.-',ms=0.5,lw=0.5) # compare
        xlim((w_0-50,w_0+50))

        # Task 1-4 OK: Get HF and envelope from S direct
        E_w_in0 = fftshift(fft(fftshift(s_w0)))/4 # the power scaling is off here still but OK for all other parameters.
        figure(tight_layout=True),title('FFT(Spectrum directly)')
        plot(kr, real(E_w_in0),'.-',markersize=0.5,linewidth=0.5)
        plot(kr, real(E_w_in), '.-',markersize=0.5,linewidth=0.5) # compare

        # What happens with the HF + env if changed in frequency?
        # Usually only the envelope is carried around.
        # Here we have the HF in the source spectrum.
        # 1) sum the S(w) + S(w)H(w)
        # 2) fftshift(fft(fftshift( ...))) note about the double fftshift here!
        # 3) then the abs(...)**2 generates the AC terms.

        # OK 2: Generate multiple spectra and the HF + Envelope
        sigma_wl = 4/log(sqrt(200))
        w_0 = 2*pi*50/2/pi
        s_1 = real(exp(-((wlr - w_0 - 100) ** 2 / (2 * sigma_wl ** 2))))
        s_2 = real(exp(-((wlr - w_0 - 144) ** 2 / (2 * sigma_wl ** 2))))
        figure(tight_layout=True), title('Sample Specs')
        plot(wlr, s_w0,'.-',ms=0.5,lw=0.5)
        plot(wlr, s_1, '.-',ms=0.5,lw=0.5)
        plot(wlr, s_2, '.-',ms=0.5,lw=0.5)

        # OK 2-1: make HF + Env according to Task 1-4
        E_w_1 = fftshift(fft(fftshift(s_1)))/4
        E_w_2 = fftshift(fft(fftshift(s_2)))/4
        figure(tight_layout=True), title('FFT(Sample Specs)')
        plot(kr, real(E_w_in0))
        plot(kr, real(E_w_1)  )
        plot(kr, real(E_w_2)  )

        # OK 2-2: sum E_1 and E_2
        figure(tight_layout=True), title('sum E_12')
        S_w = abs(E_w_in0)**2
        E_w_12 = E_w_1 + E_w_2
        # See Tomlins, assuming E_w_12 is H_w
        # However, this is not actually what we want as it is based on the mathematical model
        E_w_o = S_w * (E_w_12 + 1)**2
        plot(kr, real(E_w_o))

        # OK 2-3: AC terms when?
        # With above method no AC terms are visible!
        # Now we sum first the source spectra and then do fft and abs()**2
        # This generates the ACTs.
        # Then the theory relates to
        s_12 = 0.01*s_1 + 0.02*s_2
        E_w_12 = fftshift(fft(fftshift(s_12)))
        E_w_o = E_w_in0 + E_w_12
        figure(tight_layout=True), title('Interferrogram')
        plot(kr, abs(E_w_o)**2)

        # I_w = abs(E_w_o)**2
        I_w = E_w_o * conj(E_w_o)
        I_t = abs(fftshift(fft(I_w)))
        figure(tight_layout=True), title('FFT(sum E_12)')
        # plot(kr, abs(fftshift(fft(abs(E_w_o)**2)))) # --> AC terms
        semilogy(kr, I_t)


class Simulate_MRO(SimulationHelpers):
    '''
    # OK 0: Source S_w, E_in
    From SD-OCT

    # Test 1: Sample E_s = E_in * H_w
    SRF or Kronecker

    # Test 2: multiple reference reflections
    Use generation of a layer structure.
    E.g. Sample([0, 0.05, 0.1, 0.15, ...]) for all orders required.

    # Test 3: Sum equally like for SD-OCT.


    # Test 4: The scanning then results from E_r
    This is what we do by E_d = E_r + E_s
    Numerically the scanning is instantaneously.

    # Remarks Why does MRO has self interference?
    Consider a double layer in the sample arm and a single reference mirror.
    1) Each layer from the sample will have the same frequency.
    2) Summation of both layers does not generate beat -> auto-correlation.

    Consider PM and SRM but only single sample layer.
    1) Each scanning mirror has a different frequency.
    2) Summation of both frequencies will generate a beat frequency with (f_0 + f_1) * n.
    3) With n = 1,2,...

    Consider double sample layer and PM + SRM.
    1) The scanning fields generate beats but the sample layer remain without AC.
    2) Recall, the frequency for each sample layer is the same.

    '''
    def __init__(self):
        pass

    def test_0_MRO(self):
        um = 1e6
        nm = 1e9
        c = speed_of_light
        # Simulate for Tomlins data
        src = Source(center_wavelength=800e-9, bandwidth=50e-9)
        w_rng   = src.w_rng
        w_0     = src.w_0
        sigma_w = src.sigma_w

        # OK 0: Source S_w, E_in
        S_w, w, w_0, s_w = src.S(w_rng, w_0*2, sigma_w*2)
        WL_rng  = src.WL_rng * nm
        CWL     = src.CWL    * nm * 2
        BW      = src.BW     * nm
        # figure(tight_layout=True)
        # title('Source spectrum $S(\omega)$')
        # plot(WL_rng, S_w)
        # xlabel('Wavelength (nm)')
        # ylabel('Power (arb.)')
        # xlim((CWL - 5 * BW, CWL + 5 * BW))
        # savefig('simu_MRO_source_spectrum.pdf')

        # Test 1: Sample E_s = E_in * H_w
        # Assume we use multi layers for scanning.
        # Assume we use single mirror as sample
        # That means e.g. for two orders we should get two reflections.
        # For simplicity we do not simulate a different frequency but hi-jack the
        # spectral mixing to get the effect of self interference.
        # That means we only need to compute the position and range of the order.

        E_s = S_w # sample mirror
        H1,zr = Sample(ns=[1e3], z_widths=[5.0]).generate_H(src)
        E_r1 = S_w * H1
        aline = []
        figure(num=1,tight_layout=True)
        figure(num=2,tight_layout=True)
        img = imshow(zeros((1,src.SN)),aspect=1000,animated=True)

        for o in arange(1,8,0.1):
            H2,zr = Sample(ns=[1.01,1.02,1.03], z_widths=[5.0,10*o,5*(o+1)]).generate_H(src)
            E_r2 = S_w * H2

        # figure(tight_layout=True)
        # plot(real(E_r2)) # intermediat spectrum
            aline.append(log(abs(fftshift(fft(abs(1e-3*S_w+E_r2)**2)))))
            figure(1)
            plot(zr*um,aline[-1])
            pause(0.1)
            cla()

            figure(2)
            if any(img.get_array()): print('img',shape(img.get_array()))
            img = imshow(aline,aspect=400, vmin=-50,vmax=10)
            pause(0.1)
            # from matplotlib.image import AxesImage
            # AxesImage
            # img.set_array(aline)

            # waitforbuttonpress()

    def test_1_MRO(self):
        '''
        # Remark Why is it not possible to use mixing in the time domain?

        We could generate the reference mirrors with the Kroneckre delta and convolute them
        with the source.

        @return:
        '''
        um = 1e6
        nm = 1e9
        c = speed_of_light
        # Simulate for Tomlins data
        src = Source(center_wavelength=800e-9, bandwidth=50e-9)
        w_rng   = src.w_rng
        w_0     = src.w_0
        sigma_w = src.sigma_w

        # OK 0: Source S_w, E_in
        S_w, w, w_0, s_w = src.S(w_rng, w_0*2, sigma_w*2)
        WL_rng  = src.WL_rng * nm
        CWL     = src.CWL    * nm * 2
        BW      = src.BW     * nm
        # figure(tight_layout=True)
        # title('Source spectrum $S(\omega)$')
        # plot(WL_rng, S_w)
        # xlabel('Wavelength (nm)')
        # ylabel('Power (arb.)')
        # xlim((CWL - 5 * BW, CWL + 5 * BW))
        # savefig('simu_MRO_source_spectrum.pdf')

        E_s = S_w # sample mirror
        bframe = zeros((1,src.SN))
        lines = []
        ax_lims = None
        Sw1 = src.S(w_rng, w_0 * 2 * 1, sigma_w * 4)[0]
        Sw2 = src.S(w_rng, w_0 * 2 * 2, sigma_w * 5)[0]
        for z in arange(0,80,0.4):
            H1, zr = Sample(ns=[1.01], z_widths=[0.0+z]).generate_SRM_PM(src,[])
            H2, zr = Sample(ns=[1.01], z_widths=[0.0+z*2]).generate_SRM_PM(src,[])
            # Sw3 = src.S(w_rng,w_0*2*3, sigma_w*2)[0]
            # H3, zr = Sample(ns=[1.01], z_widths=[0.0+z*3]).generate_SRM_PM(src,[])

            # If S_w*E_r1 then no self interference occurs
            # E_r = S_w*E_r1 #+ Sw2*E_r2 #+ Sw3*E_r3
            # If Sw1*E_r1 then a first order self interference occurs
            E_w_s = Sw1*H1 + Sw2*H2 #+ Sw3*E_r3
            # I_k = abs(1e-3*S_w + E_r)**2
            # I_z = (abs(fftshift(fft(I_k))))

            aline_n = 20*log(abs(fftshift(fft(abs(real(0.1*S_w + E_w_s))))))

            # figure(0)
            # from matplotlib.pyplot import Axes
            # Axes.get_xlim()
            # ax = gca()
            # ax.clear()
            # plot(log(abs(fftshift(fft(abs(real(0.1*S_w + E_w_s)))))))
            # plot(real(E_r2))
            # plot(aline_n)
            # if ax_lims is not None: ax.set_xlim(ax_lims[0]), ax.set_ylim(ax_lims[1])
            # pause(0.1)

            # while waitforbuttonpress() is None: pass
            # ax_lims = (ax.get_xlim(),ax.get_ylim())
            bframe = concatenate((bframe,[aline_n]))


        figure(2,tight_layout=True)
        img = imshow(bframe[1:100,src.SN//2+500:src.SN//2+10000].T,aspect=1/100,cmap='CMRmap', vmin=-200,vmax=0)

        ax = gca()
        ytick_fr1 = array([9400,int(9500/6)])
        ytick_fr2 = array([150,int(150/6)])
        xtick_fr1 = array([90,int(100/6)])
        xtick_fr2 = array([150,int(150/6)])
        ax.set_yticks(range(0,ytick_fr1.sum(),ytick_fr1[1]))
        ax.set_yticklabels(range(0,ytick_fr2.sum(),ytick_fr2[1]))
        ax.set_xticks(range(0,xtick_fr1.sum(),xtick_fr1[1]))
        ax.set_xticklabels(range(0,xtick_fr2.sum(),xtick_fr2[1]))
        xlabel('mirror z-position (a.u.)')
        ylabel('PSF z-position (a.u.)')
        colorbar(ax=gca(),label='Intensity 20log$_{10}$')

        savefig('simulate_MRO_SRMPM.pdf')

            # if mod(z,20)==0: pause(0.1)

            # while waitforbuttonpress() == False: pass

    def test_1_MRO_manuscript(self):
        '''
        # Remark Why is it not possible to use mixing in the time domain?

        We could generate the reference mirrors with the Kroneckre delta and convolute them
        with the source.

        @return:
        '''
        #manuscript:lst:simulateMRO_spectral
        um = 1e6
        nm = 1e9
        c = speed_of_light
        # Simulate for Tomlins data
        src = Source(center_wavelength=800e-9, bandwidth=50e-9)
        w_rng   = src.w_rng
        w_0     = src.w_0
        sigma_w = src.sigma_w

        S_w, w, w_0, s_w = src.S(w_rng, w_0*2, sigma_w*2)
        WL_rng  = src.WL_rng * nm
        CWL     = src.CWL    * nm * 2
        BW      = src.BW     * nm

        E_s = S_w # sample mirror
        bframe = zeros((1,src.SN))
        lines = []
        ax_lims = None
        Sw1 = src.S(w_rng, w_0 * 2 * 1, sigma_w * 4)[0]
        Sw2 = src.S(w_rng, w_0 * 2 * 2, sigma_w * 5)[0]
        for z in arange(0,80,0.4):
            H1, zr = Sample(ns=[1.01], z_widths=[0.0+z]).generate_SRM_PM(src,[])
            H2, zr = Sample(ns=[1.01], z_widths=[0.0+z*2]).generate_SRM_PM(src,[])

            E_w_s = Sw1*H1 + Sw2*H2

            aline_n = log(abs(fftshift(fft(abs(real(0.1*S_w + E_w_s))))))

            bframe = concatenate((bframe,[aline_n]))

        #manuscript:lst:simulateMRO_spectral
        figure(2)
        img = imshow(bframe[1:100,src.SN//2+500:src.SN//2+10000].T,aspect=1/100, vmin=-20,vmax=10)
        savefig('simulate_MRO_SRMPM.pdf')
            # if mod(z,20)==0: pause(0.1)

            # while waitforbuttonpress() == False: pass

    def test_2_MRO(self):
        '''
        Evaluate the mixing with the Kronecker and in the time-domain.
        @return:
        '''
        um = 1e6
        nm = 1e9
        c = speed_of_light
        # Simulate for Tomlins data
        src = Source(center_wavelength=800e-9, bandwidth=50e-9)
        w_rng   = src.w_rng
        w_0     = src.w_0
        sigma_w = src.sigma_w

        # OK 0: Source S_w, E_in
        S_w = src.S(w_rng, w_0*2, sigma_w*2)[0]
        WL_rng  = src.WL_rng * nm
        CWL     = src.CWL    * nm * 2
        BW      = src.BW     * nm
        # figure(tight_layout=True)
        # title('Source spectrum $S(\omega)$')
        # plot(WL_rng, S_w)
        # xlabel('Wavelength (nm)')
        # ylabel('Power (arb.)')
        # xlim((CWL - 5 * BW, CWL + 5 * BW))
        # savefig('simu_MRO_source_spectrum.pdf')

        # Create the TD field
        # E_w_i = S_w
        E_t_i = fftshift(fft(S_w))
        f_max = 2*w_rng[-1]/2/pi
        ez = 4/f_max*src.SN * c
        ez_rng = linspace(0,ez,src.SN) # a range for the correct PSF

        ax_lims = None
        Sw1 = src.S(w_rng, w_0 * 2 * 2.4, sigma_w * 2 * 3)[0]
        Sw2 = src.S(w_rng, w_0 * 2 * 3, sigma_w * 2 * 2)[0]

        E_t_i1 = fftshift(fft(Sw1))
        E_t_i2 = fftshift(fft(Sw2))
        bframe = zeros((1,src.SN))

        for z in arange(0,280,2):
            rs_kd1, zr = Sample(ns=[1.04],z_widths=[0.0+z]).kronecker_deltas(src,new_z_rng=ez_rng)[0:2]
            rs_kd2, zr = Sample(ns=[1.05],z_widths=[0.0+z/2]).kronecker_deltas(src,new_z_rng=ez_rng)[0:2]
            # figure(1,tight_layout=True)
            # plot(zr,abs(rs_kd))

            E_t_s1 = convolve(E_t_i1, rs_kd1, mode='same')
            E_t_s2 = convolve(E_t_i2, rs_kd2, mode='same')
            # figure(2,tight_layout=True)
            # plot(zr,E_t_s)

            E_t_d1 = convolve(E_t_i, E_t_s1, mode='same')
            E_t_d2 = convolve(E_t_i, E_t_s2, mode='same')
            E_t_d = E_t_d1 + E_t_d2
            # figure(3,tight_layout=True)
            # ax = gca()
            # cla()
            # plot(zr,real(E_t_d))

            # figure(4,tight_layout=True)
            # ax = gca()
            # cla()
            # plot(zr,log(abs(real(E_t_d))**2))
            # pause(0.1)
            # if ax_lims is not None: ax.set_xlim(ax_lims[0]), ax.set_ylim(ax_lims[1])

            a_line = log(abs((E_t_d))**2)
            bframe = concatenate((bframe, [a_line]))

        figure(4)
        plot(zr,log(abs(real(E_t_d))**2))
        figure(5)
        # imshow(( bframe[:,src.SN//2:src.SN//2+10000] ),aspect=50,vmin=-55,vmax=-50)
        imshow(( bframe ),aspect=50,vmin=-55,vmax=-50)
        # colorbar(cax=gca())

            # pause(0.1)
            # while waitforbuttonpress(timeout=0) is False: pass
            # ax_lims = (ax.get_xlim(),ax.get_ylim())

    # Problem
    # Right now we generate a higher order self interference by directly adding it.
    # Can we generate the self interference before hand?

    # Remark 0: Re use previous interference I(---)P(+)
    # This idea makes it not possible to separate the direct terms in 1
    # 1) Simulate higher order self interference
    # 2) Use those terms again to create layers and interfere

    # Remark 1: Construct by first principle I(+++)P(---)

    # E_r3 = E_in * H_w(o_n=3)
    # E_r4 = E_in * H_w(o_n=4)
    # O=1 |---|
    # O=2         |-^----|
    # O=3                 |----------^-|
    # O=4                         |--^---------------------|
    #               x1               x2
    # Remark
    # Indicated are three sample mirror positions.
    # At (x1) the f2 is showing the 2nd order and f2+f1 is a scanning mirror with f=2*f2
    # showing the sample again at double the depth.
    # At (x2) this is basically the same but for f=f3+f4 showing another sample at 2*f3 or
    # also at 2*f4 achieving and accurate overlap.
    # All higher orders of mixings are possibly lost due to beam divergence.
    #
    # Then there is a next order mixing.
    # Assuming the table of first order and second order mixing then
    # signal | 1st          | 2nd
    # -------+--------------+---------
    # -      | f1           | f2 - f1 is the 1st order again
    # -      | f2           | f3 - f1 is the 2nd order again
    # -      | f3           | f4 - f1 is the 3rd order again
    # -      | f4           | f5 - f1 is the 3rd order again

    def simulate_MRO_O1(self):
        um = 1e6
        nm = 1e9
        c = speed_of_light
        # Simulate for Tomlins data
        src = Source(center_wavelength=1300e-9, bandwidth=60e-9)
        w_rng   = src.w_rng
        w_0     = src.w_0
        sigma_w = src.sigma_w
        scale_sigma = 4
        S_w = src.S(w_rng, w_0, sigma_w*2*scale_sigma)[0]
        WL_rng  = src.WL_rng * nm
        CWL     = src.CWL    * nm
        BW      = src.BW     * nm
        # figure(tight_layout=True)
        # title('Source spectrum $S(\omega)$')
        # plot(WL_rng, S_w)
        # xlabel('Wavelength (nm)')
        # ylabel('Power (arb.)')
        # xlim((CWL - 5 * BW, CWL + 5 * BW))
        # savefig('simu_MRO_source_spectrum.pdf')

        E_t_i = fftshift(fft(S_w))
        E_t_r = E_t_i/E_t_i.max() # ideal scanning mirror

        # initial z range of TD source field
        f_max = 2*w_rng[-1]/2/pi
        ez = 4/f_max*src.SN * c # set scan range
        ez_rng = linspace(0,ez,src.SN)

        Sw1 = src.S(w_rng, w_0 * 2 , sigma_w * 2 * 1 * scale_sigma)[0] # order 1
        # Sw2 = src.S(w_rng, w_0 * 2 * 2, sigma_w * 2 * 2)[0]
        E_t_i1 = fftshift(fft(Sw1))
        # E_t_i2 = fftshift(fft(Sw2))

        bframe = zeros((1,src.SN))

        for z in arange(0,1500,10):
            rs_kd1, zr = Sample(ns=[1.04],z_widths=[0.0+z]).kronecker_deltas(src,new_z_rng=ez_rng)[0:2]
            # rs_kd2, zr = Sample(ns=[1.05],z_widths=[0.0+z*2]).kronecker_deltas(src,new_z_rng=ez_rng)[0:2]
            # figure(1,tight_layout=True)
            # plot(zr,abs(rs_kd))

            E_t_s1 = convolve(E_t_i, rs_kd1, mode='same') # sample field
            # E_t_s2 = convolve(E_t_i2, rs_kd2, mode='same')
            # figure(2,tight_layout=True)
            # plot(zr,E_t_s)

            E_t_d1 = convolve(E_t_r, E_t_s1, mode='same') # scanning
            E_t_d1 /= E_t_d1.max()
            # E_t_d2 = convolve(E_t_i, E_t_s2, mode='same')
            # E_t_d = E_t_d1 + E_t_d2


            a_line = (abs((E_t_d1))**2) # linear simulate_MRO_SRMPM_td1_lin.pdf
            # a_line = 20*log10(abs((E_t_d1))**2) # log simulate_MRO_SRMPM_td1_log.pdf
            bframe = concatenate((bframe, [a_line]))

        figure(1)
        plot(zr,a_line)

        figure(gcf().number+1, tight_layout=True)
        # imshow(( bframe[:,src.SN//2:src.SN//2+10000] ),aspect=50,vmin=-55,vmax=-50)
        # imshow(( bframe[:,src.SN//2:src.SN//2+10000] ),aspect=50,vmin=-55,vmax=-50)
        # imshow(bframe, aspect=600) #vmin=0,vmax=20)
        imshow(bframe[:,src.SN//2:], aspect=220,cmap='CMRmap')#,vmin=-800,vmax=0) #,extent=(0,300,0,300))
        ax = gca()
        ax.set_xticks(range(0,34500,int(34500/6)))
        ax.set_xticklabels(range(0,180,int(180/6)))
        xlabel('mirror z-position (a.u.)')
        ylabel('PSF z-position (a.u.)')
        colorbar(ax=gca(),label='Intensity ')
        # colorbar(ax=gca(),label='Intensity 20log$_{10}$')

        savefig('simulate_MRO_SRMPM_td1_lin.pdf')
        # savefig('simulate_MRO_SRMPM_td1_log.pdf')

    def simulate_MRO_O1_manuscript(self):
        # manuscript:lst:simulateMRO_TD1mode
        um = 1e6
        nm = 1e9
        c = speed_of_light
        src = Source(center_wavelength=1300e-9, bandwidth=60e-9)
        w_rng   = src.w_rng
        w_0     = src.w_0
        sigma_w = src.sigma_w

        S_w = src.S(w_rng, w_0, sigma_w*2)[0]
        WL_rng  = src.WL_rng * nm
        CWL     = src.CWL    * nm
        BW      = src.BW     * nm

        E_t_i = fftshift(fft(S_w))
        E_t_r = E_t_i/E_t_i.max() # ideal scanning mirror

        # initial z range of TD source field
        f_max = 2*w_rng[-1]/2/pi
        ez = 4/f_max*src.SN * c # set scan range
        ez_rng = linspace(0,ez,src.SN)

        bframe = zeros((1,src.SN))

        for z in arange(0,1500,10):
            rs_kd1, zr = Sample(ns=[1.04],z_widths=[0.0+z]).kronecker_deltas(src,new_z_rng=ez_rng)[0:2]

            E_t_s1 = convolve(E_t_i, rs_kd1, mode='same') # sample field
            E_t_d1 = convolve(E_t_r, E_t_s1, mode='same') # scanning
            E_t_d1 /= E_t_d1.max()

            a_line = log10(abs((E_t_d1)**2))
            bframe = concatenate((bframe, [a_line]))
        # manuscript:lst:simulateMRO_TD1mode



    def simulate_MRO_O2(self):
        um = 1e6
        nm = 1e9
        c = speed_of_light
        # Simulate for Tomlins data
        src = Source(center_wavelength=1300e-9, bandwidth=60e-9)
        w_rng   = src.w_rng
        w_0     = src.w_0
        sigma_w = src.sigma_w
        scale = 40

        S_w = src.S(w_rng, w_0, sigma_w*2*scale/10)[0]
        # initial z range of TD source field
        f_max = 2*w_rng[-1]/2/pi
        ez = 100/f_max*src.SN * c # set scan range
        print('ez (um)',ez*1e6)
        ez_rng = linspace(0,ez,src.SN)
        WL_rng  = src.WL_rng * nm
        CWL     = src.CWL    * nm
        BW      = src.BW     * nm
        # figure(num=gcf().number+1,tight_layout=True)
        # title('Source spectrum $S(\omega)$')
        # plot(WL_rng, S_w)
        # xlabel('Wavelength (nm)')
        # ylabel('Power (arb.)')
        # xlim((CWL - 5 * BW, CWL + 5 * BW))
        # savefig('simu_MRO_source_spectrum.pdf')

        E_t_i = fftshift(fft(S_w))
        E_t_r = E_t_i/E_t_i.max() # ideal scanning mirror

        Sw1 = src.S(w_rng, w_0, sigma_w * 2 * 1 * scale/10)[0] # order 1
        Sw2 = src.S(w_rng, w_0, sigma_w * 2 * 1 * scale/10)[0]
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

            a_line = 20*log10(abs((E_t_d))**2)
            bframe = concatenate((bframe, [a_line]))

        # figure(1)
        # plot(a_line)

        figure(tight_layout=True)#num=gcf().number+1)
        # imshow(( bframe[:,src.SN//2:src.SN//2+10000] ),aspect=50,vmin=-55,vmax=-50)
        # imshow(( bframe[:,src.SN//2:src.SN//2+10000] ),aspect=50,vmin=-55,vmax=-50)
        # imshow(bframe, aspect=600)#vmin=0,vmax=20)
        imshow(bframe[0:100,src.SN//2:65000], cmap='CMRmap', aspect=310, vmin=-800,vmax=0, extent=(0,31000,100,0))
        ax = gca()
        ytick_fr1 = array([85,int(100/6)])
        ytick_fr2 = array([150,int(150/6)])
        xtick_fr1 = array([30500,int(31000/6)])
        xtick_fr2 = array([150,int(150/6)])
        ax.set_yticks(range(0,ytick_fr1.sum(),ytick_fr1[1]))
        ax.set_yticklabels(range(0,ytick_fr2.sum(),ytick_fr2[1]))
        ax.set_xticks(range(0,xtick_fr1.sum(),xtick_fr1[1]))
        ax.set_xticklabels(range(0,xtick_fr2.sum(),xtick_fr2[1]))
        xlabel('mirror z-position (a.u.)')
        ylabel('PSF z-position (a.u.)')
        colorbar(ax=gca(),label='Intensity 20log$_{10}$')

        savefig('simulate_MRO_SRMPM_td3.pdf')

    def test_tick_scaling(self):
        figure(tight_layout=True)
        plot(linspace(0,31200,110))
        ax = gca()
        tick_fr = 2
        ax.set_xticks(range(0,110+int(110/tick_fr),int(110/tick_fr)))
        ax.set_xticklabels(range(0,100+int(100/tick_fr),int(100/tick_fr)))
        ax.set_yticks(range(0,31000,int(31000/tick_fr)))
        ax.set_yticklabels(range(0,150,int(150/tick_fr)))


    def test_simple_self_interference(self):
        '''
        # Problem this simulation does not show any of generation of secondary beats.

        # Remark should it be that we have to work with beats?
        This means should we truely generate beats first?

        The proof would need to obey the input signal.
        The input signal would need to be filtered and spatially corrected.
        This should show for one particular frequency a second interface
        moving at different speed.

        We observe another frequency for instance in parallel to the 1st order.
        This second frequency is at that instant a 2nd order.
        If we spatially correct for it, it will be placed one distance further.
        So basically some higher frequency is generated that moves with the same
        speed like the 1st order, but has a frequency for 2nd or 3rd or higher order.
        @return:
        '''
        SN = 10000
        xrng = linspace(0,500,SN)
        k = 0.1
        win = tukey(SN,alpha=0.8)
        bframe = []
        lims1 = []
        lims2 = []
        filter = zeros(SN)
        win = tukey(100,alpha=0.8)
        filter[6220:6200+120] = win
        for x_pos in range(0,1000,10):
            def G(n):
                return exp(-(((x_pos + 100*n) / n  - xrng) ** 2 / (2 * (30 / n) ** 2)))
                # return exp(-(((x_pos ) - xrng) ** 2 / (2 * (30 / n) ** 2)))

            def O(n): # scanning
                # print( 2*1j*(k+n/100) )
                # print( 2*1j*k)
                # print( 2*1j*k*x_pos*n/100 )
                # Remark Convolution as a time domain amplitude modulation due to different
                # phase velocities is difficult to compute directly and
                return exp((2*1j*k + 2*1j*(k+(n)*10)) * xrng - 2*1j*k*x_pos/n*1 )
                # After spatial correction the frequency is equal!
                # return exp((2 * 1j * k + 2 * 1j )*  xrng - 2 * 1j * k * x_pos * n / 1)
            def Os(n): # sample
                return exp((2*1j*k + 2*1j*(k+(n-1)*10)) * xrng )

            intf = array([  O(n) for n in [3,6]]).T
            a_line = convolve(O(1), intf.T.sum(axis=1),mode='same')
            # a_line = convolve(G(1)*O(1),((intf)).sum(axis=1),mode='same')
            # a_line = O(1) + intf.sum(axis=1)
            do_plot = True
            if do_plot:
                fig = figure(1)
                # w = fig.canvas.manager.window
                fig.canvas.manager.window.move(300,200)
                cla()
                # plot(abs(fftshift(fft(a_line)))),plot(abs(fftshift(fft(O(1)))))
                # Remark even if we use a fast scanning this depends indeed on our step-by-step
                # motion.
                # To see the beat due to scanning it must be mixed spectrally.
                # Remark or better, the conv
                tI1 = O(1)*G(1)+O(2)*G(1)
                plot(hilbert(abs(tI1)))
                # plot(abs(fftshift(fft(tI1))))#,plot(Os(1))
                # plot(abs(fftshift(fft(hilbert(abs(tI1))))))#,plot(Os(1))
                # for i in intf.T:
                #     plot(i)
                if any( lims1 ): gca().set_xlim(lims1[0]), gca().set_ylim(lims1[1])
                pause(0.1)
                ax1 = gca()

                # fig = figure(gcf().number+1)
                # fig.canvas.manager.window.move(1500,200)

                # cla()
                # semilogy(xrng,abs(intf_fft), xrng,1000*filter)
                # semilogy(abs(intf_2))
                # if any( lims2 ): gca().set_xlim(lims2[0]), gca().set_ylim(lims2[1])
                # pause(0.1)
                while waitforbuttonpress(timeout=0) is False: pass
                lims1 = (ax1.get_xlim(),ax1.get_ylim())
                lims2 = (gca().get_xlim(),gca().get_ylim())

            bframe.append(a_line)

        fig = figure(1)
        fig.canvas.manager.window.move(300, 200)
        plot(abs(a_line))
        fig = figure(gcf().number+1)
        fig.canvas.manager.window.move(1500,200)
        imshow(array(bframe).__abs__(),cmap='CMRmap',aspect=100)

    def manuscript_plots_simple_self_interference(self):
        '''
        # Remark use the test_1_MRO simulation
        # This does show the second order using the 1st and 3rd only inculding higher ones.
        # Although it does not show the aliasing artifacts.

        # Remark the test_1_MRO does mix in the spectrum.
        # What is the ocassion this happening for MR-OCT ?
        see test_simple_self_interference
        @return:
        '''
        rcParams['text.usetex']=True
        rcParams['text.latex.preamble']=['\\usepackage{siunitx}']
        rcParams['font.size']=24
        rcParams['lines.linewidth']=1

        SN = 10000
        xrng = linspace(0,500,SN)
        k = 10
        win = tukey(SN,alpha=0.8)
        bframe = []
        lims1 = []
        lims2 = []
        filter = zeros(SN)
        win = tukey(100,alpha=0.8)
        filter[6220:6200+120] = win
        for x_pos in range(0,1000,10): #[150]:
            def G(n):
                return exp(-(((x_pos + 100*n) / n  - xrng) ** 2 / (2 * (30 / n) ** 2)))
                # return exp(-(((x_pos ) - xrng) ** 2 / (2 * (30 / n) ** 2)))

            def O(n): # scanning
                # print( 2*1j*(k+n/100) )
                # print( 2*1j*k)
                # print( 2*1j*k*x_pos*n/100 )

                return exp((2*1j*k + 2*1j*(k+(n-1)*2)*0.1) * xrng )#- 2*1j*k*x_pos/n*1 )
                # After spatial correction the frequency is equal!
                # return exp((2 * 1j * k + 2 * 1j )*  xrng - 2 * 1j * k * x_pos * n / 1)
            def Os(n): # sample
                return exp((2*1j*k ) * xrng )

            intf = array([  O(n) for n in [3,6]]).T
            a_line = convolve(O(1), intf.T.sum(axis=1),mode='same')
            # a_line = convolve(G(1)*O(1),((intf)).sum(axis=1),mode='same')
            # a_line = O(1) + intf.sum(axis=1)
            do_plot = True
            if do_plot:
                fig = figure(1,tight_layout=True)
                fig.canvas.manager.window.move(0,0)
                cla()
                # plot(abs(fftshift(fft(a_line)))),plot(abs(fftshift(fft(O(1)))))

                # manuscript:plot:nonoverlappingOrder
                tIs   = Os(1)*G(1)
                tI1o1 = O(2)+Os(1)
                tD1o1 = tI1o1*conj(tI1o1)
                tD1o1 = (tD1o1 - max(tD1o1)/2) * G(1)
                tI1o2 = O(4)+Os(1)
                tD1o2 = tI1o2*conj(tI1o2)
                tD1o2 = (tD1o2 - max(tD1o2)/2) * G(2)

                tI1 = O(2)*G(1) + O(3)*G(2) + Os(1)*G(1)
                tD1 = tI1 * conj(tI1)
                tD1 = tD1 - max(tD1)/2

                zr = linspace(0,500,len(tIs))
                plot(zr,tIs,label='HF light source')
                plot(zr,tD1o1,label='Detected signal')
                legend(fontsize=14)
                ylabel('Intensity (arb.)')
                xlabel('z (\SI{}{\micro\meter})')
                pause(0.1)
                # savefig('MRO_nonOverlappingOrder.pdf')


                # manuscript:plot:overlappingOrder


                fig = figure(gcf().number+1,tight_layout=True)
                fig.canvas.manager.window.move(600,0)
                cla()
                c2 = rcParams['axes.prop_cycle'].by_key()['color'][1]
                # plot(zr,tD1o1+tD1o2,'#ff7f0e',label='Detected signal')
                plot(zr,tD1,color=c2,label='Detected signal')
                legend(fontsize=14)
                ylabel('Intensity (arb.)')
                xlabel('z (\SI{}{\micro\meter})')
                pause(0.1)
                # savefig('MRO_withOverlappingOrder.pdf')

                # manuscript:plot:overlappingOrderFFT

                # fig = figure(gcf().number+1,tight_layout=True)
                # fig.canvas.manager.window.move(600,0)
                # plot(zr,tD1+tD1o2,label='Detected signal')
                # legend(fontsize=14)
                # ylabel('Intensity (arb.)')
                # xlabel('z (\SI{}{\micro\meter})')
                # savefig('MRO_withOverlappingOrder.pdf') # not sure yet

                kr = fftfreq(len(tD1o1),d=1)
                fig = figure(gcf().number+1,tight_layout=True)
                fig.canvas.manager.window.move(1200,0)
                cla()
                semilogy(kr,abs((fft(tD1o1))),label='Detected signal')
                legend(fontsize=14)
                xlim((0,0.1))
                ylabel('Intensity (arb.)')
                xlabel('k (1/\\textmu m)')
                pause(0.1)
                # savefig('MRO_withOverlappingOrderFFT_seperate.pdf')

                fig = figure(gcf().number+1,tight_layout=True)
                fig.canvas.manager.window.move(1200,600)
                cla()
                # semilogy(kr,abs((fft(tD1o1+tD1o2))),label='Detected signal')
                semilogy(kr,abs((fft(tD1))))
                legend(fontsize=14)
                xlim((0,0.1))
                ylabel('power (arb.)')
                xlabel('k (1/\\textmu m)')
                pause(0.1)
                # savefig('MRO_withOverlappingOrderFFT.pdf')


                # plot(abs(fftshift(fft(tI1))))#,plot(Os(1))
                # plot(abs(fftshift(fft(hilbert(abs(tI1))))))#,plot(Os(1))
                # for i in intf.T:
                #     plot(i)
                # if any( lims1 ): gca().set_xlim(lims1[0]), gca().set_ylim(lims1[1])
                # pause(0.1)
                # ax1 = gca()

                # fig = figure(gcf().number+1)
                # fig.canvas.manager.window.move(1500,200)

                # cla()
                # semilogy(xrng,abs(intf_fft), xrng,1000*filter)
                # semilogy(abs(intf_2))
                # if any( lims2 ): gca().set_xlim(lims2[0]), gca().set_ylim(lims2[1])
                # pause(0.1)
                while waitforbuttonpress(timeout=0) is False: pass
                # lims1 = (ax1.get_xlim(),ax1.get_ylim())
                # lims2 = (gca().get_xlim(),gca().get_ylim())

            # bframe.append(a_line)

        fig = figure(1)
        fig.canvas.manager.window.move(300, 200)
        plot(abs(a_line))
        fig = figure(gcf().number+1)
        fig.canvas.manager.window.move(1500,200)
        imshow(array(bframe).__abs__(),cmap='CMRmap',aspect=100)


    def simulate_MRO_O2_manuscript(self):
        # manuscript:lst:simulateMRO_TD2mode
        um = 1e6
        nm = 1e9
        scale = 30 #arbitrary to fill plot
        c = speed_of_light
        src = Source(center_wavelength=1300e-9, bandwidth=60e-9)# Simulate for Tomlins data
        w_rng   = src.w_rng
        w_0     = src.w_0
        sigma_w = src.sigma_w
        S_w = src.S(w_rng, w_0, sigma_w*2*scale/10)[0]
        f_max = 2*w_rng[-1]/2/pi # initial z range of TD source field
        ez = 100/f_max*src.SN * c # set scan range for Kronecker array
        ez_rng = linspace(0,ez,src.SN)
        WL_rng  = src.WL_rng * nm
        CWL     = src.CWL    * nm
        BW      = src.BW     * nm
        E_t_i = fftshift(fft(S_w))
        E_t_r = E_t_i/E_t_i.max() # ideal scanning mirror
        Sw1 = src.S(w_rng, w_0, sigma_w * 2 * 2 * scale/10)[0] # order 1
        Sw2 = src.S(w_rng, w_0, sigma_w * 2 * 3 * scale/10)[0]
        E_t_r1 = fftshift(fft(Sw1)) # virtual scanning field
        E_t_i2 = fftshift(fft(Sw2))
        bframe = zeros((1,src.SN))
        for z in arange(0,1500,10): # in um
            z1 = z/1 # 1st order displacement
            z2 = z/2 # 2nd order displacement
            z3 = z/3 # 3rd order displacement
            z4 = z/4 # 4th order displacement
            acd1 = z1 - z2# 1st inter order displacement
            acd2 = z2 - z4# 2nd inter order displacement
            rs_kd1, zr = Sample(ns=[1+1e-01],z_widths=[z1*scale]).kronecker_deltas(src,new_z_rng=ez_rng)[0:2]
            rs_ac1, zr = Sample(ns=[1+1e-11],z_widths=[acd1*scale]).kronecker_deltas(src,new_z_rng=ez_rng)[0:2]
            rs_ac2, zr = Sample(ns=[1+1e-15],z_widths=[acd2*scale]).kronecker_deltas(src,new_z_rng=ez_rng)[0:2]
            E_t_s1  = convolve(E_t_i, rs_kd1, mode='same') # sample field
            E_t_ac1 = convolve(E_t_i, rs_ac1, mode='same')
            E_t_ac2 = convolve(E_t_i, rs_ac2, mode='same')
            E_t_sc1  = convolve(E_t_r, E_t_s1, mode='same') # scanning
            E_t_sac1 = convolve(E_t_r1, E_t_ac1, mode='same') # scanning 1st inter order
            E_t_sac2 = convolve(E_t_r1, E_t_ac2, mode='same') # scanning 2st inter order
            E_t_d = E_t_sc1 + E_t_sac1 + E_t_sac2
            a_line = log10(abs((E_t_d))**2)
            bframe = concatenate((bframe, [a_line]))
        # manuscript:lst:simulateMRO_TD2mode



def run():

    # Manuscript_kronecker_srf()

    # Test_TD_Doppler() #todo add to manuscript the results.
    # Simulate_TD.theoretical_vs_measured_reflectivities()
    # Simulate_TD()
    # Simulate_TD_manuscript()

    # Simulate_SD()
    # Simulate_SD().test_SD_OCT()
    # Simulate_SD().test_autocorrelation_terms()
    # Simulate_SD().test_multi_sample_reflectors()
    # Simulate_SD().test_Tomlins_FD_SD()

    # Simulate_MRO().test_0_MRO()
    Simulate_MRO().test_1_MRO() # spectral mix
    # Simulate_MRO().test_2_MRO()
    # Simulate_MRO().simulate_MRO_O1()
    # Simulate_MRO().simulate_MRO_O2()
    # Simulate_MRO().test_tick_scaling()
    # Simulate_MRO().test_simple_self_interference()
    # Simulate_MRO().manuscript_plots_simple_self_interference()
    show()
run()