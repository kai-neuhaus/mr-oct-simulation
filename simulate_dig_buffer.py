from scipy import *
from scipy.constants import speed_of_light
from scipy.fftpack import *
from scipy.interpolate import interp1d
from matplotlib.pyplot import *


class Evaluation(object):
    '''
    Most of the functions here are to evaluate the simulation of the signal as it
    appears in the digitizer buffer.

    Generate a delta function for each reflection and move step by step.
    '''
    def sin_vs_lin_motion(self):
        plot(sin(linspace(0,4*pi,9)+pi/2))
        plot(linspace(0,8,100),sin(linspace(0,4*pi,100)+pi/2),'--')
        tight_layout()
        savefig('sin_vs_lin_motion.pdf')
        show()

    def SRM_vs_time(self):
        # The significance here is to understand that we only show the delay for the scanning ranges
        # and no reflector yet !!!
        # So keep this snipped as is to have something to repeat the theory
        print('Displacement of the scan range boundaries.\n')
        # The interesting artifact here is that although all scanning ranges having the same start of
        # scanning time the optical delay of the scanning ranges shows up as a displacement.
        # That means all reflectors (single SM) is in space at the same position but the orders are displaced.
        # The center of the orders are dispcaled by (N-1)D.
        SN = 100 # samples
        L_S = 90e-6 # scanning range
        D = 145e-6 # spacing
        f_SRM = 152 # scanning rate
        t_S = 1/f_SRM # linear time one scan
        L_SM = -0e-6 # sample mirror position
        N = arange(1,11)
        # The initial time delay for each order is D if the reflection is positioned at this point.
        # However, the scanning starts at the beginning of the
        s_N = (L_S/2 + (N-1)*D - N*L_S/2)
        t_N = t_S/L_S * s_N

        set_printoptions(linewidth=200)
        print('scanning time:',t_S)
        print('s_N (um)',s_N)
        print('t_N (um)',t_N)
        # print('total range: ', L_S/2 + (N-1)*D + N*L_S/2 )
        plot(s_N,'o')
        title('Start delay in buffer')
        show()

    def SM_vs_time(self):
        SN = 100 # samples
        L_S = 90e-6 # scanning range
        D = 145e-6 # spacing
        f_SRM = 152 # scanning rate
        t_S = 1/f_SRM # linear time one scan
        L_SM = -0e-6 # sample mirror position
        N = arange(1,11)

    def plot_pauls_version(self):
        '''
        What is important here?
        Please observe the generation of range_order which is different for each order!
        The ratio is found by f_SRM * t_SRM * L_S * N.
        The plots show order length vs. sample range.
        This means while the amoun of samples remains constant the range of an order increases by the ratio
        f_SRM * t_SRM * L_S * N.
        The ratio f_SRM * t_SRM is nothing else than calculating the relative sample point vs. t_SRM.
        :return:
        '''
        SN = 100 # samples
        L_S = 90e-6 # scanning range
        D = 145e-6 # spacing
        f_SRM = 152
        t_SRM = linspace(0, 1 / (2 * f_SRM), SN)

        N = arange(1,11)
        L_0 = 0
        for n in N:
            # f_SRM * t_SRM is the ratio or normalized position scaled to N * L_S
            range_order = D * (n - 1) - n * L_S * (1 / 2 - 2 * f_SRM * t_SRM)
            # To visualize the range we need to us it as x-range!
            plot(range_order, n*ones(len(range_order)))
            xlabel('samples')
            ylabel('scan range (m)')
            pause(0.001)
            waitforbuttonpress()

        show()

    def plot_pauls_version_with_reflector(self):
        '''
        What is important here!
        The sample rate for each range is the same! See the dot plot!
        However, the range values are re-calculated according to the actual order length!

        So it is making it a bit more clear here that sample rate and the value of each sample
        contributes to the placement of orders in space.

        Can we here already deduce an equation that does not require to create a matching sample space
        but instead calculates the actual position for a constant sample space?
        :return:
        '''
        # It is important to realize that pauls version does prepare the range for the Gaussian!
        # The range itself is not a signal but a scanning range representation that describes
        # an array vector as a time or spatial base.
        # If we do not use the Gaussian equation the reflector must be placed directly!
        # Furthermore we do not merge all ranges into a single buffer len yet.
        # We can simulate the Gaussian by a delta within the range.
        # Although, we lift each range by N the delta will be lifted as well.
        SN = 15000 # samples
        L_S = 100 # scanning range
        D = 100 # spacing
        f_SRM = 152
        t_SRM = linspace(0, 1 / (2 * f_SRM), SN)
        v_SRM = L_S * f_SRM
        N = arange(1,11)
        L_0 = 179

        # This simulates the orders before merged into the buffer segment.
        # Take note that this shows a true representation of orders in space.
        # The goal here is to extract the equation to describe this without the range_order!
        for n in N:
            range_order = L_S/2 + D * (n - 1) - n * L_S * (1 / 2 - 2 * f_SRM * t_SRM)
            # range_order = L_S/2 + D * (n - 1) - n * L_S / 2 + n * 2 * v_SRM * t_SRM
            # Generate a signal. Like a Gaussian we get values along range but here only there where L_0!
            # The range is lifted by n to visualize the seprate orders.
            sig = n * ones(len(range_order))
            print(abs(range_order))
            precision = L_S/SN*n
            print(precision)
            # Allocate the closest sample relative to the position.
            sig_idx = (where(abs(array(range_order)-L_0)<precision))
            print(sig_idx)
            sig[sig_idx] = sig[sig_idx] + 0.5 # set delta at reflector
            # To visualize the range we need to us it as x-range!
            plot(range_order, sig)
            ylabel('order')
            xlabel('scan range (m)')
            # pause(0.001)
            # waitforbuttonpress()
        plot([L_0, L_0],[0,N.max()])
        plot([0, 0],[0,N.max()])
        show()

    def plot_mirror_signal_by_equation(self):
        '''
        What is important here!

        The displacement in one buffer is now explicityl calculated (see x = ...) in code.
        Consequently the full mathematical description of such signal is now available.
        The x value is universal in relation to the time and space.
        :return:
        '''
        SN = 15000 # samples
        L_S = 90e-6 # scanning range
        D = 145e-6 # spacing
        f_SRM = 152
        t_SRM = linspace(0, 1 / (2 * f_SRM), SN)
        v_SRM = L_S * f_SRM
        N = arange(1,11)
        L_0 = arange(0,1500,5)*1e-6

        l_s_R = linspace(0,L_S,SN)
        for l_0 in L_0:
            signals = []
            for n in N:
                precision = L_S/SN*n

                signal = zeros(SN)

                print('n D L_S L_0', n, D, L_S, l_0, end=' ')

                # x = (l_0 - (n-1)*D - L_S/2 + n*L_S/2)/n # forward direction for step motion
                x = L_S - (l_0 - (n-1)*D - L_S/2 + n*L_S/2)/n # reverse direction of motion in buffer
                print('x', x, flush=True)

                signal[where(abs(l_s_R-x)<precision)] = 1.0
                signals.append(signal)
            cla()
            # plot(l_s_R,sum(signals,axis=0))
            plot(sum(signals,axis=0))
            while waitforbuttonpress() == 0: pass

        show()

    def plot_mirror_signal_by_equation_hold_plot(self):
        '''
        This plot shows the motion within different buffer windows respectively scan ranges.
        @return:
        '''
        SN = 15000 # samples
        L_S = 90e-6 # scanning range
        D = 145e-6 # spacing
        f_SRM = 152
        t_SRM = linspace(0, 1 / (2 * f_SRM), SN)
        v_SRM = L_S * f_SRM
        N = arange(1,11)
        L_0 = arange(0,1500,5)*1e-6

        L_S_tot = 1/2*D + (N.max()-1)*D + 1/2*D*N.max()
        l_s_R = linspace(0,L_S,SN)

        get_idx = interp1d(l_s_R,linspace(0,SN,SN))
        for l_0 in L_0:
            signals = []
            for n in N:
                precision = L_S/SN*n

                signal = -ones(SN)

                # print('n D L_S L_0', n, D, L_S, l_0, end=' ')
                # forward direction for step motion
                # x = (l_0 - (n-1)*D - L_S/2 + n*L_S/2)/n

                # reverse direction of motion in buffer
                # x = L_S - (l_0 - (n-1)*D - L_S/2 + n*L_S/2)/n

                # include the window offset.
                # that means if we would call cla() each time then we see only part of all orders.
                # That means the actual distance between the first first and next first is larger.
                x =  L_S - (l_0 - (n-1)*D - L_S/2 + n*L_S/2)/n

                # print('x', x, flush=True)
                try:
                    print('xi',get_idx(x))
                except Exception as e:
                    print('x=',x,'!')
                    pass

                signal[where(abs(l_s_R - x) < precision)] = 2.0
                signals.append(signal)
                # figure(1)
                # plot(x, x,'o')
                # pause(0.1)
                # while waitforbuttonpress() == 0: pass
            signals = array(signals).sum(axis=0)
            # signals[where(signals<1)]=nan
            figure(2)
            cla()
            # plot(l_s_R,signals, 'o')
            plot(signals,'o')
            # ylim((-0.1,1.5))
            while waitforbuttonpress() == 0: pass

        show()


    def plot_mirror_signal_by_equation_compare_sketch(self):
        '''
        What is important here!

        The displacement in one buffer is now explicityl calculated (see x = ...) in code.
        Consequently the full mathematical description of such signal is now available.
        The x value is universal in relation to the time and space.
        :return:
        '''
        SN = 15000 # samples
        L_S = 100 # scanning range
        D = 100 # spacing
        f_SRM = 152
        t_SRM = linspace(0, 1 / (2 * f_SRM), SN)
        v_SRM = L_S * f_SRM
        N = arange(1,11)
        L_0 = [179] # range(0,400,10)

        l_s_R = linspace(0,L_S,SN)
        for l_0 in L_0:
            signals = []
            for n in N:
                precision = L_S/SN*n

                signal = zeros(SN)
                x = (l_0 - (n-1)*D - L_S/2 + n*L_S/2)/n

                signal[where(abs(l_s_R-x)<precision)] = 1.0
                signals.append(signal)
            cla()
            plot(l_s_R,sum(signals,axis=0))
            while waitforbuttonpress() == 0: pass

        show()

    def plot_pauls_version_with_reflector_time_scale_change(self):
        '''
        What is important here!
        Very important the implicit change of the time scale.
        Please see how by changing the plotting range to t_SRM the actual reduction of the positions is achieved.
        This is implicit and not obvious.

        Not only are the orders aligned automatically but also the relative size has changed!

        The goal is to show that this can be achieved directly.
        :return:
        '''
        SN = 15000 # samples
        L_S = 90e-6 # scanning range
        D = 145e-6 # spacing
        f_SRM = 152

        N = arange(1,11)
        L_0 = 400e-6

        sig_tot = []
        for n in N:
            t_SRM = linspace(0, 1 / (2 * f_SRM), SN)
            range_order = D * (n - 1) - n * L_S * (1 / 2 - 2 * f_SRM * t_SRM)
            # Generate a signal. Like a Gaussian we get values along range but here only there where L_0!
            # The range is lifted by n to visualize the seprate orders.
            sig = n*ones(len(range_order))
            print(abs(range_order))
            precision = L_S/SN*n
            print(precision)
            # Allocate the closest sample relative to the position.
            sig_idx = (where(abs(array(range_order)-L_0)<precision))
            print(sig_idx)
            sig[sig_idx] = sig[sig_idx] + 0.5 # set delta at reflector
            sig_tot.append(sig)
            sig_sum = sum(sig_tot,axis=0)
            # To visualize the range we need to us it as x-range!
            # cla()
            plot(t_SRM, sig)
            ylabel('amplitude')
            xlabel('scan range (m)')
            pause(0.001)
            waitforbuttonpress()

        show()

    def plot_pauls_version_with_reflector_merged(self):
        '''
        What is important here!
        Very important the implicit change of the time scale.
        Please see how by changing the plotting range to t_SRM the actual reduction of the positions is achieved.
        This is implicit and not obvious.

        Not only are the orders aligned automatically but also the relative size has changed!

        The goal is to show that this can be achieved directly.
        :return:
        '''
        SN = 15000 # samples
        # L_S = 90e-6 # scanning range
        # D = 145e-6 # spacing
        L_S = 100
        D=100
        f_SRM = 152

        N = arange(1,11)
        L_0 = 179

        sig_tot = []
        for n in N:
            t_SRM = linspace(0, 1 / (2 * f_SRM), SN)

            range_order = L_S/2 + D * (n - 1) - n * L_S * (1 / 2 - 2 * f_SRM * t_SRM)
            # Generate a signal. Like a Gaussian we get values along range but here only there where L_0!
            sig = zeros(len(range_order))
            # print(abs(range_order))
            precision = L_S/SN*n
            print(precision)
            # Allocate the closest sample relative to the position.
            sig_idx = (where(abs(array(range_order)-L_0)<precision))
            print(sig_idx)
            sig[sig_idx] =  sig[sig_idx] + 1.0 # set delta at reflector
            sig_tot.append(sig)
            sig_sum = sum(sig_tot,axis=0)
            # Attention here. The range_order is now t_SRM!
            cla()
            plot(2*L_S*f_SRM*t_SRM, sig_sum)
            ylabel('amplitude')
            xlabel('scan range (m)')
            grid(True)
            pause(0.001)
            while waitforbuttonpress() == 0 : pass

        show()

class run_this(object):
    # Evaluation().SRM_vs_time()
    # Evaluation().plot_pauls_version_with_reflector()
    # Evaluation().plot_mirror_signal_by_equation_compare_sketch()
    # Evaluation().plot_mirror_signal_by_equation()
    Evaluation().plot_mirror_signal_by_equation_hold_plot()
    # Evaluation().plot_pauls_version_with_reflector_time_scale_change()
    # Evaluation().plot_pauls_version_with_reflector_merged()


