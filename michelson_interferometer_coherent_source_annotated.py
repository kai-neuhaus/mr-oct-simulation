from scipy.constants import speed_of_light
from scipy import *
from matplotlib.pyplot import *

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


def plot_coherent_wave():
    N = 20000  # buffer size
    SR = 20e6  # sample rate (seconds)
    nnr = linspace(0, N, N).astype(complex) # sample range
    tmr = linspace(0, N/SR, N).astype(complex) # time range
    wavelength = 1330e-9 # meter
    v_M = 0.01 # meter / second
    f_D = 2 * v_M / wavelength
    print('f_D',f_D,'Hz')
    spr = linspace(0, f_D*wavelength*N/SR, N) # spatial range
    # coherent_wave = sin(2*pi*f_D*tmr))
    coherent_wave = exp(-1j*2*pi*f_D*tmr + 1j*pi/2)
    plot(spr*1e6, coherent_wave)
    xlabel('time (s)')
    ylabel('Amplitude (arb.)')
    grid(True)
    show()


import michelson_interferometer_coherent_source as mi

class Annotate_document_plot(MPLHelper):
    def annotate_document_plot():
        xlabel('range ($\mu m$)')
        ylabel('Amplitude (arb.)')

        # figure()
        cycle_width = (where(diff(mi.coherent_wave) == 0)[0])
        # print(cycle_width)
        # spr * 1e6, coherent_wave)
        # plot(cycle_width/mi.SR*2*mi.v_M*1e6,ones(len(cycle_width)),'.')
        cycle_bounds_x = where(abs(diff(real(sqrt(mi.coherent_wave)))) < 1e-6)[0]
        cbx_single = array((cycle_bounds_x[3], cycle_bounds_x[4]))
        cbx_single_um = cbx_single/mi.SR*mi.v_M*1e6*2
        print(cycle_bounds_x)
        print(len(cycle_bounds_x))
        # plot(abs((real(sqrt(mi.coherent_wave)))))
        # plot(cbx_single_um,ones(len(cbx_single)),'-')
        gca().annotate('$\lambda$',
                       textcoords='data',
                       xycoords='data',
                       xy=(cbx_single_um[0]-0.1,1.0),
                       xytext=(cbx_single_um[1]+0.15,1.0-0.03),
                       arrowprops=dict(arrowstyle='<->'))
        tight_layout()
        savefig('monochromatic_wave_labeled.pdf')
        show()

Annotate_document_plot.annotate_document_plot()