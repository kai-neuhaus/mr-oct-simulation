from scipy import *
from scipy.fftpack import *
from scipy.signal import resample
from matplotlib.pyplot import *
import my_format_lib
writer = my_format_lib.load_video_modules()

#mro_fringe_simulation.m
#Paul McNamara
#Compact Imaging Ireland
#7 - 15th February 2018

buffer = 65580 # buffer size
SR = 20e6 # sample rate [1/s]
wavelength = 1300e-09 # [m]
wavelengthBW = 60e-9 # [m]
k = 2*pi/wavelength
delta_k=(pi/sqrt(log(2)))  * (wavelengthBW / wavelength ** 2)
Transmission=0.2

D = 140e-6 # mirror spacing [m]
s=90e-06  #Scan range in [m];
f=152 #[Hz]
t=linspace(0, 1/(2*f), buffer)  #Set up time axis

n_max=20
graph_count=0
s_nt = []
v = []
intensity = []
z = []
z_envelope = []
z_carrier = []
z_fringe = []
x = []
tMirror = []
t_envelope = []
t_carrier = []
t_fringe = []

fig = figure(num='Simulation (P)',figsize=(11, 4))
mirror_step = 5e-6 #[m]
mirror_step_start = 0
mirror_step_stop = 1000e-6

# with writer.saving(fig, "simulation_P.mp4", 100):
total_data = []
for zMirror in arange(mirror_step_start,mirror_step_stop,mirror_step):  #Position of the reference mirror

    graph_count=graph_count+1
    for n in range(1,n_max+1):
        # n_axis[n,1]=n

        s_nt.append( ( D*(n-1)  + ( -n * s/2 ))  + (2*n*f*s*t))  #Linear motion
        # s_nt.append( ( D*(n-1)) - ( (n * s/2 ) * cos( 2 * pi * f * t) ))  #Sinusoidal motion p.5-139
        # velocity_profile(:,n)=gradient(s_nt   (:,n),t)  #The derivative of the position
        # v.append(2*n*f*s)  #Average velocity of scanning mirror p.5-140

        intensity.append( Transmission**2 * (1-Transmission) ** (n-1))

        #Spatial equations
        # z.append( linspace(  ((n-1)*D)-n*s/2, ((n-1)*D)+n*s/2, buffer))  #Set up spatial axis
        # z_envelope.append( exp(-( ( (z[n-1] -  zMirror  ) / (1/delta_k)   )) ** 2    ))
        # z_carrier.append( cos(2*k*z[n-1]))
        # z_fringe.append( intensity[n-1] * z_envelope[n-1] * z_carrier[n-1])

        # if zMirror >= (D*(n-1)-(n*s/2)) and zMirror <= (D*(n-1)+(n*s/2)):  #If mirror is within range of scan
        # x.append( zMirror - ( D * (n-1)) + (n*s/2))  #p.5-136
        # tMirror.append( arccos( ( (D * (n-1) ) - zMirror ) / (n*s/2) ) / (2*pi*f))  #Sinusoidal case p. 5-141
            # tMirror(n,1)=  ( zMirror-    ((n-1)*D) + (n*s/2) ) / (v(1,n))  ;  #Linear case p.5-135-136
        # else:
        #     x.append(-1)
        #     tMirror.append(-1)

        #Temporal equations
        # print('zMirror',zMirror)
        t_envelope.append( exp( -( ( (s_nt[n-1] -  zMirror  ) / (1/delta_k)   )) ** 2    ))
        t_carrier.append( cos(    2*k*(s_nt[n-1])   ))
        t_fringe.append( intensity[n-1] * t_envelope[n-1] * t_carrier[n-1])
        t_fringe_total=sum(t_fringe,axis=0)

    total_data.append(t_fringe_total)
    tMirror = []
    t_envelope = []
    t_carrier = []
    t_fringe = []

    m=1
    l=1

    def do_plot():
        subplot(121)
        cla()
        plot(t*1e3 ,t_fringe_total[::-1])

        selected_lin_data = t_fringe_total[::-1]
        buff_seg_len = len(t_fringe_total)
        def plot_max_loc():
            envelope = abs(hilbert(selected_lin_data))
            envelope = resample(resample(envelope, num=100), num=buff_seg_len)*1.6
            # cla()
            # plot(t*1e3,envelope)

            max_idx = 0
            max_locs = []
            eps_max = 1
            env_med = median(envelope)
            # cutters = max()
            while True:
                max_val = max(envelope[max_idx:])
                level = max_val / 2
                # plot(ones(buff_seg_len) * level)
                pos = argmax(envelope[max_idx:]) + max_idx

                if pos <= max_idx: break
                if not any(where(envelope[max_idx:pos] < level)[0]): break
                # max_idx=pos
                # pos=pos+pos
                fwhm_l = max(where(envelope[max_idx:pos] < level)[0]) + max_idx
                if not any(where(envelope[pos:] < level)[0]): break
                fwhm_r = min(where(envelope[pos:] < level)[0]) + pos

                # max_locs.append({'max': level, 'max_loc': pos, 'l_loc': fwhm_l, 'r_loc': fwhm_r})
                plot([pos/SR*1e3, pos/SR*1e3], [-0.45, 0.45])
                fwhm_s = fwhm_r - fwhm_l
                gca().annotate('{:2.2f}'.format(fwhm_s/SR/max(t)*s*1e6),
                               xycoords='data',
                               textcoords='data',
                               xy=(pos/SR*1e3, max_val)
                               )

                # we half level again to catch next possible max
                if not any(where(envelope[fwhm_r:] > level / 2)[0]): break
                next_pos = max(where(envelope[fwhm_r:] > level / 2)[0]) + fwhm_r
                if next_pos > fwhm_r:
                    max_idx = fwhm_r
                else:
                    break

            print(max_locs)


        plot_max_loc()

        title(('Temporal Interference, FWHM ($\operatorname{\mu m}$)'))
        xlabel('Time (ms)')
        ylabel('Intensity')
        ylim([-0.045,0.045])
        grid(True)
        # axis([0 max(t) -intensity(1,1) intensity(1,1)]);

        subplot(122)
        cla()
        t_fringe_total_fft = abs(fft(t_fringe_total))
        fft_freq = fftfreq(n=buffer,d=1/SR*1e3)
        semilogy(fft_freq,t_fringe_total_fft)
        grid(True)
        ylim((1e-2,5e2))
        xlim((0,500))
        xlabel('Frequency (kHz)')

        pause(0.001)
        # writer.grab_frame()
        # savefig('simulation_p_{:04d}.jpg'.format(graph_count))
        # fig.canvas.manager.window.attributes('-topmost', 1)
        # fig.canvas.manager.window.attributes('-topmost', 0)
        while waitforbuttonpress() == 0: pass
    do_plot()

# save('ml_simu_data_nonlin.npy',total_data)
# from scipy.io.matlab import savemat
# savemat('ml_simu_data.mat',{'simu':total_data})

#     subplot(m,l,5);
#     hold on;
# %     plot(n_axis,x,'k+');
#     for i=1:n_max
#         plot(n_axis(i),x(1,i),'k+','Color',col(i,:),'MarkerSize',10);
#     end
#     plot([1 (zMirror+D)/(D-(s/2))],[zMirror+D+((s/2)-D) 0],'k-');  %p. 142
#     title('Distance from start of scan to mirror');
#     xlabel('Order, n');
#     ylabel('Distance (m)');
#     grid on;
#     xlim([0 n_max]);
#     ylim([0 5e-4]);

#     subplot(m,l,6);
#     hold on;
# %     plot(n_axis,tMirror,'k+');
#     for i=1:n_max
#         plot(n_axis(i),tMirror(i,1),'k+','Color',col(i,:),'MarkerSize',10);
#     end
#     title('Time of each scan to hit mirror');
#     xlabel('Order, n');
#     ylabel('Time (s)');
#     grid on;
#     xlim([0 n_max]);
#     ylim([0 max(t);]);
#
#     outfile1=sprintf('%s/Fringes_%.3d.tif',file_name,graph_count);
#     saveas(figure(1),outfile1,'tif');

