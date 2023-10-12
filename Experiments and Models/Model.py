from BLEHardware import Transmitter, Receiver
from Gain import getgain
import matplotlib.pyplot as plt
import numpy as np

class Model():
    def __init__(self,ant_type,sensorgridx,sensorgridy):
        colors = ['b','g','y','k']
        self.transmitters = []
        i = 0
        for x in sensorgridx:
            for y in sensorgridy:
                self.transmitters.append(Transmitter([x,y],ant_type=ant_type,col=colors[i]))
                i+=1

        self.receivers = []
        self.receivers.append(Receiver([50,50],[5,3]))
        for r in self.receivers:
            r.settime(0)
        for t in self.transmitters:
            t.settime(0)
        

    def plotMap(self,time):
        for t in self.transmitters:
            t.settime(time)
            t.plot()
            t.plotsignal()
        for r in self.receivers:
            r.settime(time)
            r.plot()
            for i,t in enumerate(self.transmitters):
                s = r.compute_signal(t)
                
                if s>-94:
                    #if s=-94 --> lw=1, a=0.
                    #if s=-84 --> lw=2, a=0.5,
                    #a = (s[0]+94)/20.0
                    #if a>1: a = 1
                    #if a<0: a = 0
                    a = 1
                    plt.plot([r.position[0],t.position[0]],[r.position[1],t.position[1]],lw=1+(s+94)/10,alpha=a)
                r.add_record(i,s,time,t.direction)
            
    def plotGraph(self,time):
        for i,r in self.receivers[0].record.items():            
            r = np.array(r)
            peaks = (r[10:-10]>r[20:]+15) & (r[10:-10]>r[:-20]+15) & (r[10:-10]>r[11:-9]) & (r[10:-10]>r[9:-11])
            peak_idxs = np.where(peaks)[0]+10
            if len(peak_idxs)>0:
                for r_peak in r[peak_idxs,:]:
                    self.transmitters[i].plotvector(r_peak[2])
            if len(r)<1: continue
            sig = 10**(r[:,1]/10)
            #thresh = 10**(-94/10)
            #plt.plot(r[sig>thresh,0]*20+20,sig[sig>thresh]+20,'-x')
            thresh = -110
            keep = r[:,1]>=thresh
            plt.plot(r[:,0]*10+200,100+(r[:,1]-thresh)*10,'-'+self.transmitters[i].color)
            
            #print(np.max(sig))
        plt.plot([200,2800],[100,100],'k-')
        

    def plotframe(self,i,ax=None):
        """
        Plots a single frame.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        plt.title(int(i))
        self.plotMap(i)
        self.plotGraph(i)
        
    def makemovie(self,filename=None):
        """
        Generates a diagnostic/debug movie, saved in 'filename'.
        """
        from moviepy.editor import VideoClip
        from moviepy.video.io.bindings import mplfig_to_npimage
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10,10))

        def make_frame(t):
            ax.clear()
            ax.set_aspect('equal', 'box')
            ax.set_xlim([0,3000])
            ax.set_ylim([0,3000])
            self.plotframe(t*10,ax=ax)    
            return mplfig_to_npimage(fig)

        Nframes = 25
        animation = VideoClip(make_frame, duration = Nframes)
        if filename is not None:
            animation.write_videofile(filename,fps=25,codec='mpeg4',bitrate='3000k')
        else:
            return animation.ipython_display(fps = 20, loop = False, autoplay = True)
        