from BLEHardware import Transmitter, Receiver
from Gain import getgain
import matplotlib.pyplot as plt
import numpy as np

class Model():
    def __init__(self,ant_type,sensorgridx,sensorgridy):
        colors = ['b','g','y','k']
        self.circles = []
        self.transmitters = []
        i = 0
        for x in sensorgridx:
            for y in sensorgridy:
                self.transmitters.append(Transmitter([x,y],ant_type=ant_type,col=colors[i%4]))
                i+=1

        self.receivers = []
        self.receivers.append(Receiver([50,50],[5,3]))
        for r in self.receivers:
            r.settime(0)
        for t in self.transmitters:
            t.settime(0)
        

    def plotMap(self,time,ax):
        for t in self.transmitters:
            t.settime(time)
            t.plot()
            t.plotsignal()

        for r in self.receivers:
            r.settime(time)
            r.plot()

            for i,t in enumerate(self.transmitters):
                s = r.compute_signal(t)
                rx = r.position[0]
                ry = r.position[1]
                tx = t.position[0]
                ty = t.position[1]
                transmitter_msg = str(np.floor(s)) + "dbm"
                plt.text(tx,ty,transmitter_msg)
                
                if t.ant_type == "yagi":
                    #if s=-94 --> lw=1, a=0.
                    #if s=-84 --> lw=2, a=0.5,
                    #a = (s[0]+94)/20.0
                    #if a>1: a = 1
                    #if a<0: a = 0
                    a = 1
                    plt.plot([rx,tx],[ry,ty],lw=1+(s+94)/10,alpha=a)
                    r.add_record(i,s,time,t.direction)

                """
                if (time%20 == 0):
                    norm = np.linalg.norm(r.position - t.position)
                    circle = plt.Circle((tx,ty), norm, fill=False)
                    ax.add_patch(circle)
                    r.add_omni_record(time,tx,ty,norm)

            if (time%20 == 0):
                c1 = r.omni_record[time][0]
                c2 = r.omni_record[time][1]
                c3 = r.omni_record[time][2]
                x,y = self.intersection_three_signals(c1,c2,c3)
                fly_point = plt.Circle((1000,1000), 50)
                self.circles.append(fly_point)
            """

            

    def intersection_three_signals(self,s1,s2,s3): #(x,y,r(signal strength))
        (x1,y1,r1) = s1
        (x2,y2,r2) = s2
        (x3,y3,r3) = s3

        y = ((x2-x3)*((x2**2-x1**2)+(y2**2-y1**2)+(r1**2-r2**2)) 
            - (x1-x2)*((x3**2-x2**2)+(y3**2-y2**2)+(r3**2-r2**2))) / (2*((y1-y2)*(x2-x3)-(y2-y3)*(x1-x2)))
        
        x = ((y2-y3)*((y2**2-y1**2)+(x2**2-x1**2)+(r1**2-r2**2)) 
            - (y1-y2)*((y3**2-y2**2)+(x3**2-x2**2)+(r3**2-r2**2))) / (2*((x1-x2)*(y2-y3)-(x2-x3)*(y1-y2)))
        
        return x,y

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

    def interactivePlot(self):
        plt.ion()
        fig, ax = plt.subplots()
        for t in self.transmitters:
            t.plot()
        plt.show()
    

    def plotframe(self,i,ax=None):
        """
        Plots a single frame.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        plt.title(int(i))
        self.plotMap(i,ax)
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
        