from BLEHardware import Transmitter, Receiver, BLEReciever, Antenna, Array
from Gain import getgain

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

import numpy as np

from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

class Model():
    def __init__(self,ant_type,sensorgridx,sensorgridy):
        colors = ['b','g','y','k']
        self.circles = []
        self.array = []
        i = 0
        for x in sensorgridx:
            for y in sensorgridy:
                self.array.append(Transmitter([x,y],ant_type=ant_type,col=colors[i%4]))
                i+=1

        self.receivers = []
        self.receivers.append(Receiver([50,50],[5,3]))
        for r in self.receivers:
            r.settime(0)
        for t in self.array:
            t.settime(0)
        

    def plotMap(self,time,ax):
        for t in self.array:
            t.settime(time)
            t.plot()
            t.plotsignal()

        for r in self.receivers:
            r.settime(time)
            r.plot()

            for i,t in enumerate(self.array):
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
                    self.array[i].plotvector(r_peak[2])

            if len(r)<1: continue

            sig = 10**(r[:,1]/10)
            #thresh = 10**(-94/10)
            #plt.plot(r[sig>thresh,0]*20+20,sig[sig>thresh]+20,'-x')
            thresh = -110
            keep = r[:,1]>=thresh
            plt.plot(r[:,0]*10+200,100+(r[:,1]-thresh)*10,'-'+self.array[i].color)
            
            #print(np.max(sig))
        plt.plot([200,2800],[100,100],'k-')

    def interactivePlot(self):
        plt.ion()
        fig, ax = plt.subplots()
        for t in self.array:
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
        

class NewModel:
    def __init__(self, noiseFloor,forced_array=None) -> None:
        if forced_array == None:
            self.array = [Array([0,0],0,1,None)]
        else:
            self.array = forced_array
        self.receiver = BLEReciever(self.randomPosition(28))
        self.noise_floor = noiseFloor
        self.time = 0

    def randomPosition(self, beamwidth):
        theta = np.random.random_integers(np.floor(-beamwidth/2),np.floor(beamwidth/2))
        distance = np.random.randint(15,500)
        x = np.cos(np.deg2rad(theta)) * distance
        y = np.sin(np.deg2rad(theta)) * distance
        self.test_angle = theta
        return np.array([x,y])
        
    def signalRecieved(self, transmitter : Array):
        delta = self.receiver.position - transmitter.position

        theta = np.arctan2(delta[1],delta[0])
        dist = np.linalg.norm(delta)

        (ant1, s1), (ant2, s2), (ant3, s3) = transmitter.baseGain(int(np.rad2deg(theta)))
        actualGain1 = s1#self.FSPL(s1, dist)
        actualGain2 = s2#self.FSPL(s2, dist)
        actualGain3 = s3#self.FSPL(s3, dist)

        self.receiver.addSignal({ant1.id : actualGain1, ant2.id : actualGain2, ant3.id : actualGain3})

    def FSPL(self, baseGain, dist):
        return baseGain - (20*np.log10(dist)+40.05-(self.receiver.gain))
    
    def noiser(self,gi):
        rnd = np.random.default_rng()
        if (gi - self.noise_floor) > 0:
            return gi
        else:
            return gi + (rnd.random() * (self.noise_floor - gi))
    
    def rudimentaryNoise(self,g):
        noise = np.vectorize(self.noiser)(g)
        return noise
    
    def plotarray(self):
        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
        for tran in self.array:
            tran.plot(ax)
        ax.legend()

    def plotRecievedSignal(self):
        fig, ax = plt.subplots()
        self.receiver.plotSignals(ax)

    def plotIdealMono(self,width):
        fig, ax = plt.subplots()
        monos = self.array[0].idealMonoFunction(width)
        (antA1, antB1, key1, theta1, mono1) = monos[0]
        ax.plot(theta1, mono1, label=key1)
        (andA2, antB2, key2, theta2, mono2) = monos[2]
        ax.plot(theta2, mono2, label=key2)   
        ax.legend()

    def estimateAngle(self, arr : Array) -> [float]: #{ant1 : RSSI1, ant2 : RSSI2, ant3 : RSSI3} - for simplicity at the moment
        signals = self.receiver.signal
        mono12 = self.monoFuntion(signals[arr.antennas[0].id],signals[arr.antennas[1].id])
        print(mono12)
        mono13 = self.monoFuntion(signals[arr.antennas[0].id],signals[arr.antennas[2].id])
        print(mono13)
        mono23 = self.monoFuntion(signals[arr.antennas[1].id],signals[arr.antennas[2].id])
        print(mono23)

        arrayModel = arr.idealMonoFunction(40) #20 degrees either side
        print(arrayModel)

        theta12 = []
        theta13 = []
        theta23 = []
        for (anta,antb,key,theta,mono) in arrayModel:
            if anta == arr.antennas[0].id  and antb == arr.antennas[1].id:
                theta12 = self.anglesWithMono(theta,mono,mono12)
            elif anta  == arr.antennas[0].id and antb == arr.antennas[2].id:
                theta13 = self.anglesWithMono(theta,mono,mono13)
            elif anta == arr.antennas[1].id  and antb == arr.antennas[2].id:
                theta23 = self.anglesWithMono(theta,mono,mono23)

        print("Possible angles:")
        print(theta12,theta13,theta23)

    def plotArrReciever(self):
        fig, ax = plt.subplots()
        ax.plot(self.array[0].posX, self.array[0].posY, 'x')
        ax.plot(self.receiver.posX, self.receiver.posY, 'o')
        print("angle to test:")
        print(self.test_angle)

    def monoFuntion(self, signal1, signal2): #signal 2 is always taken from signal 1
        sum = signal1 + signal2
        diff = signal1 - signal2
        return diff/sum
    
    def anglesWithMono(self,theta,modelMono,mono):
        matches = modelMono == mono
        return theta[matches]
    
    def runTest(self):
        self.signalRecieved(self.array[0])
        print("Estimating angle using received signal")
        self.estimateAngle(self.array[0])