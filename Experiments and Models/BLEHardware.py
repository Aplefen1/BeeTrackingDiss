import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Gain import getgain
import scipy.io

class Transmitter:
    def __init__(self,position,ant_type='yagi',power=10, col='k'): #dBm
        self.position = np.array(position)
        self.ant_type = ant_type
        self.power = power
        self.initdirection = np.random.rand()*np.pi*2
        self.direction = self.initdirection
        self.fwhm = np.deg2rad(36)
        self.color = col

    def plot(self):
        plt.plot(self.position[0],self.position[1],'xb')
        #plt.plot([self.position[0],self.position[0]+np.cos(self.direction)*arrowsize],[self.position[1],self.position[1]+np.sin(self.direction)*arrowsize],'-b')

    def plotsignal(self):
        ang = np.linspace(0,np.pi*2,100)
        a = (ang - self.direction)# % (np.pi*2) - np.pi
        a[a>np.pi]-=np.pi*2
        a[a<-np.pi]+=np.pi*2
        #FWHM is 2.355 sigma, so sigma = 
        #sigma = self.fwhm/2.355
        if self.ant_type=='yagi':
            s = 10**(getgain(a)/10) #np.exp(-a**2/(2*sigma**2))*400
        else:
            s = np.zeros_like(a) #assume omnidirectional
        #s = (-a**2/(2*sigma**2)) #log..
        x = self.position[0]
        y = self.position[1]
        plt.plot(x+np.cos(ang)*s*25,y+np.sin(ang)*s*25,alpha=1,lw=3,color=self.color)

    def plotvector(self,ang):
        #for ang in angs:
        x = self.position[0]
        y = self.position[1]
        plt.plot([x,x+np.cos(ang)*10000],[y,y+np.sin(ang)*10000],':'+self.color,alpha=0.5,lw=1)

    def plotcircle(self,bee):
        x = self.position[0]
        y = self.position[1]

    def getgain(self,angle):
        if self.ant_type=='yagi':
            return getgain(angle - self.direction)
        else:
            return np.zeros_like(angle)
        
    def settime(self,t):
        self.direction = (self.initdirection + t*0.15)%(np.pi*2) #radians per sec

class Receiver:
    def __init__(self,startposition,velocity):
        self.startposition = np.array(startposition)
        self.velocity = np.array(velocity)
        self.gain = -10
        self.record = {}
        self.omni_record = {}

    def add_record(self,transmitter_id,signal,time,angle):
        if transmitter_id not in self.record: self.record[transmitter_id] = []
        self.record[transmitter_id].append([time,signal,angle])

    def add_omni_record(self,time,x,y,r):
        if time not in self.omni_record: self.omni_record[time] = []
        self.omni_record[time].append((x,y,r))
        
    def plot(self):
        plt.plot(self.position[0],self.position[1],'og')

    def settime(self,t):
        #self.position = (self.startposition + t*self.velocity)
        a = -2*np.pi*t/400
        self.position = np.array([1500+200*np.cos(a),1800+200*np.sin(a)]) #(self.startposition + t*self.velocity)

    def compute_signal(self,transmitter):
        delta = self.position - transmitter.position
        #print(self.position,transmitter.position)
        #print(delta)
        ang = np.arctan2(delta[1],delta[0])
        #print(ang)
        dist = np.linalg.norm(delta)
        #print(transmitter.getgain(np.array([ang])))
        lamb = 0.125 #m (wavelength)
        #transmitter gain + transmitter power + pathloss + reciever gain(assumed to be quite poor: -10)
        return transmitter.getgain(np.array([ang])) + transmitter.power - (20*np.log10(dist)+40.05-self.gain)
    

class Antenna:
    def __init__(self, direction, power, pos, groupID, antID) -> None:
        self.direction = direction
        self.power = power
        self.position = pos
        self.id = str(groupID)+str(antID)

        wanted_keys = ('theta', 'magdB')
        mat = scipy.io.loadmat('ExampleDishFarfieldAz.mat')
        pqr = pd.Series(mat)
        mat_dict = {key : pqr.to_dict()[key] for key in wanted_keys}
        theta = []
        deg = []
        magdB = []
        zeroVal = 0

        for i in range(0, len(mat_dict['theta'][0])):
            if mat_dict['theta'][0][i] == 0: zeroVal = mat_dict['magdB'][i][0]
            if mat_dict['theta'][0][i] < 0: 
                theta.append(np.deg2rad(mat_dict['theta'][0][i] + 360))
                deg.append(mat_dict['theta'][0][i]+360)
            else:
                theta.append(np.deg2rad(mat_dict['theta'][0][i]))
                deg.append(mat_dict['theta'][0][i])
            magdB.append(mat_dict['magdB'][i][0])

        theta.append(np.pi*2)
        magdB.append(zeroVal)
        deg.append(360)
        self.df = pd.DataFrame()

        self.df["deg"] = deg
        self.df["theta"] = theta
        self.df["magdB"] = magdB

        self.df = self.df.sort_values("deg")

        self.gain = self.df["magdB"]
        self.gain = np.roll(self.gain, (self.direction))

    def baseGain(self, deg):
        gain = self.gain[deg] + self.power
        theta = np.deg2rad(deg)
        return theta, gain
    
    def polarPlot(self,ax,rot):
        ax.plot(self.df['theta'] + np.deg2rad(self.direction), self.df['magdB'])
    
    def plot(self,ax,rot):
        x = self.position[0]
        y = self.position[1]
        ax.plot(x+np.cos(self.df["theta"])*self.df["magdB"],y+np.sin(self.df["theta"])*self.df["magdB"],alpha=1,lw=3)
    

class BLEReciever:
    def __init__(self,startposition) -> None:
        self.position = startposition
        self.gain = -3 #from data sheet
        self.sensitivity = -94 #from ds

    #(antennaID, time, RSSI)

    def plotSignals(self,ax):
        ax.plot(self.recordedAngles,self.recordedSignals)

class Array:
    def __init__(self,pos,vel,id) -> None:
        self.position = pos
        self.direction = 0
        self.antennas = []
        self.angular_vel = np.pi/2
        self.id = id

        for i in range(0,3):
            dir = (-10)+(i*10) + self.direction
            self.antennas.append(Antenna(dir,14,pos,self.id,i))

    def plot(self,ax):
        for ant in self.antennas:
            ant.polarPlot(ax,0)

    def polarPlot(self,ax):
        for ant in self.antennas:
            ant.polarPlot(ax,np.pi)

    def idealMonoFunction(self, beam_width) -> [(str, [float], [float])]: #return beam width degrees to monopulse function
        half_beam = int(np.floor(beam_width/2))
        angles = [v for v in range((-half_beam),(half_beam))]
        indexes = [v for v in range(360-half_beam,360)] + [v for v in range(0,half_beam)]
        monoFunctions = []
        #A/B, A/C, B/C
        for i in range(0,len(self.antennas)-1):
            for j in range(i+1,len(self.antennas)):
                aTheta, aGain = self.antennas[i].baseGain(indexes)
                bTheta, bGain = self.antennas[j].baseGain(indexes)
                key = self.antennas[i].id + "," + self.antennas[j].id
                sum = aGain + bGain
                diff = aGain - bGain
                mono = diff / sum
                monoFunctions.append((key,angles,mono))
        
        return monoFunctions