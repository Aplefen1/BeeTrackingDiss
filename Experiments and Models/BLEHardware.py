import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Gain import getgain

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
    def __init__(self, falloff, direction, max_gain, power, pos) -> None:
        self.falloff = falloff
        self.direction = direction
        self.max_gain = max_gain
        self.power = power
        self.position = pos

    def baseGain(self, theta, rot):
        g = (self.max_gain+25) * np.power(np.cos(theta + self.direction + rot),self.falloff)
        return (g-25) + self.power
    
    def polarPlot(self,ax,rot):
        theta = np.arange(-(np.pi/2),np.pi/2,0.01)[1:]
        ax.plot(theta,self.baseGain(theta, 0))
    
    def plot(self,ax,rot):
        theta = np.arange(-(np.pi/2),np.pi/2,0.01)[1:]

        x = self.position[0]
        y = self.position[1]

        g = self.baseGain(theta, 0)
        ax.plot(x+np.cos(theta)*g,y+np.sin(theta)*g,alpha=1,lw=3)
    

class BLEReciever:
    def __init__(self,startposition) -> None:
        self.position = startposition
        self.gain = -3 #from data sheet
        self.sensitivity = -94 #from ds
        self.recordedSignals = []
        self.recordedAngles = []

    def signalRecieved(self,theta,signal):
        self.recordedSignals.append(signal)
        self.recordedAngles.append(theta)

    def plotSignals(self,ax):
        ax.plot(self.recordedAngles,self.recordedSignals)


class NewTransmitter:
    def __init__(self,pos,vel) -> None:
        self.position = pos
        self.direction = 1.8
        self.transmitters = []
        self.angular_vel = np.pi/2

        for i in range(0,3):
            dir = (-np.pi/8)+(i*np.pi/8)
            self.transmitters.append(Antenna(5,dir,14,15,pos))

    def plot(self,ax):
        for ant in self.transmitters:
            ant.polarPlot(ax,0)

    def polarPlot(self,ax):
        for ant in self.transmitters:
            ant.polarPlot(ax,0)