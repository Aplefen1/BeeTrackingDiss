import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Gain import getgain

class Transmitter:
    def __init__(self,position,ant_type='omni',power=10, col='k'): #dBm
        self.position = np.array(position)
        self.ant_type = ant_type
        self.power = power
        self.initdirection = np.random.rand()*np.pi*2
        self.direction = self.initdirection
        self.fwhm = np.deg2rad(36)
        self.color = col

    def plot(self):
        plt.plot(self.position[0],self.position[1],'xb')
        arrowsize = 50
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
        plt.plot([x,x+np.cos(ang)*10000],[y,y+np.sin(ang)*10000],'-'+self.color,alpha=0.5,lw=1)

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

    def add_record(self,transmitter_id,signal,time,angle):
        if transmitter_id not in self.record: self.record[transmitter_id] = []
        self.record[transmitter_id].append([time,signal,angle])
        
    def plot(self):
        plt.plot(self.position[0],self.position[1],'og')

    def settime(self,t):
        #self.position = (self.startposition + t*self.velocity)
        a = -2*np.pi*t/400
        self.position = np.array([1500+1300*np.cos(a),1800+1300*np.sin(a)]) #(self.startposition + t*self.velocity)

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
        return transmitter.getgain(np.array([ang]))+transmitter.power+20*np.log10(lamb/(4*np.pi*dist))+self.gain