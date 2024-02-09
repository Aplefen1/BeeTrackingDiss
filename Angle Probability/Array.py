from Antenna import Antenna
from MonoPattern import MonoPattern
import numpy as np
import matplotlib.pyplot as plt

class Array:
    def __init__(self, pos, rotation) -> None:
        self.position = np.array(pos)
        self.rotation = rotation

        ant_vector = np.array([0,0.1])
        ant_rot = np.pi/10
        self.ant_right = Antenna(self.position - ant_vector, self.rotation + (2*np.pi - ant_rot), 30, 'AR') # 20 cm to right and roated 15 degrees clockwise
        self.ant_middle = Antenna(self.position, self.rotation, 30, 'AM') # Position and rotation of Array
        self.ant_left = Antenna(self.position + ant_vector, self.rotation + ant_rot, 30, 'AL') # 20 cm to left and roated 15 degrees counter-clockwise

        self.ant_lookup = {'AL' : self.ant_left, 'AM' : self.ant_middle, 'AR' : self.ant_right}
        
        self.AL_AM_model = self.create_mono_model("AL","AM")
        self.AL_AR_model = self.create_mono_model("AL","AR")
        self.AM_AR_model = self.create_mono_model("AM","AR")
        
    ######### Ideal Mono Pulse Function ####
        
    def ideal_from_ID(self, antA_ID, antB_ID, theta):
        antA = self.ant_lookup[antA_ID]
        antB = self.ant_lookup[antB_ID]
        return self.ideal_mono(antA,antB,theta)

    def ideal_mono(self, antA : Antenna, antB : Antenna, theta):
        antA_gain = antA.get_gain(theta)
        antB_gain = antB.get_gain(theta)
        
        sum = antA_gain + antB_gain
        diff = antA_gain - antB_gain
        mono = diff / sum
        
        return diff
    
    def create_mono_model(self, antA_ID, antB_ID):
        antA = self.ant_lookup[antA_ID]
        antB = self.ant_lookup[antB_ID]
        theta = np.linspace(-np.pi/10,np.pi/10,5000)
        mono = self.ideal_mono(antA,antB,theta)
        
        monoModel = MonoPattern(theta,mono)
        return monoModel

    ########## Visualisations ##############

    def polar_plot(self):
        ax = plt.subplot(projection="polar")
        
        self.ant_left.polar_plot(ax, 'b')
        self.ant_middle.polar_plot(ax, 'g')
        self.ant_right.polar_plot(ax, 'r')
        
        ax.legend()
        
    def spatial_plot(self, ax):
        
        self.ant_left.spatial_plot(ax,'b')
        self.ant_middle.spatial_plot(ax,'g')
        self.ant_right.spatial_plot(ax,'r')
        
    def plot_ideal_mono_pair(self, low, high, ant1_id, ant2_id):
        theta = np.linspace(low,high,1000)
        mono_func = self.ideal_from_ID(ant1_id,ant2_id,theta)
        
        fig = plt.figure()
        ax = plt.subplot()
        ax.plot(theta,mono_func)
        
    def plot_all_mono_pairs(self, low, high):
        theta = np.linspace(low,high,1000)
        AL_AM_mono = self.ideal_from_ID("AL","AM",theta)
        AL_AR_mono = self.ideal_from_ID("AL", "AR", theta)
        AM_AR_mono = self.ideal_from_ID("AM", "AR", theta)
        
        ax = plt.subplot()
        ax.plot(theta,AL_AM_mono, label="AL - AM")
        ax.plot(theta,AL_AR_mono, label="AL - AR")
        ax.plot(theta,AM_AR_mono, label="AM - AR")
        ax.legend()
        