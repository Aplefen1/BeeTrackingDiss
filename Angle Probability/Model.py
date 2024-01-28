import numpy as np 
from Array import Array
from Antenna import Antenna
from Receiver import Receiver

import matplotlib.pyplot as plt

class Model:
    def __init__(self) -> None:
        self.array = Array([0,0],0)
        self.receiver = Receiver([1,1],-3)

    ####### Mono Pulse Estimation #####
        

    ######## Signal modelling #########
    def signal_model(self, base_sig, dist, rec_gain):
        loss_sig = self.free_space_loss(base_sig, dist, rec_gain)
        gauss_sig = self.gauss_noise(loss_sig)
        final_sig = self.constant_noise(gauss_sig)

        return final_sig

    def free_space_loss(signal, dist, rec_gain):
        return signal - (20*np.log10(dist)+40.05-rec_gain)

    def gauss_noise(self,signal):
        #95% between -3 and 3 dB
        mean = 0
        std_dev = 1.5
        noise = np.random.normal(mean,std_dev)
        return signal + noise
    
    def constant_noise(self, signal):
        return signal + 0
    
    ##########################
    
    ####### Visualisations #####
    
    def spatial_plot(self):
        ax = plt.subplot()
        self.array.spatial_plot(ax)
        self.receiver.plot_spatial(ax)
        
        ax.legend()
    