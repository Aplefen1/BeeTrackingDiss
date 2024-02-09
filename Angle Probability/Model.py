import numpy as np 
from Array import Array
from Antenna import Antenna
from Receiver import Receiver

import matplotlib.pyplot as plt

class Model:
    def __init__(self, reciever_angle, reciever_distance) -> None:
        self.array = Array([0,0],0)
        self.receiver_angle = reciever_angle
        self.reciever_distance = reciever_distance
        x = reciever_distance * np.cos(reciever_angle)
        y = reciever_distance * np.sin(reciever_angle)
        self.receiver = Receiver([x,y],-3)
 
    ####### Mono Pulse Estimation #####
        
    def test_estimation(self):
        L, M, R = self.recieved_signals()
        L_M = L - M
        M_R = M - R
        L_R = L - R
        
        angles1 = self.array.AL_AM_model.angles_from_mono_multi(L_M)
        angles2 = self.array.AM_AR_model.angles_from_mono_multi(M_R)
        angles3 = self.array.AL_AR_model.angles_from_mono_multi(L_R)
        
        fig = plt.figure()
        ax = plt.subplot()
        ax.set_title(str(self.receiver_angle)+"rad at "+str(self.reciever_distance) + "m")
        ax.hist(angles1, bins=100, alpha=0.5, label='AL-AM')
        ax.hist(angles2, bins=100, alpha=0.5, label='AM-AR')
        ax.hist(angles3, bins=100, alpha=0.3, label="AL-AR")
        ax.legend(loc='upper right')

    ######## Signal modelling #########
        
    def recieved_signals(self):
        deltaL = self.receiver.position - self.array.ant_left.position
        thetaL = np.arctan2(deltaL[1],deltaL[0])
        thetaAL = np.empty(1000)
        thetaAL.fill(thetaL)
        distL = np.linalg.norm(deltaL)
        
        deltaM = self.receiver.position - self.array.ant_middle.position
        thetaM = np.arctan2(deltaM[1],deltaM[0])
        thetaAM = np.empty(1000)
        thetaAM.fill(thetaM)
        distM = np.linalg.norm(deltaM)
        
        deltaR = self.receiver.position - self.array.ant_right.position
        thetaR = np.arctan2(deltaR[1],deltaR[0])
        thetaAR = np.empty(1000)
        thetaAR.fill(thetaR)
        distR = np.linalg.norm(deltaR)
        
        signalL = self.array.ant_left.get_gain(thetaAL)
        signalM = self.array.ant_middle.get_gain(thetaAM)
        signalR = self.array.ant_right.get_gain(thetaAR)
        
        signalL = self.signal_model(signalL, distL, -3)
        signalM = self.signal_model(signalM, distM, -3)
        signalR = self.signal_model(signalR, distR, -3)
        
        return signalL, signalM, signalR
        
    def signal_model(self, base_sig, dist, rec_gain):
        loss_sig = self.free_space_loss(base_sig, dist, rec_gain)
        gauss_sig = self.gauss_noise(loss_sig)
        final_sig = self.constant_noise(gauss_sig)

        return final_sig

    def free_space_loss(self, signal, dist, rec_gain):
        return signal - (20*np.log10(dist)+40.05-rec_gain)

    def gauss_noise(self,signal):
        #95% between -3 and 3 dB
        noiser = lambda x: np.random.normal(x,1.5)
        noisefunc = np.vectorize(noiser)
        return noisefunc(signal)
    
    def constant_noise(self, signal):
        return signal + 0
    ##########################
    
    ####### Visualisations #####
    
    def spatial_plot(self):
        fig = plt.figure()
        ax = plt.subplot()
        self.array.spatial_plot(ax)
        self.receiver.plot_spatial(ax)
        
        ax.legend()
        
    def polar_plot(self):
        fig = plt.figure()
        self.array.polar_plot()
        
    def plot_ideal_mono_pair(self, low, high, ant1_id, ant2_id):
        fig = plt.figure()
        self.array.plot_ideal_mono_pair(low, high, ant1_id, ant2_id)
        
    def plot_recieved_signals(self):
        fig = plt.figure()
        L, M, R = self.recieved_signals()
        plt.hist(L, bins = 200, color='b')
        plt.hist(M, bins = 200, color='g') 
        plt.hist(R, bins = 200, color='r')
        plt.show()
        
    def plot_all_mono_pairs(self,low,high):
        self.array.plot_all_mono_pairs(low,high)