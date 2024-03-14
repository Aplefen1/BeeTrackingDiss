import numpy as np 
from Array import Array
from Antenna import Antenna
from Receiver import Receiver

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm as norm

class Model:
    def __init__(self, reciever_angle, reciever_distance, array_separation) -> None:
        self.array = Array([0,0],0, array_separation)
        self.receiver_angle = reciever_angle
        self.reciever_distance = reciever_distance
        x = reciever_distance * np.cos(reciever_angle)
        y = reciever_distance * np.sin(reciever_angle)
        self.receiver = Receiver([x,y],-3)
        self.L, self.M, self.R = self.recieved_signals()
        self.L_M = self.L - self.M
        self.M_R = self.M - self.R
        self.L_R = self.L - self.R
        self.noise_std = 1.5
        
    def set_reciver_angle(self, new_angle):
        self.receiver_angle = new_angle
        x = self.reciever_distance * np.cos(new_angle)
        y = self.reciever_distance * np.sin(new_angle)
        self.receiver.set_pos([x,y])
        
    def set_antenna_separation(self, separation):
        self.array.set_separation(separation)
        
    ####### Vector Estimation #####
        
    def estimate_angle(self):
        
        ps, *a = ps.calculations(-np.pi,np.pi)
        
    def ps_calculation(self, high, low):
        recieved_strengths = self.signal_pulse().T
        v = np.ones(3)/np.sqrt(3)
        
        search_ang = np.linspace(low,high,2000)
        search_gain = self.array.get_gain(search_ang)
        
        #compute the distance from the [1,1,1] vector passing though each of the signal_mean_for_angle vectors
        #from the measured signal vector of 3 measurements
        dist = np.linalg.norm((v*((recieved_strengths-search_gain)@v)[:,None]+search_gain)-recieved_strengths,axis=1)
        
        #compute the prob. density at this distance (for a normal distribution with standard deviation equal to
        #the noise added to generate the signals
        ps = norm(0,4).pdf(dist)
        #normalise
        ps/=np.sum(ps)*(search_ang[1]-search_ang[0])
        
        return (ps, search_ang, search_gain, recieved_strengths)
        
    def plot_estimation_vectors(self, high, low):
        #store a unit vector facing along [1,1,1]
        
        ps, search_ang, search_gain, recieved_strengths = self.ps_calculation(high, low)
        plt.figure(figsize=[8,10])
        plt.subplot(2,1,1)
        plt.plot(search_ang,search_gain[:,0],'g-')
        plt.plot(search_ang,search_gain[:,1],'k-')
        plt.plot(search_ang,search_gain[:,2],'b-')

        plt.plot(self.receiver_angle,recieved_strengths[0,0],'xg')
        plt.plot(self.receiver_angle,recieved_strengths[0,1],'xk')
        plt.plot(self.receiver_angle,recieved_strengths[0,2],'xb')
        
        plt.xlabel('Angle / radians')
        plt.ylabel('Signal Strength / dBm or dB')
        plt.xlim([low,high])
        plt.subplot(2,1,2)
        plt.plot(search_ang,ps)
        plt.vlines(self.receiver_angle,0,10,'r')
        plt.xlim([low,high])
        plt.ylabel('Probability Density')
        plt.xlabel('Angle / radians')
        
 
    ####### Mono Pulse Estimation #####
        
    def difference_distributions(self):
        
        anglesL_M = self.array.AL_AM_model.angles_from_mono_multi(self.L_M)
        anglesM_R = self.array.AM_AR_model.angles_from_mono_multi(self.M_R)
        anglesL_R = self.array.AL_AR_model.angles_from_mono_multi(self.L_R)
        
        fig = plt.figure()
        ax = plt.subplot()
        ax.set_title(str(self.receiver_angle)+"rad at "+str(self.reciever_distance) + "m")
        b = 50
        bins = np.histogram(np.hstack((anglesL_M,anglesM_R,anglesL_R)), bins=b)[1]
        ax.hist(anglesL_M, bins=bins, alpha=0.5, label='AL-AM')
        ax.hist(anglesM_R, bins=bins, alpha=0.5, label='AM-AR')
        ax.hist(anglesL_R, bins=bins, alpha=0.3, label="AL-AR")
        
        ax.legend(loc='upper right')
        
    def all_probability_distributions(self):
        L_M_angle_dist = self.array.AL_AM_model.angles_from_mono_multi(self.L_M)
        M_R_angle_dist = self.array.AM_AR_model.angles_from_mono_multi(self.M_R)
        L_R_angle_dist = self.array.AL_AR_model.angles_from_mono_multi(self.L_R)
        
        L_M_prob_dist = gaussian_kde(L_M_angle_dist,bw_method=0.1)
        M_R_prob_dist = gaussian_kde(M_R_angle_dist,bw_method=0.1)
        L_R_prob_dist = gaussian_kde(L_R_angle_dist,bw_method=0.1)
        
        theta = np.linspace(-np.pi/4,np.pi/4,5000)
        fig = plt.figure()
        ax = plt.subplot()
        ax.plot(theta,L_M_prob_dist.pdf(theta),label="P(a|L-M)")
        ax.plot(theta,M_R_prob_dist.pdf(theta),label="P(a|M-R)")
        ax.plot(theta,L_R_prob_dist.pdf(theta),label="P(a|L-R)")  
        ax.legend()
        

    ######## Signal modelling #########

    def signal_pulse(self):
        deltaL = self.receiver.position - self.array.ant_left.position
        thetaL = np.arctan2(deltaL[1],deltaL[0])
        distL = np.linalg.norm(deltaL)
        
        deltaM = self.receiver.position - self.array.ant_middle.position
        thetaM = np.arctan2(deltaM[1],deltaM[0])
        distM = np.linalg.norm(deltaM)
        
        deltaR = self.receiver.position - self.array.ant_right.position
        thetaR = np.arctan2(deltaR[1],deltaR[0])
        distR = np.linalg.norm(deltaR)
        
        signalL = self.array.ant_left.get_gain(np.array([thetaL]))
        signalM = self.array.ant_middle.get_gain(np.array([thetaM]))
        signalR = self.array.ant_right.get_gain(np.array([thetaR]))
        
        signalL = self.signal_model(signalL, distL, -3)
        signalM = self.signal_model(signalM, distM, -3)
        signalR = self.signal_model(signalR, distR, -3)
        
        return np.array([signalL, signalM, signalR])
    

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
        noiser = lambda x: np.random.normal(x,4)
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
        
    def plot_against(self):
        self.array.plot_against()
        
    def plot_recieved_signals(self):
        fig = plt.figure()
        L, M, R = self.recieved_signals()
        plt.hist(L, bins = 200, color='b')
        plt.hist(M, bins = 200, color='g') 
        plt.hist(R, bins = 200, color='r')
        plt.show()
        
    def plot_all_mono_pairs(self,low,high):
        self.array.plot_all_mono_pairs(low,high)