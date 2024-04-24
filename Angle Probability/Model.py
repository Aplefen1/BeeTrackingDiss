import numpy as np 
from Array import Array
from Antenna import Antenna
from Receiver import Receiver

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm as norm
from scipy.stats import mode
from scipy.signal import find_peaks
import math


class Model:
    def __init__(self, reciever_angle, reciever_distance, array_separation, 
                  ant_types=("narrow","narrow","narrow"), eval_iterations=0) -> None:
        self.gauss_noise_add = 4
        self.array = Array(ant_types,[0,0],0, array_separation)
        self.receiver_angle = reciever_angle
        self.reciever_distance = reciever_distance
        x = reciever_distance * np.cos(reciever_angle)
        y = reciever_distance * np.sin(reciever_angle)
        self.receiver = Receiver([x,y],-3,-94)
 
        A = self.recieved_signals(1000)
        self.L = A[0]
        self.M = A[1]
        self.R = A[2]
        self.L_M = self.L - self.M
        self.M_R = self.M - self.R
        self.L_R = self.L - self.R
        self.noise_std = 1.5

        self.search_ang = np.linspace(-np.pi/4,np.pi/4,100)
        self.search_gain = self.array.get_gain(self.search_ang) # TODO simulate at 100m, no noise and remove values < 94
        
        self.MAE_vectorised = np.vectorize(self.MAE)
        self.mode_vectorised = np.vectorize(self.mode_angle)
        self.eval_iterations = eval_iterations
        
    def set_rec_dist(self, distance):
        self.reciever_distance = distance
  
    def set_reciver_angle(self, new_angle):
        '''Sets the angle of the reciever to the array at the model's set distance'''
        self.receiver_angle = new_angle
        x = self.reciever_distance * np.cos(new_angle)
        y = self.reciever_distance * np.sin(new_angle)
        self.receiver.set_pos([x,y])

    def set_antenna_separation(self, separation):
        '''Changes how far apart the side antennas are from the central antenna (radians)'''
        self.array.set_separation(separation)
        self.search_gain = self.array.get_gain(self.search_ang)
        
    def set_random_reciever(self, alow=-np.pi/4, ahigh=np.pi/4, dlow=30, dhigh=300):
        angle = np.random.uniform(low=alow,high=ahigh)
        distance = np.random.uniform(low=dlow,high=dhigh)
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        self.receiver_angle = angle
        
        self.receiver.set_pos([x,y])
        
    def rotate_array_by(self,angle):
        self.array.rotate_by(angle)
        
    ####### Probable Vector Calculation ######################
    
    def estimate_vectors(self):
        '''
            Models the system then returns the angles that corresspond to the peaks of the pdf 
            orderd from most probable to least probable.
        '''
        signal_pulse = self.signal_pulse()
        probs = self.norm_dist(signal_pulse)
        
        maxima, _ = find_peaks(probs)
        
        maxima = maxima[np.argsort(probs[maxima])[::-1]]
        
        '''
        if len(maxima) > 3:
            return self.search_ang[maxima[0:3]], probs[maxima]
        '''
        
        return self.search_ang[maxima], probs[maxima]
        
    ####### Angles using Closeness and Analysis #######
        
    def norm_dist(self, rec_strengths):
        '''Creates a distribution that relates each angle in self.search_ang to a probability
        Used in most analysis funvtions
        
        Parameters
        ----------
        rec_strengths
             noisy angles from three transmitters
        
        Returns
        -------
        Probability Distribution
            same size as self.search_ang
        '''
        
        v = np.ones(3)/np.sqrt(3)
        d = np.linalg.norm((v*((rec_strengths-self.search_gain)@v)[:,None]+self.search_gain)-rec_strengths,axis=1)
        ps = norm(0,self.gauss_noise_add).pdf(d)
        #normalise
        ps /= np.sum(ps)*(self.search_ang[1]-self.search_ang[0])
        #create discrete PDF, integral adds up to 1
        return np.multiply(ps, self.search_ang[1]-self.search_ang[0])

    def angle_analysis(self, p):
        '''
        Returns the Absolute Error for the current setup with a single pulse
        Noise is added to the signal, angle choice is simple and random based off
        distribution given by signal pulse
        '''
        ps, *a = self.ps_calculation_single()
        probabilities = np.multiply(ps, self.search_ang[1]-self.search_ang[0])
        
        est_ang = np.random.choice(self.search_ang, p=probabilities)
        
        return np.abs(np.subtract(self.receiver_angle,est_ang)) #error
    
    def find_nearest(self,angle):
        '''
        Finds "nearest" angle index in the model's searchspace, not all angles are accounted for, so this method
        effectively "quantises" the passed in angle
        '''
        array = self.search_ang
        idx = np.searchsorted(array, angle, side="left")
        if idx > 0 and (idx == len(self.search_ang) or math.fabs(angle - array[idx-1]) < math.fabs(angle - array[idx])):
            return idx-1
        else:
            return idx
        
    def abs_error(self, recieved_strengths):
        '''Very similar to Model.angle_analysis but used in optimised analysis'''
        probs = self.norm_dist(recieved_strengths)
        ang = np.random.choice(self.search_ang, p=probs)
        return np.abs(np.subtract(self.receiver_angle,ang))
    
    #Return probability of reciever angle being estimated from dristibution generated by noisy signals
    def angle_probability(self, recieved_strengths, idx):
        '''
        Returns the probability that the angle of the reciever is chosen 
        given the vector of recieved strengths 
        '''
        #PDF of angles given recieved strengths vector
        probs = self.norm_dist(recieved_strengths)
        
        #includes the probability of the angles either side to show what is "acceptable"
        #Equivalent of the area under pdf between idx-1 and idx+1
        return probs[idx] + probs[idx-1] + probs[idx+1]
   
    def ps_calculation_single(self):
        '''
        Similar to  self.norm_dist, but samples it's own noisy angle#
        
        Returns: (ps, recieved_strengths)
        -------
        ps -- PDF of possible angles given recieved_strengths
        '''
        
        recieved_strengths = self.signal_pulse()

        v = np.ones(3)/np.sqrt(3)
        #compute the distance from the [1,1,1] vector passing though each of the signal_mean_for_angle vectors
        #from the measured signal vector of 3 measurements
        dist = np.linalg.norm((v*((recieved_strengths-self.search_gain)@v)[:,None]+self.search_gain)-recieved_strengths,axis=1)
        #compute the prob. density at this distance (for a normal distribution with standard deviation equal to
        #the noise added to generate the signals
        ps = norm(0,self.gauss_noise_add).pdf(dist)
        #normalise
        ps /= np.sum(ps)*(self.search_ang[1]-self.search_ang[0])
        return (ps, recieved_strengths)
    
    #Returns the MAE for a given angle using a number of samples (iterations)
    def MAE(self, angle):
        '''Performs an analysis of the current transmitter/reciever setup, performing self.eval_iterations
        number of samples to work out the MAE of the current setup
        
        Parameters:
        angle -- angle to set the reciever to in Radians
        
        Returns:
        MAE of the system at angle'''
        self.set_reciver_angle(angle)
        recieved_signals = self.recieved_signals(self.eval_iterations)
        MAE = 0
        for sig in recieved_signals:
            MAE += self.abs_error(sig)

        return (MAE/len(recieved_signals))
    
    #Returns probaility that angle is estimated over a number of samples
    def angle_prob_iterations(self, angle):
        '''
        Performs analysis of the current transmitter/reciever setup, performing self.eval_iterations
        number of samples to work out the mean probability that the current reciever angle will be estimated by
        the model
        
        Parameters:
        -----------
        angle -- angle to set the reciever to in Radians
        
        Returns:
        --------
        [mean,s.d,max,min]
        '''
        
        self.set_reciver_angle(angle)
        #index of the nearest-matching angle
        idx = self.find_nearest(self.receiver_angle)
        
        recieved_signals = self.recieved_signals(self.eval_iterations)
        prob = np.empty_like(recieved_signals)
        
        for i in range(len(recieved_signals)):
            prob[i] = self.angle_probability(recieved_signals[i], idx)
        
        stdev = np.std(prob)
        mean = np.mean(prob)
        sdu = stdev + mean
        sdd = mean - stdev
        max = np.max(prob)
        min = np.min(prob)
            
        return np.array([mean,sdu,sdd,max,min])
    
    def mode_angle(self, angle):
        '''Performs analysis of the current transmitter/reciever setup, performing self.eval_iterations
        number of samples to work out mode angle returned by the system
        
        Parameters:
        angle -- angle to set the reciever to in Radians
        
        Returns:
        Mode angle estimated'''
        #Mode is taken to be the angle with the highest probability (for simplicity)
        self.set_reciver_angle(angle)
        recieved_signals = self.recieved_signals(self.eval_iterations)
        mde = []
        for i in range(len(recieved_signals)):
            sig = recieved_signals[i]
            probs = self.norm_dist(sig)
            max = np.argmax(probs)
            mde.append(self.search_ang[max])
        
        return mode(mde, keepdims=True)[0][0]
        

    ######## Signal modelling #########
    
    '''Group of methods and utilities that e, noise and loosing signals'''

    def signal_pulse(self, noise=True):
        '''Models a single pulse from the antenna array to the receiver
        return a vector of the Left antena, middle, right'''
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
        
        signalL = self.signal_model(signalL, distL, -3, noise)
        signalM = self.signal_model(signalM, distM, -3, noise)
        signalR = self.signal_model(signalR, distR, -3, noise)
        
        #TODO Make sure that signals are lost if they are below the reciever gain (-94), represented as -95
        
        return np.array([signalL, signalM, signalR]).T
    

    def recieved_signals(self, length) -> np.ndarray:
        '''Optimises the signal_pulse to very quickly fill out an matrix of length length
        filled with individually modelled signal pulses. I.e. each x in the return matrix
        is a unique vector of the recieved signals from the array, each modelled with noise
        and FSPL
        Used to help analyse current setups quickly
        '''
        deltaL = self.receiver.position - self.array.ant_left.position
        thetaL = np.arctan2(deltaL[1],deltaL[0])
        thetaAL = np.empty(length)
        thetaAL.fill(thetaL)
        distL = np.linalg.norm(deltaL)
        
        deltaM = self.receiver.position - self.array.ant_middle.position
        thetaM = np.arctan2(deltaM[1],deltaM[0])
        thetaAM = np.empty(length)
        thetaAM.fill(thetaM)
        distM = np.linalg.norm(deltaM)
        
        deltaR = self.receiver.position - self.array.ant_right.position
        thetaR = np.arctan2(deltaR[1],deltaR[0])
        thetaAR = np.empty(length)
        thetaAR.fill(thetaR)
        distR = np.linalg.norm(deltaR)
        
        signalL = self.array.ant_left.get_gain(thetaAL)
        signalM = self.array.ant_middle.get_gain(thetaAM)
        signalR = self.array.ant_right.get_gain(thetaAR)
        
        signalL = self.signal_model(signalL, distL, -3)
        signalM = self.signal_model(signalM, distM, -3)
        signalR = self.signal_model(signalR, distR, -3)
        
        return np.array([signalL, signalM, signalR]).T
 
    def signal_model(self, base_sig, dist, rec_gain, noise=True):
        '''Takes a base gain value and models the signal over a distance
        
        Parameters:
        base_sig -- Unmodelled gain from a transmitter
        distance -- Distance from antenna to reciever
        rec_gain -- Gain of the reciever
        
        Returns:
        Modelled signal (float)'''
        
        loss_sig = self.free_space_loss(base_sig, dist, rec_gain)
        if noise == True:
            gauss_sig = self.gauss_noise(loss_sig)
            final_sig = self.constant_noise(gauss_sig)
        else:
            final_sig = loss_sig

        return self.isSignalRecieved(final_sig)

    def free_space_loss(self, signal, dist, rec_gain):
        '''Applies the FSPL function to the signal'''
        return signal - (20*np.log10(dist)+40.05-rec_gain)

    def gauss_noise(self,signal):
        '''Adds noise gaussianly to the signal'''
        #95% between -3 and 3 dB
        noiser = lambda x: np.random.normal(x,self.gauss_noise_add)
        noisefunc = np.vectorize(noiser)
        return noisefunc(signal)

    def constant_noise(self, signal):
        '''Adds a constant amount of noise to the signal
        Set to 0 currently because no constant noise was identified'''
        return signal + 0
    
    def isSignalRecieved(self, signal):
        sens = self.receiver.sensitivity + 10
        if type(signal) == np.ndarray:
            underSens = signal <= sens
            signal[underSens] = sens-1
            return signal
        return signal if signal >= sens else sens-1
    
    ##########################
    
    def test_at_100(self):
        search_ang = np.linspace(-np.pi/4,np.pi/4,100)
        pulses = np.ndarray((len(search_ang),3))
        figa = plt.figure()
        a = plt.subplot()
        a.set_ylim(-150,150)
        a.set_xlim(-150,150)
        
        for i in range(len(search_ang)):
            self.set_reciver_angle(search_ang[i])
            self.receiver.plot_spatial(a)
            pulse = self.signal_pulse(False)
            pulses[i] = pulse
            
        figb = plt.figure()
        b = plt.subplot()
        
        b.plot(np.rad2deg(search_ang), pulses)
        
    ####### Visualisations #####
    
    def spatial_plot(self, ax):
        
        self.array.spatial_plot(ax)
        self.receiver.plot_spatial(ax)
        
        angle_vectors, probs = self.estimate_vectors()
        array_x = self.array.position[0]
        array_y = self.array.position[1]
        
        for angle, prob in zip(angle_vectors, probs):
            if prob > 0.004:
                end_x = array_x + 300 * np.cos(angle)
                end_y = array_y + 300 * np.sin(angle)
                ax.plot([array_x, end_x], [array_y, end_y], label=np.round(prob,5))
        
        ax.legend()
        
    def polar_plot(self,ax=None,legend=False):
        #fig = plt.figure()
        if ax == None:
            fig = plt.figure()
            ax = plt.subplot(projection="polar")
        self.array.polar_plot(ax)
        if legend==True : ax.legend()
        
    def plot_ideal_mono_pair(self, low, high, ant1_id, ant2_id):
        fig = plt.figure()
        self.array.plot_ideal_mono_pair(low, high, ant1_id, ant2_id)
        
    def plot_against(self):
        self.array.plot_against()
        
    def plot_recieved_signals(self):
        fig = plt.figure()
        A = self.recieved_signals(1000)
        plt.hist(A[0], bins = 200, color='b')
        plt.hist(A[1], bins = 200, color='g') 
        plt.hist(A[2], bins = 200, color='r')
        plt.show()
        
    def plot_all_mono_pairs(self,low,high):
        self.array.plot_all_mono_pairs(low,high) 
        
    def plot_estimation_vectors(self, high, low):
        '''Plots a graph from low, high of the signal patterns, the recieved noisly signals,
        the generated PDF generated by the model and a red line showing the true angle of the
        reciever'''
        
        ps, recieved_strengths = self.ps_calculation_single()
        plt.figure(figsize=[8,10])
        plt.subplot(2,1,1)
        plt.plot(self.search_ang,self.search_gain[:,0],'g-')
        plt.plot(self.search_ang,self.search_gain[:,1],'k-')
        plt.plot(self.search_ang,self.search_gain[:,2],'b-')

        plt.plot(self.receiver_angle,recieved_strengths[0,0],'xg')
        plt.plot(self.receiver_angle,recieved_strengths[0,1],'xk')
        plt.plot(self.receiver_angle,recieved_strengths[0,2],'xb')
        
        plt.xlabel('Angle / radians')
        plt.ylabel('Signal Strength / dBm or dB')
        plt.xlim([low,high])
        plt.subplot(2,1,2)
        plt.plot(self.search_ang,ps)
        plt.vlines(self.receiver_angle,0,10,'r')
        plt.xlim([low,high])
        plt.ylabel('Probability Density')
        plt.xlabel('Angle / radians')
        
    def plot_3d(self):
        ax = plt.figure().add_subplot(projection='3d')
        self.array.plot_3d(-np.pi,np.pi,ax)


    ####### Mono Pulse Estimation #########
    '''
    Now Defunct methods to use mono-pulse analysis to estimate the angle
    Not used because requires very accurate angles, slower and assumes all antenna
    pairs are independant, which cannot be proved.
    Works intuitively but doesn't work well in practice
    '''
        
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