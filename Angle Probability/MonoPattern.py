import numpy as np
import matplotlib.pyplot as plt
class MonoPattern():
    def __init__(self, theta, mono) -> None:
        self.theta = theta
        self.mono = mono
        self.low = np.min(mono)
        self.high = np.max(mono)
        self.step_size = (self.high-self.low)/len(mono)
        
    def angles_from_mono_single(self, mono_val):
        angles_above = self.mono > (mono_val - self.step_size)
        angles_below = self.mono < (mono_val + self.step_size)
        angles_in_range = np.logical_and(angles_above,angles_below)
        possible_angles = self.theta[angles_in_range]
        return np.array(possible_angles)
    
    def angles_from_mono_multi(self, mono_vals):
        angles = np.array([])
        for val in mono_vals:
            angles = np.concatenate((angles,self.angles_from_mono_single(val)))
        return angles
    
    def plot(self):
        ax = plt.subplot()
        ax.plot(self.theta,self.mono)
    