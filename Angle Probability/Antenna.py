import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt

class Antenna:

    def __init__(self,type, pos, rotation, gain, id) -> None:
        self.position = pos
        self.rotation = rotation
        self.gain = gain
        self.lobe_function = None
        self.id = id
        self.type = type
        self.df = self.data_frame_initialisation()
        self.side_lobe_function = self.side_lobe_function_init()
    

    def base_gain(self,theta):
        theta = np.mod(theta, np.pi*2)
        if self.type == "narrow":
            theta_sides = np.logical_and(theta > np.pi/16, theta < 31*np.pi/16)
            theta_sides_l = np.logical_and(theta > np.pi/16, theta <= np.pi)
            theta_sides_r = np.logical_and(theta >= np.pi, theta < 31*np.pi/16)
            
            main = np.logical_not(theta_sides)
            gain = np.zeros(theta.shape)
            
            gain[theta_sides_l] = self.side_lobe_function(theta[theta_sides_l])
            
            mirror_vals = theta[theta_sides_r] - np.pi
            
            gain[theta_sides_r] = self.side_lobe_function(np.pi - mirror_vals)
            gain[main] = self.main_lobe_approx_narrow(theta[main])
            
            return gain + self.gain
        elif self.type == "wide":
            theta_sides = np.logical_and(theta > np.pi/8, theta < (2 * np.pi - np.pi/8))
            main = np.logical_not(theta_sides)
            gain = np.zeros(theta.shape)
            
            gain[theta_sides] = self.side_lobe_function(theta[theta_sides])
            gain[main] = self.main_lobe_approx_wide(theta[main])
            
            return gain + self.gain

        elif self.type == "omni":
            gain = np.zeros(theta.shape)
            return gain + self.gain
    

    def get_gain(self, theta):
        return self.base_gain(self.relative_theta(theta))
    
    def set_separation(self, separation):
        self.rotation = separation
        
    def rotate_by(self,angle):
        self.rotation += angle
    
    #change to positive angle from if where the antenna is pointing is the origin

    def relative_theta(self, theta):
        relative = theta - self.rotation
        relative[relative < 0] += 2*np.pi
        return relative

    ######## Pattern Approximation ###########
     
    def main_lobe_approx_narrow(self, theta):
        g = 50 * np.power(np.cos(theta),57)
        return g-25
    
    def main_lobe_approx_wide(self,theta):
        g = 38 * np.power(np.cos(theta),19)
        return g-25
   
    def data_frame_initialisation(self):
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
        df = pd.DataFrame()

        df["deg"] = deg
        df["theta"] = theta
        df["magdB"] = magdB

        df = df.sort_values("deg")

        return df

    def side_lobe_function_init(self):
        left_lobe_thetas = np.array(self.df['theta']) > np.pi/16-0.01 # and np.array(df['theta']) < (np.pi*2)-np.pi/16
        right_lobe_thetas = np.array(self.df['theta']) < (np.pi*2)-np.pi/16
        side_lobe_thetas = np.logical_and(left_lobe_thetas,right_lobe_thetas)

        side_lobes = self.df['theta'][side_lobe_thetas]
        lobes_dB = self.df['magdB'][side_lobe_thetas]

        side_lobes = np.append(side_lobes,((np.pi*2)-np.pi/16))
        side_lobes = np.insert(side_lobes,0,(np.pi/16))

        lobes_dB = np.append(lobes_dB,-9)
        lobes_dB = np.insert(lobes_dB,0,-15)
        lobes_dB[1:2] = 0

        coeffs = np.polyfit(side_lobes,lobes_dB,190,rcond=0)
        p = np.poly1d(coeffs)
        
        return p
    
    ######## VISUALISATIONS ##################

    def polar_plot(self, ax, col):
        theta = np.linspace(-np.pi,np.pi,5000)
        #theta2 = np.mod(theta-self.rotation, (2*np.pi))
        gain = self.get_gain(theta)
        
        ax.plot(theta,gain, col, label="Approximation")

    def spatial_plot(self, ax, col,l):
        ax.plot(self.position[0],self.position[1],'x'+col)
        