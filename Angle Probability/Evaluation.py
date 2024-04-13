from Model import Model
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, cuda

class Evaluation:
    def __init__(self, model, sep_low, sep_high, ant_steps, rec_low_ang, rec_high_ang, rec_steps, iterations, test_type) -> None:
        """
            Instantiator for quick evaluation class
            
            Parameters
            ----------
            model:
                model to perform evaluation on
                
            sep_low: 
                Minimum separation of antennas in radians, 0 radians is antenna on top of each other
            
            sep_high:
                Maximum separation of antennas in radians
            
            ant_steps:
                Number of angles to test the antenna separation at
            
            rec_low_angle:
                Start angle of reciever to test, in radians, must be lower than rec_high_angle
            
            rec_high_angle:
                End angle of reciever to test, in radians
            
            rec_steps: 
                Number of angles to test each antenna configuration, moving the reciever
            
            iterations: 
                Number of samples the model takes at each reciever angle at each antenna setup
            
            test_type:
                - "MAE" - Averaged Abs Error for all samples, 
                - "MODE" - Abs error for the mode in sample, 
                - "PROB" - Probability that angle is chosen correctly
        """
        self.rec_low_ang = rec_low_ang
        self.rec_high_ang = rec_high_ang
        self.steps = rec_steps
        self.ant_steps = ant_steps
        self.iterations = iterations
        self.sep_low = sep_low
        self.sep_high = sep_high
        self.test_type = test_type
        
        self.rec_test_angles = np.linspace(rec_low_ang,rec_high_ang,rec_steps)
        self.ant_sep_angles = np.linspace(sep_low,sep_high,ant_steps)
        self.rec_angle_error = np.zeros(self.iterations)
        
        self.model = model
        self.i = 0
    

    def eval_model(self, ax : plt.Axes):
        vector_ang = np.vectorize(self.model.angle_analysis)
        
        def single_angle(a):
            self.model.set_reciver_angle(a)
            res = vector_ang(self.rec_angle_error)
            return np.mean(res)
        
        vector_single = np.vectorize(single_angle)
        
        model_error = vector_single(self.rec_test_angles)
        
        #compute error for every reciever angle 
        
        ax.plot(self.rec_test_angles,model_error)
        ax.set_ylabel("MAE")
        ax.set_xlabel("Angle")
    
    def opt_eval_MAE(self, ax : plt.Axes):
        MAE = self.model.MAE_vectorised(self.rec_test_angles)
        '''np.zeros_like(self.rec_test_angles)
        for i in range(len(self.rec_test_angles)):
            MAE[i] = self.model.MAE(self.rec_test_angles[i],self.iterations)
        '''
        
        ax.plot(self.rec_test_angles,MAE)
        ax.set_ylabel("mode MAE")
        ax.set_xlabel("Angle")
        
    def eval_prob(self, ax : plt.Axes):
        stat_out = np.ndarray((len(self.rec_test_angles),5))
        
        for i in range(len(self.rec_test_angles)):
            stat_out[i] = self.model.angle_prob_iterations(self.rec_test_angles[i])
        
        ax.set_ylim(0,1)
        #mean
        ax.plot(self.rec_test_angles,stat_out[:,0],'b-',label="Mean")
        #stdev
        ax.plot(self.rec_test_angles,stat_out[:,1],'b--',label="st. dev",alpha=0.2)
        ax.plot(self.rec_test_angles,stat_out[:,2],'b--',alpha=0.2)
        #max
        ax.plot(self.rec_test_angles,stat_out[:,3],'g-',label="Max", alpha=0.2)
        #min
        ax.plot(self.rec_test_angles,stat_out[:,4],'r-',label="min", alpha=0.2)
        
        ax.set_ylabel("Probability of Angle")
        ax.set_xlabel("Angle (rad)")
        ax.legend()
        
    def eval_mode(self, ax : plt.axes):
        mode = self.model.mode_vectorised(self.rec_test_angles)
        abs_difference = np.abs(np.subtract(self.rec_test_angles,mode))
        
        ax.plot(self.rec_test_angles,abs_difference)
        ax.set_ylabel("abs (actual-mode)")
        ax.set_xlabel("Angle (rad)")


    def makemovie(self,filename=None):
        """
        Generates a video saved in 'filename'.
        """
        from moviepy.editor import VideoClip
        from moviepy.video.io.bindings import mplfig_to_npimage
        import matplotlib.pyplot as plt
        fig,ax1  = plt.subplots()
        l, b, h, w = .6, .75, .3, .3
        ax2 = fig.add_axes([l,b,h,w], projection="polar")
        
        fun = None
        if self.test_type == "MODE":
            fun = self.eval_mode
        elif self.test_type == "MAE":
            fun = self.opt_eval_MAE
        elif self.test_type == "PROB":
            fun = self.eval_prob
        else:
            print("Not accepted test type, reverting to angle probability")
            fun = self.eval_prob

        def make_frame(t):
            self.model.set_antenna_separation(self.ant_sep_angles[self.i])
            ax1.clear()
            ax2.clear()
            fun(ax1)
            self.model.polar_plot(ax2)
            ax1.set_title(self.ant_sep_angles[self.i])
            self.i += 1
            return mplfig_to_npimage(fig)

        Nframes = (np.shape(self.ant_sep_angles)[0]) / 4
        animation = VideoClip(make_frame, duration = Nframes)
        
        if filename is not None:
            animation.write_videofile(filename,fps=4,codec='mpeg4',bitrate='3000k')
        else:
            return animation.ipython_display(fps = 4, loop = False, autoplay = True)