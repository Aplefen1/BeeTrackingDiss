from Model import Model
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, cuda

class Evaluation:
    def __init__(self, sep_low, sep_high, rec_low_ang, rec_high_ang, steps, iterations) -> None:
        self.rec_low_ang = rec_low_ang
        self.rec_high_ang = rec_high_ang
        self.steps = steps
        self.iterations = iterations
        self.sep_low = sep_low
        self.sep_high = sep_high
        
        self.rec_test_angles = np.linspace(rec_low_ang,rec_high_ang,steps)
        self.ant_sep_angles = np.linspace(sep_low,sep_high,100)
        
        self.model = Model(0,100,0)
        self.i = 0
        
    def eval_model(self, ax : plt.Axes):
        vector_ang = np.vectorize(self.model.angle_analysis)
        
        def single_angle(a):
            self.model.set_reciver_angle(a)
            rec_angle_error = np.zeros(self.iterations)
            res = vector_ang(rec_angle_error)
            return np.mean(res)
        
        vector_single = np.vectorize(single_angle)
        
        model_error = vector_single(self.rec_test_angles)
        
        #compute error for every reciever angle 
        '''
        for rec_ang in self.rec_test_angles:
            rec_angle_error = []
            self.model.set_reciver_angle(rec_ang)
            
            #repeat test for same angle
            for i in range(self.iterations):
                rec_angle_error.append(self.model.angle_analysis())
                
            model_error.append(np.mean(np.array(rec_angle_error)))
        '''
        ax.plot(self.rec_test_angles,model_error)
        ax.set_title("Mean abs error for each angle")
        ax.set_ylabel("MAE")
        ax.set_xlabel("Angle")
        
        
    def makemovie(self,filename=None):
        """
        Generates a diagnostic/debug movie, saved in 'filename'.
        """
        from moviepy.editor import VideoClip
        from moviepy.video.io.bindings import mplfig_to_npimage
        import matplotlib.pyplot as plt
        fig,ax1  = plt.subplots()
        l, b, h, w = .6, .75, .3, .3
        ax2 = fig.add_axes([l,b,h,w], projection="polar")

        def make_frame(t):
            self.model.set_antenna_separation(self.ant_sep_angles[self.i])
            ax1.clear()
            ax2.clear()
            self.eval_model(ax1)
            self.model.polar_plot(ax2)
            self.i += 1
            return mplfig_to_npimage(fig)

        Nframes = np.shape(self.ant_sep_angles)[0]
        animation = VideoClip(make_frame, duration = Nframes)
        
        if filename is not None:
            animation.write_videofile(filename,fps=1,codec='mpeg4',bitrate='3000k')
        else:
            return animation.ipython_display(fps = 1, loop = False, autoplay = True)