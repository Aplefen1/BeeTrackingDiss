from numba import jit, cuda
class Receiver:
    def __init__(self, position, gain) -> None:
        
        self.position = position
        self.gain = gain
        
########## Visualisations ##############
    def plot_spatial(self, ax):
        ax.plot(self.position[0],self.position[1],'oy')
        
    def set_pos(self, pos : list[float]):
        '''Pos : [x,y]'''
        self.position = pos