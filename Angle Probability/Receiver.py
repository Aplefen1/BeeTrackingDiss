
class Receiver:
    def __init__(self, position, gain, sens) -> None:
        
        self.position = position
        self.gain = gain
        self.sensitivity = sens
        
########## Visualisations ##############
    def plot_spatial(self, ax):
        ax.plot(self.position[0],self.position[1],'oy')
        
    def set_pos(self, pos : list[float]):
        '''Pos : [x,y]'''
        self.position = pos