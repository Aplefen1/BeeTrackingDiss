import numpy as np

def dist(a,b):
    d = a-b
    d[d>np.pi]-=np.pi*2
    d[d<-np.pi]+=np.pi*2
    return d

def getgain(ang):
    apr = 33.5*np.exp(-(dist(ang,0)*1.88)**2)
    apr+= 16*np.exp(-(dist(ang,np.pi*(0.72))*2)**2)
    apr+= 16*np.exp(-(dist(ang,-np.pi*(0.72))*2)**2)
    apr+= 14*np.exp(-(dist(ang,np.pi)*2.3)**2)
    apr-= 20
    return apr