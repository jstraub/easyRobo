
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from robo import *


class Mapping_Robo(DotRobo2D):
  def __init__(s,x0,sensor):
    # parameter of the Gamma distribution - prior on the rate parameters \lambda_i
    s.alpha = np.ones((400,400))*1
    s.beta = np.ones((400,400))* 10
    DotRobo2D.__init__(s,x0,sensor)

  def plot(s,axs):
    Robo2D.plot(s,axs[0])
    axs[1].imshow(s.alpha,interpolation='nearest',cmap=cm.hot)
    axs[0].imshow(s.beta,interpolation='nearest',cmap=cm.hot)
    axs[2].imshow((s.alpha-1/s.beta),interpolation='nearest',cmap=cm.hot)

#    if len(s.z) >0:
#      axs[2].imshow(s.z[-1],interpolation='nearest')

  def think(s):
    #print s.z[-1]

    o = s.z[-1][0]
    n = s.z[-1][1]
  
    iMinW = (np.floor(s.xWorld[-1][1])-o.shape[0]/2)[0]
    iMaxW = (np.floor(s.xWorld[-1][1])+o.shape[0]/2)[0]
    jMinW = (np.floor(s.xWorld[-1][0])-o.shape[1]/2)[0]
    jMaxW = (np.floor(s.xWorld[-1][0])+o.shape[1]/2)[0]
    s.alpha[iMinW:iMinW+o.shape[0], jMinW:jMinW+o.shape[1]] += o/100
    s.beta[iMinW:iMinW+n.shape[0], jMinW:jMinW+n.shape[1]] += n

    s.u.append(np.zeros((2,1))) # no movement
    s.u[-1][0] = 1.0 # speed v
    s.u[-1][1] = 0.1 # rotational speed omega 
    return s.u



