
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from robo import *


class PoissonMapping_Robo(DotRobo2D):
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


class GridMapping_Robo(DotRobo2D):
  def __init__(s,x0,sensor):
    # maximum entropy prior on all gridcells
    s.l0 = np.log(0.5/0.5)
    s.l_free = np.log(0.1/0.9)
    s.l_occ = np.log(0.9/0.1)
    s.l = np.ones((400,400))*s.l0
    DotRobo2D.__init__(s,x0,sensor)

  def plot(s,axs):
    Robo2D.plot(s,axs[0])
    p=1.0 - 1.0/(1.0+np.exp(s.l))
    axi=axs[1].imshow(p,interpolation='nearest',cmap=cm.coolwarm)
    print('pmin={}; pmax={}'.format(np.min(p),np.max(p)))

#    if len(s.z) >0:
#      axs[2].imshow(s.z[-1],interpolation='nearest')

  def think(s):
    #print s.z[-1]

    o = s.z[-1][0]
    n = s.z[-1][1]

    dl = np.zeros((o.shape))*s.l0
    dl[n>0] = s.l_free
    dl[o>0] = s.l_occ

  
    iMinW = (np.floor(s.xWorld[-1][1])-o.shape[0]/2)[0]
    iMaxW = (np.floor(s.xWorld[-1][1])+o.shape[0]/2)[0]
    jMinW = (np.floor(s.xWorld[-1][0])-o.shape[1]/2)[0]
    jMaxW = (np.floor(s.xWorld[-1][0])+o.shape[1]/2)[0]
    s.l[iMinW:iMinW+o.shape[0], jMinW:jMinW+o.shape[1]] += dl - s.l0

    s.u.append(np.zeros((2,1))) # no movement
    s.u[-1][0] = 1.0 # speed v
    s.u[-1][1] = 0.1 # rotational speed omega 
    return 
