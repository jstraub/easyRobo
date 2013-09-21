# Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
# Licensed under the MIT license. See LICENSE.txt or
# http://www.opensource.org/licenses/mit-license.php

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from aux import *

def radRange(rad):
  return (rad+np.pi)%(2*np.pi) - np.pi

class Robo2D:
  
  def __init__(s,x0,sensor):
    s.sensor = sensor
    s.xWorld = [x0]
    s.z=[]
    s.u=[np.zeros((2,1))]

  def act(s,world):
    x_n = world.move(s.xWorld[-1],s.u[-1])
    x_n = world.act(x_n)
    if x_n is not None:
      s.xWorld.append(x_n)
    else:
      s.xWorld.append(s.xWorld[-1])
    return s.xWorld

  def sense(s,world):
    s.z.append(s.sensor.sense(world,s.xWorld[-1]))
    return s.z[-1]

  def think(s):
    s.u.append(np.zeros((2,1))) # no movement
    return s.u[-1]
 
  def move(s,x,u):
    dt = 1.0
#    a=u[0]/u[1]
#    x[0] += -a * np.sin(x[2]) + a * np.sin(x[2]+u[1]*dt)
#    x[1] += a * np.cos(x[2]) - a * np.cos(x[2]+u[1]*dt)
    xPred = np.zeros((3,1))
    xPred[0] = x[0] + u[0]*dt * np.cos(x[2] + u[1]*dt/2)
    xPred[1] = x[1] + u[0]*dt * np.sin(x[2] + u[1]*dt/2)
    xPred[2] = ensureRange(x[2] + u[1]*dt)
    return xPred

  def plot(s,ax):
    r = s.sensor.maxR
    dx = np.array([np.cos(s.xWorld[-1][2]),np.sin(s.xWorld[-1][2])]) * r*0.5
    ax.plot(s.xWorld[-1][0],s.xWorld[-1][1],'rx')
    ax.plot([s.xWorld[-1][0], s.xWorld[-1][0]+dx[0] ],[s.xWorld[-1][1], s.xWorld[-1][1]+dx[1] ],'b-')

    dPhi = s.sensor.maxPhi/2.0
    n = 10
    scan = np.zeros((2,n))
    x = s.xWorld[-1][0:2].ravel()
    scan[:,0], scan[:,n-1] = x, x

    minPhi = s.xWorld[-1][2]-dPhi
    maxPhi = s.xWorld[-1][2]+dPhi
    phis = np.linspace(minPhi,maxPhi,n-2)
    for i in range(0,n-2):
      phis[i] = ensureRange(phis[i])

    for i in range(1,n-1):
      scan[:,i] = x + r*np.array([np.cos(phis[i-1]),np.sin(phis[i-1])])
    ax.plot(scan[0,:],scan[1,:],'r-')

  def status(s):
    return ' x={}\t |z|={}'.format(s.xWorld[-1].T,len(s.z[-1]))

class DotRobo2D(Robo2D):

  def __init__(s,x0,sensor):
    Robo2D.__init__(s,x0,sensor)

  def think(s):
    s.u.append(np.ones((2,1)))
    s.u[-1][0] = 1.0 # speed v
    s.u[-1][1] = 0.1 # rotational speed omega 
    return s.u[-1]

  
