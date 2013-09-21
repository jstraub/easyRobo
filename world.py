# Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
# Licensed under the MIT license. See LICENSE.txt or
# http://www.opensource.org/licenses/mit-license.php

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from aux import *

class GridWorld2D:

  def __init__(s,w,h,uCov):
    s.world = np.ones((h,w))*100
    s.world[1:h-1,1:w-1] = 0
    s.extractObstacles()
    s.uCov = uCov

  def act(s,x_n):
    dt = 1.0
    x_d = x_n[0:2] + 0.5
    x_d = np.floor(x_d)
    if s.world[int(x_d[1]),int(x_d[0])] >0:
      return None
    else:
      return x_n

  def move(s,x,u):
    dt = 1.0
    uR = np.dot(s.uCov,np.resize(np.random.randn(2),(2,1))) + u
    x[0] += uR[0]*dt * np.cos(x[2] + uR[1]*dt/2)
    x[1] += uR[0]*dt * np.sin(x[2] + uR[1]*dt/2)
    x[2] = ensureRange(x[2] + uR[1]*dt)
    return x

  def extractObstacles(s):
    s.obst=[]
    nr = 0
    for i in range(0,s.world.shape[0]):
      for j in range(0,s.world.shape[1]):
        if s.world[i,j]>0:
          s.obst.append(np.reshape(np.array([j,i,nr]),(3,1))) # [x,y,id]
          nr += 1

  def plot(s,axs):
    for ax in axs:
      ax.imshow(s.world,interpolation='nearest',cmap=cm.hot)
      ax.set_xlim([-0.5,s.world.shape[1]-0.5])
      ax.set_ylim([-0.5,s.world.shape[0]-0.5])

class SimpleWorld2D(GridWorld2D):

  def __init__(s,path):
    # load image from file as world
    GridWorld2D.extractObstacles(s)

