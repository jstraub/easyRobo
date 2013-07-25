# Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
# Licensed under the MIT license. See LICENSE.txt or
# http://www.opensource.org/licenses/mit-license.php

import numpy as np
import matplotlib.pyplot as plt

from aux import *

class RangeSensor:

  def __init__(s,maxR,maxPhi,zCov):
    s.maxR = maxR
    s.maxPhi = maxPhi
    s.zCov = zCov

  def sense(s,world,x):
    obsts = world.obst
    obs_vis = []
    for obst in obsts:
      z = np.dot(s.zCov,np.resize(np.random.randn(2),(2,1)))
      z[0] += dist(x[0:2],obst[0:2])
      z[1] = ensureRange(z[1] + angle(x,obst[0:2]))
      if z[0] <= s.maxR and np.abs(z[1]) < s.maxPhi/2.0:
        plt.plot(obst[0],obst[1],'gx')
        obs_vis.append(np.array([z[0],z[1],obst[2]]).ravel())
    return obs_vis

  def predict(s,x,l):
    # predict measurement based on robot pose x and landmark position l
    zPred = np.zeros((2,1))
    dx, dy = l[0]-x[0], l[1]-x[1]
#    print 'prediction: dx={}; dy={}; atan={}'.format(dx,dy,np.arctan2(dy,dx) )
    zPred[0], zPred[1] = np.sqrt(dx*dx+dy*dy), ensureRange(np.arctan2(dy,dx) - x[2])
    return zPred


class MultiModalRangeSensor(RangeSensor):

  def __init__(s,maxR,maxPhi,zCov):
    RangeSensor.__init__(s,maxR,maxPhi,zCov)

  def sense(s,world,x):
    obsts = world.obst
    obs_vis = []
    for obst in obsts:
      z = np.dot(s.zCov,np.resize(np.random.randn(2),(2,1)))
      z[0] += dist(x[0:2],obst[0:2])
      if np.random.rand(1)[0] > 0.5: # flip a coin to obtain multimodal outcome
        z[0] += 2
      z[1] = ensureRange(z[1] + angle(x,obst[0:2]))
      if z[0] <= s.maxR and np.abs(z[1]) < s.maxPhi/2.0:
        plt.plot(obst[0],obst[1],'gx')
        obs_vis.append(np.array([z[0],z[1],obst[2]]).ravel())
    return obs_vis

