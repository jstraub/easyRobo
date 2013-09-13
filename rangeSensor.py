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

class ScanerSensor(RangeSensor):
  def __init__(s,maxR,maxPhi,dPhi,zCov):
    s.dPhi = dPhi
    RangeSensor.__init__(s,maxR,maxPhi,zCov)

  def sense(s,world,x):
    
    # occupancy grid o
    o = np.zeros((np.ceil(s.maxR)*2,np.ceil(s.maxR)*2))
    # number of observations
    n = np.zeros((np.ceil(s.maxR)*2,np.ceil(s.maxR)*2))
    # all directions of the scanner
    phis = ensureRange(np.linspace(-s.maxPhi/2.0,s.maxPhi/2.0,s.maxPhi/s.dPhi)+x[2])
    # x0: 0 in local coordinates of occupancy grid o
    x0 = np.ones(3);
    x0[0:2] = x[0:2,0] - np.floor(x[0:2,0]) + np.array([np.floor(s.maxR),np.floor(s.maxR)])
    j0w = np.floor(x[0])
    i0w = np.floor(x[1])

    j0o = np.floor(s.maxR)
    i0o = np.floor(s.maxR)
    for phi in phis:
      xEnd = np.ones(3)
      xEnd[0] = x0[0] + np.cos(phi) * s.maxR
      xEnd[1] = x0[1] + np.sin(phi) * s.maxR
      l = np.cross(x0,xEnd)
#      print '-- phi={}'.format(toDeg(phi))
      
      wall = False
      for x in np.linspace(np.floor(x0[0]),np.floor(xEnd[0]), np.abs(np.floor(x0[0])-np.floor(xEnd[0]))+1.0 ):
        a,b = [x,0.0,1.0],[x,1.0,1.0]
        lx = np.cross(a,b)
        y = np.cross(l,lx)
        y /= y[2]
        #plt.plot(x,y[1],'rx')
        i = np.floor(y[1]-0.5)-i0o
        j = np.floor(x-0.5)-j0o
        if 0<=i+i0w and i+i0w<world.world.shape[1] and 0<=j+j0w and j+j0w<world.world.shape[0] and 0<=i+i0o and i+i0o<o.shape[1] and 0<=j+j0o and j+j0o<o.shape[0] :

#          print 'x: ind={}'.format((i,j))
#          print 'xo: ind={}'.format((i+i0o,j+j0o))
#          print 'xw: ind={}'.format((i+i0w,j+j0w))
          
          o[i+i0o,j+j0o] += world.world[int(i+i0w),int(j+j0w)]
          n[i+i0o,j+j0o] += 1
          if world.world[int(i+i0w),int(j+j0w)] > 0:
            wall = True
            break
          
      if not wall:
        for y in np.linspace(np.floor(x0[1]),np.floor(xEnd[1]), np.abs(np.floor(x0[1])-np.floor(xEnd[1]))+1.0 ):
          a,b = [0.0,y,1.0],[1.0,y,1.0]
          ly = np.cross(a,b)
          x = np.cross(l,ly)
          x /= x[2]
          #plt.plot(x[0],y,'rx')
          i = np.floor(y-0.5) - i0o
          j = np.floor(x[0]-0.5) - j0o
          if 0<=i+i0w and i+i0w<world.world.shape[1] and 0<=j+j0w and j+j0w<world.world.shape[0] and 0<=i+i0o and i+i0o<o.shape[1] and 0<=j+j0o and j+j0o<o.shape[0] :
  #          print 'y: ind={}'.format((i,j))
  #          print 'yo: ind={}'.format((i+i0o,j+j0o))
  #          print 'yw: ind={}'.format((i+i0w,j+j0w))
            o[i+i0o,j+j0o] += world.world[int(i+i0w),int(j+j0w)]
            n[i+i0o,j+j0o] += 1
            if world.world[int(i+i0w),int(j+j0w)] > 0:
              break
#    print o
    return (o,n)

      


