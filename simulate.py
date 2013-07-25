#! /usr/bin/env python

# Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
# Licensed under the MIT license. See LICENSE.txt or
# http://www.opensource.org/licenses/mit-license.php 

import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

from robo import *
from simpleWorld import *
from rangeSensor import *
from aux import *

if __name__ == '__main__':

  x0 = np.zeros((3,1))
  x0[0:2] += 100
  
  uCov = np.eye(2)*0.1 # covariance for gaussian movement simple movement model 
  uCov[1,1] *= 0.1

  zCov = np.eye(2)*0.1 # covariance for range sensor 
  zCov[1,1] *= 0.1

  xCov = np.eye(3)*2 # covariance for pose estimate
  xCov[2,2] = toRad(5)
  lCov = np.eye(2)*2 # covariance for landmark position estimate

  #robo = DotRobo2D(x0,RangeSensor(50,toRad(45)))
  #robo = SAM_Robo(x0,xCov,lCov,RangeSensor(50,toRad(90),zCov))
  robo = SAM_Robo(x0,xCov,lCov,MultiModalRangeSensor(50,toRad(90),zCov))
  world = GridWorld2D(200,200,uCov)
  for i in range(0,100):
    world.world[np.random.randint(0,199,1),np.random.randint(0,199,1)] =100
  world.extractObstacles()

  fig = plt.figure()
  axs=[plt.subplot(121), plt.subplot(122)]
  for t in range(0,20):
    axs[0].clear()
    axs[1].clear()
    world.plot(axs)
    robo.plot(axs)

    robo.sense(world)
    robo.think()
    robo.act(world)
    
    plt.title('t='+str(t))
    fig.canvas.draw()
    fig.show()
    print '--- t='+str(t)
    print robo.status()
    time.sleep(0.1)

  # extract observed landmark ids
  dr =[] # range error
  for t in range(0,len(robo.z)):
    ix = t*3
    x = robo.xx[ix:ix+3]
    for zi in robo.z[t]:
      il = robo.l[zi[2]] # index of landmarks in the mean estimate
      dr.append(dist(robo.xl[il:il+2],x[0:2])-zi[0])
      if dr[-1]>10:
        print 'x={}; l={}; dist={}; obs={}'.format(x,robo.xl[il:il+2],dist(robo.xl[il:il+2],x[0:2]),zi[0])
        plt.plot([robo.xl[il],x[0]],[robo.xl[il+1],x[1]],'r-')
      
  dr = np.array(dr)
  fig0 = plt.figure()
  plt.hist(dr,50)
  fig0.show()
  
  # save all observations to file - needed once for manifold mapping
  #pickle.dump(robo.z,open('obsHist.pickle','w'))
  
  raw_input()
