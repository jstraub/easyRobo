#! /usr/bin/env python

# Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
# Licensed under the MIT license. See LICENSE.txt or
# http://www.opensource.org/licenses/mit-license.php 

import numpy as np
import matplotlib.pyplot as plt
import time

from robo import *
from simpleWorld import *
from rangeSensor import *

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
  robo = EKF_SLAM_Robo(x0,xCov,lCov,RangeSensor(50,toRad(45),zCov))
  world = GridWorld2D(200,200,uCov)
  for i in range(0,100):
    world.world[np.random.randint(0,199,1),np.random.randint(0,199,1)] =100
  world.extractObstacles()

  fig = plt.figure()
  axs=[plt.subplot(121), plt.subplot(122)]
  for t in range(0,500):
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
  
  raw_input()
