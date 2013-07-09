# Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
# Licensed under the MIT license. See LICENSE.txt or
# http://www.opensource.org/licenses/mit-license.php

import numpy as np

def dist(a,b):
  return np.sqrt(np.sum((a-b)*(a-b)))

def angle(x,a):
  dr = a - x[0:2]
  rad = np.arctan2(dr[1],dr[0]) - x[2] # need wrapping here
  return ensureRange(rad)
 
def ensureRange(rad):
  return (rad+np.pi)%(2*np.pi) - np.pi

def toRad(deg):
  return np.pi*deg/180.0

def toDeg(rad):
  return rad*(180.0/np.pi)
