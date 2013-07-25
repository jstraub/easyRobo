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

class SAM_Robo(DotRobo2D):
  def __init__(s,x0,xCov,lCov,sensor):
    s.C = np.zeros((3,3))
    s.xx = x0 # robot 2D state 
    s.xl = np.zeros((0,1)) # landmark state
    s.l = dict() # dict of all landmark ids that are currently in the state (landmark id -> id in s.xl)
    s.bx = np.zeros((3,1)) # observations
    s.bl = np.zeros((0,1)) # observations
    s.Jxx = -np.eye(3)*1000.0
    s.Jxl = np.zeros((0,0))
    s.Jll = np.zeros((0,0))
    s.Jlx = np.zeros((0,0))
    s.nz = 0

    # prepare inverse sqrt of the ovariances for multiplication in the Jacobian
    s.xCovInvSqrt = np.real(scipy.linalg.sqrtm(np.linalg.inv(xCov)))
    s.lCovInvSqrt = np.real(scipy.linalg.sqrtm(np.linalg.inv(lCov)))

    DotRobo2D.__init__(s,x0,sensor)

  def plot(s,axs):
    Robo2D.plot(s,axs[0])
    axs[1].plot(s.xx[0::3],s.xx[1::3],'b-x')
    thetas = s.xx[2::3]
    for i in range(0,len(thetas)):
      axs[1].plot([s.xx[i*3], s.xx[i*3]+3*np.cos(thetas[i]) ],[s.xx[i*3+1], s.xx[i*3+1]+3*np.sin(thetas[i])],'g-')
    axs[1].plot(s.xl[0::2],s.xl[1::2],'ro')

  def plotSnapshot(s,J,b):
    print 'plotting Jacobian'
    print 'J.shape = {}; |l|={}; |x|={}'.format(J.shape,len(s.l),len(s.xx)/3)
    fig=plt.figure()
    plt.subplot(221)
    imgplot=plt.imshow(J,cmap=cm.gray) 
    imgplot.set_interpolation('nearest')
    plt.subplot(222)
    imgplot=plt.imshow(np.dot(J.T,J),cmap=cm.gray) 
    imgplot.set_interpolation('nearest') 
    plt.subplot(223)
    L = np.linalg.cholesky(np.dot(J.T,J))
    imgplot=plt.imshow(L,cmap=cm.gray) 
    imgplot.set_interpolation('nearest') 
    ax = plt.subplot(224)
    plt.plot(s.xx[0::3],s.xx[1::3],'b-x')
    thetas = s.xx[2::3]
    for i in range(0,len(thetas)):
      plt.plot([s.xx[i*3], s.xx[i*3]+3*np.cos(thetas[i]) ],[s.xx[i*3+1], s.xx[i*3+1]+3*np.sin(thetas[i])],'g-')
    plt.plot(s.xl[0::2],s.xl[1::2],'ro')
#    plt.xlim([0,100])
#    plt.ylim([0,100])
    ax.set_aspect(1)
    fig.show()
#    raw_input()

  def think(s,plot=None):
    # append new pose and new landmarks (if new ones were observed) 
    J,b = SAM_Robo.augmentState(s)

    for it in range(0,20):
      if it>0:
        J,b = SAM_Robo.relinearize(s)
      dx=np.linalg.solve(np.dot(J.T,J),np.dot(J.T,b))
#      L = np.linalg.cholesky(np.dot(J.T,J))
#      y=np.linalg.solve(L.T,np.dot(J.T,b))
#      dx=np.linalg.solve(L,y)
      s.xx += dx[0:len(s.xx)]
      s.xx[2::3] = ensureRange(s.xx[2::3])
      #print 'thetas={}'.format(s.xx[2::3].T)
      s.xl += dx[len(s.xx)::]

      avgDx = np.sum(np.abs(dx))/len(dx)

      #print 'x={};\ty={}\ttheta={}'.format(s.xx[0::3].T,s.xx[1::3].T,s.xx[2::3].T)
      #print 'dx={};\tdy={}\tdtheta={}'.format(dx[0:len(s.xx):3].T,dx[1:len(s.xx):3].T,dx[2:len(s.xx):3].T)
      #print 'dlx={};\tdly={}'.format(dx[len(s.xx)::2].T,dx[len(s.xx)::2].T)
      print 'avgDx={}'.format(avgDx)
      #print 'bx={};\tby={}\tbtheta={}'.format(b[0:len(s.xx):3].T,b[1:len(s.xx):3].T,b[2:len(s.xx):3].T)
      #print 'zR={};\tzPhi={}'.format(b[len(s.xx)::2].T,b[len(s.xx)::2].T/np.pi*180.0)
      if plot is not None and plot is True:
        SAM_Robo.plotSnapshot(s,J,b)

      if avgDx < 1.0e-6:
        break

    #u =  DotRobo2D.think(s)
    s.u.append(np.zeros((2,1))) # no movement
    s.u[-1][0] = 1.0 # speed v
    s.u[-1][1] = 0.1 # rotational speed omega 
    return s.u

  def relinearize(s):
    #print '-- relinearizing |u|={}; |z|={}'.format(len(s.u),len(s.z))
    # prediction of pose
    for t in range(1,len(s.u)):
      ix = t*3
      xPred = Robo2D.move(s,s.xx[ix-3:ix],s.u[t])
      dx = s.xx[ix:ix+3]-xPred
      dx[2] = ensureRange(dx[2])
      s.bx[ix:ix+3] = np.dot(s.xCovInvSqrt.T, dx)
#      print 'xx={}; ix={}'.format(s.xx,ix)
      s.Jxx[ix:ix+3,ix-3:ix] = np.dot(s.xCovInvSqrt.T, SAM_Robo.dxdx(s,s.xx[ix:ix+3],s.u[t]))
    #s.bx[2::3] = ensureRange(s.bx[2::3])
      
    ibl = 0
    for t in range(0,len(s.z)):
      ix = t*3
      for i in range(0,len(s.z[t])):  
        il = s.l[s.z[t][i][2]] 
        zPred = s.sensor.predict(s.xx[ix:ix+3], s.xl[il:il+2])
        dz = np.resize(s.z[t][i][0:2],(2,1))-zPred
        dz[1] = ensureRange(dz[1])
        s.bl[ibl:ibl+2] = np.dot(s.lCovInvSqrt.T, dz)
        s.Jlx[ibl:ibl+2, ix:ix+3] = np.dot(s.lCovInvSqrt.T, SAM_Robo.dldx(s,s.xx[ix:ix+2], s.xl[il:il+2]))
        s.Jll[ibl:ibl+2, il:il+2] = np.dot(s.lCovInvSqrt.T, SAM_Robo.dldl(s,s.xx[ix:ix+2], s.xl[il:il+2]))
        #print 'zReal={},{}; zPred={}; dz={}'.format(s.z[t][i][0],toDeg(s.z[t][i][0]),toDeg(zPred[0]),toDeg(s.z[t][i][0]-zPred[0]))
        ibl +=2
    #s.bl[1::2] = ensureRange(s.bl[1::2])

    Jx = np.concatenate((s.Jxx,s.Jxl),axis=1)
    Jl = np.concatenate((s.Jlx,s.Jll),axis=1)
    J = np.concatenate((Jx,Jl),axis=0)
#    print '{} {}: {}'.format(s.bx.shape,s.bl.shape,b.shape)
#    print '{} {}: {}'.format(Jx.shape,Jl.shape,J.shape)

    b=np.concatenate((s.bx,s.bl))
    b=np.resize(b,(len(b),1))

    return (J,b)

  def augmentState(s):
    # prediction of pose as first estimate
    ix = len(s.xx)-3
    s.xx = np.concatenate((s.xx,Robo2D.move(s,s.xx[ix:ix+3],s.u[-1])),axis=0)
#    print 'xx={}; ix={}'.format(s.xx,ix)
    s.bx = np.concatenate((s.bx, np.dot(s.xCovInvSqrt.T, np.zeros((3,1)))),axis=0)

    # append jacobian Jxx
    w,h = s.Jxx.shape[1],s.Jxx.shape[0]
    Jxx = np.zeros((h+3,w+3))
    Jxx[0:h,0:h] = s.Jxx
    Jxx[h:h+3,w:w+3] = np.dot(s.xCovInvSqrt.T, -np.eye(3))
    Jxx[h:h+3,w-3:w] = np.dot(s.xCovInvSqrt.T, SAM_Robo.dxdx(s,s.xx[ix:ix+3],s.u[-1]))
    s.Jxx = Jxx
    
    s.nz += len(s.z[-1])
    for z in s.z[-1]:
      if z[2] not in s.l:
        # new landmark -> augment state
        s.l[z[2]] = len(s.xl)
        s.xl = np.concatenate((s.xl,SAM_Robo.initLandmark(s,s.xx[ix:ix+3],z[0:2])),axis=0)
#        print s.xx[ix:ix+3]
#        print '  new l: {}@{} ({})'.format(s.l[z[2]],s.xl[s.l[z[2]]-2:s.l[z[2]]],z[0:2])

      il = s.l[z[2]]
      zPred = s.sensor.predict(s.xx[ix:ix+3], s.xl[il:il+2])
      dz = (np.resize(z[0:2],(2,1))- zPred)
      dz[1] = ensureRange(dz[1])
      s.bl = np.concatenate((s.bl, np.dot(s.lCovInvSqrt.T, dz)),axis=0)
    #s.bl[1::2] = ensureRange(s.bl[1::2])

    # jacobian dx/dl
    w,h = s.Jlx.shape[1],s.Jlx.shape[0]
    Jlx = np.zeros((h+len(s.z[-1])*2,len(s.xx)))
    Jlx[0:h,0:w] = s.Jlx
    for i in range(0,len(s.z[-1])):
      z = s.z[-1][i]
      il = s.l[z[2]] 
#      print 'z:{}, il={}, ix={}, Jlx.shape={}; l={}'.format(z,il,ix,Jlx.shape,s.xl[il:il+2])
      Jlx[h+i*2:h+i*2+2, ix:ix+3] = np.dot(s.lCovInvSqrt.T, SAM_Robo.dldx(s,s.xx[ix:ix+2], s.xl[il:il+2]))
    s.Jlx = Jlx
#    print 'Jlx=\n{}'.format(s.Jlx)

    # jacobian dl/dl
    w,h = s.Jll.shape[1],s.Jll.shape[0]
    Jll = np.zeros((h+len(s.z[-1])*2,len(s.xl)))
    Jll[0:h,0:w] = s.Jll
#    print 'Jll.shape={}'.format(Jll.shape)
    for i in range(0,len(s.z[-1])):
      z = s.z[-1][i]
      il = s.l[z[2]] 
      Jll[h+i*2:h+i*2+2, il:il+2] = np.dot(s.lCovInvSqrt.T, SAM_Robo.dldl(s,s.xx[ix:ix+2], s.xl[il:il+2]))
    s.Jll = Jll
#    print 'Jll=\n{}'.format(s.Jll)

    s.Jxl = np.zeros((s.Jxx.shape[0],s.Jll.shape[1]))
    Jx = np.concatenate((s.Jxx,s.Jxl),axis=1)
    Jl = np.concatenate((s.Jlx,s.Jll),axis=1)
    J = np.concatenate((Jx,Jl),axis=0)
#    print '{} {}: {}'.format(s.bx.shape,s.bl.shape,b.shape)
#    print '{} {}: {}'.format(Jx.shape,Jl.shape,J.shape)
    b=np.concatenate((s.bx,s.bl))
    b=np.resize(b,(len(b),1))
    return (J,b)

  
  def dxdx(s,x,u):
    # motion jacobian
    dt = 1.0
    Jm = np.eye(3)
#    Jm[0,2] = -v/omega * np.cos(theta) + v/omega * np.cos(theta+omega*dt)
#    Jm[1,2] = -v/omega * np.sin(theta) + v/omega * np.sin(theta+omega*dt)
    Jm[0,2] = -u[0]*dt * np.sin(x[2]+u[1]*dt/2.0)
    Jm[1,2] = u[0]*dt * np.cos(x[2]+u[1]*dt/2.0)
    return Jm

  def dxdl(s):
    # derivtive of the motion with respect to a landmark
    return np.zeros((3,2))
    
  def dldx(s,x,l):
    # derivatie of a landmark with respect to the state 
    dx, dy = l[0]-x[0], l[1]-x[1]
    r_sq = dx*dx + dy*dy
    r = np.sqrt(r_sq)
    Jlx = np.zeros((2,3))
    Jlx[0,0] = -dx*r
    Jlx[0,1] = -dy*r
    Jlx[0,2] = 0.0
    Jlx[1,0] = dy 
    Jlx[1,1] = -dx
    Jlx[1,2] = -r_sq
    Jlx /= r_sq
    return Jlx
    
  def dldl(s,x,l):
    # derivative of a landmark with respect to a landmark
    dx, dy = l[0]-x[0], l[1]-x[1]
    r_sq = dx*dx + dy*dy
    r = np.sqrt(r_sq)
    Jll = np.zeros((2,2))
    Jll[0,0] = dx*r
    Jll[0,1] = dy*r
    Jll[1,0] = -dy 
    Jll[1,1] = dx
    Jll /= r_sq
    return Jll
  
  def initLandmark(s,x,z):
    l = np.zeros((2,1))
    l[0] = x[0] + z[0] * np.cos(x[2]+z[1]) # z[0] is radius
    l[1] = x[1] + z[0] * np.sin(x[2]+z[1]) # z[1] is the angle
    return l

 
  
