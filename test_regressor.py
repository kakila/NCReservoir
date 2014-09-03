'''
 Copyright (C) 2014 - Juan Pablo Carbajal

 This progrm is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

# Author: Juan Pablo Carbajal <ajuanpi+dev@gmail.com>

#!/usr/env/python
from __future__ import division
 
import numpy as np
from pylab import *
import pyNCS
import sys
import lsm as L

liquid = L.Lsm() #init liquid state machine
prob_n_itter = 0.9;
t = np.linspace (0,1,1e3)[:,None]
M = np.random.randn(1,256)
x = t.dot(M)
bias = 0.05*np.random.randn(1,256)
expo = [([1]*11+range(5))*16]
zeros = np.where(np.random.rand(256,1)>prob_n_itter)[0]
def sys (x,bias,expo,zeros): 
    y = x**expo + bias
    y[:,1:10] = np.sin (2*np.pi*3*y[:,1:10]) 
    for i in zeros:
        y[:,i] = 0
    return y

y = sys (t,bias,expo,zeros)
K = np.random.randn(256,1)
z = y.dot(K[::-1])
score = []
for i in xrange(500):
    t_ = t + 0.05*np.random.randn(t.shape[0],t.shape[1])
    x_ = t_.dot(M)
    zeros = np.where(np.random.rand(256,1)>prob_n_itter)[0]
    y_ = sys (t_,bias,expo,zeros)
    z_ = y_.dot(K[::-1])
    z_ = z_ + 0.1*np.random.randn(t.shape[0],t.shape[1])  
    liquid._realtime_learn (x_,y_,z_)
    score.append([liquid._regressor["input"].score(x_,z_), liquid._regressor["output"].score(y_,z_)])

t_ = t + 0.05*np.random.randn(t.shape[0],t.shape[1])
x_ = t_.dot(M)
zeros = np.where(np.random.rand(256,1)>1)[0]
y_ = sys (t_,bias,expo,zeros)
z_ = y_.dot(K[::-1])
z_ = z_ + 0.1*np.random.randn(t.shape[0],t.shape[1])  

zh = liquid.RC_predict (x_,y_)
clf()
plot(t,z,t,zh["input"],t,zh["output"])

