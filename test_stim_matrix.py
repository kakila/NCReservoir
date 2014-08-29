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

#parameters and actions 
n_steps = 10
#init nef on neuromorphic chips
import lsm as L
liquid = L.Lsm() #init liquid state machine 

# Mean rate basis
rates = []
for n in xrange (0,3):
    rates.append(lambda t,w=n: 0.5+0.5*np.sin(2*np.pi*(w+1)*t))

# Single spatial distribution
G = [lambda x,y: np.exp (-(x**2 + y**2))]

# Multiple spatial distribution
G  = []
x0 = [-0.5, 0, 0.5] 
for n in xrange (0,3):
    G.append(lambda x,y,w=n: np.exp (-((x-x0[w])**2 + y**2)))


