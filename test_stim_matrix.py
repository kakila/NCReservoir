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
do_figs_encoding = False

#populations divisible by 2 for encoders
neuron_ids = np.linspace(0,255,256)
npops = len(neuron_ids)

#setup
prefix='./'
setuptype = './setupfiles/mc_final_mn256r1.xml'
setupfile = './setupfiles/final_mn256r1_retina_monster.xml'
nsetup = pyNCS.NeuroSetup(setuptype, setupfile, prefix=prefix)
chip = nsetup.chips['mn256r1']
#nsetup.mapper._init_fpga_mapper()
#chip.configurator._set_multiplexer(0)

#populate neurons
rcnpop = pyNCS.Population('neurons', 'for fun')
rcnpop.populate_all(nsetup,'mn256r1','excitatory')

#init nef on neuromorphic chips
import lsm as L
liquid = L.Lsm(rcnpop) #init liquid state machine 

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


