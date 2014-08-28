'''
 Copyright (C) 2014 - Federico Corradi
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


############### author ##########
# federico corradi
# federico@ini.phys.ethz.ch
# Juan Pablo Carbajal 
# ajuanpi+dev@gmail.com
#
# ===============================
#!/usr/env/python

import numpy as np
from pylab import *
import pyNCS
import sys
sys.path.append('../api/lsm/')

#parameters and actions 
n_steps = 10
do_figs_encoding = False

#populations divisible by 2 for encoders
neuron_ids = np.linspace(0,255,256)
npops = len(neuron_ids)

#setup
prefix='../'
setuptype = '../setupfiles/mc_final_mn256r1.xml'
setupfile = '../setupfiles/final_mn256r1_retina_monster.xml'
nsetup = pyNCS.NeuroSetup(setuptype, setupfile, prefix=prefix)
chip = nsetup.chips['mn256r1']
nsetup.mapper._init_fpga_mapper()
chip.configurator._set_multiplexer(0)

#populate neurons
rcnpop = pyNCS.Population('neurons', 'for fun') 
rcnpop.populate_by_id(nsetup,'mn256r1','excitatory', neuron_ids)

#init nef on neuromorphic chips
import lsm
liquid = lsm.Lsm(rcnpop) #init liquid state machine 

ntrials = 150  #same projection stimulus, but regenerated poisson trains
inputs, outputs = liquid.stimulate_reservoir(trials=ntrials)
for trial in range(ntrials):
    np.savetxt("lsm/inputs_correlated_trial_"+str(trial)+".txt", inputs[trial])
    np.savetxt("lsm/outputs_correlated_trial_"+str(trial)+".txt", outputs[trial])
