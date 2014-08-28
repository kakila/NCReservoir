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

ntrials = 3  #same projection stimulus, but regenerated poisson trains
nprojs = 2  #number of different projections from input to output
for this_stim in range(nprojs):
    inputs, outputs = liquid.stimulate_reservoir(trials=ntrials)
    for trial in range(ntrials):
        np.savetxt("lsm/inputs_proj_"+str(this_stim)+"_trial_"+str(trial)+".txt", inputs[trial])
        np.savetxt("lsm/outputs_proj_"+str(this_stim)+"_trial_"+str(trial)+".txt", outputs[trial])

