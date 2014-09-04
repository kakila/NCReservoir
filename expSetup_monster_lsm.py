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
import matplotlib

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
import lsm as L
liquid = L.Lsm(rcnpop, cee=0.8, cii=0.5) #init liquid state machine

nsetup.chips['mn256r1'].load_parameters('biases/biases_reservoir.biases')
c = 0.3
dim = np.round(np.sqrt(len(liquid.rcn.synapses['virtual_exc'].addr)*c))
### generate gestures
num_gestures = 1
gestures = []
for this_gesture in range(num_gestures):
    freqs = np.random.randint(7,size=3).tolist()   
    centers = np.random.random((3,2)).tolist()
    width = np.random.random((1,3)).tolist()
    gestures.append({'freq': freqs, 'centers': centers, 'width': width})
    
import json
json.dump(gestures, open("lsm/gestures.txt",'w'))

scores = []
func_avg = lambda t,ts: np.exp((-(t-ts)**2)/(2*150**2))
ntrials = 1
fig_h = figure()
ion()

liquid.RC_reset()
for ind,this_g in enumerate(gestures):

    M = liquid.create_stimuli_matrix(dim, this_g)
    #one stim all trials --> NO VARIABILITY IN THE INPUTs
    stimulus = liquid.create_spiketrain_from_matrix(M)
    
    for this_t in xrange(ntrials): 
    
        #stimulate
        inputs, outputs = liquid.RC_poke(stimulus)

        #if(learn_real_time == True):
        ac = np.mean(func_avg(liquid.timev[:,None], inputs[0][:,0][None,:]), axis=1) 
        ac = ac / np.max(ac)
        ac = ac[:,None]
        # Convert input and output spikes to analog signals
        X = liquid._ts2sig(inputs[0][:,0], np.floor(inputs[0][:,1]))
        Y = liquid._ts2sig(outputs[0][:,0], outputs[0][:,1])
        teach_sig = liquid.teach_generator(X* ac)[:,None] * ac **4

        #learn
        liquid._realtime_learn (X,Y,teach_sig)

        #evaluate
        this_score = [liquid._regressor["input"].score(X,teach_sig), \
                     liquid._regressor["output"].score(Y,teach_sig)]
        scores.append(this_score)

        print "we are scoring..."
        print this_score
        print "we are plotting"
        figure(fig_h.number)
        for i in range(256):
            subplot(16,16,i)
            plot(Y[:,i])
            axis('off')
            

figure()
plot(np.array(scores)[:,0], 'o-', label='input')
plot(np.array(scores)[:,1], 'o-', label='output')
legend(loc='best')

figure()
zh = liquid.RC_predict (X,Y)
clf()
plot(liquid.timev,teach_sig,liquid.timev,zh["input"],liquid.timev,zh["output"],label="ref, input, output")
legend(loc='best')



