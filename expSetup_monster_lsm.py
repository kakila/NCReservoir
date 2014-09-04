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
import lsm as L



######################################
# Configure chip
try:
  is_configured
except NameError:
  print "Configuring chip"
  is_configured = False
else:
  print "Chip is configured: ", is_configured

if (is_configured == False):
  #populations divisible by 2 for encoders
  neuron_ids = np.linspace(0,255,256)
  npops      = len(neuron_ids)

  #setup
  prefix    = '../'
  setuptype = '../setupfiles/mc_final_mn256r1.xml'
  setupfile = '../setupfiles/final_mn256r1_retina_monster.xml'
  nsetup    = pyNCS.NeuroSetup(setuptype, setupfile, prefix=prefix)
  nsetup.mapper._init_fpga_mapper()

  chip      = nsetup.chips['mn256r1']

  chip.configurator._set_multiplexer(0)

  #populate neurons
  rcnpop = pyNCS.Population('neurons', 'for fun') 
  rcnpop.populate_by_id(nsetup,'mn256r1','excitatory', neuron_ids)

  chip.load_parameters('biases/biases_reservoir.biases')

  #init liquid state machine
  liquid = L.Lsm(rcnpop, cee=0.8, cii=0.5)

  c = 0.3
  dim = np.round(np.sqrt(len(liquid.rcn.synapses['virtual_exc'].addr)*c))
  
  # do config only once
  is_configured = True
# End chip configuration
######################################

######################################
# Generate gestures parameters
num_gestures = 1 # Number of gestures
ntrials      = 3 # Number of repetitions of each gesture

gestures = []
for this_gesture in range(num_gestures):
    freqs   = np.random.randint(7,size=3).tolist()   # in Hz
    centers = np.random.random((3,2)).tolist()
    width   = np.random.random((1,3)).tolist()
    gestures.append({'freq': freqs, 'centers': centers, 'width': width})
    
import json
json.dump(gestures, open("lsm/gestures.txt",'w'))
######################################

######################################
# Generate mean rate signals representing gestures
rates = [[]*len(gestures)]
G     = [[]*len(gestures)]
for ind,this_g in enumerate(gestures):
  for f  in this_g['freq']:
      rates[ind].append(lambda t,w=f: 0.5+0.5*np.sin(2*np.pi*w*t*1e-3)) # time in ms

  # Multiple spatial distribution
  for width,pos in zip(this_g['width'], this_g['centers']):
      G[ind].append(lambda x,y,d=width,w=pos: np.exp ((-(x-w[0])**2 + (y-w[1])**2)/(np.sum(width)**2)))

# Number of time steps to sample the mean rates
nsteps = 50
######################################

# Function to calculate region of activity
func_avg = lambda t,ts: np.exp((-(t-ts)**2)/(2*150**2)) # time in ms

# Handle to figure to plot while learning
fig_h = figure()
fig_i = figure()
ion()

# Store scores of RC
scores = []
# Stimulation parameters
duration   = 1000
delay_sync = 500

# Time vector for analog signals
Fs    = 100/1e3 # Sampling frequency (in kHz)
T     = duration+delay_sync+1000
nT    = np.round (Fs*T)
timev = np.linspace(0,T,nT)

#Conversion from spikes to analog
membrane = lambda t,ts: np.atleast_2d(np.exp((-(t-ts)**2)/(2*50**2)))

liquid.RC_reset()
for ind,this_g in enumerate(gestures):

    M = liquid.create_stimuli_matrix(G[ind], rates[ind], nsteps)
    #one stimiluation for  all trials --> NO VARIABILITY IN THE INPUTs
    stimulus = liquid.create_spiketrain_from_matrix(M, \
                                                    duration=duration, \
                                                    delay_sync=delay_sync)
    
    #generate teaching signal associated with the Gesture
    gesture_teach = rates[0][ind](timev)
    
    for this_t in xrange(ntrials): 
        nsetup.chips['mn256r1'].load_parameters('biases/biases_reservoir.biases')    
        #stimulate
        inputs, outputs = liquid.RC_poke(stimulus)

        #if(learn_real_time == True):
        # Calculate activity of current inputs.
        # As of now the reservoir can only give answers during activity
        ac = np.mean(func_avg(timev[:,None], outputs[0][:,0][None,:]), axis=1) 
        ac = ac / np.max(ac)
        ac = ac[:,None]
        
        # Convert input and output spikes to analog signals
        X = L.ts2sig(timev, membrane, inputs[0][:,0], np.floor(inputs[0][:,1]))
        Y = L.ts2sig(timev, membrane, outputs[0][:,0], outputs[0][:,1])
        #if(this_t >0):
        #    print "X vs Vpre", np.sum(np.abs(Xprev-X))
        #Xprev = X 

        teach_sig = gesture_teach * ac.T**4 # Windowed by activity

        #learn
        liquid._realtime_learn (X,Y,teach_sig.T)

        #evaluate
        this_score = [liquid._regressor["input"].score(X,teach_sig.T), \
                     liquid._regressor["output"].score(Y,teach_sig.T)]
        scores.append(this_score)

        print "we are scoring..."
        print this_score
        print "we are plotting outputs"
        #figure(fig_h.number)
        #for i in range(256):
        #    subplot(16,16,i)
        #    plot(Y[:,i])
        #    axis('off')
        #print "we are plotting inputs"
        #figure(fig_i.number)
        #for i in range(256):
        #    subplot(16,16,i)
        #    plot(X[:,i])
        #    axis('off')

            

figure()
plot(np.array(scores)[:,0], 'o-', label='input')
plot(np.array(scores)[:,1], 'o-', label='output')
legend(loc='best')

figure()
zh = liquid.RC_predict (X,Y)
clf()
plot(timev,teach_sig.T,label='teach signal')
plot(timev,zh["input"], label='input')
plot(timev,zh["output"], label='output')
legend(loc='best')



