############### author ##########
# federico corradi
# federico@ini.phys.ethz.ch
# Liquid State Machine class mn256r1 
# GPL licence
# ===============================

### ========================= import packages ===============================
import random
import numpy as np
import time
import pyNCS
import matplotlib
from pylab import *

class Lsm:
    def __init__(self, population,  cee=0.5, cii=0.3):
        ### ========================= define what is needed to program the chip ====
        # resources
        self.matrix_learning_rec = np.zeros([256,256])
        self.matrix_learning_pot = np.zeros([256,256])
        self.matrix_programmable_rec = np.zeros([256,256])
        self.matrix_programmable_w = np.zeros([256,256])
        self.matrix_programmable_exc_inh = np.zeros([256,256])
        # end resources
        # network parameters
        self.cee = cee
        self.cii = cii
        self.rcn = population
        self.setup = population.setup
        # init reservoir 
        #self.setup.chips['mn256r1'].load_parameters('biases/biases_default.biases')
        self._init_lsm()
        self.program_config()
        #self.setup.chips['mn256r1'].load_parameters('biases/biases_liquid.biases')

    ### ========================= functions ===================================
    def _init_lsm(self):
        # rcn with learning synapses
        self._connect_populations_programmable(self.rcn,self.rcn,self.cee,[2])
        self._connect_populations_programmable_inh(self.rcn,self.rcn,self.cii,[2])
        return 

    def _connect_populations_programmable_inh(self, pop_pre,pop_post,connectivity,w):
        '''
            Connect two populations via programmable synapses with specified connectivity and w
        '''    
        if(np.shape(w)[0] == 2):
            w_min = w[0]        
            w_max = w[1]
            random_w = True
        elif(np.shape(w)[0] == 1 ):
            random_w = False
        else:
            print 'w should be np.shape(2), [w_min,w_max]'

        #loop trought the pop and connect with probability connectivity
        for pre in pop_pre.soma.addr['neu']:
            for post in pop_post.soma.addr['neu']:
                coin = np.random.rand()
                if(coin < connectivity):
                    #we connect this pre with this post
                    self.matrix_programmable_exc_inh[post,pre] = 0  
                    self.matrix_programmable_rec[post,pre] = 1    
                    if(random_w):
                        self.matrix_programmable_w[post,pre] = np.random.randint(w_max)+w_min
                    else:
                        self.matrix_programmable_w[post,pre] = w[0]

    def _connect_populations_learning(self, pop_pre,pop_post,connectivity,pot):
        '''
            Connect two populations via learning synapses with specified connectivity and pot
        '''    
        #loop trought the pop and connect with probability connectivity
        for pre in pop_pre.soma.addr['neu']:
            for post in pop_post.soma.addr['neu']:
                coin = np.random.rand()
                if(coin < connectivity):
                    #we connect this pre with this post
                    self.matrix_learning_rec[post,pre] = 1 
                    coin = np.random.rand()
                if(coin < pot):  
                    self.matrix_learning_pot[post,pre] = 1

    def _connect_populations_programmable(self, pop_pre, pop_post,connectivity,w):
        '''
            Connect two populations via programmable synapses with specified connectivity and w
        '''    
        if(np.shape(w)[0] == 2):
            w_min = w[0]        
            w_max = w[1]
            random_w = True
        elif(np.shape(w)[0] == 1 ):
            random_w = False
        else:
            print 'w should be np.shape(2), [w_min,w_max]'

        #loop trought the pop and connect with probability connectivity
        for pre in pop_pre.soma.addr['neu']:
            for post in pop_post.soma.addr['neu']:
                coin = np.random.rand()
                if(coin < connectivity):
                    #we connect this pre with this post
                    self.matrix_programmable_exc_inh[post,pre] = 1
                    self.matrix_programmable_rec[post,pre] = 1   
                    if(random_w):
                        self.matrix_programmable_w[post,pre] = np.random.randint(w_max)+w_min
                    else:
                        self.matrix_programmable_w[post,pre] = w[0]

    def stimulate_reservoir(self, nsteps = 3, max_freq = 1500, min_freq = 500, duration = 1000, c=0.5, trials=5):
        '''
        stimulate reservoir via virtual input synapses
        '''
        vsyn = 4
        somach = self.rcn.soma.channel
        inputch = 1
        nsyn_tot = len(self.rcn.synapses['virtual_exc'].addr)
        nsyn = int(round(nsyn_tot * c))
        index_syn = np.random.randint(nsyn_tot,size=(nsyn))
        index_syn = np.unique(index_syn)
        nsyn = len(index_syn)
        tot_outputs = []
        tot_inputs = []
        #stim_matrix = r_[[500*np.random.random(len(self.rcn.soma.addr)*vsyn)]*nsteps]
        #stim_matrix = r_[[np.linspace(min_freq,max_freq,len(self.rcn.soma.addr)*vsyn)]*nsteps]
        stim_matrix = np.r_[[np.linspace(min_freq,max_freq,nsyn)]*nsteps]
        timebins = np.linspace(0, duration, nsteps)
        syn = self.rcn.synapses['virtual_exc'][index_syn]
        
        for this_stim in range(trials):
            spiketrain = syn.spiketrains_inh_poisson(stim_matrix.T,timebins)
            out = self.setup.stimulate(spiketrain, send_reset_event=False, duration=duration)
            out = out[somach]
            out.t_start = np.max(out.raw_data()[:,0])-duration
            raw_out = out.raw_data()
            raw_out[:,0] = raw_out[:,0]-np.min(raw_out[:,0]) 
            tot_outputs.append(raw_out)
            tot_inputs.append(spiketrain[inputch].raw_data())

        return tot_inputs, tot_outputs

    def plot_inout(self, inputs,outputs):
        '''
        just plot input and output trials of reservoir after stimulate reservoir
        '''
        trials = len(inputs)
        figure()
        for i in range(trials):
            plot(inputs[i][:,0],inputs[i][:,1],'o')
            xlabel('time [s]')
            ylabel('neuron id')
            title('input spike trains')
        for i in range(trials):
            figure()
            plot(outputs[i][:,0],outputs[i][:,1],'o')
            xlabel('time [s]')
            ylabel('neuron id')
            title('output spike trains')

    def load_config(self, directory='lsm/'):
        '''
            load configuration from folder 
        '''
        self.popsne = np.loadtxt(directory+'popse.txt')
        self.popsni = np.loadtxt(directory+'popsi.txt')
        self.matrix_learning_rec = np.loadtxt(directory+'conf_matrix_learning_rec.txt')
        self.matrix_learning_pot = np.loadtxt(directory+'conf_matrix_learning_pot.txt')
        self.matrix_programmable_rec = np.loadtxt(directory+'conf_matrix_programmable_rec.txt')
        self.matrix_programmable_w = np.loadtxt(directory+'conf_matrix_programmable_w.txt')
        self.matrix_programmable_exc_inh = np.loadtxt(directory+'conf_matrix_matrix_programmable_exc_inh.txt')

    def save_config(self, directory = 'lsm/'):
        '''
            save matrices configurations
        '''
        np.savetxt(directory+'conf_matrix_learning_rec.txt', self.matrix_learning_rec)
        np.savetxt(directory+'conf_matrix_learning_pot.txt', self.matrix_learning_pot)
        np.savetxt(directory+'conf_matrix_programmable_rec.txt', self.matrix_programmable_rec)
        np.savetxt(directory+'conf_matrix_programmable_w.txt', self.matrix_programmable_w)
        np.savetxt(directory+'conf_matrix_matrix_programmable_exc_inh.txt', self.matrix_programmable_exc_inh)

    def program_config(self):
        '''
        upload configuration matrices on the neuromorphic chip mn256r1
        '''
        self.setup.mapper._program_onchip_weight_matrix_programmable(self.matrix_programmable_w)
        self.setup.mapper._program_onchip_programmable_connections(self.matrix_programmable_rec)
        self.setup.mapper._program_onchip_exc_inh(self.matrix_programmable_exc_inh)
        self.setup.mapper._program_onchip_learning_state(self.matrix_learning_pot)
        self.setup.mapper._program_onchip_plastic_connections(self.matrix_learning_rec)
      
    def _ismember(self, a,b):
        '''
        as matlab: ismember
        '''
        # tf = np.in1d(a,b) # for newer versions of numpy
        tf = np.array([i in b for i in a])
        u = np.unique(a[tf])
        index = np.array([(np.where(b == i))[0][-1] if t else 0 for i,t in zip(a,tf)])
        return tf, index

    def mean_neu_firing(self, spike_train, n_neurons,nbins=10):
        '''
        return mean neu firing matrix
        '''
        simulation_time = [np.min(spike_train[:,0]),np.max(spike_train[:,0])]
        un, bins = np.histogram(simulation_time,nbins)
        mean_rate = np.zeros([len(n_neurons),nbins])
        for b in range(nbins-1):
            #simulation_time = [np.min(spike_train[0][:]), np.max(spike_train[0][:])]
            for i in range(len(n_neurons)):
                index_neu = np.where(np.logical_and(spike_train[:,1] == n_neurons[i], np.logical_and(spike_train[:,0] >     bins[b] , spike_train[:,0] < bins[b+1] )) )
                mean_rate[i,b] = len(index_neu[0])*1000.0/(bins[b+1]-bins[b]) # time unit: ms
        return mean_rate
