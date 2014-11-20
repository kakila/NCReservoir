## Copyright (C) 2014 - Federico Corradi
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program. If not, see <http://www.gnu.org/licenses/>.

## Author: (2014) Federico Corradi <federico@ini.phys.ethz.ch>

import numpy as np
import matplotlib
from matplotlib.pyplot import *
import scipy as sp
from scipy import linalg
from sklearn.linear_model import RidgeCV
from sklearn import metrics
import time

class reservoir:
    def __init__(self, resSize = 256, inSize = 1, outSize = 1, scale_w=1.8):
    
        np.random.seed(42)     # seed the random generator with the answer to all universe... etc.
        self.Win = (np.random.rand(resSize,inSize)-0.5) * 0.4     # input weights
        self.W = np.random.rand(resSize,resSize)-0.5              # recurrent weights
        # 'Computing spectral radius...'
        self.rhoW = max(abs(linalg.eig(self.W)[0]))
        self.W *= scale_w / self.rhoW
        self.resSize = resSize
        self.inSize = inSize
        self.leaky_rate = 0.5
        self.samples = 0 #real time learning
        self.x = np.zeros((self.resSize,1))
        alpha = np.logspace (-10,100,50) # Regularization parameters: 50 values
        
        Nn2_out = [self.resSize,self.resSize]
        Nn2_in = [self.inSize,self.inSize]
        
        self.CovMatrix  = {"input":np.zeros(Nn2_in),"output":np.zeros(Nn2_out)} # Covariance matrix of inputs and outputs
        self.ReadoutW   = {"input":np.zeros([self.resSize,1]),"output":np.zeros([self.resSize,1])}     # Readout weights
        self.ProjTeach  = {"input":np.zeros([self.inSize,1]),"output":np.zeros([self.resSize,1])}     #
        

        self._regressor = {"input":RidgeCV(alphas=alpha,normalize=False, fit_intercept=False), \
                           "output":RidgeCV(alphas=alpha,normalize=False, fit_intercept=False)}
        
    def stimulate(self, stimulus):
        # allocated memory for the design (collected states) matrix
        inSize, nT = np.shape(stimulus)
        self.X = np.zeros([self.resSize,nT])
        # set the corresponding target matrix directly
        self.Yt = stimulus.T
        
        #very important
        self.x = np.zeros((self.resSize,1))
        
        # run the reservoir with the data (training input u) and collect X (activation states)   
        for t in range(nT):
            sum_inputs = np.dot( self.Win, stimulus.T[t] )
            self.x = np.reshape((1-self.leaky_rate)*self.x,[self.resSize]) + self.leaky_rate*np.tanh( sum_inputs + np.reshape(np.dot( self.W, self.x ), [self.resSize]) )
            self.X[:,t] = self.x
    

    def train(self, teach_sig):
        '''
        Regression of teach_sig using inputs (self.Yt) and outputs (self.X)
        '''
        #inits
        nT,Nn        = np.shape(self.X.T)
        nTtot        = self.samples + nT
        w            = (self.samples/nTtot, 1.0/nTtot)
        
        CInputs = np.zeros([nT,Nn])
        
        # Covariance matrix
        Cx = np.dot (self.X, self.X.T) # output
        C  = np.dot (self.Yt.T, self.Yt) # input
        # Projection of data
        Zx = np.dot (self.X, teach_sig) #output
        Z  = np.dot (self.Yt.T, teach_sig) # input
        # Update cov matrix
        #raise Exception
        self.CovMatrix["output"]  = w[0]*self.CovMatrix["output"] + w[1]*Cx
        self.CovMatrix["input"]  = w[0]*self.CovMatrix["input"] + w[1]*C
        self.ProjTeach["output"]  = w[0]*self.ProjTeach["output"] + w[1]*Zx
        self.ProjTeach["input"]  = w[0]*self.ProjTeach["input"] + w[1]*Z
        # Update weights
        self._regressor["input"].fit(self.CovMatrix["input"], self.ProjTeach["input"])
        self._regressor["output"].fit(self.CovMatrix["output"], self.ProjTeach["output"])
        self.ReadoutW["input"]  = self._regressor["input"].coef_.T
        self.ReadoutW["output"] = self._regressor["output"].coef_.T
        # Update samples
        self.samples = nTtot

    def predict (self, initNt=0):
        Z = {"input":  self._regressor["input"].predict(self.Yt[initNt::,:]), \
             "output": self._regressor["output"].predict(self.X.T[initNt::,:])}
        return Z
                           
    def create_stimuli_matrix (self, G, rates, nT, nx=16, ny=16) :
        '''
        Generates a matrix the mean rates of the input neurons defined in
        nT time intervals.
        
        ** Inputs **
        G: A list with G_i(x,y), each element is a function 
                G_i: [-1,1]x[-1,1] --> [0,1] 
           defining the intensity of mean rates on the (-1,1)-square.
        nx,ny: Number of neurons in the x and y direction (default 16).

        rates: A list with the time variations of the input mean rates.
               Each element is
                f_i: [0,1] --> [0,1]
        
        nT: Number of time intervals.

        ** Outputs **
        '''
        
        nR = len(rates) # Number of rates

        x,y = np.meshgrid(np.linspace(-1,1,nx), np.linspace(-1,1,ny))
        t   = np.linspace(0,1,nT,endpoint=False)
        V   = np.array([r(t) for r in rates])

        M = np.zeros ([nx*ny, nT])
        GG = np.zeros ([nx,ny])
        for g,r in zip(G,V):
            M += np.array(g(x,y).ravel()[:,None] * r) / sum (g(x,y).ravel()[:,None])
            GG += g(x,y)

        return M
        
    def generates_gestures(self, num_gestures, n_components, max_f = 8, min_f = 1,  nScales = 4):
        '''
        generates gesture with n_components in frequency
        '''
        gestures = []
        for this_gesture in range(num_gestures):
            freqs   = (np.random.randint(min_f, high=max_f,size=n_components)+1).tolist()   # in Hz
            centers = (-1+2*np.random.random((n_components,2))).tolist()
            width   = (0.5+np.random.random(n_components)).tolist()
            gestures.append({'freq': freqs, 'centers': centers, 'width': width})   
        rates = []
        G     = []
        for ind,this_g in enumerate(gestures):
          for f  in this_g['freq']:
              rates.append(lambda t,w=f: 0.5+0.5*np.sin(2*np.pi*w*t)) 
          # Multiple spatial distribution
          for width,pos in zip(this_g['width'], this_g['centers']):
              G.append(lambda x,y,d=width,w=pos: np.exp ((-(x-w[0])**2 + (y-w[1])**2)/d**2))
    
        return G, rates, gestures

    def generates_G_rates(self, gestures): 
        rates = []
        G     = []
        for ind,this_g in enumerate(gestures):
          for f  in this_g['freq']:
              rates.append(lambda t,w=f: 0.5+0.5*np.sin(2*np.pi*w*t)) 
          # Multiple spatial distribution
          for width,pos in zip(this_g['width'], this_g['centers']):
              G.append(lambda x,y,d=width,w=pos: np.exp ((-(x-w[0])**2 + (y-w[1])**2)/d**2))
        return G, rates
        
    def generate_teacher(self, gesture, rates, n_components, nT, nScales, timev):
        #generate teaching signal associated with the Gesture
        teach_sig = np.zeros([nT, nScales])
        for this_component in range(n_components):
            #sum all frequencies with distance dependent from centers
            this_centers = np.array(gesture['centers'])
            #rates are weighted with their respective euclidan distance from center
            rate_w = np.sqrt(this_centers[this_component,0]**2 + this_centers[this_component,1]**2)
            teach_sig += rates[this_component]((teach_scale*timev[:,None])*1e-3)*rate_w          
        return teach_sig
        
    def root_mean_square(self, ideal, measured):
        ''' calculate RMSE 
        ie: root_mean_square(ideal,measured)
        numpy vector in 
        float out'''
        import numpy as np
        return np.sqrt(((ideal - measured) ** 2).mean())

                
#test experiment
if __name__ == '__main__': 

    ion() 
    
    ######################################
    # Gestures and network parameters
    ######################################
    #  ~~~~~~ TRAIN
    num_gestures = 1        # number of gestures
    repeat_same = 10         # n trial
    n_components = 2       # number frequency components in a single gestures
    max_f =        10        # maximum value of frequency component in Hz
    min_f =        1        # minimum value of frequency component in Hz
    nx_d = 16               # 2d input grid of neurons
    ny_d = 16   
    nScales = 5             # number of scales (parallel teaching on different scales, freqs)
    scale_w_rhow = 1.48     # a fundamental parameters... if it below 1.5 train is horrible
    teach_scale = np.linspace(0.1,5, nScales)   # the scales in parallel
    plot_teaching = False
    #  ~~~~~~ TEST
    num_gestures_test = 1           # number of test gestures 
    perturbe_test = True            # perturb a previously learned gestures?
    n_perturbations = 3
    center_pert_value = np.linspace(0.0001,0.008,n_perturbations)     # perturbation value, moves the centes of the freq components
    plot_all_test = True
    # other parameters 
    nT = 500                        # Number of time steps to sample the mean rates
    initNt = 0                      # Time to init the reservoir, not training in this initTime 
    timev = np.linspace(0,nT-1,nT)  # our time vector   
    n_neu_in_reservoir = 256        # this can be of any size neurons are tanh units
    neutoplot = 10                  # display reservoir neuron activity, pick one in the reservoir
    
    #####################################
    # init reservoir
    #####################################
    res = reservoir(resSize = n_neu_in_reservoir, inSize=nx_d*ny_d, scale_w=scale_w_rhow) 
   

    #######
    # ~~ TRAIN STIM
    #######
    G, rates, gestures = res.generates_gestures( num_gestures, n_components, max_f = max_f, min_f = min_f, nScales = nScales)
    M_tot = np.zeros([nx_d*ny_d, nT, num_gestures])
    for ind in range(num_gestures):
        this_g = [G[(ind*n_components)+this_c] for this_c in range(n_components)]
        this_r = [rates[(ind*n_components)+this_c] for this_c in range(n_components)]
        M_tot[:,:,ind] = res.create_stimuli_matrix(this_g, this_r, nT, nx=nx_d ,ny=ny_d)
    # -------------
              
    ##################################################
    # Train the Network with all train gestures
    ##################################################
    fig_a = figure()
    rmse_tot = np.zeros([nScales, num_gestures])
    for this_g in range(num_gestures):
        for this_repeat in range(repeat_same):
            # This should be a for loop.... over all gestures 
            #poke network
            res.stimulate(M_tot[:,:,this_g])   
            #generate associated teacher signal
            teach_sig = res.generate_teacher(gestures[this_g], rates, n_components, nT, nScales, timev)        
            res.train(teach_sig)          
            zh = res.predict(initNt=initNt)
                      
            if plot_teaching:
                figure()
                title('training error')
            print '####### gesture n', this_g, ' trial ', this_repeat
            for i in range(nScales):
                this_rmse = res.root_mean_square(teach_sig[initNt::,i], zh["output"][:,i])
                print '### SCALE n', i, ' RMSE ', this_rmse
                rmse_tot[i, this_g] = this_rmse
                if plot_teaching:
                    subplot(nScales,1,i+1)
                    plot(timev[initNt::],teach_sig[initNt::,i],label='teach signal')
                    plot(timev[initNt::],zh["input"][:,i], label='input')
                    plot(timev[initNt::],zh["output"][:,i], label='output')
            if plot_teaching:        
                legend(loc='best')
                figure(fig_a.number)
                plot(res.X[neutoplot,:],label='train')
    figure()
    for this_s in range(nScales):
        semilogy(rmse_tot[this_s,:], 'o-', label='nScale n'+str(this_s))
    xlabel('Gesture Num')
    ylabel('RMSE')
    legend(loc='best')
            
    #######        
    # ~~ TEST STIM
    #######
    if perturbe_test:
        print 'we perturb teached gestures'
        #perturbe teached gestures
        gestures_pert = (np.repeat(gestures, len(center_pert_value))).copy()
        gestures_final = []
        for this_test in range(len(center_pert_value)): #loop over gestures
            freqs   = gestures_pert[this_test]["freq"]  # in Hz
            width   = gestures_pert[this_test]["width"] 
            centers = []
            for this_component in range(n_components):                
                centers.append([gestures_pert[this_test]["centers"][this_component][0]+ center_pert_value[this_test]*np.random.random_sample()- center_pert_value[this_test], gestures_pert[this_test]["centers"][this_component][1]+ center_pert_value[this_test]*np.random.random_sample()- center_pert_value[this_test]]) 
            gestures_final.append({'freq': freqs, 'centers': centers, 'width': width}) 
        G_test, rates_test =  res.generates_G_rates(gestures_final)
        gestures_pert = gestures_final
    else:
        #generate new sets of gestures
        G_test, rates_test, gestures_test = res.generates_gestures( num_gestures_test, n_components, max_f = max_f, min_f = min_f, nScales = nScales)
        gestures_pert = gestures_test

    M_tot_test = np.zeros([nx_d*ny_d, nT, len(center_pert_value)])
    for ind in range(len(center_pert_value)):
        this_g = [G_test[(ind*n_components)+this_c] for this_c in range(n_components)]
        this_r = [rates_test[(ind*n_components)+this_c] for this_c in range(n_components)]
        M_tot_test[:,:,ind] = res.create_stimuli_matrix(this_g, this_r, nT, nx=nx_d ,ny=ny_d)
    # -------------
                    
    ##################################################
    # TEST the Network with all test gestures
    ##################################################
    rmse_over_perturnations = []
    for this_g in range(len(gestures_final)):
        # This should be a for loop.... over all gestures 
        #poke network
        res.stimulate(M_tot_test[:,:,this_g]) 
        
        #generate teaching signal associated with the Gesture
        teach_sig = res.generate_teacher(gestures_final[this_g], rates_test, n_components, nT, nScales, timev) 
        zh = res.predict(initNt=initNt)
        
        if(plot_all_test):
            figure()
            title('test error')
        rmse_this_g = []
        for i in range(nScales):
            if(plot_all_test):
                subplot(nScales,1,i+1)
                plot(timev[initNt::],teach_sig[initNt::,i],label='test target signal')
                plot(timev[initNt::],zh["input"][:,i], label='input')
                plot(timev[initNt::],zh["output"][:,i], label='output')
            print "TESTING ERROR RMSE:", res.root_mean_square(teach_sig[initNt::,i], zh["output"][:,i])
            rmse_this_g.append(res.root_mean_square(teach_sig[initNt::,i], zh["output"][:,i])) 
        if(plot_all_test):    
            legend(loc='best')       
            figure(fig_a.number)
            plot(res.X[neutoplot,:], 'o-', label='test')


     
        rmse_over_perturnations.append([np.mean(rmse_this_g), np.std(rmse_this_g)])
    if(plot_all_test):  
        legend(loc='best')
    
    figure()
    rmse_over_perturnations = np.array(rmse_over_perturnations)
    errorbar(center_pert_value*100, rmse_over_perturnations[:,0], yerr=rmse_over_perturnations[:,1] , fmt='--o')
    xlabel('pertubation centers norm. distance')
    ylabel('RMSE')
    
    
    
    
