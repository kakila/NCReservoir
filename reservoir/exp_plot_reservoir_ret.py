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
from __future__ import division

import numpy as np
from pylab import *
import pyNCS
import sys
import matplotlib
sys.path.append('../api/reservoir/')
sys.path.append('../api/retina/')
sys.path.append('../gui/reservoir_display/')
import reservoir as L
import time
import retina as ret
import glob        #command line parser
from mpl_toolkits.mplot3d import Axes3D

ion()

directory = 'lsm_ret/'
fig_dir = 'figures/'            #make sure they exists
extension = "txt"

plot_omega_grid = True


def _ismember(a, b):
    '''
    as matlab: ismember
    '''
    tf = np.array([i in b for i in a])
    u = np.unique(a[tf])
    index = np.array([(np.where(b == i))[0][-1] if t else 0 for i,t in zip(a,tf)])
    return tf, index

print "loading files... "
### folder parser
filenames_dat = []          #without extensions and dirs
inputs_dat = []
outputs_dat = []
omegas_dat = []
gestures = []
trials = [] 
omegas = []
for f in sorted(glob.iglob(directory+"*."+extension)):
    
    f_raw = f
    f = f.replace(".txt","") #remove extension
    f = f.split("/")         #get directories
    if( f[1] != 'phi' and f[1] != 'theta' and f[1] != 'timev'): 
        ndirs  = np.shape(f)
        ndirs = int(''.join(map(str,ndirs)))
        filenames_dat.append(f[ndirs-1])
        if( f[ndirs-1].split("_")[0] == 'inputs'):
            inputs_dat.append(f_raw)
        if( f[ndirs-1].split("_")[0] == 'outputs'):    
            outputs_dat.append(f_raw)
        if( f[ndirs-1].split("_")[0] == 'omegas'):    
            omegas_dat.append(f_raw)    
            this_omegas = np.loadtxt(f_raw)
            omegas.append(this_omegas)

        this_gesture = int(f[ndirs-1].split("_")[2])
        gestures.append(this_gesture)
        this_trial = int(f[ndirs-1].split("_")[4])
        trials.append(this_trial)

omegas = np.array(omegas)    
omegas_shape = np.shape(omegas)
trials = np.array(trials)
gestures = np.array(gestures)    
ngestures = len(np.unique(gestures))    
ntrials =  len(np.unique(trials))   
print "found ", ngestures, " gestures"
print "found ", ntrials,   " trials per gesture"

## choose where to train and test    
timev = np.loadtxt(directory+"timev.txt")  
theta = np.loadtxt(directory+"theta.txt")   
phi = np.loadtxt(directory+"phi.txt")
[T,P] = np.meshgrid(theta,phi)
wX = (np.sin(P)*np.cos(T)).ravel() 
wY = (np.sin(P)*np.sin(T)).ravel() 
wZ = np.cos(P).ravel()
fig = figure()
ax = Axes3D(fig)
for i in range(len(wX)):
    if(i%2):
        ax.scatter(wX[i],wY[i],wZ[i],c='r',marker='^')
    else:
        ax.scatter(wX[i],wY[i],wZ[i],c='b',marker='o')
title('teaching on triangles and training on dots')    

#this will be the teachings
wx_teaching = wX[::2]
wy_teaching = wY[::2]
wz_teaching = wZ[::2]

wx = []
wy = []
wz = []
index_teaching = []
index_testing = []
for i in range(omegas_shape[0]):
    tf,idx = _ismember(wx_teaching, [omegas[i,0]])
    if(np.sum(tf)>0):
        index_teaching.append(i)
    else:
        index_testing.append(i)
    wx.append(omegas[i,0])
    wy.append(omegas[i,1])
    wz.append(omegas[i,2])
wx = np.array(wx)
wy = np.array(wy)
wz = np.array(wz)
index_teaching = np.array(index_teaching)
index_testing = np.array(index_testing)

#order wx as ... just to check
#fig = figure()
#ax = Axes3D(fig)
#for i in range(len(index_teaching)):
#    ax.scatter(wx[index_teaching[i]],wy[index_teaching[i]],wz[index_teaching[i]],c='r',marker='^')            
#for i in range(len(index_testing)):    
#    ax.scatter(wx[index_testing[i]],wy[index_testing[i]],wz[index_testing[i]],c='g',marker='o') 

####################
#init reservoir
res = L.Reservoir() #object without population does not need the chip offline build

#membrane timev
membrane = lambda t,ts: np.atleast_2d(np.exp((-(t-ts)**2)/(2*35**2)))
func_avg = lambda t,ts: np.exp((-(t-ts)**2)/(2*35**2)) # Function to calculate region of activity

#### TEACHING
n_teach = len(index_teaching)
teach_base = np.linspace(0,np.max(timev)/500.0,len(timev))
figure()
n_subplots = np.ceil(np.sqrt(n_teach))
rmse_teachings = []
for this_teach in range(len(index_teaching)):
    #inputs  = np.loadtxt(inputs_dat[index_teaching[this_teach]])
    outputs = np.loadtxt(outputs_dat[index_teaching[this_teach]])
    omegas = np.loadtxt(omegas_dat[index_teaching[this_teach]])

    print "train offline reservoir on teaching signal.. ", this_teach, " of ", n_teach
    X = L.ts2sig(timev, membrane, outputs[:,0], outputs[:,1], n_neu = 256)
    #Yt = L.ts2sig(timev, membrane, inputs[:,0], inputs[:,1], n_neu = 256)  
            
    #build teaching signal
    teach_sig = omegas[0]*np.sin(teach_base)+omegas[1]*np.sin(teach_base)+omegas[2]*np.sin(teach_base)
    #inputs are "fake" just to speed up
    Yt = omegas[0]*np.cos(teach_base)+omegas[1]*np.cos(teach_base)+omegas[2]*np.cos(teach_base) 
    
    Yt = np.repeat(Yt,256).reshape([len(timev),256])
    #plot(teach_sign)
    
    if(this_trial == 0):
        # Calculate activity of current inputs.
        # As of now the reservoir can only give answers during activity
        tmp_ac = np.mean(func_avg(timev[:,None], outputs[:,0][None,:]), axis=1) 
        tmp_ac = tmp_ac / np.max(tmp_ac)
        ac = tmp_ac[:,None]
        teach_sig = teach_sig[:,None] * ac**4 # Windowed by activity
        print "teach_sign", np.shape(teach_sig)
        
    res.train(X,Yt,teach_sig[:,None])   
    zh = res.predict(X, Yt)
   
    
    this_rmse = res.root_mean_square(teach_sig, zh["output"][:,0])
    this_rmse_input = res.root_mean_square(teach_sig, zh["input"][:,0])

    print "### RMSE outputs", this_rmse
    print "### RMSE inputs", this_rmse_input

    rmse_teachings.append(this_rmse)

    subplot(n_subplots,n_subplots,this_teach)
    plot(timev,zh["output"])
    plot(timev,teach_sig)
    axis('off')

#### TESTING
n_tests = len(index_testing)
rmse_testings = []
n_subplots = np.ceil(np.sqrt(n_tests))
for this_test in range(len(index_testing)):
    outputs = np.loadtxt(outputs_dat[index_testing[this_test]])
    omegas = np.loadtxt(omegas_dat[index_testing[this_test]])

    print "test offline reservoir on testing signal.. ", this_test, " of ", n_tests
    
    X = L.ts2sig(timev, membrane, outputs[:,0], outputs[:,1], n_neu = 256)
    target_sig = omegas[0]*np.sin(teach_base)+omegas[1]*np.sin(teach_base)+omegas[2]*np.sin(teach_base)
    Yt = omegas[0]*np.cos(teach_base)+omegas[1]*np.cos(teach_base)+omegas[2]*np.cos(teach_base)
    Yt = np.repeat(Yt,256).reshape([len(timev),256])
    
    zh = res.predict(X, Yt)     
    this_rmse = res.root_mean_square(target_sig, zh["output"][:,0])
    print "### RMSE outputs", this_rmse
    
    rmse_testings.append(this_rmse)
 
    subplot(n_subplots,n_subplots,this_test)
    plot(timev,zh["output"])
    plot(timev,target_sig)
    axis('off')
    
#order wx as ... just to check cmap=plt.cm.get_cmap('jet'),
fig = figure()
ax = Axes3D(fig)
color_teachings = (rmse_teachings/np.max(rmse_teachings))*30
color_testings = (rmse_testings/np.max(rmse_testings))*30
scatter1 = ax.scatter(wx[index_teaching],wy[index_teaching],wz[index_teaching],c=color_teachings*10,marker='^',s=50, label='teaching')            
scatter2 = ax.scatter(wx[index_testing],wy[index_testing],wz[index_testing],c=color_testings,marker='o',s=50,label='testing') 
    
scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c='b', marker = '^')
scatter2_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c='g', marker = 'o')

ax.legend([scatter1_proxy, scatter2_proxy], ['teaching', 'testing'], numpoints = 1)

    
    
