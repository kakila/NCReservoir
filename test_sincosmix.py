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
# Test singal generators
# ===============================
from __future__ import division
import numpy as np
import sig_gen as s
import matplotlib.pyplot as plt

t  = np.linspace(0,10,1e3)[:,None] # time vector
f  = np.arange(10,25,0.1)[None,:] # frequency band 5-25 Hz

plt.ion()
for shift in np.linspace(-5,5,3):
    w = s.sincosmix(t,f+shift)
    u = np.fft.rfft(w,axis=0)
    ff = np.linspace(0,t.shape[0]/2/t[-1],u.shape[0])
    plt.semilogy (ff,u*np.conjugate(u))


