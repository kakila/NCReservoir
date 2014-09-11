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
# Singal generators
# ===============================
from __future__ import division
import numpy as np

def sinmix_int (t, n_max=50, n_min=1, A=None):
    '''
     Supperposition of sines with integer frequencies.
     t -> time vector (should be vertical and 2D).
     n_max -> maximum integer for frequencies
     n_min -> minimum integer for frequencies
     A -> specify the weights of each frequency (should be 2D and vertical)
    '''
    
    f = np.arange(n_min, n_max)[None,:];
    N = f.shape[1]
    
    omega = 2*np.pi*f;
    omeg2 = omega**2;
    if A is None:
        A = np.random.randn (N,1);
    
    T = t[-1];

    s = np.sin (omega*t) / omega;
    s = np.atleast_2d(s)

    # Remove the mean value
    avgW    = ( 1 - np.cos (omega*T) ) / omeg2;
    W       = np.dot(s-avgW, A);
    
    # Scale to [0 1]
    w_min = np.min (W)
    w_max = np.max (W)
    W     = (W - w_min) / (w_max - w_min)
    
    return W,A

def sinmix (t, f, A=None):
    '''
     Supperposition of sines with given frequencies.
     t -> time vector (should be 2D and vertical).
     f -> frequency vector (should be 2D and horizontal)
     A -> specify the weights of each frequency (should be 2D and vertical)
    '''
    
    N = f.shape[1]
    
    omega = 2*np.pi*f;
    omeg2 = omega**2;
    if A is None:
        A = np.random.randn (N,1);
    
    T = t[-1];

    s = np.sin (omega*t) / omega;
    s = np.atleast_2d(s)

    # Remove the mean value
    avgW    = ( 1 - np.cos (omega*T) ) / omeg2;
    W       = np.dot(s-avgW, A);
    
    # Scale to [0 1]
    w_min = np.min (W)
    w_max = np.max (W)
    W     = (W - w_min) / (w_max - w_min)
    
    return W,A

def sincosmix(t, f, A=None):
    '''
     Supperposition of sines and cosines with given frequencies.
     t -> time vector (should be 2D and vertical).
     f -> frequency vector (should be 2D and horizontal)
     A -> specify the weights of each frequency (should be 2D,vertical and double length as f)
    '''
    
    N = f.shape[1]
    
    omega = 2*np.pi*f;
    omeg2 = omega**2;
    if A is None:
        A = np.random.randn (2*N,1);
    
    T = t[-1];

    s = np.sin (omega*t) / omega;
    s = np.atleast_2d(s)
    c = np.cos (omega*t) / omega;
    c = np.atleast_2d(c)

    # Remove the mean value
    avgW    = ( 1 - np.cos (omega*T) ) / omeg2;
    W       = s-avgW;
    avgW    = np.sin (omega*T) / omeg2;
    W       = np.hstack ((W,c-avgW))
    W       = np.dot(W, A)
    
    # Scale to [0 1]
    w_min = np.min (W)
    w_max = np.max (W)
    W     = (W - w_min) / (w_max - w_min)
    
    return W,A

