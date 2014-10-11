## Copyright (C) 2014 - Juan Pablo Carbajal
##
## This progrm is free software; you can redistribute it and/or modify
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

## Author: Juan Pablo Carbajal <ajuanpi+dev@gmail.com>

# Scrip to convert data to GNU Octave
# and make easier processing.
pkg load parallel general

folder = "../data/lsm_ret";
## Number of neurons in each direction
ny = nx = 16;
N  = nx*ny;

# Find maximum and minimum values of the whole data set.
Ng = 10; #Number of gestures [0:N-1] 
Nt = 3; #Number of trials  [0:N-1]
if exist("redo")
  OUTdata = INdata  = struct ("Tmax",0, ...          # Maximum value of time vector
                              "Nid" ,[1e4 -1], ...   # Min max neuron id
                              "ind",[], ... # Linear index of active neurons
                              "nid",[]);    # List of active neurons
  INdata.ind = cell (Ng,Nt);
  INdata.nid = cell (Ng,Nt);
  OUTdata.ind = cell (Ng,Nt);
  OUTdata.nid = cell (Ng,Nt);

  for g = 1:Ng;
    for tt = 1:Nt;
      [X Y] = loaddata (g-1,tt-1, folder);
      INdata.Tmax = max (max (X.t), INdata.Tmax);
      INdata.Nid(1) = min (min (X.n_id), INdata.Nid(1)) + 1;
      INdata.Nid(2) = max (max (X.n_id), INdata.Nid(2)) + 1;

      INdata.nid{g,tt} = unique (X.n_id) + 1;
      Nn    = length (INdata.nid{g,tt});
      I     = repmat (INdata.nid{g,tt}, Nn, 1);
      J     = INdata.nid{g,tt}(kron ((1:Nn).', ones(Nn,1)));
      INdata.ind{g,tt} = sub2ind ([N,N], I, J);

      OUTdata.Tmax = max (max (Y.t), OUTdata.Tmax);
      OUTdata.Nid(1) = min (min (Y.n_id), OUTdata.Nid(1));
      OUTdata.Nid(2) = max (max (Y.n_id), OUTdata.Nid(2));

      OUTdata.nid{g,tt} = unique (Y.n_id) + 1;
      Nn    = length (OUTdata.nid{g,tt});
      I     = repmat (OUTdata.nid{g,tt}, Nn, 1);
      J     = OUTdata.nid{g,tt}(kron ((1:Nn).', ones(Nn,1)));
      OUTdata.ind{g,tt} = sub2ind ([N,N], I, J);
#      if g==1 && tt==1
#        keyboard
#      endif
    endfor # over trials
  endfor # over gestures

  fname  = fullfile (folder, "metadata.dat");
  save (fname,"INdata","OUTdata");
endif

clear -x N Ng Nt nx ny folder
fname  = fullfile (folder, "metadata.dat");
load (fname);
## Prepare analog time vector
T  = OUTdata.Tmax / 1e3; # in seconds
Fs = 25; # minimum sampling Freq in Hz
nT = ceil (T*Fs);
t  = linspace (0,T,nT).';
dt = t(2) - t(1);

# Conversion form Sike to Analog
t_width = 0.5; # time width in seconds
tw2     = t_width^2;
v       = @(t,ts) exp (-(t-ts).^2/2/tw2);   % .* cos(2*pi*(t-ts)/3/s);
v_avg   = @(t,ts) exp (-(t-ts).^2/2/tw2/10);% .* cos(2*pi*(t-ts)/3/s);

fname = @(p,t) fullfile (folder, ...
                         sprintf ("gesture_%d_trial_%d.dat",p,t));
x_active = zeros (nT,Ng);
for g = 1:Ng
  for tt = 1:Nt
    [X Y] = loaddata (g-1,tt-1, folder);
    Ya = ts2sig_v2 (t,Y.t/1e3,Y.n_id,v,[nx,ny])/N;
    Xa = ts2sig_v2 (t,X.t/1e3,X.n_id,v,[nx,ny])/N;
    save (fname(g-1,tt-1), "t", "Ya", "Xa");

    # input Activity signal
    x_active(:,g) += mean (ts2sig_v2 (t,X.t/1e3,X.n_id,v_avg,[nx,ny]),2);
  endfor # over trials
endfor # over gestures
x_active = x_active ./ max (x_active);
save (fullfile (folder, "input_activity.dat"),"x_active");

