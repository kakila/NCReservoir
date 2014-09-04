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


## Load data
#g=0;
[X Y] = loaddata (g,tt);
##Y = loaddata (g,tt);

## Time information
T  = max (Y.t);
# Conversion form Sike to Analog
v = @(t,ts,s) exp (-(t-ts).^2/2/s^2);% .* cos(2*pi*(t-ts)/3/s);
t_width = 50;
#mts = max(arrayfun (@(i)sum(Y.n_id==i), unique(Y.n_id)));
#nT = mts;
#t  = linspace (0,T,nT).'; # time vector
t = (0:t_width/10:T).';
nT = length (t);

% Plot temporal reaction
#figure (1)
#h = plot (t,v(t,T/2,t_width));
#set (h,"linewidth",2);
#xlabel ("time")
#title ("Temporal component")

Ya = ts2sig_v2 (t,Y.t,Y.n_id,@(t,ts)v(t,ts,t_width),[nx,ny]);
Xa = ts2sig_v2 (t,X.t,X.n_id,@(t,ts)v(t,ts,t_width),[nx,ny]);

% sort by activity
#ac = sumsq(Ya) / nT;
#[~,idx] = sort (ac,"descend");
#Ya_s      = Ya(:,idx);

# Covariance
M_in = (Xa.'*Xa)/nT;
M  = (Ya.'*Ya)/nT;
[ux,s_in,~] = svd (M_in);
[u,s,~]   = svd (M);


# A' * teach signal
#P_in = Xa.'*z;
#P    = Ya.'*y;

#subplot(2,1,1)
#hold on
#plot(t,Ya(:,1))
#subplot(2,1,2)
#hold on
#plot(t,Xa(:,17))
