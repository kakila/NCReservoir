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

## Number of neurons in each direction
ny = nx = 16;
N  = nx*ny;

# Find maximum and minimum values of the whole data set.
Ng = 25; #Number of gestures [0:N-1] 
Nt = 4; #Number of trials  [0:N-1]
Ntrain = 2;
Ntest  = 1;

C_in = zeros (N);
C    = zeros (N);

Z_in = zeros (N,1);
Z    = zeros (N,1);

folder = "../../chip_MN256R01/data/tracking_10gestures_3trials";

fname = @(p,t) fullfile (folder, ...
                         sprintf ("gesture_%d_trial_%d.dat",p,t));
### Teaching signal
if exist("recode")
# Amplitude encoding
z = @(i,t) i*ones(size(t));
  
# Frequency encoding
#if !exist("recode")
#  freq = linspace (1/t(end),6,50);
#  Nf   = length (freq);
#  A    = randn (Ng,2*Nf); 
#  A    = A ./ sqrt (sumsq(A,2));
#  for g=1:Ng
#    [~,idx] = sort(abs(A(g,1:Nf)),"descend");
#    A(g,1:Nf) = A(g,1:Nf)(idx);;

#    [~,idx] = sort(abs(A(g,Nf+(1:Nf))),"descend");
#    A(g,Nf+(1:Nf)) = A(g,Nf+(1:Nf))(idx);
#  end
#scale = 0.5;
#bias= max (-1,0)
#z = @(i,t) [sin(2*pi*(scale*freq+bias).*t) cos(2*pi*(scale*freq+bias).*t)]*A(i,:).';

#source("./data/synthetic_gestures/gestures.m")
#for g=1:Ng
#  
#endfor
endif # recode teach signal

# Covariance matrix of learning set
C = C_in = zeros (N);
Z = Z_in = zeros (N,1);
load(fullfile (folder,"metadata.dat"));
load(fullfile (folder, "input_activity.dat"));
nT = length(x_active);
for g=1:Ng;
  for tt=1:Ntrain; 
    load(fname(g-1,tt-1)); 
    C_in(INdata.ind{g,tt}) += (Xa.'*Xa)(:)/nT; 
    C(OUTdata.ind{g,tt})   += (Ya.'*Ya)(:)/nT;

    zz = z(g,t);%.*x_active(:,g);
    Z_in(INdata.nid{g,tt}) += Xa.'*zz;
    Z(OUTdata.nid{g,tt})   += Ya.'*zz;
    
  endfor #over trials 
endfor #over gestures

# Train
W = W_in  = zeros (N,1); 
lambda = logspace (-6,2,50);

# Cov matrix has zero rows and cols due to unactive neurons
# We remove those zero rows and cols.
nz_in       = find(sum(abs(C_in),2)>1e-4); # non zeros rows of Cov matrix
K           = submat (C_in,nz_in,nz_in,"mode","keep","eco"); 
[W_tmp,lambda0_in]       = xridgereg (K, Z_in(nz_in),lambda); 
W_in(nz_in) = W_tmp;

nz    = find(sum(abs(C),2)>1e-4); # non zeros rows of Cov matrix
K     = submat (C,nz,nz,"mode","keep","eco");
[W_tmp,lambda0] = xridgereg (K, Z(nz),lambda); 
W(nz) = W_tmp;

INerror = OUTerror = struct ("train",zeros(Ng-1,Nt-1),"test", zeros (1,1));
# Train Error
for g = 1:Ng;
  figure (g)
  for tt=1:Ntrain;
    load (fname(g-1,tt-1)); 
    zh_in = Xa * W_in(INdata.nid{g,tt})/nT;
    zh    = Ya * W(OUTdata.nid{g,tt})/nT;
    zz    = z(g,t);%.*x_active(:,g);
    
    INerror.train(g,tt)  = mean ((zz-zh_in).^2) / mean(zz.^2);
    OUTerror.train(g,tt) = mean ((zz-zh).^2) / mean(zz.^2);  
    subplot(Nt,1,tt)
    plot(t,zh_in,'.r',t,zh,'.g',t,zz,'-k');
    axis tight
    %axis ([0 max(t) min(zz) max(zz)]);
  endfor #over trials 
endfor #over gestures

# Test Error
for g=1:Ng;
  figure (g)
  for tt=Ntrain+1:Ntrain+Ntest;
    load (fname(g-1,tt-1));
    zh_in = Xa * W_in(INdata.nid{g,tt})/nT;
    zh    = Ya * W(OUTdata.nid{g,tt})/nT;
    zz = z(g,t);%.*x_active(:,g);

    INerror.test(g,tt)  = mean ((zz-zh_in).^2) / mean(zz.^2);
    OUTerror.test(g,tt) = mean ((zz-zh).^2) / mean(zz.^2);  
    subplot(Nt,1,tt)
    plot(t,zh_in,'.r',t,zh,'.g',t,zz,'-k');
    axis ([0 max(t) min(zz) max(zz)]);

    [tmp F] = normFFT (t,[zh_in zh zz]);
    FFTtest_in(:,g) = tmp(:,1);
    FFTtest(:,g) = tmp(:,2);
    FFT(:,g) = tmp (:,3); 
  endfor #over trials 
endfor #over gestures

figure(Ng+1)
for g=1:Ng
  subplot(2,2,g);
  tmp = [FFTtest_in(:,g) FFTtest(:,g) FFT(:,g)];
  h = semilogy(F,abs(tmp)); legend(h,{"in","out","des"});
  axis tight
  %axis([0 2*max(freq) 1e-6 1]);
endfor
