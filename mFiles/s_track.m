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
Ng = 10; #Number of gestures [0:N-1] 
Nt = 3; #Number of trials  [0:N-1]
Ntrain = 9;
Ntest  = 1;

C_in = zeros (N);
C    = zeros (N);

Z_in = zeros (N,1);
Z    = zeros (N,1);

folder = "../../chip_MN256R01/data/tracking_10gestures_3trials";

fname = @(p,t) fullfile (folder, ...
                         sprintf ("gesture_%d_trial_%d.dat",p,t));

load(fullfile (folder,"metadata.dat"));
load(fullfile (folder, "input_activity.dat"));
nT = length(x_active);

### Teaching signal
if exist("recode","var")
  clear z
  for g=1:Ng
    for tt=1:Nt
      for i=1:3
        tmp = load(fullfile(folder,...
                             sprintf ("teaching_signals_gesture_%d_teach_input_%d_trial_%d.txt",g-1,i-1,tt-1)));
        if !exist ("z","var")
          nnT = size (tmp,1);
          z   = zeros (nT, 3, Ng, Nt);
          t_  = linspace (0,1,nnT).';
          ti  = linspace (0,1,nT).';
        endif
        z(:,i,g,tt) = interp1 (t_,tmp,ti);
      endfor
    endfor
  endfor
  clear recode
endif # recode teach signal

# Covariance matrix of learning set
C = C_in = zeros (N);
Z = Z_in = zeros (N,3);
for g=1:Ntrain;
  for tt=1:Nt;
    load(fname(g-1,tt-1)); 
    C_in(INdata.ind{g,tt}) += (Xa.'*Xa)(:)/nT; 
    C(OUTdata.ind{g,tt})   += (Ya.'*Ya)(:)/nT;

    zz = z(:,:,g,tt);%.*x_active(:,g);
    Z_in(INdata.nid{g,tt},:) += Xa.'*zz;
    Z(OUTdata.nid{g,tt},:)   += Ya.'*zz;
    
  endfor #over trials 
endfor #over gestures

# Train
W = W_in  = zeros (N,3); 
lambda = logspace (-6,2,50);

# Cov matrix has zero rows and cols due to unactive neurons
# We remove those zero rows and cols.
nz_in       = find(sum(abs(C_in),2)>1e-4); # non zeros rows of Cov matrix
K           = submat (C_in,nz_in,nz_in,"mode","keep","eco"); 
[W_tmp,lambda0_in]       = xridgereg (K, Z_in(nz_in,:),lambda); 
W_in(nz_in,:) = W_tmp;

nz    = find(sum(abs(C),2)>1e-4); # non zeros rows of Cov matrix
K     = submat (C,nz,nz,"mode","keep","eco");
[W_tmp,lambda0] = xridgereg (K, Z(nz,:),lambda); 
W(nz,:) = W_tmp;

INerror = OUTerror = struct ("train",zeros(Ntrain,Nt,3),"test", zeros (Ntest,Nt,3));
# Train Error
for g = 1:Ntrain;
  for tt=1:Nt;
    figure (tt)
    load (fname(g-1,tt-1)); 
    zh_in = Xa * W_in(INdata.nid{g,tt},:)/nT;
    zh    = Ya * W(OUTdata.nid{g,tt},:)/nT;
    zz    = z(:,:,g,tt);%.*x_active(:,g);
    
    INerror.train(g,tt,:)  = mean ((zz-zh_in).^2) ./ mean(zz.^2);
    OUTerror.train(g,tt,:) = mean ((zz-zh).^2) ./ mean(zz.^2);  

    subplot(Ng,1,g)
    plot(t,zh_in,'-r',t,zh,'-g',t,zz,'-k');
    axis ([0 max(t) min(zz(:)) max(zz(:))]);

  endfor #over trials 
endfor #over gestures

# Test Error
for g=Ntrain+1:Ntrain+Ntest;
  for tt=1:Nt;
    figure (tt)
    load (fname(g-1,tt-1));
    zh_in = Xa * W_in(INdata.nid{g,tt},:)/nT;
    zh    = Ya * W(OUTdata.nid{g,tt},:)/nT;
    zz    = z(:,:,g,tt);%.*x_active(:,g);

    INerror.test(g-Ntrain,tt,:)  = mean ((zz-zh_in).^2) ./ mean(zz.^2);
    OUTerror.test(g-Ntrain,tt,:) = mean ((zz-zh).^2) ./ mean(zz.^2);  

    subplot(Ng,1,g)
    plot(t,zh_in,'-r',t,zh,'-g',t,zz,'-k');
    axis ([0 max(t) min(zz(:)) max(zz(:))]);

  endfor #over trials 
endfor #over gestures
