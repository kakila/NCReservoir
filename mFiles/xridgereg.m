## Copyright (C) 2008-2014 Juan Pablo Carbajal <ajuanpi+dev@gmail.com>
## 
## This program is free software; you can redistribute it and/or modify it under
## the terms of the GNU General Public License as published by the Free Software
## Foundation; either version 3 of the License, or (at your option) any later
## version.
## 
## This program is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
## FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
## details.
## 
## You should have received a copy of the GNU General Public License along with
## this program; if not, see <http://www.gnu.org/licenses/>.

## Copyright (C) 2013 - Juan Pablo Carbajal
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

## Author: Juan Pablo Carbajal <ajuanpi+dev@gmail.com>

## -*- texinfo -*-
## @deftypefn {Function File} {[@var{b}, @var{l}, @var{err}] =} xridgereg (@var{A}, @var{x},@var{l}=[], @var{k}=10, @var{efunc}=[])
## Ridge regresssion A*b = x with cross validation.
##
## @end deftypefn


function [b l err] = xridgereg (data,x,l=[],k=10, efunc=[]);

  [nT nS] = size (data);

  if (nT <= k)
    warning (["subset bigger (k=%d) than whole data set (n=%d)!\n" ...
             "Taking k=n-1\n"],k,nT);
    k = nT-1;
  endif

  
  n      = floor (nT/k);
  nT     = k*n;
  data_s = data(1:nT,:);
  rest   = [];%data(nT+(1:nT-k*n),:);
  x_s    = x(1:nT,:);
  x_rest = [];%x(nT+(1:nT-k*n),:);

  K = mat2cell (reshape (randperm (nT), k, n),ones (k,1),n);

  U  = S = ST = W = cell (k,1);
  ind = 1:k;
  for i=1:k
    T = cell2mat (K(ind != i))(:);
    [u,s,w] = svd ([data_s(T,:); rest],true);
    U{i}  = u.';
    S{i}  = s;
    ST{i} = s.';
    W{i}  = w;
  end

  if !isempty (l)

    err = arrayfun (@(z)xvaliderror (z,data_s,rest,x,x_rest,U,S,ST,W,K,efunc),l);
    [~,il] = min (err);
    l      = l(il);
    
  else #Optimize

    l = 10.^[-6:2];
    [~,l] = xridgereg (data,x,l,k);
    [l err st] = sqp (l, @(z)xvaliderror (z,data_s,rest,x,x_rest,U,S,ST,W,K),[],[],0,[]);
    if st != 101
      warning ("SQP terminated abnormally!! info: %d",st);
    endif

  endif

  [U,S,W] = svd (data, true);
  ST = S.';
  U = U.';
  I = eye(size(S,1));
  b = W * ((ST * S + l^2 * I) \ (ST * U * x));

endfunction

function err = xvaliderror (l,data,rest,x,x_rest,U,S,ST,W,K,efunc=[])

  k = numel(K);
  n = size(data,2);

  ind = 1:k;

  err = 0;
  for i=1:k
    T = cell2mat (K(ind != i))(:);
    b = W{i}* ((ST{i} * S{i} + l^2 * eye (size(S{i},1))) \ (ST{i} * U{i} * x(T,:)));
    V = K{i}(:);
    if efunc
      err += efunc ([x(V,:); x_rest], [data(V,:); rest]*b);
    else
      err += sumsq (([x(V,:); x_rest] - [data(V,:); rest]*b)(:));
    endif
  end
  err /= k;

endfunction

%!demo
%! nT = 100; n = 15;
%! t = linspace (0,1,nT)';
%! tr = ismember(1:nT,randperm (nT,n-1));
%!
%! y = sin(2*pi*2*t);
%! x = y + 0.1*randn (nT,1);
%! A = t.^[0:n-1];
%!
%! [b l] = xridgereg (A(tr,:),x(tr),[],7);
%! b_ = A(tr,:) \ x(tr);
%!
%! plot(t(tr),x(tr),'.;data;',t(!tr),x(!tr),'.g;unseen;',...
%!      t,A*b,'-r;optim;',t,A*b_,'-m;bckslh;',...
%!        t,y,'-k;true;');
%! axis([t(1) t(end) min(x) max(x)])
%! e = sumsq(y - A*[b b_]) / sumsq(y);
%! printf("Lambda: %g\nError: %g %g(bckslh)\n",l,e);
