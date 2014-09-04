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

## -*- texinfo -*-
## @deftypefn {Function File} {[@var{x},@var{y}] =} id2xy (@var{id}, @var{dim})
## Converts from id to x,y in [-1,1].
## Scaled version of ind2sub.
## @end deftypefn

function [x,y] = id2xy (id,dim) 
  [x,y] = ind2sub (dim,double(id)+1);
  x     = 2*(x-1)/(dim(1)-1) - 1;
  y     = 2*(y-1)/(dim(2)-1) - 1;
endfunction
