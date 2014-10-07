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
## @deftypefn {Function File} {[@var{S}, @var{Pl}, @var{Pr}] = submat (@var{A},@var{r},@var{k})
## @deftypefnx {Function File} {[@dots{}] = submat (@dots{},@asis{"mode"},@var{value})
## @deftypefnx {Function File} {[@dots{}] = submat (@dots{},@asis{"eco"})
## Extracts submatrix from @var{A} by removing/keeping columns and rows.
##
## The matrix @var{S} is the submatrix of @var{A} in which rows @var{r} and
## columns @{k} where removed/selected. By default the function removes rows and columns.
##
## Optional parameters are:
## @table @asis
## @item "mode"
## Controls how the submatrix is built. If followed by any of
## @asis{"erase", "-", "remove"} rows and columns are removed. If followed
## by any of @asis{"select", "+", "keep"} rows and clumns are kept.
## @item "eco"
## By default the submatrix has the same size as @var{A}. To remove the zero
## rows and columns pass the string @asis{"eco"}.
## @end table
##
## The function uses the matrices @var{Pl}, @var{Pr}
## that select rows and colums respectively. The result is
## @asis{@var{S} = @var{Pl}*@var{A}*@var{Pr}}. The returned @var{Pl}, @var{Pr}
## matrices are sparse.
##
## Example:
## @example
## A = reshape (1:3*4,3,4)
## A =
##    1    4    7   10
##    2    5    8   11
##    3    6    9   12
##
## S = submat (A,3,[2 4],"mode","keep")
## S =
##    0    0    0    0
##    0    0    0    0
##    0    6    0   12
##
## S = submat (A,3,[2 4],"eco")
## S =
##   1   7
##   2   8
## @end example
## @end deftypefn

function [S Pl Pr] = submat (A,r,k,varargin)

  parser = inputParser ();
  parser = addParamValue (parser,'Mode', "-" , @ischar);
  parser = addSwitch (parser,'Eco');
  parser = parse(parser,varargin{:});

  mode = parser.Results.Mode;
  eco  = parser.Results.Eco;

  clear parser

  valid_mode.erase  = {"erase", "-", "remove"};
  valid_mode.select = {"select", "+", "keep"};

  [n,m] = size(A);

  ld = ismember (1:n,r);
  rd = ismember (1:m,k);

  switch tolower(mode)
    case valid_mode.erase
     ld = !ld;
     rd = !rd;
    case valid_mode.select
     1;
    otherwise
     error ("Octave:invalid-input-arg","Invalid mode.");
  endswitch

  Pl = sparse (diag (ld));
  Pr = sparse (diag (rd));

  S = Pl * A * Pr;

  if eco
    S = S(ld,rd);
  endif

endfunction

%!demo
%! disp ("Original matrix")
%! A = reshape (1:4*5,4,5)
%! disp ("Sumatrix removing row 3 and columns 2 and 4")
%! S = submat (A,3,[2 4])
%! disp ("Erasing zeros rows and columns")
%! S = submat (A,3,[2 4],"eco")
%! disp ("Sumatrix keeping row 3 and columns 2 and 4")
%! S = submat (A,3,[2 4],"mode","keep")
%! # -------------------------------------------------
