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
## @deftypefn {Function File} {@var{Y} =} loaddata (@var{proj},@var{trial})
## @deftypefnx {Function File} {[@var{X} @var{Y}] =} loaddata (@dots{})
## Loads recordigs from project @var{proj} and trial @var{trial}.
## Loads only output unless two output arguments are requiered.
## @end deftypefn

function [varargout] = loaddata (proj, trial, folder)
  ofname = @(p,t) sprintf ("outputs_gesture_%d_trial_%d.txt",p,t);
  ifname = @(p,t) sprintf ("inputs_gesture_%d_trial_%d.txt",p,t);

  # t: timestamp (double)
  # n_id: neuron id (uint8, [0,255]);
  # s_id: synapse id (double) TODO make int when knowing the range.
  X  = struct ("t",[],"n_id",[],"s_id",[]); # inputs
  Y  = struct ("t",[],"n_id",[]);                    # outputs

  if nargout > 1
    # Load input
    fname  = fullfile (folder, ifname(proj,trial));
    printf ("Loading from %s\n",fname); fflush(stdout);
    tmp    = load (fname);
    
    X.t    = tmp(:,1); 
    nid    = fix (tmp(:,2));
    X.n_id = uint8 (nid);
    X.s_id = tmp(:,2) - nid;
  endif
            
  # Load output
  fname  = fullfile (folder, ofname(proj,trial));
  printf ("Loading from %s\n",fname); fflush(stdout);
  tmp    = load (fname);
  
  Y.t    = tmp(:,1); 
  Y.n_id = uint8 (fix (tmp(:,2)));

  if nargout > 1
    varargout{1} = X;
    varargout{2} = Y;
  else
    varargout{1} = Y;
  endif
endfunction
