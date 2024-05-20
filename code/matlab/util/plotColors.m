function colcell = plotColors(inds,n,type,inv)
% function colcell = plotColors(mu, ci, varargin)
% plotColors -- call plotColor function to create multiple color sets
%
% [Outputs]
%   colcell:color cell
%
%
% Written by Tomoyasu Horikawa horikawa.t@gmail.com 2023/12/14
% 
% 
if ~exist('n','var')||isempty(n)
    n = length(inds);
end
if ~exist('type','var')||isempty(type)
    type = 'Specral_r';
end
if ~exist('inv','var')||isempty(inv)
    inv = 0;
end
for i = length(inds):-1:1
    colcell{i} = plotColor(inds(i),n,type,inv);
end
