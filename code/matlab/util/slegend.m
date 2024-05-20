function hh = slegend(varargin)
% function hh = slegend(varargin)
% slegend shorten the length of legned items
% 
% [Input]
%   - varargin: various arguments for legend function
% 
% [Output]
%   - hh: figure handle
% 
% 
% Tomoyasu Horikawa 20211022
% 
% 
if any(strcmp(varargin,'ncol')) || any(strcmp(varargin,'nColumns'))
    ncolidx = min([find(strcmp(varargin,'ncol')),find(strcmp(varargin,'nColumns'))]);
    ncol = varargin{ncolidx+1};
    varargin(ncolidx:(ncolidx+1)) = [];
else
    ncol = 1;
end
hh = legend(varargin{:});
% hh.ItemTokenSize = hh.ItemTokenSize/2;
hh.ItemTokenSize = hh.ItemTokenSize./[4;2];

hh.NumColumns = ncol;
