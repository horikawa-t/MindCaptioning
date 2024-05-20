function [xq,yq,z] = computeGrid(x1,x2,fout,nbins)
if ~exist('nbins','var') || isempty(nbins)
    nbins = 100;
end
x = linspace(min(x1),max(x1),nbins);
y = linspace(min(x2),max(x2),nbins);
[xq,yq] = meshgrid(x,y);
orig_state = warning;
warning('off','all');
z = griddata(double(x1),double(x2),fout,xq,yq);
warning(orig_state);