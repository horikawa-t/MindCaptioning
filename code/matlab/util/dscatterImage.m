function [h,params] = dscatterImage(x,y,varargin)
% function [h,ax,params] = dscatterImage(x,y,varargin)
% densemap: draw density map for 2D scores as image
%
% [input]
% 	x: score 1
% 	y: score 2
% 
%  [optional input]
%     nbin: n bins for making histogram
%     rangx: range of x axis
%     rangy: range of y axis
%     bandwidth: bandwidth for estimating kernel density
%     colname: color name for the density 
%     drawtype: types for drawing (default='density' or 'frequency') 
%
% [output]
% 	h: figure handle
% 	params: parameters to draw image
%         params.edgesx : edges of x axis
%         params.edgesy : edges of y axis
%         params.scalex : scale of x axis 
%         params.scaley : scale of y axis 
%         params.to0shiftx : a scalar value to shift to 0 on x axis
%         params.to0shifty : a scalar value to shift to 0 on y axis
%         params.toMeanshiftx : a scalar value to shift to mean of x
%         params.toMeanshifty : a scalar value to shift to mean of y
%
% [example]
% % without intercept
% n = 10000;
% noise = gaussGen(n,[1,1]',[1,0.5;0.5,1]);
% % noise = gaussGen(n,[1,1]',[1,0.999;0.999,1]);
% x = noise(:,1);
% y = noise(:,2);
% bandwidth = 0.05;
% bandwidth = 0.1;
% rangx = -3:0.1:5;
% rangy = -3:0.1:5;
% figure;
% subplot(1,2,1);
% dscatterImage(x,y,'nbin',50,'rangx',rangx,'rangy',rangy,'bandwidth',bandwidth,'colname','magma','drawtype','frequency');
% axis square
% % odlinetight(axis)
% subplot(1,2,2);
% dscatterImage(x,y,'nbin',50,'rangx',rangx,'rangy',rangy,'bandwidth',bandwidth,'colname','magma','drawtype','density');
% axis square
% % odlinetight(axis)
% figure;
% [h,ps] = dscatterImage(x,y,'nbin',50,'rangx',rangx,'rangy',rangy,'bandwidth',bandwidth,'colname','magma','drawtype','density');
% vline(ps.to0shiftx,'-k')
% hline(ps.to0shifty,'-k')
% vline(ps.toMeanshiftx,'--r')
% hline(ps.toMeanshifty,'--r')
% [slopeDR,int,st,xymu] = demingRegression(x,y,1,1,1,[],100);
% hold on
% rangslope = rangx*ps.scalex+ps.to0shiftx;
% plot(rangslope,(rangx*slopeDR+int)*ps.scaley+ps.to0shifty+xymu(2)*ps.scaley-ps.toMeanshifty+ps.to0shifty,'-','Color',[1,1,1]*0.8)
% angle = rad2deg(atan(slopeDR));
% axname(linspace(rangx(1),rangx(end),5),1,linspace(rangx(1),rangx(end),5)*ps.scalex+ps.to0shiftx)
% axname(linspace(rangy(1),rangy(end),5),2,linspace(rangy(1),rangy(end),5)*ps.scaley+ps.to0shifty)
% 
% % with intercept
% n = 10000;
% noise = gaussGen(n,[1,2]',[1,0.5;0.5,1]);
% % noise = gaussGen(n,[1,2]',[1,0.999;0.999,1]);
% x = noise(:,1);
% y = noise(:,2)*2;
% bandwidth = 0.1;
% rangx = -3:0.1:5;
% rangy = -4:0.1:12;
% figure;
% subplot(1,2,1);
% dscatterImage(x,y,'nbin',50,'rangx',rangx,'rangy',rangy,'bandwidth',bandwidth,'colname','magma','drawtype','frequency');
% axis square
% % odlinetight(axis)
% subplot(1,2,2);
% dscatterImage(x,y,'nbin',50,'rangx',rangx,'rangy',rangy,'bandwidth',bandwidth,'colname','magma','drawtype','density');
% axis square
% % odlinetight(axis)
% figure;
% [h,ps] = dscatterImage(x,y,'nbin',50,'rangx',rangx,'rangy',rangy,'bandwidth',bandwidth,'colname','magma','drawtype','density');
% vline(ps.to0shiftx,'-k')
% hline(ps.to0shifty,'-k')
% vline(ps.toMeanshiftx,'--r')
% hline(ps.toMeanshifty,'--r')
% [slopeDR,int,st,xymu] = demingRegression(x,y,0,1,1,[],100);
% hold on
% rangslope = rangx*ps.scalex+ps.to0shiftx;
% plot(rangslope,(rangx*slopeDR+int)*ps.scaley+ps.to0shifty+xymu(2)*ps.scaley-ps.toMeanshifty+ps.to0shifty,'-','Color',[1,1,1]*0.8)
% angle = rad2deg(atan(slopeDR));
% axname(linspace(rangx(1),rangx(end),5),1,linspace(rangx(1),rangx(end),5)*ps.scalex+ps.to0shiftx)
% axname(linspace(rangy(1),rangy(end),5),2,linspace(rangy(1),rangy(end),5)*ps.scaley+ps.to0shifty)
% 
% written by horikawa.t@gmail.com 20220204

%% initial setting
nArgin = nargin;
idx = find(strcmp(varargin,'nbin'),1);
if ~isempty(idx)
    nbin = varargin{idx+1};
    varargin([idx,idx+1]) = [];
    nArgin = nArgin-2;
else
    nbin = 200;
end
idx = find(strcmp(varargin,'bandwidth'),1);
if ~isempty(idx)
    bandwidth = varargin{idx+1};
    varargin([idx,idx+1]) = [];
    nArgin = nArgin-2;
else
    bandwidth = 0.01;
end
idx = find(strcmp(varargin,'rangx'),1);
if ~isempty(idx)
    rangx = varargin{idx+1};
    varargin([idx,idx+1]) = [];
    nArgin = nArgin-2;
else
    rangx = linspace(min(x,[],1),max(x,[],1),min([length(x),200]));
end

idx = find(strcmp(varargin,'rangy'),1);
if ~isempty(idx)
    rangy = varargin{idx+1};
    varargin([idx,idx+1]) = [];
    nArgin = nArgin-2;
else
    rangy = linspace(min(y,[],1),max(y,[],1),min([length(y),200]));
end
idx = find(strcmp(varargin,'colname'),1);
if ~isempty(idx)
    colname = varargin{idx+1};
    varargin([idx,idx+1]) = [];
    nArgin = nArgin-2;
else
    colname = 'magma';
end
idx = find(strcmp(varargin,'drawtype'),1);
if ~isempty(idx)
    drawtype = varargin{idx+1};
    varargin([idx,idx+1]) = [];
    nArgin = nArgin-2;
else
    drawtype = 'density';
end
idx = find(strcmp(varargin,'logscale'),1);
if ~isempty(idx)
    logscale = varargin{idx+1};
    varargin([idx,idx+1]) = [];
    nArgin = nArgin-2;
else
    logscale = 0;
end


%% draw map
edgesx = linspace(rangx(1),rangx(end), nbin);
edgesy = linspace(rangy(1),rangy(end), nbin);
[x1,x2] = meshgrid(edgesx, edgesy);
xi = [x1(:),x2(:)];
X = [x,y];
[fout,xout,u,plottype] = mvksdensity(X,xi,'bandwidth',bandwidth);
[xq,yq,z] = computeGrid(xout(:,1),xout(:,2),fout,nbin);
hh3 = hist3([y,x], 'Edges',{edgesy,edgesx});
z(hh3(:) == 0) = nan;
if logscale
    z(z > 0) = log10(z(z > 0));
    hh3(hh3 > 0) = log10(hh3(hh3 > 0));
end
switch drawtype
    case 'density'
        h = imagesc([0,nbin],[nbin,0],flipud(z),[0,max(z(:))]);
    case 'frequency'
        h = imagesc([0,nbin],[nbin,0],flipud(hh3),[0,max(hh3(:))]);
end
set(gca,'YDir','normal')
coltab = colmap(colname);
coltab = matScale(coltab,1000,1);
coltab(1,:) = [1,1,1];
colormap(coltab)

% get additional info.
params.edgesx = edgesx;
params.edgesy = edgesy;
params.scalex = nbin/(rangx(end)-rangx(1));
params.scaley = nbin/(rangy(end)-rangy(1));
params.to0shiftx = (0-mean(edgesx(1:2)))*params.scalex;
params.to0shifty = (0-mean(edgesy(1:2)))*params.scaley;
params.toMeanshiftx = (mean(x)-mean(edgesx(1:2)))*params.scalex;
params.toMeanshifty = (mean(y)-mean(edgesy(1:2)))*params.scaley;
% vline(params.to0shiftx)
% hline(params.to0shiftx)

%%

