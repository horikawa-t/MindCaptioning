function hh = bandebarplot(mus, cis, cols, varargin)
% function hh = bandebarplot(mus, cis, cols, varargin)
%
% [Inputs]
%   -mus: mean of multiple blocks (nblock cells containing mlines)
%   -cis: ci of multiple blocks (nblock cells containing mlines)
%   -cols: color cells (nblock cell)
%
%
% [Outputs]
%   -h: figure handle
%
%
% [howto]
% norgDat = 100;
% nblocks = 10;
% nelms = 5;
% sd = 3;
% clear mus cis
% for i = nblocks:-1:1
%     [cis{i},mus{i}] = ciestim3(randn(norgDat,nelms)*sd+i);
% end
% cols = plotColors(1:length(mus),length(mus),'Spectral_r');
% bandebarplot(mus,cis,cols)
%
% Written by Tomoyasu Horikawa horikawa.t@gmail.com 2023/12/14
%

nblocks = length(mus);
if ~exist('cols','var')||isempty(cols)
    cols = plotColors(1:nblocks,nblocks,'Spectral_r');
end

if ~iscell(cols)
    if size(cols,1) > 1 && size(cols,1) == nblocks
        cols = num2cell(cols',1);
    else
        for j = nblocks:-1:1
            cols_new{j} = cols(1,:); 
        end
        cols = cols_new;
    end
end
nArgin = nargin;
idx = find(strcmp(varargin,'BlockAverage'),1); % TH231221
if ~isempty(idx)
    blockaverage = varargin{idx+1};
    varargin([idx,idx+1]) = [];
    nArgin = nArgin-2;
else
    blockaverage = 1;
end
idx = find(strcmp(varargin,'FlipColor'),1); % TH231221
if ~isempty(idx)
    FlipColor = varargin{idx+1};
    varargin([idx,idx+1]) = [];
    nArgin = nArgin-2;
else
    FlipColor = 0;
end

for j = 1:nblocks
    nlines = size(mus{j}, 2);
    wid = 1;
    dif = wid / (nlines + 1);
    base = -wid / 2;
    % block average
    if blockaverage
        [ci, mu] = ciestim3(mus{j}');
        xx = [j + base + dif * 1, j + base + dif * nlines];
        if FlipColor
        hh{1}{j} = fill([xx, flip(xx, 2)], [mu - ci, mu - ci, flip(mu + ci, 1), flip(mu + ci, 1)], [1,1,1], 'EdgeColor', cols{j});
        %set(hh{1}{j}, 'EdgeAlpha', 0.2)
        else
        hh{1}{j} = fill([xx, flip(xx, 2)], [mu - ci, mu - ci, flip(mu + ci, 1), flip(mu + ci, 1)], cols{j}, 'EdgeColor', 'none');
        end
        set(hh{1}{j}, 'FaceAlpha', 0.2)
        %AnnotationOff(hh);
        hold on
        hh{2}{j} = plot([xx, flip(xx, 2)], [mu, mu, mu, mu], 'Color', cols{j},varargin{:});
        AnnotationOff(hh{2}{j});
    end
    
    for i = 1:nlines
        x0 = j + base + dif * i;
        y = [mus{j}(i) + cis{j}(i), mus{j}(i) - cis{j}(i)];
        x = [x0, x0];
        hh{3}{j,i} = plot(x, y, 'Color', cols{j}, varargin{:});
        AnnotationOff(hh{3}{j,i});
        hold on
        if FlipColor
        hh{4}{j,i} = plot(x0, mus{j}(i), plotMarker2(i), 'Color', cols{j}, 'MarkerFaceColor', [1,1,1], varargin{:});
        else
        hh{4}{j,i} = plot(x0, mus{j}(i), plotMarker2(i), 'Color', cols{j}, 'MarkerFaceColor', cols{j}, varargin{:});
        end
        AnnotationOff(hh{4}{j,i});
    end
end
