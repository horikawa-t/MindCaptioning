function h = bandplot3(xax,mx,vx,col,varargin)
% bandplot--plot mean and draw sd with band
% function bandplot(X,Y,E)
%
% [Inputs]
%     -mx:mean of the data [NxM]
%     -vx:SD of the data [NxM]
%     -col:color [cell]
%
%
% [Example]
% x1=cumsum(randn(3,100)')'; % original data
% mx1=mean(x1); % mean
% vx1=sqrt(var(x1))/2; % sd
% x2=cumsum(randn(3,100)')'; % original data
% mx2=mean(x2); % mean
% vx2=sqrt(var(x2))/2; % sd
% mx=[mx1;mx2]';
% vx=[vx1;vx2]';
% bandplot(mx,vx)
% legend({'a','b'})
%
%  Created by Tomoyasu Horikawa horikawa.t@gmal.com 2010/07/01
%  Modified to use appropriate legend property TH 2011/7/7
%

if ~exist('col','var')||isempty(col)
    % draw area
    % r b g y c m
    % col={[0.9 0 0.9] [0.8 0 0] [0 0 0.8] [0 0.8 0] [0.9 0.9 0] [0 0.9 0.9]};
    % % r g b m c y
    col={[0.9 0.9 0] [0.8 0 0] [0 0.8 0] [0 0 0.8],[0.9 0 0.9] [0 0.9 0.9]};
    % hold off
end
h=[];
if ~iscell(col)
    col={col};
end

for itr=1:size(mx,2)
    hold on
    h=fill([xax,flipdim(xax,2)],...
        [mx(:,itr)-vx(:,itr);flipdim(mx(:,itr)+vx(:,itr),1)]',col{mod(itr,length(col))+1},'EdgeColor','none');
    
    %set(h,'EdgeAlpha',0.2)
    set(h,'FaceAlpha',0.2)
    % no annotation
    hh=plot(xax,mx(:,itr),'Color',col{mod(itr,length(col))+1}*0.8,varargin{:});
    %h=plot(xax,mx(:,itr),'Color',col{mod(itr,length(col))+1},varargin{:});
    x=get(hh,'Annotation');
    x.LegendInformation.IconDisplayStyle='off';
    withcontour = 0;
    if withcontour
        % remove edges
        h=vline([1 xax(size(mx(:,itr),1))],'w');
        x=get(h,'Annotation');
        for i=1:length(x)
            x{i}.LegendInformation.IconDisplayStyle='off';
        end
        h=plot(xax,mx(:,itr)+vx(:,itr),'Color',col{mod(itr,length(col))+1});
        x=get(h,'Annotation');
        x.LegendInformation.IconDisplayStyle='off';
        h=plot(xax,mx(:,itr)-vx(:,itr),'Color',col{mod(itr,length(col))+1});
        x=get(h,'Annotation');
        x.LegendInformation.IconDisplayStyle='off';
    end
end
% legend(num2str((1:10)'))

hold off
% alpha(0.2);


