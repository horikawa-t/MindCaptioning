function odlinetight(ax,flip)
% odlinetight -- draw off diagonal line
%
% [Input]
%   -ax:axis
%
%
%
%
%
%
%
%
% Written by Tomoyasu horikawa horikawa-t@atr.jp 2011/10/05
%
%
if ~exist('flip','var') || isempty(flip)
    flip = 0;
end

hld=ishold;
hold on
min2max=[min([ax(1),ax(3)]),max([ax(2),ax(4)])];
if length(ax) > 4
    %h=plot3(min2max,min2max,[ax(5),ax(6)],'-k');
    h=plot3(min2max,min2max,[ax(5),ax(5)],'-k'); % draw 2d diag line at the bottom
%     h=plot3(min2max,min2max,[ax(6),ax(6)],'-k'); % draw 2d diag line at the top
else
    if flip
    h=plot(min2max(end:-1:1),min2max,'-k');
    else
    h=plot(min2max,min2max,'-k');
    end
end
x=get(h,'Annotation');
axis([min2max,min2max])
if iscell(x)
    for i=1:length(x)
        x{i}.LegendInformation.IconDisplayStyle='off';
    end
else
    x.LegendInformation.IconDisplayStyle='off';
end

if ~hld
    hold off
end

