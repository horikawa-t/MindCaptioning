function hout=pathtitle(str)
%SUPTITLE Puts a title above all subplots.
%	SUPTITLE('text') adds text to the top of the figure
%	above all subplots (a "super title"). Use this function
%	after all subplot commands.

% Drea Thomas 6/15/95 drea@mathworks.com
% John Cristion 12/13/00 modified
% Mark Histed 03/13/04 histed@mit.edu: fix disappearing legend on last plot
%
% $Id: suptitle.m,v 1.2 2004/03/13 22:17:47 histed Exp $

% Warning: If the figure or axis units are non-default, this
% will break.
% Modified by Tomoyasu Horikawa 2012/12/26

% TH121226
set(0,'DefaultTextInterpreter','none')
str={which(str)};


% Parameters used to position the supertitle.

% Amount of the figure window devoted to subplots
plotregion = 1.0;

% Y position of title in normalized coordinates
titleypos  = .02;

% Fontsize for supertitle
%fs = get(gcf,'defaultaxesfontsize')+4;

fs = get(gcf,'defaultaxesfontsize');

% Fudge factor to adjust y spacing between subplots
fudge=1;

haold = gca;
figunits = get(gcf,'units');

% Get the (approximate) difference between full height (plot + title
% + xlabel) and bounding rectangle.

if (~strcmp(figunits,'pixels')),
    set(gcf,'units','pixels');
    pos = get(gcf,'position');
    set(gcf,'units',figunits);
else,
    pos = get(gcf,'position');
end
ff = (fs-4)*1.27*5/pos(4)*fudge;

% The 5 here reflects about 3 characters of height below
% an axis and 2 above. 1.27 is pixels per point.

% Determine the bounding rectange for all the plots

% h = findobj('Type','axes');

% findobj is a 4.2 thing.. if you don't have 4.2 comment out
% the next line and uncomment the following block.

h = findobj(gcf,'Type','axes');  % Change suggested by Stacy J. Hills

% If you don't have 4.2, use this code instead
%ch = get(gcf,'children');
%h=[];
%for i=1:length(ch),
%  if strcmp(get(ch(i),'type'),'axes'),
%    h=[h,ch(i)];
%  end
%end




max_y=0;
min_y=1;

oldtitle =0;
for i=1:length(h),
    if (~strcmp(get(h(i),'Tag'),'pathtitle')),
        pos=get(h(i),'pos');
        if (pos(2) < min_y), min_y=pos(2)-ff/5*3;end;
        if (pos(4)+pos(2) > max_y), max_y=pos(4)+pos(2)+ff/5*2;end;
    else,
        oldtitle = h(i);
    end
end

% if max_y > plotregion,
% 	scale = (plotregion-min_y)/(max_y-min_y);
% 	for i=1:length(h),
% 		pos = get(h(i),'position');
% 		pos(2) = (pos(2)-min_y)*scale+min_y;
% 		pos(4) = pos(4)*scale-(1-scale)*ff/5*3;
% 		set(h(i),'position',pos);
% 	end
% end

np = get(gcf,'nextplot');
set(gcf,'nextplot','add');
% if (oldtitle),
%     delete(oldtitle);
% end
ha=axes('pos',[0 1 1 1],'visible','off','Tag','pathtitle');
ht=text(.5,titleypos-1,str);set(ht,'horizontalalignment','center','fontsize',fs);
set(gcf,'nextplot',np);
axes(haold);

% fix legend if one exists
if 0  % TH20211110 for matlab2019b
legH = legend;
if ~isempty(legH)
    try
        axes(legH);
    catch
        if ~isempty(legH.String) % TH20210610 for matlab2019b
            try
                axes(legH.String);
            end
        end
    end
end
end
if nargout,
    hout=ht;
end

