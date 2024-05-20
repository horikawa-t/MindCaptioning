function htxt = figtitle(hfig, str, pos)
% figtitle    Draw title text on the figure
%
% This file is a part of BrainDecoderToolbox2.
%
% Inputs:
%
% - hfig : Figure handle
% - str  : Title string
% - pos  : Position of the title ('top' or 'bottom', default: 'top')
%
% Outputs:
%
% - htxt : Axis handle
%
% Note:
%
% This function is based on `suptitle` developed by Drea Thomas, John Cristion,
% and Mark Histed (suptitle.m,v 1.2 2004/03/13 22:17:47 histed Exp).
%
% Modified by Tomoyasu Horikawa 20230829 : add a line break at the half point if length(str) > 150
%

if ~exist('pos', 'var')
    pos = 'top';
end

if isequal(pos, 'top')
    titleypos = 0.98;
    titleypos = 0.99;
elseif isequal(pos, 'bottom')
    titleypos = 0.02;
else
    error('Unsupported title position');
end

fontSize = get(hfig, 'defaultaxesfontsize');

np = get(hfig, 'nextplot');
set(hfig, 'nextplot', 'add');

% Draw the title
ha = axes('Position', [0, 0, 1, 1], 'Visible', 'off', 'Tag', 'figtitle');
if length(str) > 150 % TH230829
    str = sprintf('%s\n%s',str(1:round(length(str)/2)),str(round(length(str)/2)+1:end));
    hshift = 0.01;
else
    hshift = 0;
end
htxt = text(0.5, titleypos-hshift, str, ...
            'Interpreter', 'none', ...
            'FontSize', fontSize, ...
            'HorizontalAlignment', 'center');

set(hfig, 'nextplot', np);
