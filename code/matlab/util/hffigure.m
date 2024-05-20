function [h,scrsz]=hffigure(varargin)
% ffigure - build half of fullscreen-size figure window
% function [h,scrsz]=ffigure
% 
% 
% [Outputs]
%     h: figure handle
%     scrsz: screen size
%     
%     
% Tomoyasu Horikawa horikawa-t@atr.jp
% 
% 
scrsz = get(0,'ScreenSize');
shift = 200;
position = ceil([1 scrsz(4) scrsz(3) scrsz(4)]/2)+[shift,0,0,0];
h = figure('Position',position,'DefaultAxesFontSize',8,'DefaultTextFontSize',8,varargin{:});

