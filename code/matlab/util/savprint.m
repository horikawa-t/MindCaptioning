function verb=savprint(handle,savName,orientation,verbose)
% function savprint(handle,savName,orientation,verbose)
% 
% 
% [input]
%  -handle: figure handle or gcf
%  -savName: file name 
%  -orientation: orientation of pdf (default = 'landscape')
%   'portrait': tall and thin
%   'landscape': short and wide
% 
% 
% Tomoyasu Horikawa horikawa-t@atr.jp 2010/10/26
% modified by horikawa.t@gmail.com 20210106: add orientation argument
% 

if ~exist('orientation','var') || isempty(orientation)
    orientation = 'landscape';
end

setdir(fileparts(savName));
set(handle, 'PaperType', 'A4');
set(handle,'PaperOrientation',orientation);
set(handle,'PaperUnits','normalized');
set(handle,'PaperPosition',[0 0 1 1]);
if strcmp(savName(end-2:end),'png')
    print(handle,'-dpng',savName);
end
if strcmp(savName(end-2:end),'pdf')
%     print(handle,'-dpdf',savName);
    print(handle,'-painters','-dpdf',savName);
%     print(handle,'-painters','-r300','-dpdf',savName);
end
if strcmp(savName(end-3:end),'tiff')
    print(handle,'-dtiff',savName);
end
if strcmp(savName(end-1:end),'ps')
    print(handle,'-dps',savName);
end
if strcmp(savName(end-3:end),'jpeg')||strcmp(savName(end-3:end),'jpg')
    print(handle,'-djpeg',savName);
end
verb=sprintf(savName);
if exist('verbose','var') && verbose
    fprintf(verb)
    fprintf('\n')
end
%%
% hf = gcf;
% hf.Renderer = 'painters';
% print('im.pdf','-dpdf','-bestfit')