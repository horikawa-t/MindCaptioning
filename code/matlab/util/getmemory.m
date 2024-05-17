function [mf,mt] = getmemory(printOn)
% function [mf,mt] = getmemory
% getmemory get current free memory on the machine
% 
% [ouput]
%  mf: free memory
%  mt: total memory
% 
% written by horikawa.t@gmail.com 20220801
% 
if ~exist('printOn','var') || isempty(printOn)
    printOn = 0;
end
try
    [status,cmdout] = system('grep MemFree /proc/meminfo');
    for i = length(cmdout):-1:1
        mfidx(i) = contains(cmdout(i),{'0','1','2','3','4','5','6','7','8','9'});
    end
    mf = str2double(cmdout(find(mfidx,1,'first'):find(mfidx,1,'last')));
    if printOn
    fprintf(cmdout)
    end
    
    [status,cmdout] = system('grep MemTotal /proc/meminfo');
    for i = length(cmdout):-1:1
        mtidx(i) = contains(cmdout(i),{'0','1','2','3','4','5','6','7','8','9'});
    end
    mt = str2double(cmdout(find(mtidx,1,'first'):find(mtidx,1,'last')));
    if printOn
    fprintf(cmdout)
    end

catch me
    fprintf('Failed to get memory size.')
    mf = nan;
    mt = nan;
end
