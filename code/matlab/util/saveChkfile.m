function saveChkfile(saveFnameChk)
% function saveChkfile(saveFnameChk)
% 
% this function save log file with empty variable 'tmp'
% 
% [Input]
%  -saveFnameChk: filename
% 
% 
% 
% Written by Tomoyasu Horikawa 20231003
% 
tmp = [];
save(saveFnameChk,'tmp','-ascii')

