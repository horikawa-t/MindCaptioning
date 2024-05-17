% function dchk
% delete chk files
% 
warning off
try
fprintf('Delete: %s\n',saveFnameChkx)
delete(saveFnameChkx)
end
try
fprintf('Delete: %s\n',saveFnameChk)
delete(saveFnameChk)
end
warning on