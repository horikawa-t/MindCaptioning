function state=structout(strc,name,fields)
% structout -- create variables in structures with the field names
% 
% [Inputs]
%     -strc:structrue
%     -name:name of structure
%     -fields: if not assigned, all fields are created
%     
% [Outputs]
%     -state:state to create variables
% 
% 
% [usage]
% strc.a=1;
% strc.b='test';
% strc.c=magic(10);
% eval(structout(strc,'strc'))
% 
% 
% 
% Written by Tomoyasu Horikawa horikawa-t@atr.jp
% Modified by Tomoyasu Horikawa horikawa-t@atr.jp
% -adding the 'fields' option.
% 
% 

fname=fieldnames(strc);

if exist('fields','var')
    ind=ismember(fname,fields);
    fname=fname(ind);
end

state=[];
for itr=1:length(fname)
    sub=sprintf('%s=%s.%s;',fname{itr},name,fname{itr});
    state=[state,sub];
end
    
    