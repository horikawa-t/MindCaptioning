function A=merge(D,direction,connection)
% this function merge multicell matrix to one matrix
%
% direction:direction ==1 ?? direction ==2 -> (default=1)
% connection: can add something between elements
%
% HORIKAWA tomoyasu 09/09/03
% modified by tomoyasu horikawa 090917; direction was inversed
% modified by tomoyasu horikawa 230904; connection var was added
% modified by tomoyasu horikawa 231005; return immediately if D is not cell
len=numel(D);
if ~iscell(D)
    A = D;
    return
end
if len == 0
    A = [];
else
    A=D{1};
    if ~exist('connection','var') || isempty(connection)
        if isnumeric(A)
            connection = [];
        elseif ischar(A)
            connection = '';
        else
            connection = [];
        end
    end
    if exist('direction','var')==1
        if direction==1
            for i=2:len
                A=[A;connection;D{i}];
            end
        else
            for i=2:len
                A=[A,connection,D{i}];
            end
        end
    else
        for i=2:len
            A=[A;D{i}];
        end
        
    end
end