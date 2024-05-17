function [dpath,id]=setdir(dpath)
% setdir - output dpath and make directory if the directory did not exist.
% function [dpath,id]=setdir(dpath)
%
%
%
%
%
%  Created by Tomoyasu Horikawa horikawa-t@atr.jp 2011/01/21
%
%
warning off
id = exist(dpath,'dir');
if ~id
    dpath = strrep(dpath,'//','/');
    try
        eval(sprintf('!mkdir %s',dpath)) % TH231222
%         fileattrib(dpath,'+w +x')
    end
    try
        mkdir(dpath) % TH231222
    end
    try
        flag = 1;
        dpaths = {dpath};
        cnt = 1;
        while flag
            cnt = cnt+1;
            dpaths{cnt} = fileparts(dpaths{cnt-1});
            if strcmp(dpaths{cnt}, dpaths{cnt-1})
                flag = 0;
            end
        end
        for i = 1:(length(dpaths)-6)
            dpathx = dpaths{i};
%             try
%                 fileattrib(dpathx,'+w +x');
%             end
%             eval(sprintf('!chmod +t %s',dpathx)) %modified by SY
            eval(sprintf('!chmod 777 %s',dpathx))
        end
    end
end

if ~strcmp(dpath(end),'/') % modified by TH230131
    dpath = [dpath,'/'];
end

warning on