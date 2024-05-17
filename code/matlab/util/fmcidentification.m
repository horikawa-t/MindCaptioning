function [crs, crs_all] = fmcidentification(sim2true,sim2false,nclass,nrep,topk)
% function [crs, crs_all] = = fmcidentification(sim2true,sim2false,nclass,nrep,topk)
%
% fmcidentification performs fast multi-class identification
%
% [Input]
% sim2true: similarity scores to true candidates [nTestSamp x 1]
% sim2false: similarity scores to false candidates [nTestSamp x nCandidates]
% nclass: # of classes [vector] (default=nCandidates)
% nrep: # of repetitions [scalar] (default=100)
% topk: parameter for top-k accuracy [scalar] (default=1)
%
% [Output]
% crs: correct rate averaged across repetitions
% crs_all: correct rate for all repetitions
%
%
% Written by
% horikawa.t@gmail.com 20230830
%
%% compute identificaiton accuracy
% match size
if size(sim2true,2) > 1
    warning('sim2true has more than 1 columns. Please check the input format. Perform analysis with transposed matrix.')
    sim2true = sim2true';
end
if size(sim2true,1) ~= size(sim2false,1)
    error('Sample numbers of true and flase data is not the same.')
end
[nsamps,ncands] = size(sim2false);
sim2true = repmat(sim2true,1,ncands);

if ~exist('nclass','var')
    nclass = ncands;
end
if ~exist('nrep','var')
    nrep = 2;
end
if ~exist('topk','var')
    topk = 1;
end
if topk > 1 && any(nclass < topk+1)
    %warning('class and top-k parameter can produce all 100% accuracy for some class conditions.\n')
end
%%
comp_table = sim2true > sim2false;
crs_all = zeros(nsamps,length(nclass));
for i = nrep:-1:1
    idx = randsample(ncands,ncands);
    for citr = length(nclass):-1:1
        if nclass(citr) == 2
            if topk > 1 % always 100%
                crs_all(:,citr) = crs_all(:,citr)+1;
            else
                crs_all(:,citr) = crs_all(:,citr)+sum(comp_table,2)/ncands;
            end
        else
            
            crs_all(:,citr) = crs_all(:,citr)+(sum(comp_table(:,idx(1:nclass(citr))),2) > (nclass(citr)-topk));
        end
    end
end
crs = crs_all/nrep;

