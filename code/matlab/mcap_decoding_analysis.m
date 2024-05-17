% function mcap_decoding_analysis
%
% This script is written to perform decoding analysis in
%  Horikawa, T. (2024) Mind captioning: Evolving descriptive text of mental
%  content from human brain activity. bioRxiv. 
%
% written by Tomoyasu Horikawa horikawa.t@gmail.com 2024/05/13
%

%% initialize
clear all, close all

% general info.
cd('/home/psi/horikawa-t/toolbox/public/mcap/');
p.rootPath = './';
p.analysisType = 'decoding';
addpath(genpath([p.rootPath,'code/']));

%% set paramters
p = mcap_setParams(p);

% overwrite for demo (relatively faster)
do_demo = 1; % set 0, if you want to completely reproduce the manuscript results 
if do_demo
    % perform only 'WB' analysis, skipping ROI-wise analysis
    p.rparam.genDecSkipROITypes = {'WBnoLang','WBnoSem','WBnoVis','Lang'};
    %[4,4,5] can be used for demo [faster almost equivalent to our results]
    p.aparam.l2.nparamLogSearch = 4;
    p.aparam.l2.lowL = 4;
    p.aparam.l2.highL = 5;
end

%% perform crossvalidation decoding analysis
% cv decoding is time consuming. 
% Perform this, only if you are interested in the results of validatoin
% and results in Extended Data Fig.7e.
do_cv = 0;
if do_cv; mcap_cv_decoding(p); end

%% perform generalization decoding analysis
mcap_gen_decoding(p)

%%
done('End decoding analysis')
