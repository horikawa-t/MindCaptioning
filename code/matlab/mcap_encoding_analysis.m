% function mcap_encoding_analysis
%
% This script is written to perform encoding analysis in
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
p.analysisType = 'encoding';
addpath(genpath([p.rootPath,'code/']));

%% set paramters
p = mcap_setParams(p);

% overwrite for demo (relatively faster)
do_demo = 1; % set 0, if you want to completely reproduce the manuscript results 
if do_demo
    %[4,4,5] can be used for demo [faster almost equivalent to our results]
    p.aparam.l2.nparamLogSearch = 4;
    p.aparam.l2.lowL = 4;
    p.aparam.l2.highL = 5;
end

%% perform cross-validation encoding analyses
mcap_cv_encoding(p)

%% perform generalization encoding analyses
mcap_gen_encoding(p)

%%
done('End encoding analysis')
