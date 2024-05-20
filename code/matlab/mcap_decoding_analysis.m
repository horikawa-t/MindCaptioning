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

% set path to MindCaptioning directory
p.rootPath = './';
p.analysisType = 'decoding';
addpath(genpath([p.rootPath,'code/']));

%% set paramters
p = mcap_setParams(p);

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
