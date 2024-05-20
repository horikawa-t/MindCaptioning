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

% set path to MindCaptioning directory
p.rootPath = './';
p.analysisType = 'encoding';
addpath(genpath([p.rootPath,'code/']));

%% set paramters
p = mcap_setParams(p);

%% perform cross-validation encoding analyses
mcap_cv_encoding(p)

%% perform generalization encoding analyses
mcap_gen_encoding(p)

%%
done('End encoding analysis')
