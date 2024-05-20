% function mcap_summary_encoding
%
% This script is written to summarize encoding results in
%  Horikawa, T. (2024) Mind captioning: Evolving descriptive text of mental
%  content from human brain activity. bioRxiv. 
%
% written by Tomoyasu Horikawa horikawa.t@gmail.com 2024/05/13
%
%% initialize
clear all, close all

% set path to MindCaptioning directory
cd('/home/psi/horikawa-t/toolbox/public/mcap/');
p.rootPath = './';
addpath(genpath([p.rootPath,'code/']));

%% set paramters for encoding summary
p.analysisType = 'encoding';
p = mcap_setParams(p);

%% summarize results 
[res, p] = mcap_summary_encoding_summarize(p);

%% summarize whole brain encoding accuracy
mcap_summary_encoding_draw_scatter(res,p);
mcap_summary_encoding_draw_accuracy(res,p);
mcap_summary_encoding_draw_bestlayer(res,p);



%%
