% function mcap_summary_decoding
%
% This script is written to summarize decoding results in
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
addpath(genpath([p.rootPath,'code/']));

%% set paramters for decoding(text_generation) summary
p.analysisType = 'text_generation';
p = mcap_setParams(p);

%% parameter settings
p.mlmType = 'roberta-large';
p.lmType = 'deberta-large';
p.roiTypes = p.rparam.roiTypes;
p.dataTypes = {'testPerception','testImagery'};

%% load text generation results
res = mcap_summary_decoding_summarize(p);

%% draw figures for text generation results
mcap_summary_decoding_draw_generatedtext(res,p);
mcap_summary_decoding_draw_similarity(res,p);
mcap_summary_decoding_draw_idenacc(res,p);

%%
