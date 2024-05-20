function p = mcap_setParams(p)
%
% This script is used to set parameters
%
%
% written by Tomoyasu Horikawa horikawa.t@gmail.com 2024/05/13
%

%% analyses setting
% misc setting
p.del = 0; % set 1 if you want to clean log files (delete log files without res files)
p.checkChkfile = 0; % if 0 skip file check (fast) .
p.checkModeRes = 0; % if 1 check by result file, if 0 check by log file.
p.integrateRes = 1; % if 1 integrate results across nparse

%% set path information
% set directories
p.savdir = setdir(sprintf('%s/res/%s/',p.rootPath,p.analysisType)); % analysis results
p.figdir = setdir(sprintf('%s/fig/%s/',p.rootPath,p.analysisType)); % result figures

p.fmridir = setdir(sprintf('%s/data/fmri/',p.rootPath));
p.featdir = setdir(sprintf('%s/data/feature/',p.rootPath));

%% data information
% subject info
p.sbjID = {'S1','S2','S3','S4','S5','S6'};

%% label group types
p.modelTypes = {'deberta-large','timesformer'};

%% analysis parameters
% cross-validation parameters
p.cparam.cv.run2FoldAssignIdx = [1,11,21,31,41,51;10,20,30,40,50,58];

% reguralization paramters settings
%[10,1,6] are the actual paramters used in our manuscript
p.aparam.l2.nparamLogSearch = 10; %  You can reduce computation time by setting smaller # of exploration parameters.
p.aparam.l2.lowL = 1;
p.aparam.l2.highL = 6;

do_demo = 0; % set 0, if you want to completely reproduce the manuscript results 
if do_demo
    %[4,4,5] can be used for demo [faster but with minor difference from our results]
    p.aparam.l2.nparamLogSearch = 4;
    p.aparam.l2.lowL = 4;
    p.aparam.l2.highL = 5;
end

% decoding analysis parameters
p.aparam.nSelectVoxels = 50000; % # of selected voxels (n = 50000 in our study). You can reduce computation time by setting smaller # of voxels (e.g., 5000) here.

% ROI settings
p.rparam.roiTypes = {'WB','WBnoLang','WBnoSem','WBnoVis','Lang'};
p.rparam.cvDecSkipROITypes = {'WBnoLang','WBnoSem','WBnoVis','Lang'};
p.rparam.genDecSkipROITypes = {};
p.rparam.language = {'temporal_language','frontal_language'};

% feature types
p.fparam.feature_path_template = '%s/data/feature/video/%s/layer*.mat'; % 

% misc
p.misc.decSkipModels = {'timesformer'};
