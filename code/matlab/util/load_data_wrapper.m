function [braindat, metainf, labels, unilabels, nStim, nVox, nSample] = load_data_wrapper(dpath, label_type, verbose)
% LOAD_DATA_WRAPPER Loads brain data and associated metadata.
%
% [braindat, metainf, labels, unilabels, nStim, nVox, nSample] = load_data_wrapper(dpath, label_type)
%
% This wrapper function loads brain data and associated metadata from the specified data path (dpath).
%
% Input:
%   - dpath: Data path where the brain data and metadata are stored.
%   - label_type: Label type used to construct labels (optional).
%
% Output:
%   - braindat: Brain data.
%   - metainf: Metadata information for brain data.
%   - labels: Labels corresponding to the specified label type (if provided).
%   - unilabels: Unique labels present in the data (if labels are provided).
%   - nStim: Number of unique labels (if labels are provided).
%   - nVox: Number of voxels (features) in the brain data.
%   - nSample: Number of samples in the brain data.
%
% Written by Tomoyasu Horikawa on 2023-10-02.
%
% Example usage:
% [braindat, metainf, labels, unilabels, nStim, nVox, nSample] = load_data_wrapper('data_path', 'label_type');

d_tmp = load(dpath,'braindat','metainf');
braindat = d_tmp.braindat;
metainf = d_tmp.metainf;
clear d_tmp

% get label information from each subject
if nargin > 1
    stimIDidx = ismember(metainf.label_type,{label_type});
    labels = metainf.Label(:,stimIDidx);
    unilabels = unique(labels); % get unique labels
    nStim = length(unilabels);
    nVox = size(braindat,2);
    nSample = size(braindat,1);
end

if exist('verbose','var') && verbose
    tims
end