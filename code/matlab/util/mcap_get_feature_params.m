function [featTypes, nlayers] = mcap_get_feature_params(rootPath,fparam,modelType)
%
% this function get feature parameters under specified path.
%
% [Input]
%   -fparam.feature_path_template: template path string
%    e.g., feature_path_template = '%s/data/feature/%s/video/layer*.mat';
%   -rootPath: framework of the model (e.g., pytorch)
%   -modelType: model name (e.g., deberta-large)
%
% [Output]
%  -featTypes: feautres types of the model (e.g., {'layer1', ..., 'layer12'})
%  -nlayers: number of layers
%
% Written by Tomoyasu Horikawa 20240515
%
featnames = dir(sprintf(fparam.feature_path_template,rootPath,modelType));
if isempty(featnames)
    warning('No features found in specified path.')
    featTypes = []; 
    nlayers = [];
    return
end
for i = length(featnames):-1:1
    [fpath,featTypes{i},ext] = fileparts(featnames(i).name);
end
nlayers = length(featTypes);
