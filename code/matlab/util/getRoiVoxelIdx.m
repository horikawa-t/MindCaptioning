function [voxidx] = getRoiVoxelIdx(metainf,roiset,hemitype)
% function [voxidx] = getRoiVoxelIdx(metainf,roiset,hemitype)
%
% this function get voxel indices specified by roiset and hemitype
%
% [Input]
%  -metainf: metainf
%     -roiname: roiname
%     -roiind_value: roi2voxel index
%  -roiset: roiset used to detect index
%  -hemitype: hemisphere type (default = 'both'; 'right','left')
%
% [Output]
%  -voxidx: voxel index
%
% Written by Tomoyasu Horikawa 20231219
%
if ~exist('hemitype','var') || isempty(hemitype)
    hemitype = 'both';
end

switch hemitype
    case 'both'
        prefixs = {'localizer_r_rh','localizer_r_lh'};
    case 'right'
        prefixs = {'localizer_r_rh'};
    case 'left'
        prefixs = {'localizer_r_lh'};
    otherwise
        error('invalid hemitype')
end
roinames = cell(length(prefixs),length(roiset));
for hemitr = 1:length(prefixs)
    prefix = prefixs{hemitr};
    for roitr = 1:length(roiset)
        roi = roiset{roitr};
        roinames{hemitr,roitr} = [prefix,'.',roi];
    end
end

roi_Idx = ismember(metainf.roiname,roinames(:));
voxidx = any(metainf.roiind_value(roi_Idx,:),1);

