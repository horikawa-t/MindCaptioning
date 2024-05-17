function [profile, pattern, iden_acc] = evaluate_accuracy(pred,true,type)
% function evaluate_accuracy(pred,true,type)
%
% this function compute accuracy (profile, pattern, iden.acc.) of
% prediction using 'type' measure
%
% [Input]
%  -pred: prediction valus (N sample x D dimension)
%  -true: true values (N sample x D dimension)
%  -type: similarity metric (default = 'corr')
%
% [Input]
%  -pred: prediction valus
%  -true: true values
%  -type: similarity metric (default = 'corr')
%
% Written by Tomoyasu Horikawa 20231006
%
if ~exist('type','var') || isempty(type)
    type = 'corr';
end
switch type
    case 'corr'
        profile = single(fcorrdiag(pred,true));
        corrMat = fcorr(pred',true');
        pattern = single(diag(corrMat));
        iden_acc = single(fmcidentification(diag(corrMat),rmvDiag(corrMat),2,1));
    otherwise
        error('not yet implemented.')
end


