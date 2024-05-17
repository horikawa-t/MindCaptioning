function [acc, pred, yparm] = gen_regression_anlaysis(Xtr,Ytr,Xte,Yte,p)
% gen_regression_anlaysis Perform generalization regression analysis.
%
%   acc = gen_regression_anlaysis(Xtr,Ytr,Xte,Yte,p) performs
%   generalization regression analysis on the input data Xtr and Ytr
%   using parameters defined in the structure p.
%
%   Input arguments:
%   - Xtr: The training X data (e.g., brain data).
%   - Xte: The test X data (e.g., brain data).
%   - Ytr: The training Y data (e.g., brain data).
%   - Yte: The test Y data (e.g., brain data).
%   - p: A structure containing optional parameters.
%     - p.algType (optional): Algorithm type ('l2' by default).
%     - p.lambda (optional): Regularization parameter (0 by default).
%
%   Output:
%   - acc: Prediction accuracy as a correlation coefficient and identification accuracy.
%   - pred: Prediction.
%
%   [note]
%   - Prediction and their accuracy were evaluated in non-normalized space.
%   
%   See also: NORMALIZE_DATA, ADDBIAS, FLINREG_L2.

if ~isfield(p,'algType')
    p.algType = 'l2';
end
if strcmp(p.algType,'l2') && ~isfield(p,'lambda')
    p.lambda = 0;
end

% normalization
% regressor normalization
parm.norm_mode = 1;
[Xtr,xparm] = normalize_data(Xtr',parm.norm_mode);
Xte = normalize_data(Xte',parm.norm_mode,xparm);

% target normalization
[Ytr,yparm] = normalize_data(Ytr',parm.norm_mode);

% convert to single to reduce file size
Xtr = single(Xtr);
Xte = single(Xte);
Ytr = single(Ytr);
Yte = single(Yte);

switch p.algType
    case 'l2'
        % perform l2 regression
        % training & test
        nsamp_test = size(Xte,2);
        % add intercept
        Xtr = addBias(Xtr');
        % training & test
        pred = flinreg_l2(p.lambda,Xtr'*Xtr,Xtr,Ytr',Xte');
        pred = pred'.*repmat(yparm.xnorm,1,nsamp_test) + repmat(yparm.xmean,1,nsamp_test);
end
% summary
pred = pred';
[acc.profile,acc.pattern,acc.iden_acc] = evaluate_accuracy(pred,Yte);

%%