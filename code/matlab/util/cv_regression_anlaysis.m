function [acc, pred] = cv_regression_anlaysis(Xdat,Ydat,runIdx,cvIdx,p)
% cv_regression_anlaysis Perform cross-validation regression analysis.
%
%   acc = cv_regression_anlaysis(Xdat, Ydat, runIdx, cvIdx, p) performs
%   cross-validation regression analysis on the input data Xdat and Ydat
%   using the specified cross-validation indices runIdx and cvIdx, and
%   parameters defined in the structure p.
%
%   Input arguments:
%   - Xdat: The input data (e.g., brain data).
%   - Ydat: The target data (e.g., scores or labels).
%   - runIdx: The run indices for cross-validation.
%   - cvIdx: The cross-validation indices.
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
inds_all = 1:size(Xdat,1);
nfolds = size(cvIdx,2);

for ixx = nfolds:-1:1
    % separate training and test data
    % Get test and training index
    inds_te = find(ismember(runIdx, cvIdx(1, ixx):cvIdx(2, ixx)));
    inds_tr = setdiff(inds_all, inds_te);
    
    % Extract training and testing data
    Xtr = Xdat(inds_tr, :);
    Xte = Xdat(inds_te, :);
    Ytr = Ydat(inds_tr, :);
    
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
    
    switch p.algType
        case 'l2'
            % perform l2 regression
            % training & test
            nsamp_test = size(Xte,2);
            % add intercept
            Xtr = addBias(Xtr');
            % training & test
            res{ixx} = flinreg_l2(p.lambda,Xtr'*Xtr,Xtr,Ytr',Xte')';
            res{ixx} = res{ixx}.*repmat(yparm.xnorm,1,nsamp_test) + repmat(yparm.xmean,1,nsamp_test);
    end
end
% summary
pred = merge(res,2)';
[acc.profile,acc.pattern,acc.iden_acc] = evaluate_accuracy(pred,Ydat);

%%