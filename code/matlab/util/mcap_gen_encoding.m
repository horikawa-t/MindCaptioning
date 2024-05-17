function mcap_gen_encoding(p)
%
% This script is written to perform cross-validation encoding analysis in
%  Horikawa, T. (2024) Mind captioning: Evolving descriptive text of mental
%  content from human brain activity. bioRxiv.
%
% written by Tomoyasu Horikawa horikawa.t@gmail.com 2024/05/13
%

%% settinig
warning off
save_log = 1;
thmem = 50; %[GB]
trainType = 'trainPerception';
testTypes = {'testPerception'}; % {'testPerception','testImagery'}
algType = 'l2';
warning off

% set parameters
eval(structout(p,'p'))

% create all condition list
% create all condition list
vars = {modelTypes, testTypes,  sbjID};
C.cond_names = {'modelType','testType','sbj'};
C.cond_list = generateCombinations(vars,'random');


%% start loop for multiple conditions
for cix = 1:length(C.cond_list)
    % set conditions
    for cixx = 1:length(C.cond_list{cix})
        eval(sprintf('%s = C.cond_list{cix}{cixx};',C.cond_names{cixx}))
    end
    
    % set data and feature params
    [featTypes, nlayers] = mcap_get_feature_params(rootPath,fparam,modelType);
    if isempty(featTypes); continue; end
    
    % crossvalidation params
    cp = cparam.cv;
    nfolds = size(cp.run2FoldAssignIdx,2);
    
    % set algorithm params
    ap = aparam.(algType);
    lambda = logspace(ap.lowL,ap.highL,ap.nparamLogSearch);
    
    % set save info. for summary results over all layers
    suffix_summary = sprintf('%s/%s/%s/',testType,modelType,sbj);
    saveFnameChk = setPath2file(sprintf('%s/%s/res_summary_log.txt',savdir,suffix_summary)); % log files
    saveFname = setPath2file(sprintf('%s/%s/res_summary.mat',savdir,suffix_summary)); % log files
    if checkModeRes, chkfile = saveFname; else, chkfile = saveFnameChk; end
    
    fCheck = zeros(nlayers,1); % check variable
    for fitr = randsample(1:nlayers,nlayers)%1:nlayers
        featType = featTypes{fitr};
        
        %% Save info. for final results
        saveFnameChkx = setPath2file(sprintf('%s/%s/%s_log.txt',savdir,suffix_summary,featType)); % log files
        saveFnamex = setPath2file(sprintf('%s/%s/%s.mat',savdir,suffix_summary,featType)); % res files
        if checkModeRes,chkfilex = saveFnamex; else, chkfilex = saveFnameChkx;end
        
        
        %% Start analyses
        if exist(chkfilex,'file')
            fCheck(fitr) = exist(saveFnamex,'file') > 0;
            if del && ~exist(saveFnamex,'file') && exist(saveFnameChkx,'file')
                fprintf('Delete: %s\n',saveFnameChkx)
                delete(saveFnameChkx)
            end
        elseif ~del
            % skip check to avoide memory over [1000000=GB]
            if getmemory < thmem*1000000; continue; end
            fprintf('Start:%s\n',saveFnameChkx)
            if save_log
                saveChkfile(saveFnameChkx) % save log file
            end
            fprintf('Condition ====================\n')
            for cixx = 1:length(C.cond_list{cix})
                fprintf('%s = %s\n',C.cond_names{cixx},merge(merge(eval(C.cond_names{cixx}),2),2))
            end
            fprintf('==============================\n')
            
            
            % =======================
            % load training data
            fprintf('Load training data...\n%s',sbj)
                dpath = sprintf('%s/preprocessed/%s_%s.mat',fmridir,trainType,sbj);
            [braindat_train, metainf_train, labels_tr, unilabels_tr, nStim_tr, nVox, nSample_tr] = load_data_wrapper(dpath, 'Condition',1);
            % load test data
            fprintf('Load test data...\n')
                dpath = sprintf('%s/preprocessed/%s_%s.mat',fmridir,testType,sbj);
            [braindat_test, metainf_test, labels_te, unilabels_te, nStim_te, nVox, nSample_te] = load_data_wrapper(dpath, 'Condition',1);
            
            % load features
            % train
            fpath = sprintf('%s/%s/video/%s.mat',featdir,modelType,featType);
            L_train = load(fpath);
            if isempty(L_train.feat); delete(saveFnameChkx); continue; end
            L_train.feat = L_train.feat(labels_tr,:);
            
            % test
            fpath = sprintf('%s/%s/video/%s.mat',featdir,modelType,featType);
            L_test = load(fpath);
            if isempty(L_test.feat); delete(saveFnameChkx); continue; end
            L_test.feat = L_test.feat(labels_te,:);
            
            %% analysis loop
            fprintf('Start CV analysis within training for parameter determination: %dCV:\n',nfolds)
            % ========================
            % start train CV analyses
            % set params and perform regression
            clear res_cv ptmp
            ptmp.algType = algType;
            for pitr = ap.nparamLogSearch:-1:1
                ptmp.lambda = lambda(pitr);
                acc = cv_regression_anlaysis(L_train.feat,braindat_train,metainf_train.Run,cp.run2FoldAssignIdx,ptmp);
                res_cv.profile(pitr,:) = acc.profile;
                res_cv.pattern(pitr,:) = acc.pattern;
                res_cv.iden_acc(pitr,:) = acc.iden_acc;
                fprintf('trainCV[lambda=%.5f]mean r(profile) = %.4f, r(pattern) = %.4f; iden acc = %.2f%%:%s\n',...
                    ptmp.lambda,mean(res_cv.profile(pitr,:)),mean(res_cv.pattern(pitr,:)),mean(res_cv.iden_acc(pitr,:))*100,tims(1))
            end
            clear ptmp
            % end train CV analyses
            
            % ========================
            
            % start GEN analyses ========================
            fprintf('Parameter determination and compute final prediction\n')
            clear norms
            % get best param index
            [mxval,mxindcv] = max(mean(res_cv.profile,2));
            
            % set params and perform regression
            ptmp.algType = algType;
            ptmp.lambda = lambda(mxindcv);
            [res_gen,preds_gen,ynorms] = gen_regression_anlaysis(L_train.feat,braindat_train,L_test.feat,braindat_test,ptmp);
            
            % get normalized prediction
            norm_mode = 1;
            preds_norm = single(normalize_data(preds_gen',norm_mode,ynorms)');
            trues_norm = single(normalize_data(braindat_test',norm_mode,ynorms)');
            
            % preserve params
            norms.mu = single(ynorms.xmean);
            norms.sd = single(ynorms.xnorm);
            fprintf('GEN[best:lambda=%.5f]mean r(profile) = %.4f, r(pattern) = %.4f; iden acc = %.2f%%:%s\n',...
                ptmp.lambda,mean(res_gen.profile),mean(res_gen.pattern),mean(res_gen.iden_acc)*100,tims(1))
            
            % evaluate performance with normalized prediction
            [res_norm.profile,res_norm.pattern,res_norm.iden_acc] = evaluate_accuracy(trues_norm,preds_norm);
            fprintf('GEN[best:lambda=%.5f]mean r(profile) = %.4f, r(pattern) = %.4f; iden acc = %.2f%%:%s\n',...
                ptmp.lambda,mean(res_norm.profile),mean(res_norm.pattern),mean(res_norm.iden_acc)*100,tims(1))
            % end GEN analyses
            % ========================
            
            % Check trial-averaged results ===========
            clear pred_raw pred_norm true_raw_rep true_norm_rep res_raw_rep res_norm_rep
            nRep = nSample_te/nStim_te;
            for uitr = nStim_te:-1:1
                idx = find(labels_te == unilabels_te(uitr));
                pred_raw(uitr,:) = mean(preds_gen(idx(1),:),1);
                pred_norm(uitr,:) = mean(preds_norm(idx(1),:),1);
                for repitr = nRep:-1:1
                    true_raw_rep{repitr}(uitr,:) = mean(braindat_test(idx(1:repitr),:),1);
                    true_norm_rep{repitr}(uitr,:) = mean(trues_norm(idx(1:repitr),:),1);
                end
            end
            
            % evaluation with identiication among test samples
            fprintf('Evaluation among test samples\n')
            for repitr = nRep:-1:1
                [res_raw_rep.profile{repitr},res_raw_rep.pattern{repitr},res_raw_rep.iden_acc{repitr}] = evaluate_accuracy(pred_raw,true_raw_rep{repitr});
                fprintf('GEN[raw :rep%d]mean r(profile) = %.4f, r(pattern) = %.4f; iden acc = %.2f%%:%s\n',...
                    repitr,mean(res_raw_rep.profile{repitr}),mean(res_raw_rep.pattern{repitr}),mean(res_raw_rep.iden_acc{repitr})*100,tims(1))
            end
            for repitr = nRep:-1:1
                [res_norm_rep.profile{repitr},res_norm_rep.pattern{repitr},res_norm_rep.iden_acc{repitr}] = evaluate_accuracy(pred_norm,true_norm_rep{repitr});
                fprintf('GEN[norm:rep%d]mean r(profile) = %.4f, r(pattern) = %.4f; iden acc = %.2f%%:%s\n',...
                    repitr,mean(res_norm_rep.profile{repitr}),mean(res_norm_rep.pattern{repitr}),mean(res_norm_rep.iden_acc{repitr})*100,tims(1))
            end
            
            % evaluation with identification beyond test samples (all ckmovie)
            fprintf('Evaluation among all candidate samples\n')
            for repitr = nRep:-1:1
                corrMat = fcorr(pred_raw',true_raw_rep{repitr}');
                corrMat_full = fcorr(pred_raw',braindat_train');
                res_raw_rep.iden_acc_full{repitr} = single(fmcidentification(diag(corrMat),[rmvDiag(corrMat),corrMat_full],2,1));
                fprintf('GEN[raw :rep%d] iden acc with full candidates = %.2f%%:%s\n',...
                    repitr,mean(res_raw_rep.iden_acc_full{repitr})*100,tims(1))
            end
            cands_norm = single(normalize_data(braindat_train',norm_mode,ynorms)');
            for repitr = nRep:-1:1
                corrMat = single(fcorr(pred_norm',true_norm_rep{repitr}'));
                corrMat_full = fcorr(pred_norm',cands_norm');
                res_norm_rep.iden_acc_full{repitr} = single(fmcidentification(diag(corrMat),[rmvDiag(corrMat),corrMat_full],2,1));
                fprintf('GEN[norm :rep%d] iden acc with full candidates = %.2f%%:%s\n',...
                    repitr,mean(res_norm_rep.iden_acc_full{repitr})*100,tims(1))
            end
            % ========================
            
            % prepare data for further analysis
            % norm params estimated by training data
            [normed,mu,sd] = zscore(braindat_train);
            labels = unilabels_te;
            
            % save resutls
            fprintf('Save:%s\n',saveFnamex)
            if save_log
                if ~exist(saveFnamex,'file')
                    save(saveFnamex,...
                        'res_norm','res_gen','res_cv',...
                        'preds_gen','preds_norm','true_raw_rep','true_norm_rep','mxindcv','norms',...
                        'res_norm_rep','res_raw_rep',...
                        'pred_raw','pred_norm','mu','sd','labels','-v7.3')
                end
            end
            clear braindat_train braindat_test ptmp trues_gen preds_gen res_norm res_gen res_cv
            clear feat_pred feat_pred pred_raw pred_norm
            
            tims
        end
    end
    
    % summarize all layer results =================================
    if all(fCheck==1) && ~exist(chkfile,'file') && ~del && integrateRes
        fprintf('Summarize all layer results [%s: %s]\n',sbj,modelType)
        saveChkfile(saveFnameChk) % save log file
        
        % prepare data
        dpath = sprintf('%s/preprocessed/%s_%s.mat',fmridir,testType,sbj);
        [braindat_test, metainf_test, labels_te, unilabels_te, nStim_te, nVox, nSample_te] = load_data_wrapper(dpath, 'Condition',1);
        nRep = nSample_te/nStim_te;
        
        % load all results
        clear r_gen r_cv_best r_optim dataFnames_all
        for fitr = nlayers:-1:1
            featType = featTypes{fitr};
            dataFnamex = setPath2file(sprintf('%s/%s/%s.mat',savdir,suffix_summary,featType)); % res files
            dataFnames_all{fitr} = dataFnamex;
            try
                r = load(dataFnamex,'res_norm_rep','res_cv','mxindcv');
            catch me
                fprintf('Failed to load:%s\n',dataFnamex)
                break
            end
            for repitr = nRep:-1:1
                r_gen.profile{repitr}(fitr,:) = r.res_norm_rep.profile{repitr};
                r_gen.pattern{repitr}(fitr,:) = r.res_norm_rep.pattern{repitr};
                r_gen.iden_acc{repitr}(fitr,:) = r.res_norm_rep.iden_acc{repitr};
            end
            r_cv_best.profile(fitr,:) = r.res_cv.profile(r.mxindcv,:);
            r_cv_best.pattern(fitr,:) = r.res_cv.pattern(r.mxindcv,:);
            r_cv_best.iden_acc(fitr,:) = r.res_cv.iden_acc(r.mxindcv,:);
            fprintf('trainCV[Layer%02d:mean] r(profile) = %.4f, r(pattern) = %.4f; iden acc = %.2f%%:%s\n',...
                fitr,mean(r_cv_best.profile(fitr,:)),mean(r_cv_best.pattern(fitr,:)),mean(r_cv_best.iden_acc(fitr,:))*100,tims(1))
        end
        
        % estimate best layers for each voxel
        [mx,r_cv_best.bestLayerIdx] = max(r_cv_best.profile,[],1);
        r_cv_best.bestLayerIdx = single(r_cv_best.bestLayerIdx);
        for repitr = length(r_gen.profile):-1:1
            [mx,r_gen.bestLayerIdx(repitr,:)] = max(r_gen.profile{repitr},[],1);
        end
        r_gen.bestLayerIdx = single(r_gen.bestLayerIdx);
        
        fprintf('Construct best predictions from all layers\n')
        clear best_preds best_preds_norm
        for repitr = nRep:-1:1
            best_preds{repitr} = zeros(nStim_te,nVox,'single');
            best_preds_norm{repitr} = zeros(nStim_te,nVox,'single');
        end
        for fitr = nlayers:-1:1
            fprintf('%d ',fitr)
            if fitr == 1
                load(dataFnames_all{fitr},'pred_raw','pred_norm','true_raw_rep','true_norm_rep');
            else
                load(dataFnames_all{fitr},'pred_raw','pred_norm');
            end
            
            % fill matrix by prediction of best layers for each repetition
            for repitr = nRep:-1:1
                voxInds = r_cv_best.bestLayerIdx == fitr;
                best_preds{repitr}(:,voxInds) = pred_raw(:,voxInds);
                best_preds_norm{repitr}(:,voxInds) = pred_norm(:,voxInds);
            end
        end
        fprintf('\n')
        
        % summarize final prediction
        for repitr = nRep:-1:1
            [r_optim.profile{repitr},r_optim.pattern{repitr},r_optim.iden_acc{repitr}] = evaluate_accuracy(true_raw_rep{repitr},best_preds{repitr});
            [r_optim_norm.profile{repitr},r_optim_norm.pattern{repitr},r_optim_norm.iden_acc{repitr}] = evaluate_accuracy(true_norm_rep{repitr},best_preds_norm{repitr});
            fprintf('GEN[%s][optim:raw:rep%d]mean r(profile) = %.4f, r(pattern) = %.4f; iden acc = %.2f%%:%s\n',...
                testType,repitr,mean(r_optim.profile{repitr}),mean(r_optim.pattern{repitr}),mean(r_optim.iden_acc{repitr})*100,tims(1))
            fprintf('GEN[%s][optim:norm:rep%d]mean r(profile) = %.4f, r(pattern) = %.4f; iden acc = %.2f%%:%s\n',...
                testType,repitr,mean(r_optim_norm.profile{repitr}),mean(r_optim_norm.pattern{repitr}),mean(r_optim_norm.iden_acc{repitr})*100,tims(1))
        end
        
        if ~exist(saveFname,'file')
            fprintf('Save:%s\n',saveFname);
            save(saveFname,'best_preds','best_preds_norm','r_optim','r_optim_norm','r_cv_best','r_gen','-v7.3')
        end
        
        % end summary =============================================
        
    end
    clear saveFname*
end
clear C

%%


