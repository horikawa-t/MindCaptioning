function mcap_gen_decoding(p)
%
% This script is written to perform cross-validation encoding analysis in
%  Horikawa, T. (2024) Mind captioning: Evolving descriptive text of mental
%  content from human brain activity. bioRxiv. 
%
% written by Tomoyasu Horikawa horikawa.t@gmail.com 2024/05/13
%

%% settinig
thmem = 50; % [GB]
trainType = 'trainPerception';
testTypes = {'testPerception','testImagery'};
algType = 'l2';
warning off
save_log = 1;

% set parameters
eval(structout(p,'p'))

% create all condition list
vars = {modelTypes, testTypes, rparam.roiTypes, sbjID};
C.cond_names = {'modelType','testType','roiType','sbj'};
C.cond_list = generateCombinations(vars,'random');


%% start loop for multiple conditions
for cix = 1:length(C.cond_list)
    % set conditions
    for cixx = 1:length(C.cond_list{cix})
        eval(sprintf('%s = C.cond_list{cix}{cixx};',C.cond_names{cixx}))
    end
    % check condition
    switch modelType; case misc.decSkipModels; continue; end
    switch roiType; case rparam.genDecSkipROITypes; continue; end
    
    % set data and feature params
    [featTypes, nlayers] = mcap_get_feature_params(rootPath,fparam,modelType);
    if isempty(featTypes); continue; end

    for fitr = randsample(1:nlayers,nlayers)%1:nlayersw
        featType = featTypes{fitr};
        
        % set cv fold params
        cp = cparam.cv;
        nfolds = size(cp.run2FoldAssignIdx,2);
        
        % set algorithm params
        ap = aparam.(algType);
        lambda = logspace(ap.lowL,ap.highL,ap.nparamLogSearch);
        
        % check if encoding result was saved
        suffix_summary_enc = sprintf('%s/%s/%s/',trainType,modelType,sbj);
        saveFnameEnc = setPath2file(sprintf('%s/res/encoding/%s/%s.mat',rootPath,suffix_summary_enc,featType)); % res files
        if ~exist(saveFnameEnc,'file')
            fprintf('Not yet prepared:%s\n',saveFnameEnc)
            continue
        end
        
        % check if encoding results of base and comp models are already summarized.
        switch roiType
            case 'WBnoVis'
                base_comp_models = {modelType,'timesformer'};
            case 'WBnoSem'
                base_comp_models = {'timesformer',modelType};
            otherwise
                base_comp_models = {};
        end
        proceedflag = 1;
        for ix = length(base_comp_models):-1:1
            suffix_summary_enc = sprintf('%s/%s/%s/',trainType,base_comp_models{ix},sbj);
            dataFnameEncPath = setPath2file(sprintf('%s/res/encoding/%s/res_summary.mat',rootPath,suffix_summary_enc)); % res files
            if exist(dataFnameEncPath,'file')
                dataFnameEncPaths{ix} = dataFnameEncPath;
            else
                fprintf('Not yet prepared:%s\n',dataFnameEncPath)
                proceedflag = 0;
            end
        end
        if proceedflag == 0,continue,end
        
        %% Save info. for final results
        suffix_summary = sprintf('%s/%s/%s/%s/',testType,modelType,sbj,roiType);
        saveFnameChk = setPath2file(sprintf('%s/%s/%s_log.txt',savdir,suffix_summary,featType)); % log files
        saveFname = setPath2file(sprintf('%s/%s/%s.mat',savdir,suffix_summary,featType)); % res files
        if checkModeRes, chkfile = saveFname; else, chkfile = saveFnameChk; end
        
        
        %% Start analyses
        if exist(chkfile,'file')
            if del && ~exist(saveFname,'file') && exist(saveFnameChk,'file')
                fprintf('Delete: %s\n',saveFnameChk)
                delete(saveFnameChk)
            end
        elseif ~del
            % skip check to avoide memory over [1000000=GB]
            if getmemory < thmem*1000000; continue; end
            fprintf('Start:%s\n',saveFnameChk)
            if save_log
                saveChkfile(saveFnameChk) % save log file
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
            
            % remove voxels within specified brain areas
            switch roiType
                case {'WBnoVis','WBnoSem'}
                    for ix = length(dataFnameEncPaths):-1:1
                        eres = load(dataFnameEncPaths{ix},'r_cv_best');
                        try cvEncodingAcc{ix} = max(eres.r_cv_best.profile,[],1);
                        catch, dchk, fprintf('result variable needs to be saved.\nSkip: %s\n',saveFnameChk), break, end
                    end
                    % voxels better by comp model is removed
                    rmvVoxIdx = cvEncodingAcc{1} < cvEncodingAcc{2};
                    clear eres cvEncodingAcc
                case 'WBnoLang'
                    rmvVoxIdx = getRoiVoxelIdx(metainf_train,rparam.language);
                case 'Lang'
                    rmvVoxIdx = true(1,nVox);
                    langVoxIdx = getRoiVoxelIdx(metainf_train,rparam.language);
                    rmvVoxIdx(langVoxIdx) = 0;
                otherwise
                    rmvVoxIdx = [];
            end
            % training
            braindat_train(:,rmvVoxIdx) = [];
            braindat_test(:,rmvVoxIdx) = [];
            rmvfieldnames = {'xyz','volInds','roiind_value','voxind_all'};
            for rmitr = 1:length(rmvfieldnames)
                metainf_train.(rmvfieldnames{rmitr})(:,rmvVoxIdx);
                metainf_test.(rmvfieldnames{rmitr})(:,rmvVoxIdx);
            end
            
            % load encoding accuracy and get acc by best param for voxel selection
            try eres = load(saveFnameEnc,'res_cv','res_nestcv','mxindcv');
            catch, dchk, fprintf('Encoding result variable needs to be saved.\nSkip: %s\n',saveFnameChk), break, end
            cvEncodingAccBest = cellmean(eres.res_cv.profile);
            % remove voxels within specified brain areas
            for folditr = nfolds:-1:1
                eres.res_nestcv.profile{folditr}(:,rmvVoxIdx) = [];
            end
            cvEncodingAccBest(rmvVoxIdx) = [];
            clear useVoxelsCV
            for folditr = nfolds:-1:1
                [s,o]= sort(eres.res_nestcv.profile{folditr}(eres.mxindcv(folditr),:),2,'descend');
                useVoxelsCV{folditr} = o(1:min(aparam.nSelectVoxels,length(o)));
            end
            [s,o]= sort(cvEncodingAccBest,1,'descend');
            useVoxelsGEN = o(1:min(aparam.nSelectVoxels,length(o)));
            clear eres
            
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
            ptmp.algType = algType;
            clear res_cv
            for pitr = ap.nparamLogSearch:-1:1
                ptmp.lambda = lambda(pitr);
                clear preds_cv trues_cv
                for folditr = nfolds:-1:1
                    % get sample index for each cv
                    inds_te = find(ismember(metainf_train.Run, cp.run2FoldAssignIdx(1,folditr):cp.run2FoldAssignIdx(2,folditr)));
                    inds_tr = setdiff(1:nSample_tr, inds_te);
                    % train CV ===============================
                    % perform regression after voxel selection
                    useVoxels = useVoxelsCV{folditr};
                    [acc,preds_tmp,ynorms] = gen_regression_anlaysis(braindat_train(inds_tr,useVoxels),L_train.feat(inds_tr,:),braindat_train(inds_te,useVoxels),L_train.feat(inds_te,:),ptmp);
                    
                    % get normalized prediction
                    norm_mode = 1;
                    preds_cv{folditr} = single(normalize_data(preds_tmp',norm_mode,ynorms)');
                    trues_cv{folditr} = single(normalize_data(L_train.feat(inds_te,:)',norm_mode,ynorms)');
                end
                % evaluate performance with normalized prediction
                preds_cv = merge(preds_cv,1);
                trues_cv = merge(trues_cv,1);
                [res_cv.profile(pitr,:),res_cv.pattern(pitr,:),res_cv.iden_acc(pitr,:)] = evaluate_accuracy(trues_cv,preds_cv);
                fprintf('trainCV[lambda=%.5f]mean r(profile) = %.4f, r(pattern) = %.4f; iden acc = %.2f%%:%s\n',...
                    ptmp.lambda,mean(res_cv.profile(pitr,:)),mean(res_cv.pattern(pitr,:)),mean(res_cv.iden_acc(pitr,:))*100,tims(1))
            end
            clear ptmp preds_tmp
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
            [res_gen,preds_gen,ynorms] = gen_regression_anlaysis(braindat_train(:,useVoxelsGEN),L_train.feat,braindat_test(:,useVoxelsGEN),L_test.feat,ptmp);
            
            % get normalized prediction
            norm_mode = 1;
            preds_norm = single(normalize_data(preds_gen',norm_mode,ynorms)');
            trues_norm = single(normalize_data(L_test.feat',norm_mode,ynorms)');
            
            % preserve params
            norms.mu = single(ynorms.xmean);
            norms.sd = single(ynorms.xnorm);
            %fprintf('GEN[raw:best:lambda=%.5f]mean r(profile) = %.4f, r(pattern) = %.4f; iden acc = %.2f%%:%s\n',...
            %    ptmp.lambda,mean(res_gen.profile),mean(res_gen.pattern),mean(res_gen.iden_acc)*100,tims(1))
            
            % evaluate performance with normalized prediction
            [res_norm.profile,res_norm.pattern,res_norm.iden_acc] = evaluate_accuracy(trues_norm,preds_norm);
            fprintf('GEN[norm:best:lambda=%.5f]mean r(profile) = %.4f, r(pattern) = %.4f; iden acc = %.2f%%:%s\n',...
                ptmp.lambda,mean(res_norm.profile),mean(res_norm.pattern),mean(res_norm.iden_acc)*100,tims(1))
            % end GEN analyses
            % ========================
            
            % Check trial-averaged results ===========
            clear pred_raw_rep pred_norm_rep true_raw true_norm res_raw_rep res_norm_rep
            nRep = nSample_te/nStim_te;
            for uitr = nStim_te:-1:1
                idx = find(labels_te == unilabels_te(uitr));
                true_raw(uitr,:) = L_test.feat(idx(1),:);
                true_norm(uitr,:) = trues_norm(idx(1),:);
                for repitr = nRep:-1:1
                    pred_raw_rep{repitr}(uitr,:) = mean(preds_gen(idx(1:repitr),:),1);
                    pred_norm_rep{repitr}(uitr,:) = mean(preds_norm(idx(1:repitr),:),1);
                end
            end
            
            % evaluation with identiication among test samples
            fprintf('Evaluation among test samples\n')
            for repitr = nRep:-1:1
                [res_raw_rep.profile{repitr},res_raw_rep.pattern{repitr},res_raw_rep.iden_acc{repitr}] = evaluate_accuracy(pred_raw_rep{repitr},true_raw);
                %fprintf('GEN[raw :rep%d]mean r(profile) = %.4f, r(pattern) = %.4f; iden acc = %.2f%%:%s\n',...
                %    repitr,mean(res_raw_rep.profile{repitr}),mean(res_raw_rep.pattern{repitr}),mean(res_raw_rep.iden_acc{repitr})*100,tims(1))
            end
            for repitr = nRep:-1:1
                [res_norm_rep.profile{repitr},res_norm_rep.pattern{repitr},res_norm_rep.iden_acc{repitr}] = evaluate_accuracy(pred_norm_rep{repitr},true_norm);
                fprintf('GEN[norm:rep%d]mean r(profile) = %.4f, r(pattern) = %.4f; iden acc = %.2f%%:%s\n',...
                    repitr,mean(res_norm_rep.profile{repitr}),mean(res_norm_rep.pattern{repitr}),mean(res_norm_rep.iden_acc{repitr})*100,tims(1))
            end
            
            % evaluation with identification beyond test samples (all ckmovie)
            fprintf('Evaluation among all candidate samples\n')
            for repitr = nRep:-1:1
                corrMat = fcorr(pred_raw_rep{repitr}',true_raw');
                corrMat_full = fcorr(pred_raw_rep{repitr}',L_train.feat');
                res_raw_rep.iden_acc_full{repitr} = single(fmcidentification(diag(corrMat),[rmvDiag(corrMat),corrMat_full],2,1));
                %fprintf('GEN[raw :rep%d] iden acc with full candidates = %.2f%%:%s\n',...
                %   repitr,mean(res_raw_rep.iden_acc_full{repitr})*100,tims(1))
            end
            cands_norm = single(normalize_data(L_train.feat',norm_mode,ynorms)');
            for repitr = nRep:-1:1
                corrMat = single(fcorr(pred_norm_rep{repitr}',true_norm'));
                corrMat_full = fcorr(pred_norm_rep{repitr}',cands_norm');
                res_norm_rep.iden_acc_full{repitr} = single(fmcidentification(diag(corrMat),[rmvDiag(corrMat),corrMat_full],2,1));
                fprintf('GEN[norm :rep%d] iden acc with full candidates = %.2f%%:%s\n',...
                    repitr,mean(res_norm_rep.iden_acc_full{repitr})*100,tims(1))
            end
            % ========================
            
            % prepare data for further analysis
            % norm params estimated by training data
            [normed,mu,sd] = zscore(L_train.feat); % get labels
            labels = unilabels_te;
            labels_org = labels_te;
            runIdx = ones(length(labels),1);
            feat_pred = pred_raw_rep{end};
            feat_pred_norm = pred_norm_rep{end};
            
            % save resutls
            fprintf('Save:%s\n',saveFname)
            if save_log
                if ~exist(saveFname,'file')
                    save(saveFname,...
                        'res_norm','res_gen','res_cv',...
                        'preds_gen','preds_norm','mxindcv','norms',...
                        'res_norm_rep','res_raw_rep',...
                        'feat_pred','feat_pred_norm','pred_raw_rep','pred_norm_rep','mu','sd','labels','labels_org','runIdx','-v7.3')
                end
            end
            clear braindat_train braindat_test ptmp trues_gen preds_gen res_norm res_gen res_cv
            clear feat_pred feat_pred pred_raw_rep pred_norm_rep labels_org
            
            tims
        end
        clear saveFname*
    end
end
clear C

%%


