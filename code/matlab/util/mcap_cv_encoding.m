function mcap_cv_encoding(p)
%
% This script is written to perform cross-validation encoding analysis in
%  Horikawa, T. (2024) Mind captioning: Evolving descriptive text of mental
%  content from human brain activity. bioRxiv. 
%
% written by Tomoyasu Horikawa horikawa.t@gmail.com 2024/05/13
%

%% settinig
thmem = 30; % [GB]
dataType = 'trainPerception';
algType = 'l2';
warning off

% set parameters
eval(structout(p,'p'))

% create all condition list
vars = {modelTypes, sbjID};
C.cond_names = {'modelType', 'sbj'};
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
    cp = cparam.cv;
    nfolds = size(cp.run2FoldAssignIdx,2);
    
    % set algorithm params
    ap = aparam.(algType);
    lambda = logspace(ap.lowL,ap.highL,ap.nparamLogSearch);
    
    % set save info. for summary results over all layers
    suffix_summary = sprintf('%s/%s/%s/',dataType,modelType,sbj);
    saveFnameChk = setPath2file(sprintf('%s/%s/res_summary_log.txt',savdir,suffix_summary)); % log files
    saveFname = setPath2file(sprintf('%s/%s/res_summary.mat',savdir,suffix_summary)); % log files
    if checkModeRes, chkfile = saveFname; else, chkfile = saveFnameChk; end
    
    for fitr = randsample(1:nlayers,nlayers)%1:nlayers
        featType = featTypes{fitr};
        
        % set save info. for each layer results
        saveFnameChkx = setPath2file(sprintf('%s/%s/%s_log.txt',savdir,suffix_summary,featType)); % log files
        saveFnamex = setPath2file(sprintf('%s/%s/%s.mat',savdir,suffix_summary,featType)); % res files
        
        %  Start analyses
        if checkModeRes,chkfilex = saveFnamex;else,chkfilex = saveFnameChkx;end
        if exist(chkfilex,'file')
            if checkChkfile && del && ~exist(saveFnamex,'file') && exist(saveFnameChkx,'file')
                fprintf('Delete: %s\n',saveFnameChkx)
                delete(saveFnameChkx)
            end
        elseif ~del
            % skip check to avoide memory over [1000000=GB]
            if getmemory < thmem*1000000; continue; end
            fprintf('Start:%s\n',saveFnameChkx)
            saveChkfile(saveFnameChkx) % save log file
            fprintf('Condition ====================\n')
            for cixx = 1:length(C.cond_list{cix})
                fprintf('%s = %s\n',C.cond_names{cixx},merge(eval(C.cond_names{cixx}),2))
            end
            fprintf('==============================\n')
            
            % load data and get params for all subjects
            fprintf('Load data...\n%s',sbj)
            try
                dpath = sprintf('%s/preprocessed/%s_%s.mat',fmridir,dataType,sbj);
                [braindat_all, metainf_all, labels, unilabels, nStim, nVox, nSample] = load_data_wrapper(dpath, 'Condition',1);
            catch
                delete(saveFnameChkx)
                break
            end
            
            % load features
            fpath = sprintf('%s/%s/video/%s.mat',featdir,modelType,featType);
            L = load(fpath);
            if isempty(L.feat); delete(saveFnameChkx); continue; end
            L.feat = L.feat(labels,:);            
            
            %% analysis loop
            fprintf('Start inner CV analysis for parameter determination\n')
            clear res_nestcv res_cv res_all
            for folditr = nfolds:-1:1
                % nested CV ===============================
                % get sample index for each cv
                inds_te = find(ismember(metainf_all.Run, cp.run2FoldAssignIdx(1,folditr):cp.run2FoldAssignIdx(2,folditr)));
                inds_tr = setdiff(1:nSample, inds_te);
                trainRunIdx = ~ismember(1:nfolds,folditr);
                % prepare index for nested cv
                nestcvIdx = [cp.run2FoldAssignIdx(1,trainRunIdx);cp.run2FoldAssignIdx(2,trainRunIdx)];
                nestrunIdx = metainf_all.Run(inds_tr);
                
                ptmp.algType = algType;
                for pitr = ap.nparamLogSearch:-1:1
                    ptmp.lambda = lambda(pitr);
                    acc = cv_regression_anlaysis(L.feat(inds_tr,:),braindat_all(inds_tr,:),nestrunIdx,nestcvIdx,ptmp);
                    res_nestcv.profile{folditr}(pitr,:) = acc.profile;
                    fprintf('innerCV[%dth fold][lambda=%.5f]mean r = %.4f:%s\n',folditr,ptmp.lambda,mean(res_nestcv.profile{folditr}(pitr,:)),tims(1))
                end
                clear ptmp
                % end nested CV analyses
                % ========================
            end
            
            fprintf('Parameter determination and compute final prediction\n')
            clear preds_folds true_folds
            for folditr = nfolds:-1:1
                % get best param index
                [mxval,mxindcv(folditr)] = max(mean(res_nestcv.profile{folditr},2));
                
                % get sample index
                inds_te = find(ismember(metainf_all.Run, cp.run2FoldAssignIdx(1,folditr):cp.run2FoldAssignIdx(2,folditr)));
                inds_tr = setdiff(1:nSample, inds_te);
                trainRunIdx = ~ismember(1:nfolds,folditr);
                
                % set params and perform regression
                ptmp.algType = algType;
                ptmp.lambda = lambda(mxindcv(folditr));
                [acc,preds_tmp,ynorms] = gen_regression_anlaysis(L.feat(inds_tr,:),braindat_all(inds_tr,:),L.feat(inds_te,:),braindat_all(inds_te,:),ptmp);
                
                % get normalized prediction
                norm_mode = 1;
                preds_folds{folditr} = single(normalize_data(preds_tmp',norm_mode,ynorms))';
                trues_folds{folditr} = single(normalize_data(braindat_all(inds_te,:)',norm_mode,ynorms))';
                
                % preserve params and results
                norms.mu{folditr} = ynorms.xmean;
                norms.sd{folditr} = ynorms.xnorm;
                res_cv.profile{folditr} = acc.profile;
                res_cv.pattern{folditr} = acc.pattern;
                res_cv.iden_acc{folditr} = acc.iden_acc;
                fprintf('outerCV[%dth fold][lambda=%.5f]mean r(profile) = %.4f, r(pattern) = %.4f; iden acc = %.2f%%:%s\n',...
                    folditr,ptmp.lambda,mean(res_cv.profile{folditr}),mean(res_cv.pattern{folditr}),mean(res_cv.iden_acc{folditr})*100,tims(1))
                
                clear ptmp
            end
            % evaluate performance with normalized prediction
            preds_norm = merge(preds_folds,1);
            trues_norm = merge(trues_folds,1);
            [res_all.profile,res_all.pattern,res_all.iden_acc] = evaluate_accuracy(preds_norm,trues_norm);
            fprintf('All[best]mean r(profile) = %.4f, r(pattern) = %.4f; iden acc = %.2f%%:%s\n',...
                mean(res_all.profile),mean(res_all.pattern),mean(res_all.iden_acc)*100,tims(1))
            
            try
                if ~exist(saveFnamex,'file')
                    save(saveFnamex,'res_all','res_nestcv','res_cv','preds_folds','mxindcv','norms','-v7.3')
                end
            catch me
                me
                delete(saveFnameChkx)
            end
            tims
            clear res preds_folds trues_folds norms preds_norm trues_norm
            clear res_nestcv res_cv res_all
        end
    end
    
    % check results
    fCheck = zeros(nlayers,1); % check variable
    for fitr = randsample(1:nlayers,nlayers)%1:nlayers
        featType = featTypes{fitr};
        % set save info. for each layer results
        saveFnameChkx = setPath2file(sprintf('%s/%s/%s_log.txt',savdir,suffix_summary,featType)); % log files
        saveFnamex = setPath2file(sprintf('%s/%s/%s.mat',savdir,suffix_summary,featType)); % res files
        %  Start analyses
        if checkModeRes,chkfilex = saveFnamex;else,chkfilex = saveFnameChkx;end
        if exist(chkfilex,'file')
            fCheck(fitr) = exist(saveFnamex,'file') > 0;
        end
    end
        
    % summarize all layer results =================================
    if all(fCheck==1) && ~exist(chkfile,'file') && ~del && integrateRes
        fprintf('Summarize all layer results [%s: %s]\n',sbj,modelType)
        saveChkfile(saveFnameChk) % save log file
        
        % load all results
        clear r_cv_best r_nestcv_best r_optim dataFnames_all
        for fitr = nlayers:-1:1
            featType = featTypes{fitr};
            dataFnamex = setPath2file(sprintf('%s/%s/%s.mat',savdir,suffix_summary,featType)); % res files
            dataFnames_all{fitr} = dataFnamex;
            try
                r = load(dataFnamex,'res_all','res_nestcv','mxindcv');
            catch me
                fprintf('Failed to load:%s\n',dataFnamex)
                break
            end
            r_cv_best.profile(fitr,:) = r.res_all.profile;
            r_cv_best.pattern(fitr,:) = r.res_all.pattern;
            r_cv_best.iden_acc(fitr,:) = r.res_all.iden_acc;
            fprintf('Layer%02d:mean r(profile) = %.4f, r(pattern) = %.4f; iden acc = %.2f%%:%s\n',...
                fitr,mean(r_cv_best.profile(fitr,:)),mean(r_cv_best.pattern(fitr,:)),mean(r_cv_best.iden_acc(fitr,:))*100,tims(1))
            
            % preserve best nested cv accuracy for each fold
            for folditr = nfolds:-1:1
                r_nestcv_best.profile{folditr}(fitr,:) = r.res_nestcv.profile{folditr}(r.mxindcv(folditr),:);
            end
        end
        % estimate best layers for each voxel in each folds
        for folditr = nfolds:-1:1
            [mx,r_cv_best.bestLayerIdx(folditr,:)] = max(r_nestcv_best.profile{folditr},[],1);
        end
        r_cv_best.bestLayerIdx = single(r_cv_best.bestLayerIdx);
        
        fprintf('Construct best predictions from all layers\n')
        dpath = sprintf('%s/preprocessed/%s_%s.mat',fmridir,dataType,sbj);
        [braindat_all, metainf_all, labels, unilabels, nStim, nVox, nSample] = load_data_wrapper(dpath, 'Condition',1);
        best_preds = zeros(size(braindat_all),'single');
        for fitr = nlayers:-1:1
            fprintf('%d ',fitr)
            if fitr == 1
                load(dataFnames_all{fitr},'preds_folds','norms');
            else
                load(dataFnames_all{fitr},'preds_folds');
            end
            % fill matrix by prediction of best layers for each fold
            cntte = 0;
            for folditr = 1:nfolds
                nsamp_fold = size(preds_folds{folditr},1);
                sampInds = (1:nsamp_fold)+cntte;
                voxInds = r_cv_best.bestLayerIdx(folditr,:) == fitr;
                best_preds(sampInds,voxInds) = preds_folds{folditr}(:,voxInds);
                cntte = cntte + nsamp_fold;
                
                % normalize measured brain data
                if fitr == 1
                    braindat_all(sampInds,:) = (braindat_all(sampInds,:)-repmat(norms.mu{folditr}',nsamp_fold,1))./repmat(norms.sd{folditr}',nsamp_fold,1);
                end
            end
        end
        
        % summarize final prediction
        [r_optim.profile,r_optim.pattern,r_optim.iden_acc] = evaluate_accuracy(braindat_all,best_preds);
        fprintf('\nAll[optim]mean r(profile) = %.4f, r(pattern) = %.4f; iden acc = %.2f%%:%s\n',...
            mean(r_optim.profile),mean(r_optim.pattern),mean(r_optim.iden_acc)*100,tims(1))

        if ~exist(saveFname,'file')
            fprintf('Save:%s\n',saveFname);
            save(saveFname,'best_preds','r_optim','r_cv_best','r_nestcv_best','-v7.3')
        end
        
        clear braindat_all best_preds  r_cv_best r_optim
        % end summary =============================================
        
    end
    clear saveFname*
end
%%
warning on
done

