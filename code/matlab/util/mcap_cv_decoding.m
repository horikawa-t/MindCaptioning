function mcap_cv_decoding(p)
%
% This script is written to perform cross-validation encoding analysis in
%  Horikawa, T. (2024) Mind captioning: Evolving descriptive text of mental
%  content from human brain activity. bioRxiv. 
%
% written by Tomoyasu Horikawa horikawa.t@gmail.com 2024/05/13
%

%% settinig
thmem = 50; % [GB]
dataType = 'trainPerception';
algType = 'l2';
warning off
save_log = 1;

% set parameters
eval(structout(p,'p'))

% create all condition list
vars = {modelTypes, rparam.roiTypes, sbjID};
C.cond_names = {'modelType', 'roiType','sbj'};
C.cond_list = generateCombinations(vars,'random');

%% start loop for multiple conditions
for cix = 1:length(C.cond_list)
    % set conditions
    for cixx = 1:length(C.cond_list{cix})
        eval(sprintf('%s = C.cond_list{cix}{cixx};',C.cond_names{cixx}))
    end
    % check condition
    switch modelType; case misc.decSkipModels; continue; end
    switch roiType; case rparam.cvDecSkipROITypes; continue; end
    
    % set data and feature params
    [featTypes, nlayers] = mcap_get_feature_params(rootPath,fparam,modelType);
    if isempty(featTypes); continue; end
    
    for fitr = randsample(1:nlayers,nlayers)%1:nlayers
        featType = featTypes{fitr};
        
        % set cv fold params
        cp = cparam.cv;
        nfolds = size(cp.run2FoldAssignIdx,2);
        
        % set algorithm params
        ap = aparam.(algType);
        lambda = logspace(ap.lowL,ap.highL,ap.nparamLogSearch);
        
        % load features
        fpath = sprintf([fileparts(fparam.feature_path_template),'/%s.mat'],rootPath,modelType,featType);
        L_org = load(fpath);
        if isempty(L_org.feat); continue; end
        
        % check if encoding result was saved
        suffix_summary_enc = sprintf('%s/%s/%s/',dataType,modelType,sbj);
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
            suffix_summary_tmp = sprintf('%s/%s/%s/',dataType,base_comp_models{ix},sbj);
            dataFnameEncPath = setPath2file(sprintf('%s/res/encoding/%s/res_summary.mat',rootPath,suffix_summary_tmp)); % res files
            if exist(dataFnameEncPath,'file')
                dataFnameEncPaths{ix} = dataFnameEncPath;
            else
                fprintf('Not yet prepared:%s\n',dataFnameEncPath)
                proceedflag = 0;
            end
        end
        if proceedflag == 0,continue,end
        
        % Save info. for summary results over all layers
        suffix_summary = sprintf('%s/%s/%s/%s/',dataType,modelType,sbj,roiType);
        saveFnameChk = setPath2file(sprintf('%s/%s/%s_log.txt',savdir,suffix_summary,featType)); % log files
        saveFname = setPath2file(sprintf('%s/%s/%s.mat',savdir,suffix_summary,featType)); % res files
        if checkModeRes, chkfile = saveFname; else, chkfile = saveFnameChk; end
        
        %% Start analyses
        fCheck = zeros(nfolds,1); % check variable
        for folditr = randsample(1:nfolds,nfolds)
            saveFnameChkx = setPath2file(sprintf('%s/%s/%s_cv%02d_log.txt',savdir,suffix_summary,featType,folditr)); % log files
            saveFnamex = setPath2file(sprintf('%s/%s/%s_cv%02d.mat',savdir,suffix_summary,featType,folditr)); % res files
            if checkModeRes,chkfilex = saveFnamex; else, chkfilex = saveFnameChkx;end
            
            if exist(chkfilex,'file')
                fCheck(folditr) = exist(saveFnamex,'file') > 0;
                if checkChkfile && del && ~exist(saveFnamex,'file') && exist(saveFnameChkx,'file')
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
                    fprintf('%s = %s\n',C.cond_names{cixx},merge(eval(C.cond_names{cixx}),2))
                end
                fprintf('==============================\n')
                
                % =======================
                % load data and get params
                fprintf('Load data...\n%s',sbj)
                dpath = sprintf('%s/preprocessed/%s_%s.mat',fmridir,dataType,sbj);
                [braindat_all, metainf_all, labels, unilabels, nStim, nVox, nSample] = load_data_wrapper(dpath, 'Condition',1);
                
                % load encoding accuracy and get acc by best
                % param in nested CV accuracy
                try
                    eres = load(saveFnameEnc,'res_nestcv','mxindcv');
                    cvEncodingAcc = eres.res_nestcv.profile;
                    mxindcv = eres.mxindcv;
                catch
                    fprintf('result variable needs to be saved.\nSkip: %s\n',saveFnameChkx)
                    dchk
                    break
                end
                cvEncodingAccBest = cvEncodingAcc{folditr}(mxindcv(folditr),:);
                clear eres cvEncodingAcc
                
                % remove language areas if WBnoLang
                switch roiType
                    case {'WBnoVis','WBnoSem'}
                        for ix = length(dataFnameEncPaths):-1:1
                            eres = load(dataFnameEncPaths{ix},'r_nestcv_best');
                            try cvEncodingAcc{ix} = max(eres.r_nestcv_best.profile{folditr},[],1);
                            catch, dchk, fprintf('result variable needs to be saved.\nSkip: %s\n',saveFnameChkx), break, end
                        end
                        % voxels better by comp model is removed
                        rmvVoxIdx = cvEncodingAcc{1} < cvEncodingAcc{2};
                        clear eres cvEncodingAcc
                    case 'WBnoLang'
                        rmvVoxIdx = getRoiVoxelIdx(metainf_all,rparam.language);
                    case 'Lang'
                        rmvVoxIdx = true(1,nVox);
                        langVoxIdx = getRoiVoxelIdx(metainf_all,rparam.language);
                        rmvVoxIdx(langVoxIdx) = 0;
                    otherwise
                        rmvVoxIdx = [];
                end
                braindat_all(:,rmvVoxIdx) = [];
                rmvfieldnames = {'xyz','volInds','roiind_value','voxind_all'};
                for rmitr = 1:length(rmvfieldnames)
                    metainf_all.(rmvfieldnames{rmitr})(:,rmvVoxIdx);
                end
                cvEncodingAccBest(:,rmvVoxIdx) = [];
                
                % copy features
                clear L
                L.feat = L_org.feat(labels,:);
                
                %% analysis loop
                fprintf('Start inner CV analysis for parameter determination: %d/%dCV:\n',folditr,nfolds)
                % get sample index for each cv
                inds_te = find(ismember(metainf_all.Run, cp.run2FoldAssignIdx(1,folditr):cp.run2FoldAssignIdx(2,folditr)));
                inds_tr = setdiff(1:nSample, inds_te);
                trainRunIdx = ~ismember(1:nfolds,folditr);
                
                % nested CV ===============================
                % voxel selection
                [s,o]= sort(cvEncodingAccBest,2,'descend');
                useVoxels = o(1:min(aparam.nSelectVoxels,length(o)));
                
                % prepare index for nested cv
                nestcvIdx = [cp.run2FoldAssignIdx(1,trainRunIdx);cp.run2FoldAssignIdx(2,trainRunIdx)];
                nestrunIdx = metainf_all.Run(inds_tr);
                
                ptmp.algType = algType;
                for pitr = ap.nparamLogSearch:-1:1
                    ptmp.lambda = lambda(pitr);
                    acc = cv_regression_anlaysis(braindat_all(inds_tr,useVoxels),L.feat(inds_tr,:),nestrunIdx,nestcvIdx,ptmp);
                    res_nestcv.profile(pitr,:) = acc.profile;
                    fprintf('innerCV[%dth fold:raw][lambda=%.5f]mean r = %.4f:%s\n',folditr,ptmp.lambda,mean(res_nestcv.profile(pitr,:)),tims(1))
                end
                clear ptmp
                % end nested CV analyses
                % ========================
                
                % res-start CV analyses (not nestedCV)========================
                fprintf('Parameter determination and compute final prediction\n')
                clear preds_folds true_folds norms
                % get best param index
                [mxval,mxindcv] = max(mean(res_nestcv.profile,2));
                
                % set params and perform regression
                ptmp.algType = algType;
                ptmp.lambda = lambda(mxindcv);
                [res_cv,preds_tmp,ynorms] = gen_regression_anlaysis(braindat_all(inds_tr,useVoxels),L.feat(inds_tr,:),braindat_all(inds_te,useVoxels),L.feat(inds_te,:),ptmp);
                
                % get normalized prediction
                norm_mode = 1;
                preds_folds = single(normalize_data(preds_tmp',norm_mode,ynorms)');
                trues_folds = single(normalize_data(L.feat(inds_te,:)',norm_mode,ynorms)');
                
                % preserve params
                norms.mu = ynorms.xmean;
                norms.sd = ynorms.xnorm;
                fprintf('outerCV[%dth fold][lambda=%.5f]mean r(profile) = %.4f, r(pattern) = %.4f; iden acc = %.2f%%:%s\n',...
                    folditr,ptmp.lambda,mean(res_cv.profile),mean(res_cv.pattern),mean(res_cv.iden_acc)*100,tims(1))
                
                % evaluate performance with normalized prediction
                [res_all.profile,res_all.pattern,res_all.iden_acc] = evaluate_accuracy(preds_folds,trues_folds);
                fprintf('All[best]mean r(profile) = %.4f, r(pattern) = %.4f; iden acc = %.2f%%:%s\n',...
                    mean(res_all.profile),mean(res_all.pattern),mean(res_all.iden_acc)*100,tims(1))
                
                % save resutls for one voxel group
                fprintf('Save:%s\n',saveFnamex)
                if save_log
                    if ~exist(saveFnamex,'file')
                        save(saveFnamex,'res_all','res_nestcv','res_cv','preds_folds','mxindcv','norms','-v7.3')
                    end
                end
                clear train_brain test_brain trues_folds preds_folds ptmp
            end
        end
        
        
        % summarize all layer results =================================
        if all(fCheck==1)&& ~exist(chkfile,'file') && ~del && integrateRes
            fprintf('Summarize all fold results [%s: %s: %s]\n',sbj,modelType,featType)
            saveChkfile(saveFnameChk) % save log file
            stopflag = 0;
            
            fprintf('Prepare true features...\n')
            dpath = sprintf('%s/preprocessed/%s_%s.mat',fmridir,dataType,sbj);
            [braindat_all, metainf_all, labels, unilabels, nStim, nVox, nSample] = load_data_wrapper(dpath, 'Condition',1);
            clear braindat_all metainf_all
            L.feat = L_org.feat(labels,:);
            L.norm = L_org.feat(labels,:);
            nFeature = size(L.feat,2);
            
            fprintf('Load predictions...\n')
            cntte = 0;
            preds_norm = zeros(nStim,nFeature,'single');
            preds_raw = zeros(nStim,nFeature,'single');
            mu = zeros(nfolds,nFeature,'single');
            sd = zeros(nfolds,nFeature,'single');
            runIdx = zeros(nStim,1,'single');
            for folditr = 1:nfolds
                dataFnamex = setPath2file(sprintf('%s/%s/%s_cv%02d.mat',savdir,suffix_summary,featType,folditr)); % res files
                try
                    r = load(dataFnamex,'preds_folds','norms');
                    nsamp_fold = size(r.preds_folds,1);
                    sampInds = (1:nsamp_fold)+cntte;
                    cntte = cntte + nsamp_fold;
                    
                    % preserve normalized and base pred/true features
                    preds_norm(sampInds,:) = r.preds_folds;
                    preds_raw(sampInds,:) = r.preds_folds.*repmat(r.norms.sd',nsamp_fold,1) + repmat(r.norms.mu',nsamp_fold,1);
                    L.norm(sampInds,:) = (L.feat(sampInds,:)-repmat(r.norms.mu',nsamp_fold,1))./repmat(r.norms.sd',nsamp_fold,1);
                    
                    runIdx(sampInds) = folditr;
                    mu(folditr,:) = r.norms.mu;
                    sd(folditr,:) = r.norms.sd;
                    
                catch
                    fprintf('Failed to load result data.\n')
                    delete(saveFnameChk)
                    stopflag = 1;
                    break
                end
            end
            if stopflag,continue,end
            
            % evaluate performance
            % raw
            [res_raw.profile,res_raw.pattern,res_raw.iden_acc] = evaluate_accuracy(preds_raw,L.feat);
            %fprintf('All[raw]mean r(profile) = %.4f, r(pattern) = %.4f; iden acc = %.2f%%:%s\n',...
            %    mean(res_raw.profile),mean(res_raw.pattern),mean(res_raw.iden_acc)*100,tims(1))
            % normalized
            [res_norm.profile,res_norm.pattern,res_norm.iden_acc] = evaluate_accuracy(preds_norm,L.norm);
            fprintf('All[norm]mean r(profile) = %.4f, r(pattern) = %.4f; iden acc = %.2f%%:%s\n',...
                mean(res_norm.profile),mean(res_norm.pattern),mean(res_norm.iden_acc)*100,tims(1))
            
            % save decoding score
            if ~exist(saveFname,'file')
                fprintf('%s\n',saveFname)
                save(saveFname,'preds_norm','preds_raw','labels','res_norm','res_raw','mu','sd','runIdx','-v7.3')
            end
        end
        tims
        clear saveFname*
    end
end
clear braindat_all metainf_all C

%%

