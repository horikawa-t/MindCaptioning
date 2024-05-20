function [res_enc,p] = mcap_summary_encoding_summarize(p)

%%  set parameters
eval(structout(p,'p'))

% set ROI params
p.roiSets = { % {'roiname',{used set},{exclude set}}
    {'V1',{'localizer_r_lh.V1d','localizer_r_lh.V1v','localizer_r_rh.V1d','localizer_r_rh.V1v'},{}}
    {'V2',{'localizer_r_lh.V2d','localizer_r_lh.V2v','localizer_r_rh.V2d','localizer_r_rh.V2v'},{}}
    {'V3',{'localizer_r_lh.V3d','localizer_r_lh.V3v','localizer_r_rh.V3d','localizer_r_rh.V3v'},{}}
    {'V3AB',{'localizer_r_lh.V3A','localizer_r_rh.V3A','localizer_r_lh.V3B','localizer_r_rh.V3B'},{}}
    {'V7',{'localizer_r_lh.V7','localizer_r_rh.V7'},{}}
    {'V4',{'localizer_r_lh.hV4','localizer_r_rh.hV4'},{}}
    {'LO1',{'localizer_r_lh.LO1','localizer_r_rh.LO1'},{}}
    {'LO2',{'localizer_r_lh.LO2','localizer_r_rh.LO2'},{}}
    {'FFA1',{'localizer_r_lh.FFA1','localizer_r_rh.FFA1'},{}}
    {'FFA2',{'localizer_r_lh.FFA2','localizer_r_rh.FFA2'},{}}
    {'OFA',{'localizer_r_lh.OFA','localizer_r_rh.OFA'},{}}
    {'fSTS',{'localizer_r_lh.fSTS','localizer_r_rh.fSTS'},{}}
    {'Face',{'localizer_r_lh.FFA1','localizer_r_lh.FFA2','localizer_r_lh.OFA','localizer_r_lh.fSTS','localizer_r_rh.FFA1','localizer_r_rh.FFA2','localizer_r_rh.OFA','localizer_r_rh.fSTS'},{}}
    {'EBA',{'localizer_r_lh.EBA','localizer_r_rh.EBA'},{}}
    {'FBA',{'localizer_r_lh.FBA','localizer_r_rh.FBA'},{}}
    {'Body',{'localizer_r_lh.EBA','localizer_r_lh.FBA','localizer_r_rh.EBA','localizer_r_rh.FBA'},{}}
    {'Object',{'localizer_r_lh.LOC','localizer_r_rh.LOC'},{}}
    {'Motion',{'localizer_r_lh.hMT','localizer_r_rh.hMT'},{}}
    {'VWFA1',{'localizer_r_lh.VWFA1','localizer_r_rh.VWFA1'},{}}
    {'VWFA2',{'localizer_r_lh.VWFA2','localizer_r_rh.VWFA2'},{}}
    {'VWFA12',{'localizer_r_lh.VWFA1','localizer_r_lh.VWFA2','localizer_r_rh.VWFA1','localizer_r_rh.VWFA2'},{}}
    {'OWFA',{'localizer_r_lh.OWFA','localizer_r_rh.OWFA'},{}}
    {'Word',{'localizer_r_lh.VWFA1','localizer_r_lh.VWFA2','localizer_r_lh.OWFA','localizer_r_rh.VWFA1','localizer_r_rh.VWFA2','localizer_r_rh.OWFA'},{}}
    {'PPA',{'localizer_r_lh.PPA','localizer_r_rh.PPA'},{}}
    {'MPA',{'localizer_r_lh.MPA','localizer_r_rh.OPA'},{}}
    {'OPA',{'localizer_r_lh.OPA','localizer_r_rh.MPA'},{}}
    {'Place',{'localizer_r_lh.PPA','localizer_r_lh.OPA','localizer_r_lh.MPA','localizer_r_rh.PPA','localizer_r_rh.OPA','localizer_r_rh.MPA'},{}}
    {'Auditory',{'localizer_r_lh.AC','localizer_r_rh.AC'},{}}
    {'tLang',{'localizer_r_lh.temporal_language','localizer_r_rh.temporal_language'},{}}
    {'fLang',{'localizer_r_lh.frontal_language','localizer_r_rh.frontal_language'},{}}
    {'Language',{'localizer_r_lh.temporal_language','localizer_r_lh.frontal_language','localizer_r_rh.temporal_language','localizer_r_rh.frontal_language'},{}}
    {'lTOM',{'localizer_r_lh.lateral_TOM','localizer_r_rh.lateral_TOM'},{}}
    {'mTOM',{'localizer_r_lh.medial_TOM','localizer_r_rh.medial_TOM'},{}}
    {'TOM',{'localizer_r_lh.lateral_TOM','localizer_r_lh.medial_TOM','localizer_r_rh.lateral_TOM','localizer_r_rh.medial_TOM'},{}}
    {'LVC',{'localizer_r_lh.V1d','localizer_r_lh.V1v','localizer_r_rh.V1d','localizer_r_rh.V1v',...
    'localizer_r_lh.V2d','localizer_r_lh.V2v','localizer_r_rh.V2d','localizer_r_rh.V2v',...
    'localizer_r_lh.V3d','localizer_r_lh.V3v','localizer_r_rh.V3d','localizer_r_rh.V3v'},{}}
    {'HVC',{'localizer_r_lh.FFA1','localizer_r_lh.FFA2','localizer_r_lh.OFA','localizer_r_lh.fSTS','localizer_r_rh.FFA1','localizer_r_rh.FFA2','localizer_r_rh.OFA','localizer_r_rh.fSTS',...
    'localizer_r_lh.EBA','localizer_r_lh.FBA','localizer_r_rh.EBA','localizer_r_rh.FBA',...
    'localizer_r_lh.LOC','localizer_r_rh.LOC',...
    'localizer_r_lh.hMT','localizer_r_rh.hMT',...
    'localizer_r_lh.VWFA1','localizer_r_lh.VWFA2','localizer_r_lh.OWFA','localizer_r_rh.VWFA1','localizer_r_rh.VWFA2','localizer_r_rh.OWFA',...
    'localizer_r_lh.PPA','localizer_r_lh.OPA','localizer_r_lh.MPA','localizer_r_rh.PPA','localizer_r_rh.OPA','localizer_r_rh.MPA'},...
    {'localizer_r_lh.V1d','localizer_r_lh.V1v','localizer_r_rh.V1d','localizer_r_rh.V1v',...
    'localizer_r_lh.V2d','localizer_r_lh.V2v','localizer_r_rh.V2d','localizer_r_rh.V2v',...
    'localizer_r_lh.V3d','localizer_r_lh.V3v','localizer_r_rh.V3d','localizer_r_rh.V3v'}}
    {'WB',{},{}}
    };

dataType = 'trainPerception';
vars = {modelTypes};
C.cond_names = {'modelType'};
C.cond_list = generateCombinations(vars);

%% load results
for cix = 1:length(C.cond_list)
    % set conditions
    for cixx = 1:length(C.cond_list{cix})
        eval(sprintf('%s = C.cond_list{cix}{cixx};',C.cond_names{cixx}))
    end
    modelName = strrep(modelType,'-','_');
    
    saveFname = sprintf('%s/%s_summary.mat',savdir,modelType);
    setdir(fileparts(saveFname));
    
    if exist(saveFname,'file')
        fprintf('Load:%s\n',saveFname)
        load(saveFname,'res');
        res_enc.(modelName) = res;
    else
        clear res
        fprintf('Start:%s\n',saveFname)
        for sbjitr = length(sbjID):-1:1
            sbj = sbjID{sbjitr};
            
            % load data and get params
            clear metainf
            switch dataType
                case 'trainPerception'
                    dpath = sprintf('%s/preprocessed/%s_%s.mat',fmridir,dataType,sbj);
                    load(dpath, 'metainf');
                    fname = sprintf('%s/%s/%s/%s/res_summary.mat',savdir,dataType,modelType,sbj);
                    eval_scores = {'r_optim','r_cv_best'};
                otherwise
                    continue
            end
            tmp = load(fname,eval_scores{:});
            
            for eitr = 1:length(eval_scores)
                eval_score = eval_scores{eitr};
                if ~isfield(tmp,eval_score)
                    continue
                end
                switch eval_score
                    case {'r_optim'}
                        [res.(eval_score).profile.ci(:,sbjitr),res.(eval_score).profile.mu(:,sbjitr)] = ciestim3(tmp.(eval_score).profile);
                        [res.(eval_score).pattern.ci(:,sbjitr),res.(eval_score).pattern.mu(:,sbjitr)] = ciestim3(tmp.(eval_score).pattern);
                        [res.(eval_score).iden_acc.ci(:,sbjitr),res.(eval_score).iden_acc.mu(:,sbjitr)] = ciestim3(tmp.(eval_score).iden_acc*100);
                        profile_acc = tmp.(eval_score).profile;
                        for roitr = length(roiSets):-1:1
                            roiname = roiSets{roitr}{1};
                            useSet = roiSets{roitr}{2};
                            exSet = roiSets{roitr}{3};
                            inVoxIdx = any(metainf.roiind_value(ismember(metainf.roiname,useSet),:),1);
                            outVoxIdx = any(metainf.roiind_value(ismember(metainf.roiname,exSet),:),1);
                            res.roinames{roitr} = roiname;
                            switch roiname
                                case 'WB'
                                    res.(eval_score).roi_acc{roitr,sbjitr} = profile_acc;
                                otherwise
                                    res.(eval_score).roi_acc{roitr,sbjitr} = profile_acc(inVoxIdx & ~outVoxIdx);
                            end
                        end
                    case {'r_cv_best'}
                        [res.(eval_score).profile.ci(:,sbjitr),res.(eval_score).profile.mu(:,sbjitr)] = ciestim3(tmp.(eval_score).profile,2);
                        [res.(eval_score).pattern.ci(:,sbjitr),res.(eval_score).pattern.mu(:,sbjitr)] = ciestim3(tmp.(eval_score).pattern,2);
                        [res.(eval_score).iden_acc.ci(:,sbjitr),res.(eval_score).iden_acc.mu(:,sbjitr)] = ciestim3(tmp.(eval_score).iden_acc*100,2);
                        profile_acc = tmp.(eval_score).profile;
                        bestLayerIdx = tmp.(eval_score).bestLayerIdx;
                        for roitr = length(roiSets):-1:1
                            roiname = roiSets{roitr}{1};
                            useSet = roiSets{roitr}{2};
                            exSet = roiSets{roitr}{3};
                            inVoxIdx = any(metainf.roiind_value(ismember(metainf.roiname,useSet),:),1);
                            outVoxIdx = any(metainf.roiind_value(ismember(metainf.roiname,exSet),:),1);
                            res.roinames{roitr} = roiname;
                            switch roiname
                                case 'WB'
                                    res.(eval_score).roi_acc{roitr,sbjitr} = profile_acc';
                                    res.(eval_score).roi_best_layers{roitr,sbjitr} = bestLayerIdx';
                                otherwise
                                    res.(eval_score).roi_acc{roitr,sbjitr} = profile_acc(:,inVoxIdx & ~outVoxIdx)';
                                    res.(eval_score).roi_best_layers{roitr,sbjitr} = bestLayerIdx(:,inVoxIdx & ~outVoxIdx)';
                            end
                        end
                end
            end
            fprintf('%s:%s:%s\n',sbj,modelType,dataType)
            tims
        end
        fprintf('Save:%s\n',saveFname)
        save(saveFname,'res','-v7.3');
        res_enc.(modelName) = res;
    end
end
