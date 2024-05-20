function res_all = mcap_summary_decoding_summarize(p)

%%  set parameters
eval(structout(p,'p'))

eval_scores = {'best_cands','videoidx','scores_all'};
iden_params = [2,5,10:10:100];
vars = {dataTypes,roiTypes};
C.cond_names = {'dataType','roiType'};
C.cond_list = generateCombinations(vars);



%%
for cix = 1:length(C.cond_list)
    % set conditions
    for cixx = 1:length(C.cond_list{cix})
        eval(sprintf('%s = C.cond_list{cix}{cixx};',C.cond_names{cixx}))
    end
    
    mlmName = strrep(mlmType,'-','_');
    lmName = strrep(lmType,'-','_');
    
    switch dataType
        case {'testPerception','testImagery'}
            nSamples = 72;
        otherwise
            fprintf('dataType is not valid.\n')
            continue
    end
    saveFname = sprintf('%s/res_summary_%s_mlm_%s_lm_%s_%s.mat',savdir,dataType,mlmType,lmType,roiType);
    setdir(fileparts(saveFname));
    
    if exist(saveFname,'file')
        fprintf('Load:%s\n',saveFname)
        load(saveFname,'res');
        res_all.(dataType).(mlmName).(lmName).(roiType) = res;
    else
        save_flag = 1;
        clear res
        fprintf('Start:%s\n',saveFname)
        for sbjitr = length(sbjID):-1:1
            sbj = sbjID{sbjitr};
            
            for ix = nSamples:-1:1
                fname = sprintf('%s/%s/mlm_%s/lm_%s/%s/%s/res/res_samp%04d.mat',savdir,dataType,mlmType,lmType,sbj,roiType,ix);
                try
                    tmp = load(fname,eval_scores{:});
                catch
                    fprintf('Failed to load results.\n')
                    save_flag = 0;
                    break
                end
                
                
                for eitr = 1:length(eval_scores)
                    eval_score = eval_scores{eitr};
                    if ~isfield(tmp,eval_score)
                        continue
                    end
                    switch eval_score
                        case {'scores_all'}
                            res.(eval_score){1,sbjitr}(ix,:) = tmp.(eval_score);
                            res.scores_best{1,sbjitr}(ix,:) = tmp.(eval_score)(end);
                        case {'videoidx'}
                            res.(eval_score){1,sbjitr}(ix,:) = tmp.(eval_score);
                            res.(eval_score){1,sbjitr}(ix,:) = tmp.(eval_score);
                        case 'best_cands'
                            for nitr = size(tmp.(eval_score),1):-1:1
                                res.gentext{1,sbjitr}{ix,nitr} = strrep(tmp.(eval_score)(nitr,:),'  ','');
                            end
                        otherwise
                            continue
                    end
                end
                
            end
            
            % load similarity scores to both correct and incorrect reference to perform multi-class identification
            metricTypes = {'featcorr','bleu','meteor','rouge','cider','F1','R','P'};
            fname = sprintf('%s/%s/mlm_%s/lm_%s/%s/%s/res/res_summary.mat',savdir,dataType,mlmType,lmType,sbj,roiType);
            tmp = load(fname);
            for metitr = 1:length(metricTypes)
                metricType = metricTypes{metitr};
                switch metricType
                    case {'featcorr'}
                        truescores_mean = tmp.(['scores_gen2ref_eval_true_means'])';
                        falsescores_mean = tmp.(['scores_gen2ref_eval_false_means']);
                        truescores_max = tmp.(['scores_gen2ref_eval_true_maxs'])';
                        falsescores_max = tmp.(['scores_gen2ref_eval_false_maxs']);
                        
                        res.([metricType,'_true_mean']){1,sbjitr} = truescores_mean;
                        res.([metricType,'_true_max']){1,sbjitr} = truescores_max;
                        res.([metricType,'_false_mean']){1,sbjitr} = mean(falsescores_mean,2);
                        res.([metricType,'_false_max']){1,sbjitr} = mean(falsescores_max,2);
                        
                        rand('seed',42)
                        [ci,mu] = ciestim3(fmcidentification(truescores_mean,falsescores_mean,iden_params,100));
                        res.([metricType,'_mciden_mu_mean']){1,sbjitr} = mu*100;
                        res.([metricType,'_mciden_ci_mean']){1,sbjitr} = ci*100;
                        rand('seed',42)
                        [ci,mu] = ciestim3(fmcidentification(truescores_max,falsescores_max,iden_params,100));
                        res.([metricType,'_mciden_mu_max']){1,sbjitr} = mu*100;
                        res.([metricType,'_mciden_ci_max']){1,sbjitr} = ci*100;
                        fprintf('video iden acc[%s][100]: cr=%.2f%%[max], cr=%.2f%%[mean]\n',metricType,res.([metricType,'_mciden_mu_max']){1,sbjitr}(end),res.([metricType,'_mciden_mu_mean']){1,sbjitr}(end))
                        
                    case {'bleu','meteor','rouge','cider'}
                        switch metricType
                            case {'bleu','meteor','rouge'}
                                truescores = tmp.([metricType,'_scores_true_refs'])';
                            case 'cider'
                                truescores = tmp.([metricType,'_scores_true_refs']);
                        end
                        falsescores = tmp.([metricType,'_scores_false_refs']);
                        
                        res.([metricType,'_true']){1,sbjitr} = truescores;
                        res.([metricType,'_false']){1,sbjitr} = mean(falsescores,2);
                        
                        rand('seed',42)
                        [ci,mu] = ciestim3(fmcidentification(truescores,falsescores,iden_params,100));
                        res.([metricType,'_mciden_mu']){1,sbjitr} = mu*100;
                        res.([metricType,'_mciden_ci']){1,sbjitr} = ci*100;
                        fprintf('video iden acc[%s][100]: cr=%.2f%%\n',metricType,res.([metricType,'_mciden_mu']){1,sbjitr}(end))
                        
                    case {'F1','R','P'}
                        truescores_mean = mean(tmp.([metricType,'_scores_true_refs']),2);
                        falsescores_mean = tmp.([metricType,'_scores_false_refs_mean']);
                        truescores_max = max(tmp.([metricType,'_scores_true_refs']),[],2);
                        falsescores_max = tmp.([metricType,'_scores_false_refs_max']);
                        res.([metricType,'_true_mean']){1,sbjitr} = truescores_mean;
                        res.([metricType,'_true_max']){1,sbjitr} = truescores_max;
                        res.([metricType,'_false_max']){1,sbjitr} = mean(falsescores_mean,2);
                        res.([metricType,'_false_mean']){1,sbjitr} = mean(falsescores_max,2);
                        
                        rand('seed',42)
                        [ci,mu] = ciestim3(fmcidentification(truescores_mean,falsescores_mean,iden_params,100));
                        res.([metricType,'_mciden_mu_mean']){1,sbjitr} = mu*100;
                        res.([metricType,'_mciden_ci_mean']){1,sbjitr} = ci*100;
                        rand('seed',42)
                        [ci,mu] = ciestim3(fmcidentification(truescores_max,falsescores_max,iden_params,100));
                        res.([metricType,'_mciden_mu_max']){1,sbjitr} = mu*100;
                        res.([metricType,'_mciden_ci_max']){1,sbjitr} = ci*100;
                        
                        fprintf('video iden acc[%s][100]: cr=%.2f%%[max], cr=%.2f%%[mean]\n',metricType,res.([metricType,'_mciden_mu_max']){1,sbjitr}(end),res.([metricType,'_mciden_mu_mean']){1,sbjitr}(end))
                end
            end
            
            fprintf('%s:%s:%s:%s\n',sbj,mlmType,lmType,dataType)
            tims
        end
        
        if save_flag
            fprintf('Save:%s\n',saveFname)
            save(saveFname,'res','-v7.3');
            res_all.(dataType).(mlmName).(lmName).(roiType) = res;
        end
    end
end
%%

