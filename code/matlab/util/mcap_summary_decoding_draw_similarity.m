function mcap_summary_decoding_draw_similarity(res,p)
% draw results of similarity scores (e.g., Fig 2d)
eval(structout(p,'p'))

%% draw results
% display settings
c = 5;
r = 4;
[r,c,o] = setrc2(r*c,'ltr',[r,c],[1]);
fsize = 5;
msize = 2;

scoreTypes = {'featcorr','BLEU','METEOR','ROUGE','CIDEr','F1','R','P'};
evalTypes = {'raw','cohend'};
roiType = 'WB';


mlmName = strrep(mlmType,'-','_');
lmName = strrep(lmType,'-','_');

close all
h = hffigure;
cnt = 0;
draw_suffix = sprintf('%s_%s_%s_%s_%s_%s_%s',mlmType,lmType,roiType);
for evalitr = 1:length(evalTypes)
    evalType = evalTypes{evalitr};
    for datitr = 1:length(dataTypes)
        dataType = dataTypes{datitr};
        rs = res.(dataType).(mlmName).(lmName).(roiType);
        
        cnt = cnt+1;
        subplottight(r,c,o(cnt),0.15);
        clear mu_alls ci_alls muf_alls cif_alls
        for scoritr = length(scoreTypes):-1:1
            scoreType = scoreTypes{scoritr};
            clear legNames
            clear mu_all ci_all muf_all cif_all
            for sbjitr = length(sbjID):-1:1
                switch scoreType
                    case 'featcorr'
                        scores = rs.featcorr_true_max;
                        scores_f = rs.featcorr_false_max;
                        rang = -0.1:0.1:0.6;
                    case 'BLEU'
                        scores = rs.bleu_true;
                        scores_f = rs.bleu_false;
                        rang = -0.05:0.05:0.25;
                    case 'METEOR'
                        scores = rs.meteor_true;
                        scores_f = rs.meteor_false;
                        rang = -0.1:0.1:0.4;
                    case 'ROUGE'
                        scores = rs.rouge_true;
                        scores_f = rs.rouge_false;
                        rang = -0.1:0.1:0.4;
                    case 'CIDEr'
                        scores = rs.cider_true;
                        scores_f = rs.cider_false;
                        rang = -0.05:0.05:0.15;
                    case 'F1'
                        scores = rs.F1_true_max;
                        scores_f = rs.F1_false_max;
                        rang = -0.1:0.1:0.6;
                    case 'R'
                        scores = rs.R_true_max;
                        scores_f = rs.R_false_max;
                        rang = -0.1:0.1:0.6;
                    case 'P'
                        scores = rs.P_true_max;
                        scores_f = rs.P_false_max;
                        rang = -0.1:0.1:0.6;
                end
                chanceline = 0;
                
                switch evalType
                    case 'cohend'
                        [mu,ci] = cohend(scores{:,sbjitr},scores_f{:,sbjitr},1,0.99);
                        muf = zeros(size(mu));
                        cif = zeros(size(mu));
                        
                    case 'raw'
                        [ci,mu] = ciestim3(double(scores{:,sbjitr}),1);
                        [cif,muf] = ciestim3(double(scores_f{:,sbjitr}),1);
                        
                end
                ci_all(:,sbjitr) = ci(:,end);
                mu_all(:,sbjitr) = mu(:,end);
                cif_all(:,sbjitr) = cif(:,end);
                muf_all(:,sbjitr) = muf(:,end);
                
            end
            
            mu_alls{scoritr} = mu_all;
            ci_alls{scoritr} = ci_all;
            muf_alls{scoritr} = muf_all;
            cif_alls{scoritr} = cif_all;
        end
        
        bandebarplot(mu_alls,ci_alls,plotColors(1:length(scoreTypes),length(scoreTypes),'rainbow'),'MarkerSize',msize,'LineWidth',0.2);
        hold on
        switch evalType
            case 'cohend'
            otherwise
                bandebarplot(muf_alls,cif_alls,[1,1,1]*0.5,'MarkerSize',msize,'LineWidth',0.2);
        end
        
        set(gca,'FontSize',fsize)
        axname(scoreTypes);
        xlim([0,length(scoreTypes)+1]);
        switch evalType
            case 'raw'
                switch dataType
                    case 'testPerception'
                        rang = -0.1:0.1:0.6;
                    case 'testImagery'
                        rang = -0.1:0.1:0.6;
                end
                axname(rang,2,rang);
                hh = hline(rang(2:end),'k');
                set(hh,'Color',[1,1,1]*0.8)
                ylim([rang(1),rang(end)])
                ylabel('Score')
            case 'cohend'
                switch dataType
                    case 'testPerception'
                        rang = 0:6;
                    case 'testImagery'
                        rang = 0:4;
                end
                ylim([rang(1),rang(end)])
                hh = hline(rang(2:end),'k');
                set(hh,'Color',[1,1,1]*0.8)
                axname(rang,2,rang)
                ylim([rang(1),rang(end)])
                ylabel("Cohen's d")
        end
        xticklabel_rotate([],45)
        hline(chanceline,'-k');
        title(sprintf('%s:%s',dataType,evalType));
        set(gca, 'Box', 'off' ); % here gca means get current axis
        set(gca, 'TickDir', 'out');
        
    end
    % figure adjust
    cnt = cnt+1;
end
suptitle(sprintf('Sentence similarity results:%s',draw_suffix));
savname = [figdir,'/similarity',draw_suffix,'.pdf'];
setdir(fileparts(savname));
fprintf([savprint(h,savname),'\n']);


close all


%%