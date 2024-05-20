function mcap_summary_decoding_draw_idenacc(res,p)
% draw results of identification accuracy (e.g., Fig 2e)
eval(structout(p,'p'))

%% draw results
% display settings
c = 10;
r = 4;
[r,c,o] = setrc2(r*c,'ltr',[r,c],[0,2]);
fsize = 5;
msize = 2.5;

scoreTypes = {'featcorr','BLEU','METEOR','ROUGE','CIDEr','F1','R','P'};
nclass = [2,5,10:10:100];
ax = [2,20,50,100];


vars = {roiTypes,dataTypes};
C.cond_names = {'roiType','dataType'};
C.cond_list = generateCombinations(vars);

for cix = 1:length(C.cond_list)
    % set conditions
    for cixx = 1:length(C.cond_list{cix})
        eval(sprintf('%s = C.cond_list{cix}{cixx};',C.cond_names{cixx}))
    end
    
    
    mlmName = strrep(mlmType,'-','_');
    lmName = strrep(lmType,'-','_');
    
    % initialize
    close all
    h = hffigure;
    cnt = 0;
    
    rs = res.(dataType).(mlmName).(lmName).(roiType);
    draw_suffix = sprintf('%s-%s-%s-%s',dataType,mlmName,lmName,roiType);
    
    clear cis mus
    for sbjitr = 1:length(sbjID)
        sbj = sbjID{sbjitr};
        cnt = cnt+1;
        subplottight(r,c,o(cnt),0.15);
        cntleg = 0;
        clear mus_all cis_all ps_all
        for scoritr = length(scoreTypes):-1:1
            scoreType = scoreTypes{scoritr};
            
            cntleg = cntleg+1;
            switch scoreType
                case 'featcorr'
                    mu = rs.featcorr_mciden_mu_mean{sbjitr};
                    ci = rs.featcorr_mciden_ci_mean{sbjitr};
                case 'BLEU'
                    mu = rs.bleu_mciden_mu{sbjitr};
                    ci = rs.bleu_mciden_ci{sbjitr};
                case 'METEOR'
                    mu = rs.meteor_mciden_mu{sbjitr};
                    ci = rs.meteor_mciden_ci{sbjitr};
                case 'ROUGE'
                    mu = rs.rouge_mciden_mu{sbjitr};
                    ci = rs.rouge_mciden_ci{sbjitr};
                case 'CIDEr'
                    mu = rs.cider_mciden_mu{sbjitr};
                    ci = rs.cider_mciden_ci{sbjitr};
                case 'F1'
                    mu = rs.F1_mciden_mu_mean{sbjitr};
                    ci = rs.F1_mciden_ci_mean{sbjitr};
                case 'R'
                    mu = rs.R_mciden_mu_mean{sbjitr};
                    ci = rs.R_mciden_ci_mean{sbjitr};
                case 'P'
                    mu = rs.P_mciden_mu_mean{sbjitr};
                    ci = rs.P_mciden_ci_mean{sbjitr};
                    
            end
            mus_all{scoritr} = mu;
            cis_all{scoritr} = ci;
        end
        
        for scoritr = 1:length(scoreTypes)
            mu = mus_all{scoritr};
            ci = cis_all{scoritr};
            hh = bandplot3(nclass,mu',ci',plotColor(scoritr,length(scoreTypes),'rainbow'));
            AnnotationOff(hh);
            hold on
            hh = plot(nclass,mu',plotMarker2(scoritr),'Color',plotColor(scoritr,length(scoreTypes),'rainbow')*0.8,'MarkerSize',msize);
            set(gca,'FontSize',fsize)
        end
        plot(nclass,100./nclass,'--k')
        axname(ax,1,ax)
        xlim([-5,nclass(end)+5])
        ylim([0,100])
        hh = hline(20:20:100,'-k');
        set(hh,'Color',[1,1,1]*0.8)
        ylabel('Identiciation accuracy (%)')
        title(sprintf('%s',sbj));
        slegend(scoreTypes);
        ffine(h)
        
        % figure adjust
        cnt = cnt+0;
    end
    
    suptitle(sprintf('Video identification accuracy via generated text :%s',draw_suffix));
    savname = [figdir,'/identification_',draw_suffix,'.pdf'];
    setdir(fileparts(savname));
    fprintf([savprint(h,savname),'\n']);
    
end
close all


%%