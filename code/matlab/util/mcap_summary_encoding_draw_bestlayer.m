function mcap_summary_encoding_draw_bestlayer(res,p)
% draw results of scatter (Fig 3f)
eval(structout(p,'p'))

%% draw results
% display settings
c = 6;
r = 6;
[r,c,o] = setrc2(r*c,'ltd',[r,c],[2]);
addmargin = 1;
fsize = 5;
msize = 1;

mRoiSets = merge(roiSets);
colname = 'rainbow';

close all
h = hffigure;
cnt = 0;
draw_suffix = sprintf('');


for modelitr = length(modelTypes):-1:1
    model = strrep(modelTypes{modelitr},'-','_');
    bl = res.(model).r_cv_best.roi_best_layers;
    acc = res.(model).r_optim.roi_acc;
    nlayers = size(res.(model).r_cv_best.profile.mu,1);
    
    usedRois = {'LVC','HVC','Language'};
    usedRoiNames = {'LVC','HVC','Lang.'};
    [ism,roiIdx] = ismember(usedRois,mRoiSets(:,1));
    nrois = length(roiIdx);
    
    clear mu ci ps
    for sbjitr = length(sbjID):-1:1
        for roitr = length(roiIdx):-1:1
            th = -1;
            highaccVoxelIdx = acc{roiIdx(roitr),sbjitr} > th;
            [ci{roitr}(sbjitr),mu{roitr}(sbjitr)] = ciestim3(asvector(bl{roiIdx(roitr),sbjitr}(highaccVoxelIdx,:))./nlayers);
        end
    end
    
    cnt = cnt+1;
    subplottight(r,c,o(cnt),0.15);
    cols = plotColors(1:11,11,colname,1);
    cols = cols([1,8,11]);
    bandebarplot(mu,ci,cols,'MarkerSize',msize);
    
    set(gca,'FontSize',fsize)
    xlim([0,max([9,nrois])+1])
    switch model
        case 'timesformer'
            rang = 0.4:0.1:0.8;
            ylim([0.4,0.8])
            hh = hline(0.4:0.1:0.8,'-k');
        case 'deberta_large'
            rang = 0.4:0.05:0.6;
            ylim([rang(1),rang(end)])
            hh = hline(rang,'-k');
    end
    
    set(hh,'Color',[1,1,1]*0.8)
    axname(usedRoiNames,1,1:length(usedRoiNames))
    axname(rang,2,rang)
    xticklabel_rotate([],45)
    xlabel('Area')
    ylabel(sprintf('Best layer(relative depth)'))
    title(sprintf('%s',model))
    set(gca, 'Box', 'off' );
    set(gca, 'TickDir', 'out');
    
end
suptitle(sprintf('Best layers for individual ROIs :%s',draw_suffix));
savname = [figdir,'/bestLayers_',draw_suffix,'.pdf'];
setdir(fileparts(savname));
fprintf([savprint(h,savname),'\n']);


close all

%%


%%