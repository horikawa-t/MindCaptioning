function mcap_summary_encoding_draw_scatter(res,p)
% draw results of scatter (Fig 3c)
eval(structout(p,'p'))

%% draw results
% display settings
c = 11;
r = 5;
fsize = 6;

drawSbj = [sbjID,'Pooled'];
drawSbj = {'Pooled'};
usedRois = {'LVC','V4','Place','Word','Motion','Object','Face','Body','Language'};
mRoiSets = merge(roiSets);

for sbjitr = 1:length(drawSbj)
    sbj = drawSbj{sbjitr};
    
    close all
    h = hffigure;
    cnt = 0;
    draw_suffix = sprintf('%s',sbj);
    rang = -0.2:0.1:0.8;
    
    model1 = strrep(modelTypes{2},'-','_');
    model2 = strrep(modelTypes{1},'-','_');
    res1 = res.(model1).r_optim.roi_acc;
    res2 = res.(model2).r_optim.roi_acc;
    [ism,roiIdx] = ismember(usedRois,mRoiSets(:,1));
    nrois = length(roiIdx);
    cols = plotColors(1:11,11,'rainbow_r');
    cols = cols([2,4:11]);
    
    [r,c,o] = setrc2(r*c,'ltr',[r,c],[1,3]);
    
    angleX = cell(nrois,1);
    angleY = cell(nrois,1);
    pss = cell(nrois,1);
    for roiidx = 1:nrois
        switch sbj
            case 'Pooled'
                acc1 = merge(res1(roiIdx(roiidx),:));
                acc2 = merge(res2(roiIdx(roiidx),:));
            otherwise
                acc1 = res1{roiIdx(roiidx),sbjitr};
                acc2 = res2{roiIdx(roiidx),sbjitr};
                
        end
        %
        cnt = cnt+1;
        subplottight(r,c,o(cnt),0.15);
        nbin = 200;
        bandwidth = 0.01;
        colname = 'magma';
        [hh,ps] = dscatterImage(acc1,acc2,'bins',nbin,'rangx',rang,'rangy',rang,'bandwidth',bandwidth,'colname',colname,'drawtype','density');
        pss{roiidx} = ps;
        axis square
        odlinetight(axis)
        set(gca,'FontSize',fsize)
        xlabel(sprintf('%s',model1))
        ylabel(sprintf('%s',model2))
        
        vline(ps.to0shiftx,'-k')
        hline(ps.to0shiftx,'-k')
        vline(ps.toMeanshiftx,'--r')
        hline(ps.toMeanshifty,'--r')
        
        [slopeDR,int,st,xymu] = demingRegression(acc1,acc2,1,1,1,[],100); % assume no intercept for model comparison
        hold on
        rangslope = rang*ps.scalex+ps.to0shiftx;
        plot(rangslope,(rang*ps.scalex)*slopeDR+ps.to0shiftx+xymu(2)*ps.scaley-ps.toMeanshifty+ps.to0shifty,'-','Color',cols{roiidx})
        angleX{roiidx} = rang;
        angleY{roiidx} = slopeDR;
        angle = rad2deg(atan(slopeDR));
        axname(rang(1:2:end),1,rang(1:2:end)*ps.scalex+ps.to0shiftx)
        axname(rang(1:2:end),2,rang(1:2:end)*ps.scaley+ps.to0shifty)
        text(0.7*ps.scalex+ps.to0shiftx,-0.1*ps.scaley+ps.to0shifty,usedRois{roiidx},'HorizontalAlignment','right','FontSize',fsize)
        title(sprintf('%s:a=%.1f;p=%.5f',usedRois{roiidx},angle,st.pval))
        set(gca, 'Box', 'off' );
        set(gca, 'TickDir', 'out');
        
    end
    
    
    suptitle(sprintf('Encoding accuracy for individual ROIs :%s',draw_suffix));
    savname = [figdir,'/scatter_',draw_suffix,'.pdf'];
    setdir(fileparts(savname));
    fprintf([savprint(h,savname),'\n']);
end

close all


%%