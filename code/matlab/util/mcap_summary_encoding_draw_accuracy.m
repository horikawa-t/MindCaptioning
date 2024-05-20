function mcap_summary_encoding_draw_accuracy(res,p)
% draw results of scatter (Fig 3d,e)
eval(structout(p,'p'))

%% draw results
% display settings
c = 5;
r = 6;
[r,c,o] = setrc2(r* c,'ltr',[r,c],[1,1]);
fsize = 5;
msize = 1;
lwidth = 0.2;

colname = 'rainbow';


close all
h = hffigure;
cnt = 0;
draw_suffix = sprintf('');

rang = 0:0.1:0.4;

model1 = strrep(modelTypes{2},'-','_');
model2 = strrep(modelTypes{1},'-','_');
res1 = res.(model1).r_optim.roi_acc;
res2 = res.(model2).r_optim.roi_acc;
mRoiSets = merge(roiSets);

usedRois = {'V1','V2','V3','V4','Place','Word','Motion','Object','Face','Body','Language'};
usedRoiNames = {'V1','V2','V3','V4','Place','Word','Motion','Object','Face','Body','Lang.'};
[ism,roiIdx] = ismember(usedRois,mRoiSets(:,1));
nrois = length(roiIdx);

clear parity_disp ses pvals
for sbjitr = 1:length(sbjID)
    sbj = sbjID{sbjitr};
    for roitr = length(roiIdx):-1:1
        acc1 = res1{roiIdx(roitr),sbjitr};
        acc2 = res2{roiIdx(roitr),sbjitr};
        [slopeDR,int,st] = demingRegression(acc1,acc2,1,1,1,[],100);
        parity_disp{roitr}(sbjitr) = rad2deg(atan(slopeDR))-45;
        ses{roitr}(sbjitr) = st.se;
        pvals{roitr}(sbjitr) = st.pval;
    end
end


% compare TSF vs DL
cols = cmap3('rb22');
cnt = cnt+1;
subplottight(r,c,o(cnt),0.15);
[ci1,mu1] = cellfun(@ciestim3,res1(roiIdx,:));
bandebarplot(num2cell(mu1,2),num2cell(ci1,2),cols{2},'MarkerSize',msize,'LineWidth',lwidth);
hold on
[ci2,mu2] = cellfun(@ciestim3,res2(roiIdx,:));
bandebarplot(num2cell(mu2,2),num2cell(ci2,2),cols{1},'MarkerSize',msize,'LineWidth',lwidth);
set(gca,'FontSize',fsize)
hline(0,'-k')
xlabel('Area')
ylabel(sprintf('Correlation coefficient'))
ylim([rang(1),rang(end)])
hh = hline(rang,'-k');
set(hh,'Color',[1,1,1]*0.8)
axname(usedRoiNames)
axname(rang,2,rang)
xticklabel_rotate([],45);
title(sprintf('Accuracy:%s(blue) vs. %s(red)',model1,model2))
set(gca, 'Box', 'off' );
set(gca, 'TickDir', 'out');

% slope
cnt = cnt+1;
subplottight(r,c,o(cnt),0.15);
bandebarplot(parity_disp,ses,plotColors(1:nrois,nrois,colname,1),'MarkerSize',msize);
set(gca,'FontSize',fsize)
xlabel('Area')
ylabel(sprintf('Deviation from parity\n(slope angle-45)'))
ylim([-16,16])
hh = hline(-16:4:16,'-k');
set(hh,'Color',[1,1,1]*0.8)
hline(0,'-k')
axname(usedRoiNames)
axname(-16:8:16,2,-16:8:16)
xticklabel_rotate([],45);
title(sprintf('Slope:%s(<0) vs. %s(>0)',model1,model2))
set(gca, 'Box', 'off' );
set(gca, 'TickDir', 'out');

suptitle(sprintf('Summary of encoding accuracy for individual ROIs :%s',draw_suffix));
savname = [figdir,'/meanacc_angle_',draw_suffix,'.pdf'];
setdir(fileparts(savname));
fprintf([savprint(h,savname),'\n']);

close all
%%


%%