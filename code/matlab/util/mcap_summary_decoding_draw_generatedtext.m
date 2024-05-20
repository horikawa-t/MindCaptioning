function mcap_summary_decoding_draw_generatedtext(res,p)
% draw results of generated text (Fig 2b)
eval(structout(p,'p'))

%% get movie frames
try
    load(sprintf('%s/data/misc/video_thumbnails.mat',rootPath))
catch
    error("Can't find video_thumbnails.mat. Please download the file and from our repository and put it under ./data/misc/.")
end

vars = {dataTypes,roiTypes};
C.cond_names = {'dataType','roiType'};
C.cond_list = generateCombinations(vars);

%% visualize images
% figure settings
r = 18+2;
c = 4;
wcanvas = 2000;
lheight = 36;
fsize = 2.5;

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
    end
    [r,c,o] = setrc2(r*c,'ltr',[r,c],[1,0]);
    close all
    h = hffigure;
    
    rs = res.(dataType).(mlmName).(lmName).(roiType);
    draw_suffix = sprintf('%s_%s_%s_%s',mlmName,lmType,dataType,roiType);
    
    cnt = 0;
    cntsamp = 0;
    for ix = 1:nSamples
        cnt = cnt + 1;
        subplottight(r,c,o(cnt),0.05);
        [h0,w,d] = size(Is{ix});
        I2 = [Is{ix},ones(h0,wcanvas,d)*255];
        imagesc(I2);
        axis image off
        
        txt = sprintf('[video:%d]',ix);
        text(w+lheight,5,txt,'FontSize',fsize)
        cntsamp = cntsamp+1;
        cntvar = 0;
        for sbjitr = 1:length(sbjID)
            cntvar = cntvar+1;
            caps = rs.gentext{1,sbjitr};
            txt = sprintf('S%d: %s',sbjitr,caps{cntsamp,end});
            text(w+lheight,lheight*cntvar,txt,'FontSize',fsize)
        end
    end
suptitle(sprintf('List of generated captions:%s',draw_suffix));
savname = [figdir,'/ListOfGeneratedDescriptions',draw_suffix,'.pdf'];
setdir(fileparts(savname));
fprintf([savprint(h,savname),'\n']);
end

close all

%%